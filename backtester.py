#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting module for testing trading strategies on historical data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools

from data_fetcher import DataFetcher
from strategies import StrategyFactory
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

class Backtester:
    """
    Class for backtesting trading strategies on historical data.
    """
    def __init__(self, config=None):
        """
        Initialize the Backtester with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.risk_manager = RiskManager(config)
        
        # Default backtesting parameters if config not provided
        if not config:
            self.initial_capital = 100000
            self.commission_pct = 0.001  # 0.1% commission
            self.slippage_pct = 0.0005   # 0.05% slippage
        else:
            self.initial_capital = getattr(config, 'INITIAL_CAPITAL', 100000)
            self.commission_pct = getattr(config, 'COMMISSION_PCT', 0.001)
            self.slippage_pct = getattr(config, 'SLIPPAGE_PCT', 0.0005)
    
    def run_backtest(self, strategy_name, symbols, start_date, end_date, 
                     timeframe='1d', use_stops=True, data_source='yfinance'):
        """
        Run a backtest for a given strategy, symbols, and date range.
        
        Args:
            strategy_name (str): Name of the strategy to test
            symbols (list): List of symbols to test
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            timeframe (str): Timeframe for the data
            use_stops (bool): Whether to use stop-loss and take-profit
            data_source (str): Source of the data ('yfinance' or 'alpaca')
            
        Returns:
            dict: Dictionary with backtest results
        """
        logger.info(f"Running backtest for {strategy_name} on {symbols} from {start_date} to {end_date}")
        
        # Create strategy instance
        strategy = StrategyFactory.create_strategy(strategy_name, self.config)
        
        # Initialize result tracking
        results = {
            'strategy': strategy_name,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': 0,
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'trades': [],
            'positions': [],
            'equity_curve': pd.DataFrame(),
            'performance_by_symbol': {}
        }
        
        # Get historical data for all symbols
        data_dict = self.data_fetcher.get_multiple_symbols_data(
            symbols, timeframe, None, start_date, end_date, data_source
        )
        
        # Check if data was retrieved successfully
        if not data_dict:
            logger.error("No data retrieved for any symbols. Backtest failed.")
            return results
        
        # Prepare for backtesting each symbol
        all_positions = []
        all_trades = []
        all_equity_curves = {}
        
        # Run backtest for each symbol
        for symbol, data in data_dict.items():
            if data is None or data.empty:
                logger.warning(f"No data retrieved for {symbol}. Skipping.")
                continue
            
            # Add indicators based on strategy
            data = strategy.add_indicators(data)
            
            # Generate signals
            signal_data = strategy.generate_signals(data)
            
            # Run the trading simulation for this symbol
            symbol_results = self._run_symbol_backtest(
                symbol, signal_data, self.initial_capital / len(symbols), use_stops
            )
            
            # Add symbol results to overall results
            all_positions.extend(symbol_results['positions'])
            all_trades.extend(symbol_results['trades'])
            all_equity_curves[symbol] = symbol_results['equity_curve']
            
            # Store symbol-specific performance
            results['performance_by_symbol'][symbol] = {
                'total_return': symbol_results['total_return'],
                'max_drawdown': symbol_results['max_drawdown'],
                'sharpe_ratio': symbol_results['sharpe_ratio'],
                'num_trades': len(symbol_results['trades']),
                'win_rate': symbol_results['win_rate']
            }
        
        # Combine equity curves
        if all_equity_curves:
            # Reindex to common date range and forward fill
            dfs = [ec.set_index('date')['equity'] for ec in all_equity_curves.values()]
            combined = pd.concat(dfs, axis=1)
            combined.columns = list(all_equity_curves.keys())
            combined = combined.fillna(method='ffill')
            
            # Calculate portfolio equity
            portfolio_equity = combined.sum(axis=1)
            equity_curve = pd.DataFrame({
                'date': portfolio_equity.index,
                'equity': portfolio_equity.values
            })
            
            # Calculate returns
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            equity_curve['cumulative_returns'] = (1 + equity_curve['returns']).cumprod()
            
            # Calculate drawdown
            equity_curve['peak'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak']
            
            results['equity_curve'] = equity_curve
            results['final_capital'] = equity_curve['equity'].iloc[-1] if not equity_curve.empty else self.initial_capital
            results['total_return'] = (results['final_capital'] / self.initial_capital) - 1
            
            # Calculate other metrics
            if len(equity_curve) > 1:
                # Annual return
                days = (equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days
                if days > 0:
                    results['annual_return'] = ((1 + results['total_return']) ** (365 / days)) - 1
                
                # Max drawdown
                results['max_drawdown'] = equity_curve['drawdown'].min()
                
                # Sharpe ratio (assuming risk-free rate of 0)
                daily_returns = equity_curve['returns'].dropna()
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    results['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        
        # Store trades and positions
        results['trades'] = all_trades
        results['positions'] = all_positions
        
        # Calculate win rate
        if all_trades:
            winning_trades = sum(1 for trade in all_trades if trade['pnl'] > 0)
            results['win_rate'] = winning_trades / len(all_trades)
        else:
            results['win_rate'] = 0
        
        logger.info(f"Backtest completed. Total return: {results['total_return']:.2%}, Sharpe ratio: {results['sharpe_ratio']:.2f}")
        
        return results
    
    def _run_symbol_backtest(self, symbol, data, initial_capital, use_stops=True):
        """
        Run backtest for a single symbol.
        
        Args:
            symbol (str): Symbol to backtest
            data (pd.DataFrame): DataFrame with price and signal data
            initial_capital (float): Initial capital for this symbol
            use_stops (bool): Whether to use stop-loss and take-profit
            
        Returns:
            dict: Dictionary with symbol backtest results
        """
        # Initialize tracking variables
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_date = None
        stop_loss = None
        take_profit = None
        shares = 0
        
        trades = []
        positions = []
        equity_curve = []
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Iterate through each bar
        for i, (date, row) in enumerate(data.iterrows()):
            # Track equity for equity curve
            current_value = capital
            if position != 0:
                current_value = capital + (shares * row['close'] * position)
            
            equity_curve.append({
                'date': date,
                'equity': current_value
            })
            
            # Check stop loss and take profit if in a position
            if position != 0 and use_stops:
                # Check if stop loss hit
                if stop_loss is not None:
                    if (position == 1 and row['low'] <= stop_loss) or \
                       (position == -1 and row['high'] >= stop_loss):
                        # Execute at stop price
                        exit_price = stop_loss
                        
                        # Calculate P&L
                        if position == 1:
                            pnl = (exit_price - entry_price) * shares
                        else:  # short position
                            pnl = (entry_price - exit_price) * shares
                        
                        # Account for commission
                        commission = exit_price * shares * self.commission_pct
                        pnl -= commission
                        
                        # Update capital
                        capital += pnl
                        
                        # Record trade
                        trades.append({
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'exit_date': date,
                            'position': 'long' if position == 1 else 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'pnl': pnl,
                            'return': pnl / (entry_price * shares),
                            'exit_reason': 'stop_loss'
                        })
                        
                        # Reset position
                        position = 0
                        stop_loss = None
                        take_profit = None
                        continue
                
                # Check if take profit hit
                if take_profit is not None:
                    if (position == 1 and row['high'] >= take_profit) or \
                       (position == -1 and row['low'] <= take_profit):
                        # Execute at take profit price
                        exit_price = take_profit
                        
                        # Calculate P&L
                        if position == 1:
                            pnl = (exit_price - entry_price) * shares
                        else:  # short position
                            pnl = (entry_price - exit_price) * shares
                        
                        # Account for commission
                        commission = exit_price * shares * self.commission_pct
                        pnl -= commission
                        
                        # Update capital
                        capital += pnl
                        
                        # Record trade
                        trades.append({
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'exit_date': date,
                            'position': 'long' if position == 1 else 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'pnl': pnl,
                            'return': pnl / (entry_price * shares),
                            'exit_reason': 'take_profit'
                        })
                        
                        # Reset position
                        position = 0
                        stop_loss = None
                        take_profit = None
                        continue
            
            # Get current signal
            signal = row['signal']
            
            # Process signals for entry/exit
            if position == 0:  # No position
                if signal == 1:  # Buy signal
                    # Calculate entry price with slippage
                    entry_price = row['close'] * (1 + self.slippage_pct)
                    entry_date = date
                    
                    # Calculate position size
                    shares = int(capital / entry_price)
                    
                    if shares > 0:
                        # Calculate stop loss and take profit
                        if use_stops:
                            if 'atr' in row and not np.isnan(row['atr']):
                                stop_loss = self.risk_manager.calculate_stop_loss(
                                    entry_price, 'long', row['atr']
                                )
                                take_profit = self.risk_manager.calculate_take_profit(
                                    entry_price, 'long', risk_reward_ratio=2.0
                                )
                            else:
                                stop_loss = self.risk_manager.calculate_stop_loss(
                                    entry_price, 'long'
                                )
                                take_profit = self.risk_manager.calculate_take_profit(
                                    entry_price, 'long'
                                )
                        
                        # Account for commission
                        commission = entry_price * shares * self.commission_pct
                        capital -= commission
                        
                        # Update position
                        position = 1
                        
                        # Record position
                        positions.append({
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'position': 'long',
                            'entry_price': entry_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                
                elif signal == -1:  # Sell short signal
                    # Calculate entry price with slippage
                    entry_price = row['close'] * (1 - self.slippage_pct)
                    entry_date = date
                    
                    # Calculate position size
                    shares = int(capital / entry_price)
                    
                    if shares > 0:
                        # Calculate stop loss and take profit
                        if use_stops:
                            if 'atr' in row and not np.isnan(row['atr']):
                                stop_loss = self.risk_manager.calculate_stop_loss(
                                    entry_price, 'short', row['atr']
                                )
                                take_profit = self.risk_manager.calculate_take_profit(
                                    entry_price, 'short', risk_reward_ratio=2.0
                                )
                            else:
                                stop_loss = self.risk_manager.calculate_stop_loss(
                                    entry_price, 'short'
                                )
                                take_profit = self.risk_manager.calculate_take_profit(
                                    entry_price, 'short'
                                )
                        
                        # Account for commission
                        commission = entry_price * shares * self.commission_pct
                        capital -= commission
                        
                        # Update position
                        position = -1
                        
                        # Record position
                        positions.append({
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'position': 'short',
                            'entry_price': entry_price,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
            
            elif position == 1:  # Long position
                if signal == -1:  # Exit signal
                    # Calculate exit price with slippage
                    exit_price = row['close'] * (1 - self.slippage_pct)
                    
                    # Calculate P&L
                    pnl = (exit_price - entry_price) * shares
                    
                    # Account for commission
                    commission = exit_price * shares * self.commission_pct
                    pnl -= commission
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trades.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'position': 'long',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'return': pnl / (entry_price * shares),
                        'exit_reason': 'signal'
                    })
                    
                    # Reset position
                    position = 0
                    stop_loss = None
                    take_profit = None
            
            elif position == -1:  # Short position
                if signal == 1:  # Exit signal
                    # Calculate exit price with slippage
                    exit_price = row['close'] * (1 + self.slippage_pct)
                    
                    # Calculate P&L
                    pnl = (entry_price - exit_price) * shares
                    
                    # Account for commission
                    commission = exit_price * shares * self.commission_pct
                    pnl -= commission
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trades.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'position': 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'return': pnl / (entry_price * shares),
                        'exit_reason': 'signal'
                    })
                    
                    # Reset position
                    position = 0
                    stop_loss = None
                    take_profit = None
        
        # Close any open position at the end of the backtest
        if position != 0:
            # Use the last close price
            exit_price = data['close'].iloc[-1]
            exit_date = data.index[-1]
            
            # Calculate P&L
            if position == 1:
                pnl = (exit_price - entry_price) * shares
            else:  # short position
                pnl = (entry_price - exit_price) * shares
            
            # Account for commission
            commission = exit_price * shares * self.commission_pct
            pnl -= commission
            
            # Update capital
            capital += pnl
            
            # Record trade
            trades.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'position': 'long' if position == 1 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'return': pnl / (entry_price * shares),
                'exit_reason': 'end_of_backtest'
            })
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        total_return = (capital / initial_capital) - 1
        
        # Max drawdown
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio
        daily_returns = equity_df['returns'].dropna()
        sharpe_ratio = 0
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        
        # Win rate
        win_rate = 0
        if trades:
            winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            win_rate = winning_trades / len(trades)
        
        return {
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'trades': trades,
            'positions': positions,
            'equity_curve': equity_df
        }
    
    def optimize_strategy(self, strategy_name, symbols, start_date, end_date, 
                          param_grid, timeframe='1d', metric='sharpe_ratio', n_jobs=-1):
        """
        Optimize a strategy by testing a grid of parameters.
        
        Args:
            strategy_name (str): Name of the strategy to optimize
            symbols (list): List of symbols to test
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            param_grid (dict): Dictionary of parameter grids to test
            timeframe (str): Timeframe for the data
            metric (str): Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            n_jobs (int): Number of parallel jobs. -1 means use all processors.
            
        Returns:
            dict: Dictionary with optimization results
        """
        logger.info(f"Optimizing {strategy_name} on {symbols} from {start_date} to {end_date}")
        
        # Generate all parameter combinations
        param_names = sorted(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        # Track best parameters and results
        best_params = None
        best_metric_value = float('-inf') if metric != 'max_drawdown' else float('inf')
        all_results = []
        
        # Define optimization function for parallel processing
        def evaluate_params(params):
            # Create parameter dictionary
            param_dict = {name: value for name, value in zip(param_names, params)}
            
            # Create config with these parameters
            temp_config = self.config
            if temp_config is None:
                # Create a simple object to hold parameters
                from types import SimpleNamespace
                temp_config = SimpleNamespace()
            
            # Set parameters in config
            for name, value in param_dict.items():
                setattr(temp_config, name.upper(), value)
            
            # Create backtester with this config
            backtester = Backtester(temp_config)
            
            # Run backtest
            results = backtester.run_backtest(
                strategy_name, symbols, start_date, end_date, timeframe, True
            )
            
            # Extract metric
            if metric == 'sharpe_ratio':
                metric_value = results['sharpe_ratio']
            elif metric == 'total_return':
                metric_value = results['total_return']
            elif metric == 'max_drawdown':
                metric_value = results['max_drawdown']
            else:
                raise ValueError(f"Unknown optimization metric: {metric}")
            
            return {
                'params': param_dict,
                'metric': metric,
                'value': metric_value,
                'results': results
            }
        
        # Run optimization in parallel
        with tqdm(total=len(param_combinations), desc="Optimizing") as pbar:
            def update_pbar(*args):
                pbar.update()
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_params)(params) for params in param_combinations
            )
        
        # Find best parameters
        for result in results:
            metric_value = result['value']
            
            # Update best if better (or lower for drawdown)
            if (metric != 'max_drawdown' and metric_value > best_metric_value) or \
               (metric == 'max_drawdown' and metric_value < best_metric_value):
                best_metric_value = metric_value
                best_params = result['params']
                best_results = result['results']
            
            all_results.append(result)
        
        # Sort results
        if metric != 'max_drawdown':
            all_results.sort(key=lambda x: x['value'], reverse=True)
        else:
            all_results.sort(key=lambda x: x['value'])
        
        logger.info(f"Optimization completed. Best {metric}: {best_metric_value}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_metric_value': best_metric_value,
            'best_results': best_results,
            'all_results': all_results
        }
    
    def plot_equity_curve(self, results, title=None, figsize=(12, 8)):
        """
        Plot equity curve from backtest results.
        
        Args:
            results (dict): Backtest results dictionary
            title (str, optional): Plot title
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if results['equity_curve'].empty:
            logger.warning("No equity curve data to plot")
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        equity_curve = results['equity_curve']
        axes[0].plot(equity_curve['date'], equity_curve['equity'], label='Portfolio Value')
        
        # Add initial capital line
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label=f'Initial Capital (${self.initial_capital:,.0f})')
        
        # Configure plot
        if title:
            axes[0].set_title(title)
        else:
            strategy = results['strategy']
            symbols = ', '.join(results['symbols'])
            start = results['start_date']
            end = results['end_date']
            axes[0].set_title(f"{strategy} Strategy Backtest on {symbols} from {start} to {end}")
        
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot drawdown
        axes[1].fill_between(equity_curve['date'], equity_curve['drawdown'], 0, color='r', alpha=0.3)
        axes[1].set_ylabel('Drawdown')
        axes[1].set_xlabel('Date')
        axes[1].grid(True)
        
        # Set y-axis format to percentage
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add key metrics as text
        metrics_text = (
            f"Total Return: {results['total_return']:.2%}\n"
            f"Annual Return: {results['annual_return']:.2%}\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown']:.2%}\n"
            f"Win Rate: {results.get('win_rate', 0):.2%}"
        )
        
        # Place text in the upper left of the equity curve plot
        axes[0].text(
            0.02, 0.95, metrics_text,
            transform=axes[0].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )
        
        plt.tight_layout()
        return fig
    
    def plot_trades(self, results, symbol=None, figsize=(12, 8)):
        """
        Plot trades on a price chart.
        
        Args:
            results (dict): Backtest results dictionary
            symbol (str, optional): Symbol to plot trades for. If None, plots first symbol.
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if not results['trades']:
            logger.warning("No trades to plot")
            return None
        
        # If symbol not specified, use the first symbol in the results
        if symbol is None:
            symbol = results['symbols'][0]
        
        # Filter trades for the specific symbol
        trades = [t for t in results['trades'] if t['symbol'] == symbol]
        
        if not trades:
            logger.warning(f"No trades for symbol {symbol}")
            return None
        
        # Get data for the symbol
        data = self.data_fetcher.get_historical_data(
            symbol, 
            timeframe='1d', 
            start_date=results['start_date'],
            end_date=results['end_date']
        )
        
        if data is None or data.empty:
            logger.warning(f"No price data for symbol {symbol}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price data
        ax.plot(data.index, data['close'], label=f'{symbol} Close Price', color='black', alpha=0.5)
        
        # Plot buy trades
        for trade in trades:
            if trade['position'] == 'long':
                ax.scatter(
                    trade['entry_date'], 
                    trade['entry_price'],
                    marker='^', 
                    color='green', 
                    s=100, 
                    label='Buy'
                )
                ax.scatter(
                    trade['exit_date'], 
                    trade['exit_price'],
                    marker='v' if trade['pnl'] >= 0 else 'X', 
                    color='blue' if trade['pnl'] >= 0 else 'red', 
                    s=100, 
                    label='Sell (Profit)' if trade['pnl'] >= 0 else 'Sell (Loss)'
                )
            else:  # short position
                ax.scatter(
                    trade['entry_date'], 
                    trade['entry_price'],
                    marker='v', 
                    color='red', 
                    s=100, 
                    label='Short'
                )
                ax.scatter(
                    trade['exit_date'], 
                    trade['exit_price'],
                    marker='^' if trade['pnl'] >= 0 else 'X', 
                    color='blue' if trade['pnl'] >= 0 else 'red', 
                    s=100, 
                    label='Cover (Profit)' if trade['pnl'] >= 0 else 'Cover (Loss)'
                )
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Configure plot
        ax.set_title(f"Trades for {symbol} from {results['start_date']} to {results['end_date']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        
        plt.tight_layout()
        return fig 