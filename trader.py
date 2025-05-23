#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trader module for executing trades through Alpaca Markets API based on signals.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaTrader:
    """Class for executing trades via Alpaca Markets API based on signals."""
    
    def __init__(self):
        """Initialize the AlpacaTrader with API credentials."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')  # Default to paper trading
        
        # Validate credentials
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API credentials not found. Check your .env file.")
            raise ValueError("Alpaca API credentials not found. Check your .env file.")
        
        # Initialize API - remove additional v2 in api_version if base_url already contains it
        if 'v2' in self.base_url:
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version=''  # Empty since v2 is in the URL
            )
        else:
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
        
        # Check connection
        try:
            # Use a direct call to the API endpoint to avoid potential path issues
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Cash balance: ${float(account.cash):.2f}")
            logger.info(f"Portfolio value: ${float(account.portfolio_value):.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {str(e)}")
            logger.warning("Make sure your .env file has correct API credentials and the API URL is correct")
            raise
        
        # Initialize strategy performance tracking
        self.strategy_performance = {}
        self.position_history = []
        
        # Initialize a record of our current positions
        self.current_positions = {}
        self.update_positions()
        
    def update_positions(self):
        """Update the current positions dictionary."""
        try:
            positions = self.api.list_positions()
            self.current_positions = {p.symbol: {
                'qty': int(p.qty), 
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            } for p in positions}
            logger.info(f"Updated positions: {len(self.current_positions)} active positions")
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def get_buying_power(self):
        """Get the current buying power."""
        try:
            account = self.api.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Error getting buying power: {str(e)}")
            return 0
    
    def get_portfolio_value(self):
        """Get the current portfolio value."""
        try:
            account = self.api.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {str(e)}")
            return 0
    
    def execute_trade(self, symbol, signal, quantity=None, strategy='auto', risk_pct=0.02):
        """
        Execute a trade based on a signal.
        
        Args:
            symbol (str): Stock symbol
            signal (int): Signal type (1=buy, -1=sell, 0=hold)
            quantity (int, optional): Number of shares to trade. If None, calculate based on risk.
            strategy (str): Strategy that generated the signal
            risk_pct (float): Percentage of portfolio to risk on a single trade
            
        Returns:
            dict: Trade information
        """
        # Always log the signal for debugging
        signal_types = {1: "BUY", -1: "SELL", 0: "HOLD"}
        signal_text = signal_types.get(signal, f"UNKNOWN({signal})")
        logger.info(f"SIGNAL RECEIVED: {signal_text} for {symbol} from {strategy}")
        
        if signal == 0:  # Hold signal
            logger.info(f"HOLD signal for {symbol} - no action taken")
            return {'symbol': symbol, 'action': 'HOLD', 'status': 'no_action', 'strategy': strategy}
        
        # For testing - if we're not connected to Alpaca, simulate the trade
        if not hasattr(self, 'api') or self.api is None:
            logger.warning(f"No Alpaca API connection - SIMULATING {signal_text} for {symbol}")
            return {
                'symbol': symbol, 
                'action': signal_text,
                'quantity': quantity or 10,  # Default quantity for simulation
                'price': 100.0,  # Dummy price
                'status': 'simulated',
                'strategy': strategy
            }
        
        try:
            # Get account information
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Get current position if any
            try:
                position = self.api.get_position(symbol)
                current_position = int(position.qty)
                position_exists = True
            except:
                current_position = 0
                position_exists = False
            
            # Get current price
            try:
                ticker_data = self.api.get_latest_trade(symbol)
                current_price = float(ticker_data.price)
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {str(e)}")
                # Fallback to Yahoo Finance
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period='1d').iloc[-1]['Close']
                except:
                    logger.error(f"Could not get price for {symbol} from any source")
                    return {'symbol': symbol, 'action': 'ERROR', 'status': 'price_error', 'strategy': strategy}
            
            # Calculate quantity if not provided
            if quantity is None:
                if signal == 1:  # Buy
                    # Use a percentage of buying power based on risk
                    trade_value = buying_power * risk_pct
                    quantity = max(1, int(trade_value / current_price))
                    
                    # Log buying power calculation for debugging
                    logger.info(f"Buying power calculation for {symbol}: buying_power=${buying_power:.2f}, risk_pct={risk_pct:.3f}, trade_value=${trade_value:.2f}, price=${current_price:.2f}, calculated_qty={quantity}")
                    
                    # Check if even one share is affordable
                    if trade_value < current_price:
                        logger.warning(f"Trade value ${trade_value:.2f} less than price ${current_price:.2f} for {symbol} - trying minimum 1 share")
                        quantity = 1
                        
                elif signal == -1 and position_exists:  # Sell
                    # Sell all shares
                    quantity = abs(current_position)
            
            # Execute the trade
            if signal == 1:  # Buy
                # Check if we can afford the calculated quantity
                total_cost = current_price * quantity
                if buying_power < total_cost:
                    logger.warning(f"Insufficient buying power (${buying_power:.2f}) to buy {quantity} shares of {symbol} at ${current_price:.2f} (total: ${total_cost:.2f})")
                    return {'symbol': symbol, 'action': 'BUY', 'status': 'insufficient_funds', 'strategy': strategy}
                
                try:
                    # Place a market order to buy
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"BUY order placed for {quantity} shares of {symbol} at ~${current_price:.2f} (total: ${total_cost:.2f})")
                    
                    # Update strategy performance tracking
                    self._update_strategy_performance(strategy, 'buy', symbol, quantity, current_price)
                    
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': current_price,
                        'order_id': order.id,
                        'status': 'submitted',
                        'strategy': strategy
                    }
                except Exception as e:
                    logger.error(f"Error placing buy order for {symbol}: {str(e)}")
                    # Log more details about the error
                    import traceback
                    logger.error(f"Full error traceback: {traceback.format_exc()}")
                    return {'symbol': symbol, 'action': 'BUY', 'status': 'order_error', 'error': str(e), 'strategy': strategy}
                    
            elif signal == -1:  # Sell
                if not position_exists or current_position <= 0:
                    logger.warning(f"No position in {symbol} to sell")
                    return {'symbol': symbol, 'action': 'SELL', 'status': 'no_position', 'strategy': strategy}
                
                try:
                    # Place a market order to sell
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"SELL order placed for {quantity} shares of {symbol} at ~${current_price:.2f}")
                    
                    # Update strategy performance tracking
                    self._update_strategy_performance(strategy, 'sell', symbol, quantity, current_price)
                    
                    return {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': current_price,
                        'order_id': order.id,
                        'status': 'submitted',
                        'strategy': strategy
                    }
                except Exception as e:
                    logger.error(f"Error placing sell order for {symbol}: {str(e)}")
                    # Log more details about the error
                    import traceback
                    logger.error(f"Full error traceback: {traceback.format_exc()}")
                    return {'symbol': symbol, 'action': 'SELL', 'status': 'order_error', 'error': str(e), 'strategy': strategy}
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return {'symbol': symbol, 'action': signal_text, 'status': 'error', 'error': str(e), 'strategy': strategy}
    
    def process_signals(self, signals_data, strategy_selector=None):
        """
        Process multiple signals and execute appropriate trades.
        
        Args:
            signals_data (dict): Dictionary with symbols as keys and signal DataFrames as values
            strategy_selector (callable, optional): Function to select the best strategy for each symbol
            
        Returns:
            list: List of executed trades information
        """
        executed_trades = []
        
        for symbol, signals in signals_data.items():
            try:
                # Get the last signal for each available strategy
                strategy_signals = {}
                
                # Check if signals is a dictionary (multiple strategies) or DataFrame (single strategy)
                if isinstance(signals, dict):
                    # Multiple strategies
                    for strategy_name, signal_df in signals.items():
                        if not signal_df.empty:
                            strategy_signals[strategy_name] = signal_df['signal'].iloc[-1]
                else:
                    # Single strategy
                    if not signals.empty:
                        strategy_signals['default'] = signals['signal'].iloc[-1]
                
                if not strategy_signals:
                    logger.warning(f"No valid signals found for {symbol}")
                    continue
                
                # Check if we already have a position in this symbol
                position_exists = symbol in self.current_positions
                
                # If we don't have a position, filter out SELL and HOLD signals
                if not position_exists:
                    # Look for BUY signals only - ignore SELL and HOLD
                    buy_strategies = {name: signal for name, signal in strategy_signals.items() 
                                     if signal == 1}
                    
                    if buy_strategies:
                        # Pick the first buy strategy (preferring trend-following strategies)
                        best_strategy = next(iter(buy_strategies.keys()))
                        signal = 1
                        
                        # Execute the BUY trade
                        trade_info = self.execute_trade(symbol, signal, strategy=best_strategy)
                        executed_trades.append(trade_info)
                    else:
                        # No BUY signals for a stock we don't own - skip
                        logger.info(f"No BUY signals for {symbol} and no existing position - skipping")
                        continue
                else:
                    # If we have a position, we can process any signal type
                    if strategy_selector and len(strategy_signals) > 1:
                        best_strategy = strategy_selector(symbol, strategy_signals, self.strategy_performance)
                    else:
                        best_strategy = next(iter(strategy_signals.keys()))
                    
                    signal = strategy_signals.get(best_strategy, 0)
                    
                    # Execute the trade (BUY more, SELL, or HOLD)
                    trade_info = self.execute_trade(symbol, signal, strategy=best_strategy)
                    executed_trades.append(trade_info)
                
            except Exception as e:
                logger.error(f"Error processing signals for {symbol}: {str(e)}")
        
        return executed_trades
    
    def _update_strategy_performance(self, strategy, action, symbol, quantity, price):
        """Update the strategy performance tracking."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit_loss': 0.0,
                'symbols': {}
            }
        
        # Initialize symbol tracking if needed
        if symbol not in self.strategy_performance[strategy]['symbols']:
            self.strategy_performance[strategy]['symbols'][symbol] = {
                'position': 0,
                'avg_entry': 0.0,
                'trades': []
            }
        
        symbol_data = self.strategy_performance[strategy]['symbols'][symbol]
        
        # Update based on action
        if action == 'buy':
            # Calculate new average entry price
            current_value = symbol_data['position'] * symbol_data['avg_entry']
            new_value = quantity * price
            new_position = symbol_data['position'] + quantity
            
            if new_position > 0:
                new_avg_entry = (current_value + new_value) / new_position
            else:
                new_avg_entry = 0.0
            
            # Update position
            symbol_data['position'] = new_position
            symbol_data['avg_entry'] = new_avg_entry
            
            # Record trade
            symbol_data['trades'].append({
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'timestamp': datetime.now().isoformat()
            })
            
            self.strategy_performance[strategy]['trades'] += 1
            
        elif action == 'sell':
            # Only count profit/loss if we had a position
            if symbol_data['position'] > 0:
                # Calculate P&L
                pl_per_share = price - symbol_data['avg_entry']
                total_pl = pl_per_share * quantity
                
                # Update overall strategy performance
                self.strategy_performance[strategy]['profit_loss'] += total_pl
                
                if total_pl > 0:
                    self.strategy_performance[strategy]['wins'] += 1
                else:
                    self.strategy_performance[strategy]['losses'] += 1
                
                # Record trade
                symbol_data['trades'].append({
                    'action': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'pl_per_share': pl_per_share,
                    'total_pl': total_pl,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update position
                symbol_data['position'] -= quantity
                if symbol_data['position'] <= 0:
                    symbol_data['position'] = 0
                    symbol_data['avg_entry'] = 0.0
                
                self.strategy_performance[strategy]['trades'] += 1
    
    def get_strategy_performance(self):
        """Get the performance metrics for all strategies."""
        performance = {}
        
        for strategy, data in self.strategy_performance.items():
            win_rate = 0
            if data['trades'] > 0:
                win_rate = data['wins'] / data['trades'] * 100
                
            performance[strategy] = {
                'trades': data['trades'],
                'wins': data['wins'],
                'losses': data['losses'],
                'win_rate': win_rate,
                'profit_loss': data['profit_loss']
            }
        
        return performance
    
    def generate_trading_summary(self):
        """
        Generate a comprehensive trading summary for saving to file.
        
        Returns:
            dict: Trading summary with all trades, positions, and performance metrics
        """
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trades': 0,
            'total_profit_loss': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'strategies': {},
            'symbols': {},
            'all_trades': []
        }
        
        # Collect all trades and performance data
        for strategy, data in self.strategy_performance.items():
            strategy_pl = data['profit_loss']
            summary['total_profit_loss'] += strategy_pl
            summary['total_trades'] += data['trades']
            summary['winning_trades'] += data['wins']
            summary['losing_trades'] += data['losses']
            
            # Strategy-level summary
            summary['strategies'][strategy] = {
                'total_trades': data['trades'],
                'wins': data['wins'],
                'losses': data['losses'],
                'win_rate': (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0,
                'profit_loss': strategy_pl
            }
            
            # Symbol-level details
            for symbol, symbol_data in data['symbols'].items():
                if symbol not in summary['symbols']:
                    summary['symbols'][symbol] = {
                        'trades': [],
                        'current_position': 0,
                        'total_bought': 0,
                        'total_sold': 0,
                        'realized_pl': 0.0
                    }
                
                # Add trades for this symbol
                for trade in symbol_data['trades']:
                    trade_record = {
                        'symbol': symbol,
                        'strategy': strategy,
                        'action': trade['action'],
                        'quantity': trade['quantity'],
                        'price': trade['price'],
                        'timestamp': trade['timestamp']
                    }
                    
                    if trade['action'] == 'buy':
                        summary['symbols'][symbol]['total_bought'] += trade['quantity']
                    elif trade['action'] == 'sell':
                        summary['symbols'][symbol]['total_sold'] += trade['quantity']
                        if 'total_pl' in trade:
                            summary['symbols'][symbol]['realized_pl'] += trade['total_pl']
                            trade_record['profit_loss'] = trade['total_pl']
                    
                    summary['symbols'][symbol]['trades'].append(trade_record)
                    summary['all_trades'].append(trade_record)
                
                summary['symbols'][symbol]['current_position'] = symbol_data['position']
        
        # Calculate overall win rate
        summary['overall_win_rate'] = (summary['winning_trades'] / summary['total_trades'] * 100) if summary['total_trades'] > 0 else 0
        
        return summary
    
    def save_trading_summary(self, filename=None):
        """
        Save a trading summary to a file.
        
        Args:
            filename (str, optional): Filename to save to. If None, uses timestamp.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_summary_{timestamp}.txt"
        
        # Create summaries directory if it doesn't exist
        summaries_dir = "summaries"
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)
            logger.info(f"Created directory: {summaries_dir}")
        
        filepath = os.path.join(summaries_dir, filename)
        
        try:
            summary = self.generate_trading_summary()
            
            with open(filepath, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("TRADING SESSION SUMMARY\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {summary['timestamp']}\n\n")
                
                # Overall Performance
                f.write("OVERALL PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Trades: {summary['total_trades']}\n")
                f.write(f"Winning Trades: {summary['winning_trades']}\n")
                f.write(f"Losing Trades: {summary['losing_trades']}\n")
                f.write(f"Win Rate: {summary['overall_win_rate']:.1f}%\n")
                f.write(f"Total Profit/Loss: ${summary['total_profit_loss']:.2f}\n\n")
                
                # Strategy Performance
                if summary['strategies']:
                    f.write("STRATEGY PERFORMANCE:\n")
                    f.write("-" * 30 + "\n")
                    for strategy, metrics in summary['strategies'].items():
                        f.write(f"{strategy}:\n")
                        f.write(f"  Trades: {metrics['total_trades']}\n")
                        f.write(f"  Win Rate: {metrics['win_rate']:.1f}%\n")
                        f.write(f"  P&L: ${metrics['profit_loss']:.2f}\n\n")
                
                # Symbol Details
                if summary['symbols']:
                    f.write("SYMBOL TRADING DETAILS:\n")
                    f.write("-" * 30 + "\n")
                    for symbol, data in summary['symbols'].items():
                        f.write(f"{symbol}:\n")
                        f.write(f"  Total Bought: {data['total_bought']} shares\n")
                        f.write(f"  Total Sold: {data['total_sold']} shares\n")
                        f.write(f"  Current Position: {data['current_position']} shares\n")
                        f.write(f"  Realized P&L: ${data['realized_pl']:.2f}\n")
                        
                        if data['trades']:
                            f.write(f"  Recent Trades:\n")
                            # Show last 5 trades for this symbol
                            recent_trades = data['trades'][-5:]
                            for trade in recent_trades:
                                f.write(f"    {trade['action'].upper()} {trade['quantity']} @ ${trade['price']:.2f}")
                                if 'profit_loss' in trade:
                                    f.write(f" (P&L: ${trade['profit_loss']:.2f})")
                                f.write(f" [{trade['timestamp']}]\n")
                        f.write("\n")
                
                # All Trades Summary (limited to last 20 for brevity)
                if summary['all_trades']:
                    f.write("RECENT TRADES (Last 20):\n")
                    f.write("-" * 30 + "\n")
                    recent_all_trades = summary['all_trades'][-20:]
                    for trade in recent_all_trades:
                        f.write(f"{trade['timestamp']}: {trade['action'].upper()} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
                        if 'profit_loss' in trade:
                            f.write(f" (P&L: ${trade['profit_loss']:.2f})")
                        f.write(f" [{trade['strategy']}]\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("End of Summary\n")
                f.write("=" * 60 + "\n")
            
            logger.info(f"Trading summary saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving trading summary: {str(e)}")
            return None

class StrategySelector:
    """Class for selecting the best trading strategy based on historical performance."""
    
    def __init__(self, lookback_period=14):
        """
        Initialize the StrategySelector.
        
        Args:
            lookback_period (int): Number of days to look back for performance evaluation
        """
        self.lookback_period = lookback_period
        self.strategy_metrics = {}
        self.market_conditions = {}
    
    def select_best_strategy(self, symbol, signals, performance_data):
        """
        Select the best strategy for a given symbol based on historical performance.
        
        Args:
            symbol (str): Stock symbol
            signals (dict): Dictionary of strategy signals
            performance_data (dict): Historical performance data
            
        Returns:
            str: Name of the best strategy
        """
        # If we only have one strategy, use it
        if len(signals) == 1:
            return next(iter(signals.keys()))
        
        # If we have performance data, use it
        best_strategy = None
        best_score = -float('inf')
        
        for strategy in signals.keys():
            score = self._calculate_strategy_score(strategy, symbol, performance_data)
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        # If we couldn't determine a best strategy, use the one with the strongest signal
        if best_strategy is None:
            best_strategy = max(signals.items(), key=lambda x: abs(x[1]))[0]
        
        return best_strategy
    
    def _calculate_strategy_score(self, strategy, symbol, performance_data):
        """Calculate a score for a strategy based on performance data."""
        if strategy not in performance_data:
            return 0
        
        strategy_data = performance_data[strategy]
        
        # Calculate basic score based on win rate and profit/loss
        win_rate = 0
        if strategy_data['trades'] > 0:
            win_rate = strategy_data['wins'] / strategy_data['trades']
        
        # Profit factor
        profit_factor = 1.0
        if strategy_data['profit_loss'] != 0:
            profit_factor = abs(strategy_data['profit_loss']) / 1000  # Scale factor
            if strategy_data['profit_loss'] < 0:
                profit_factor = -profit_factor
        
        # Weight recent trades more heavily
        recency_factor = 1.0
        symbol_data = strategy_data.get('symbols', {}).get(symbol)
        if symbol_data and 'trades' in symbol_data:
            recent_trades = symbol_data['trades'][-min(5, len(symbol_data['trades'])):]
            if recent_trades:
                recent_pl = sum(trade.get('total_pl', 0) for trade in recent_trades if 'total_pl' in trade)
                recency_factor = 1.0 + (recent_pl / 1000)  # Scale factor
        
        # Combine factors into final score
        score = (win_rate * 10) + profit_factor + recency_factor
        
        return score

def best_strategy_factory(market_regime='auto'):
    """
    Factory function to create a strategy selector with the specified market regime bias.
    
    Args:
        market_regime (str): Market regime to bias toward ('trending', 'ranging', or 'auto')
        
    Returns:
        function: Strategy selector function
    """
    def strategy_selector(symbol, signals, performance_data):
        """Select the best strategy based on market regime and performance."""
        # If auto, try to detect market regime
        current_regime = market_regime
        if current_regime == 'auto':
            # Simple market regime detection based on performance
            trending_strategies = ['MA Crossover', 'Momentum Strategy']
            ranging_strategies = ['RSI Strategy', 'Mean Reversion Strategy']
            
            trending_perf = sum(performance_data.get(s, {}).get('profit_loss', 0) for s in trending_strategies)
            ranging_perf = sum(performance_data.get(s, {}).get('profit_loss', 0) for s in ranging_strategies)
            
            current_regime = 'trending' if trending_perf > ranging_perf else 'ranging'
        
        # More balanced weights - don't heavily favor any one strategy
        weights = {
            'MA Crossover': 1.0,
            'Momentum Strategy': 1.0,
            'Breakout Strategy': 1.0,
            'Dual Strategy System': 1.0,
            'RSI Strategy': 1.0,
            'Mean Reversion Strategy': 1.0
        }
        
        # Only slightly adjust based on market regime
        if current_regime == 'trending':
            weights['MA Crossover'] = 1.2
            weights['Momentum Strategy'] = 1.2
            weights['RSI Strategy'] = 0.9
            weights['Mean Reversion Strategy'] = 0.9
        else:  # ranging
            weights['MA Crossover'] = 0.9
            weights['Momentum Strategy'] = 0.9
            weights['RSI Strategy'] = 1.2
            weights['Mean Reversion Strategy'] = 1.2
        
        # Calculate scores
        scores = {}
        for strategy, signal in signals.items():
            # Get base score from signal strength
            base_score = abs(signal)
            
            # Get weight for this strategy
            weight = weights.get(strategy, 1.0)
            
            # Get performance score
            performance_score = 0
            if strategy in performance_data:
                perf = performance_data[strategy]
                if perf['trades'] > 0:
                    win_rate = perf['wins'] / perf['trades']
                    performance_score = win_rate * perf['profit_loss'] / (perf['trades'] + 1)
            
            # Calculate final score
            scores[strategy] = base_score * weight + performance_score
        
        # Return strategy with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            # Default to first strategy
            return next(iter(signals.keys()))
    
    return strategy_selector


# Example usage
if __name__ == "__main__":
    # Create a trader instance
    trader = AlpacaTrader()
    
    # Print account info
    account = trader.api.get_account()
    print(f"Account ID: {account.id}")
    print(f"Cash: ${float(account.cash):.2f}")
    print(f"Buying Power: ${float(account.buying_power):.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
    
    # List current positions
    positions = trader.api.list_positions()
    print("\nCurrent Positions:")
    for position in positions:
        print(f"{position.symbol}: {position.qty} shares @ ${float(position.avg_entry_price):.2f} - P/L: ${float(position.unrealized_pl):.2f}")
    
    # Test strategy selector
    selector = best_strategy_factory(market_regime='auto')
    print("\nStrategy selector test:")
    test_signals = {
        'MA Crossover': 1,
        'RSI Strategy': 0,
        'Momentum Strategy': 1,
        'Mean Reversion Strategy': -1
    }
    test_performance = {
        'MA Crossover': {'trades': 10, 'wins': 6, 'profit_loss': 500},
        'RSI Strategy': {'trades': 8, 'wins': 5, 'profit_loss': 400},
        'Momentum Strategy': {'trades': 5, 'wins': 3, 'profit_loss': 300},
        'Mean Reversion Strategy': {'trades': 7, 'wins': 4, 'profit_loss': 350}
    }
    best_strategy = selector('AAPL', test_signals, test_performance)
    print(f"Best strategy for AAPL: {best_strategy}") 