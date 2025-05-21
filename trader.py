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
        
        # Initialize API
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
        
        # Check connection
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Cash balance: ${float(account.cash):.2f}")
            logger.info(f"Portfolio value: ${float(account.portfolio_value):.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {str(e)}")
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
        if signal == 0:  # Hold signal, do nothing
            logger.info(f"HOLD signal for {symbol} - no action taken")
            return {'symbol': symbol, 'action': 'HOLD', 'status': 'no_action', 'strategy': strategy}
        
        try:
            # Check if market is open
            clock = self.api.get_clock()
            if not clock.is_open:
                next_open = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
                logger.warning(f"Market is closed. Next open: {next_open}")
                return {'symbol': symbol, 'action': 'NONE', 'status': 'market_closed', 'strategy': strategy}
            
            # Get current price
            last_quote = self.api.get_latest_quote(symbol)
            current_price = (float(last_quote.ap) + float(last_quote.bp)) / 2  # Midpoint between ask and bid
            
            # Check current position
            position_exists = symbol in self.current_positions
            current_position = self.current_positions.get(symbol, {'qty': 0})
            current_shares = current_position.get('qty', 0)
            
            # Determine action based on signal and current position
            if signal == 1:  # Buy signal
                if position_exists and current_shares > 0:
                    logger.info(f"Already long {current_shares} shares of {symbol} - no additional buy")
                    return {'symbol': symbol, 'action': 'HOLD', 'status': 'already_long', 'strategy': strategy}
                
                # Calculate quantity if not specified
                if quantity is None:
                    portfolio_value = self.get_portfolio_value()
                    risk_amount = portfolio_value * risk_pct
                    # Use a simple position sizing based on risk percentage
                    quantity = max(1, int(risk_amount / current_price))
                
                # Place buy order
                logger.info(f"Placing market order to BUY {quantity} shares of {symbol} at ~${current_price:.2f}")
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                # Record the trade
                trade_info = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'order_id': order.id,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'strategy': strategy,
                    'status': 'submitted'
                }
                self.position_history.append(trade_info)
                logger.info(f"Buy order placed for {symbol}: {order.id}")
                
                # Update strategy performance
                self._update_strategy_performance(strategy, 'buy', symbol, quantity, current_price)
                
                # Update our positions
                self.update_positions()
                
                return trade_info
                
            elif signal == -1:  # Sell signal
                if not position_exists or current_shares <= 0:
                    logger.info(f"No position in {symbol} to sell")
                    return {'symbol': symbol, 'action': 'HOLD', 'status': 'no_position', 'strategy': strategy}
                
                # Determine quantity to sell
                if quantity is None or quantity > current_shares:
                    quantity = current_shares
                
                # Place sell order
                logger.info(f"Placing market order to SELL {quantity} shares of {symbol} at ~${current_price:.2f}")
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                # Record the trade
                trade_info = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': current_price,
                    'order_id': order.id,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'strategy': strategy,
                    'status': 'submitted'
                }
                self.position_history.append(trade_info)
                logger.info(f"Sell order placed for {symbol}: {order.id}")
                
                # Update strategy performance
                self._update_strategy_performance(strategy, 'sell', symbol, quantity, current_price)
                
                # Update our positions
                self.update_positions()
                
                return trade_info
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return {'symbol': symbol, 'action': 'ERROR', 'status': 'error', 'error': str(e), 'strategy': strategy}
    
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
                
                # Select the best strategy if a selector is provided
                if strategy_selector and len(strategy_signals) > 1:
                    best_strategy = strategy_selector(symbol, strategy_signals, self.strategy_performance)
                else:
                    # Otherwise, just use the first available strategy
                    best_strategy = next(iter(strategy_signals.keys()))
                
                # Get the signal from the selected strategy
                signal = strategy_signals.get(best_strategy, 0)
                
                # Execute the trade
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
        
        # Adjust weights based on regime
        weights = {}
        if current_regime == 'trending':
            weights = {
                'MA Crossover': 1.5,
                'Momentum Strategy': 1.3,
                'Breakout Strategy': 1.2,
                'Dual Strategy System': 1.0,
                'RSI Strategy': 0.7,
                'Mean Reversion Strategy': 0.5
            }
        else:  # ranging
            weights = {
                'Mean Reversion Strategy': 1.5,
                'RSI Strategy': 1.3,
                'Dual Strategy System': 1.0,
                'Breakout Strategy': 0.8,
                'MA Crossover': 0.6,
                'Momentum Strategy': 0.5
            }
        
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