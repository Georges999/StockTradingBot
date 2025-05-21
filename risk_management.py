#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk management module for the StockTradingBot.
Handles position sizing, portfolio allocation, and risk limits.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskManager:
    """Class for managing trading risk and position sizing."""
    
    def __init__(self, config):
        """Initialize the RiskManager with configuration."""
        self.config = config
        self.portfolio_value = None
        self.positions = {}  # Currently open positions
        self.portfolio_history = []  # Track portfolio value over time
        self.trade_history = []  # Track completed trades
    
    def evaluate_trades(self, signal_data: Dict, market_data: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Evaluate trading signals and apply risk management rules.
        
        Args:
            signal_data: Dictionary with trading signals from the strategy.
            market_data: DataFrame with price and indicator data.
            symbol: The ticker symbol.
            
        Returns:
            List of trade decisions with position size and risk parameters.
        """
        if market_data.empty or 'signal' not in signal_data:
            return []
        
        # Get the latest market data
        latest_data = market_data.iloc[-1]
        current_price = latest_data['close']
        
        # Initialize trade decisions list
        trade_decisions = []
        
        # Check if we already have a position in this symbol
        existing_position = self.positions.get(symbol)
        
        # Get the trading signal
        signal = signal_data['signal']
        signal_strength = signal_data.get('strength', 0)
        signal_direction = signal_data.get('direction', 'neutral')
        
        # Decision logic based on signal and existing position
        if signal == 'buy' and signal_direction == 'long':
            # Long entry logic
            if existing_position is None:
                # No existing position, consider opening a new long position
                if self._can_open_new_position(symbol):
                    # Calculate position size
                    position_size = self._calculate_position_size(symbol, current_price, signal_strength)
                    
                    if position_size > 0:
                        # Calculate stop loss and take profit levels
                        stop_loss = current_price * (1 - self.config.STOP_LOSS_PCT)
                        take_profit = current_price * (1 + self.config.TAKE_PROFIT_PCT)
                        
                        # Create trade decision
                        trade_decision = {
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': position_size,
                            'price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'timestamp': datetime.now().isoformat(),
                            'reason': signal_data.get('reason', 'Signal generated'),
                            'risk_pct': self.config.STOP_LOSS_PCT * 100,
                            'reward_pct': self.config.TAKE_PROFIT_PCT * 100
                        }
                        
                        trade_decisions.append(trade_decision)
            
            elif existing_position['side'] == 'short':
                # Have a short position, consider closing it (exit short)
                trade_decision = {
                    'symbol': symbol,
                    'action': 'buy_to_cover',  # Buy to cover short position
                    'quantity': existing_position['quantity'],
                    'price': current_price,
                    'timestamp': datetime.now().isoformat(),
                    'reason': f"Exit short: {signal_data.get('reason', 'Signal generated')}",
                    'pnl': (existing_position['price'] - current_price) * existing_position['quantity']
                }
                
                trade_decisions.append(trade_decision)
        
        elif signal == 'sell' and signal_direction == 'short':
            # Short entry logic
            if existing_position is None:
                # No existing position, consider opening a new short position
                if self._can_open_new_position(symbol) and self.config.ALLOW_SHORT_SELLING:
                    # Calculate position size
                    position_size = self._calculate_position_size(symbol, current_price, signal_strength)
                    
                    if position_size > 0:
                        # Calculate stop loss and take profit levels
                        stop_loss = current_price * (1 + self.config.STOP_LOSS_PCT)
                        take_profit = current_price * (1 - self.config.TAKE_PROFIT_PCT)
                        
                        # Create trade decision
                        trade_decision = {
                            'symbol': symbol,
                            'action': 'sell_short',
                            'quantity': position_size,
                            'price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'timestamp': datetime.now().isoformat(),
                            'reason': signal_data.get('reason', 'Signal generated'),
                            'risk_pct': self.config.STOP_LOSS_PCT * 100,
                            'reward_pct': self.config.TAKE_PROFIT_PCT * 100
                        }
                        
                        trade_decisions.append(trade_decision)
            
            elif existing_position['side'] == 'long':
                # Have a long position, consider closing it (exit long)
                trade_decision = {
                    'symbol': symbol,
                    'action': 'sell',  # Sell to close long position
                    'quantity': existing_position['quantity'],
                    'price': current_price,
                    'timestamp': datetime.now().isoformat(),
                    'reason': f"Exit long: {signal_data.get('reason', 'Signal generated')}",
                    'pnl': (current_price - existing_position['price']) * existing_position['quantity']
                }
                
                trade_decisions.append(trade_decision)
        
        # Check for stop loss and take profit triggers for existing positions
        if existing_position:
            exit_decision = self._check_exit_conditions(symbol, existing_position, current_price)
            if exit_decision:
                trade_decisions.append(exit_decision)
        
        return trade_decisions
    
    def _can_open_new_position(self, symbol: str) -> bool:
        """
        Check if a new position can be opened based on portfolio constraints.
        
        Args:
            symbol: The ticker symbol.
            
        Returns:
            Boolean indicating if a new position can be opened.
        """
        # Check if we already have too many open positions
        if len(self.positions) >= self.config.MAX_OPEN_POSITIONS:
            logger.info(f"Cannot open position in {symbol}: Maximum number of positions reached")
            return False
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            logger.info(f"Cannot open position in {symbol}: Position already exists")
            return False
        
        # Additional checks could include:
        # - Sector exposure limits
        # - Correlation with existing positions
        # - Volatility constraints
        
        return True
    
    def _calculate_position_size(self, symbol: str, price: float, signal_strength: float) -> int:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            symbol: The ticker symbol.
            price: Current price of the asset.
            signal_strength: Strength of the signal (0-1).
            
        Returns:
            Number of shares to trade.
        """
        # Get portfolio value
        portfolio_value = self._get_portfolio_value()
        
        # Base position size on maximum position size parameter
        base_position_value = portfolio_value * self.config.MAX_POSITION_SIZE
        
        # Scale position size by signal strength
        adjusted_position_value = base_position_value * signal_strength
        
        # Convert to number of shares
        shares = int(adjusted_position_value / price)
        
        # Ensure minimum position size (at least 1 share)
        shares = max(1, shares)
        
        # Log the position sizing calculation
        logger.info(f"Position sizing for {symbol}: Portfolio=${portfolio_value:.2f}, " +
                   f"Signal strength={signal_strength:.2f}, Shares={shares} at ${price:.2f}")
        
        return shares
    
    def _check_exit_conditions(self, symbol: str, position: Dict, current_price: float) -> Optional[Dict]:
        """
        Check if any exit conditions are met for a position.
        
        Args:
            symbol: The ticker symbol.
            position: Dictionary with position details.
            current_price: Current price of the asset.
            
        Returns:
            Trade decision dictionary if exit condition is met, None otherwise.
        """
        # Skip if no position exists
        if not position:
            return None
        
        # Check stop loss
        if position['side'] == 'long' and current_price <= position.get('stop_loss', 0):
            return {
                'symbol': symbol,
                'action': 'sell',
                'quantity': position['quantity'],
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'reason': 'Stop loss triggered',
                'pnl': (current_price - position['price']) * position['quantity']
            }
        
        if position['side'] == 'short' and current_price >= position.get('stop_loss', float('inf')):
            return {
                'symbol': symbol,
                'action': 'buy_to_cover',
                'quantity': position['quantity'],
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'reason': 'Stop loss triggered',
                'pnl': (position['price'] - current_price) * position['quantity']
            }
        
        # Check take profit
        if position['side'] == 'long' and current_price >= position.get('take_profit', float('inf')):
            return {
                'symbol': symbol,
                'action': 'sell',
                'quantity': position['quantity'],
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'reason': 'Take profit triggered',
                'pnl': (current_price - position['price']) * position['quantity']
            }
        
        if position['side'] == 'short' and current_price <= position.get('take_profit', 0):
            return {
                'symbol': symbol,
                'action': 'buy_to_cover',
                'quantity': position['quantity'],
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'reason': 'Take profit triggered',
                'pnl': (position['price'] - current_price) * position['quantity']
            }
        
        # No exit conditions met
        return None
    
    def update_portfolio(self, trade_result: Dict) -> None:
        """
        Update portfolio after a trade is executed.
        
        Args:
            trade_result: Dictionary with trade execution details.
        """
        symbol = trade_result['symbol']
        action = trade_result['action']
        quantity = trade_result['quantity']
        price = trade_result['price']
        
        # Update positions dictionary
        if action == 'buy':
            # Opening a long position
            self.positions[symbol] = {
                'side': 'long',
                'quantity': quantity,
                'price': price,
                'value': price * quantity,
                'stop_loss': trade_result.get('stop_loss'),
                'take_profit': trade_result.get('take_profit'),
                'open_time': datetime.now()
            }
        
        elif action == 'sell_short':
            # Opening a short position
            self.positions[symbol] = {
                'side': 'short',
                'quantity': quantity,
                'price': price,
                'value': price * quantity,
                'stop_loss': trade_result.get('stop_loss'),
                'take_profit': trade_result.get('take_profit'),
                'open_time': datetime.now()
            }
        
        elif action == 'sell' and symbol in self.positions:
            # Closing a long position
            position = self.positions[symbol]
            if position['side'] == 'long':
                # Calculate PnL
                pnl = (price - position['price']) * quantity
                
                # Add to trade history
                self.trade_history.append({
                    'symbol': symbol,
                    'side': 'long',
                    'entry_price': position['price'],
                    'exit_price': price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'pnl_pct': (price / position['price'] - 1) * 100,
                    'open_time': position['open_time'],
                    'close_time': datetime.now(),
                    'duration': (datetime.now() - position['open_time']).total_seconds() / 3600  # Hours
                })
                
                # Remove from active positions
                del self.positions[symbol]
                
                logger.info(f"Closed long position in {symbol}: {quantity} shares, PnL: ${pnl:.2f}")
        
        elif action == 'buy_to_cover' and symbol in self.positions:
            # Closing a short position
            position = self.positions[symbol]
            if position['side'] == 'short':
                # Calculate PnL
                pnl = (position['price'] - price) * quantity
                
                # Add to trade history
                self.trade_history.append({
                    'symbol': symbol,
                    'side': 'short',
                    'entry_price': position['price'],
                    'exit_price': price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'pnl_pct': (position['price'] / price - 1) * 100,
                    'open_time': position['open_time'],
                    'close_time': datetime.now(),
                    'duration': (datetime.now() - position['open_time']).total_seconds() / 3600  # Hours
                })
                
                # Remove from active positions
                del self.positions[symbol]
                
                logger.info(f"Closed short position in {symbol}: {quantity} shares, PnL: ${pnl:.2f}")
        
        # Update portfolio value
        self._update_portfolio_value()
    
    def _get_portfolio_value(self) -> float:
        """
        Get the current portfolio value.
        
        Returns:
            Current portfolio value.
        """
        if self.portfolio_value is None:
            # Initialize with configuration value
            self.portfolio_value = self.config.INITIAL_CAPITAL
            
        return self.portfolio_value
    
    def _update_portfolio_value(self) -> None:
        """Update the portfolio value based on current positions."""
        # This would typically involve getting current market prices
        # and calculating the value of all positions plus cash
        # For simplicity, we're just using the sum of position values
        position_value = sum(position['value'] for position in self.positions.values())
        cash = self.config.INITIAL_CAPITAL - position_value
        
        self.portfolio_value = cash + position_value
        
        # Record portfolio value history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': self.portfolio_value,
            'positions': len(self.positions)
        })
        
        logger.info(f"Updated portfolio value: ${self.portfolio_value:.2f}")
        
        # Check for portfolio stop loss
        if self.portfolio_value < self.config.INITIAL_CAPITAL * (1 - self.config.PORTFOLIO_STOP_LOSS):
            logger.warning(f"Portfolio stop loss triggered! Current value: ${self.portfolio_value:.2f}")
            # In a real implementation, this would close all positions
    
    def adjust_position(self, symbol: str, new_stop_loss: float = None, new_take_profit: float = None) -> None:
        """
        Adjust an existing position's risk parameters.
        
        Args:
            symbol: The ticker symbol.
            new_stop_loss: New stop loss price level.
            new_take_profit: New take profit price level.
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot adjust position for {symbol}: No position exists")
            return
        
        if new_stop_loss is not None:
            self.positions[symbol]['stop_loss'] = new_stop_loss
            logger.info(f"Adjusted stop loss for {symbol} to ${new_stop_loss:.2f}")
        
        if new_take_profit is not None:
            self.positions[symbol]['take_profit'] = new_take_profit
            logger.info(f"Adjusted take profit for {symbol} to ${new_take_profit:.2f}")
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics from trading history.
        
        Returns:
            Dictionary of performance metrics.
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit_per_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Basic metrics
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        total_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_profit = sum(trade['pnl'] for trade in self.trade_history) / total_trades if total_trades > 0 else 0
        
        # Drawdown calculation (simplified)
        max_drawdown = 0
        peak = self.config.INITIAL_CAPITAL
        for history in self.portfolio_history:
            if history['value'] > peak:
                peak = history['value']
            drawdown = (peak - history['value']) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio calculation (simplified)
        if len(self.portfolio_history) > 1:
            returns = [
                (self.portfolio_history[i]['value'] / self.portfolio_history[i-1]['value']) - 1
                for i in range(1, len(self.portfolio_history))
            ]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'avg_profit_per_trade': avg_profit,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio
        }
    
    def save_performance_to_csv(self, filename: str) -> None:
        """
        Save trade history and performance metrics to CSV.
        
        Args:
            filename: Name of the CSV file to save.
        """
        # Create DataFrame from trade history
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            trade_df.to_csv(f"trade_history_{filename}", index=False)
            
            # Create DataFrame from portfolio history
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df.to_csv(f"portfolio_history_{filename}", index=False)
            
            # Get performance metrics
            metrics = self.get_performance_metrics()
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f"performance_metrics_{filename}", index=False)
            
            logger.info(f"Saved performance data to {filename}")
        else:
            logger.warning("No trade history to save") 