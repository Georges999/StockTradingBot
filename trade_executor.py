#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trade executor module for executing trades via Alpaca API or simulation.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import alpaca_trade_api as tradeapi
import random
import numpy as np

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Class for executing trades via Alpaca API or simulation.
    """
    def __init__(self, config=None):
        """
        Initialize the TradeExecutor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.alpaca_api = None
        self.paper_trading = True
        self.positions = {}  # For paper trading simulation
        self.orders = {}     # For order tracking
        self.initial_capital = 100000.0  # Default initial capital for paper trading
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.transaction_history = []
        
        # Initialize Alpaca API if credentials available
        if config and hasattr(config, 'ALPACA_API_KEY') and hasattr(config, 'ALPACA_SECRET_KEY'):
            if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
                self.alpaca_api = tradeapi.REST(
                    config.ALPACA_API_KEY,
                    config.ALPACA_SECRET_KEY,
                    config.ALPACA_BASE_URL,
                    api_version='v2'
                )
                # Check if using paper trading
                self.paper_trading = 'paper' in config.ALPACA_BASE_URL.lower()
                
                # Get account info for real trading
                if not self.paper_trading:
                    try:
                        account = self.alpaca_api.get_account()
                        self.initial_capital = float(account.portfolio_value)
                        self.cash = float(account.cash)
                        self.portfolio_value = float(account.portfolio_value)
                    except Exception as e:
                        logger.error(f"Error fetching Alpaca account: {e}")
            else:
                logger.warning("Alpaca API credentials not provided. Using paper trading simulation.")
    
    def submit_order(self, symbol, qty, side, order_type='market', 
                     time_in_force='day', limit_price=None, stop_price=None):
        """
        Submit an order to buy or sell a security.
        
        Args:
            symbol (str): Symbol of the security
            qty (int): Quantity to buy/sell
            side (str): 'buy' or 'sell'
            order_type (str): 'market', 'limit', 'stop', 'stop_limit'
            time_in_force (str): 'day', 'gtc', 'ioc', 'fok'
            limit_price (float, optional): Limit price for limit orders
            stop_price (float, optional): Stop price for stop orders
            
        Returns:
            dict: Order information
        """
        order_id = f"order_{int(time.time() * 1000)}"
        
        # Process using Alpaca API if available
        if self.alpaca_api is not None:
            try:
                logger.info(f"Submitting {side} order for {qty} shares of {symbol}")
                
                # Submit order to Alpaca
                order = self.alpaca_api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price
                )
                
                # Store order details
                order_info = {
                    'id': order.id,
                    'symbol': symbol,
                    'qty': qty,
                    'side': side,
                    'type': order_type,
                    'limit_price': limit_price,
                    'stop_price': stop_price,
                    'submitted_at': datetime.now(),
                    'status': order.status
                }
                
                self.orders[order.id] = order_info
                return order_info
                
            except Exception as e:
                logger.error(f"Error submitting order via Alpaca: {e}")
                return None
        
        # Paper trading simulation
        else:
            logger.info(f"Paper trading: {side} {qty} shares of {symbol}")
            
            # Simulate a market order execution for paper trading
            try:
                # Get current price (in real implementation, this would come from a data source)
                # Here we're just simulating with a placeholder
                current_price = self._get_paper_trading_price(symbol)
                
                if current_price <= 0:
                    logger.error(f"Invalid price for {symbol}: {current_price}")
                    return None
                
                # Simulate order execution
                executed_price = current_price
                
                # For limit orders, check if price is favorable
                if order_type == 'limit':
                    if (side == 'buy' and limit_price < current_price) or \
                       (side == 'sell' and limit_price > current_price):
                        logger.info(f"Paper trading: Limit order not executed, current price not favorable")
                        
                        # Store pending order
                        order_info = {
                            'id': order_id,
                            'symbol': symbol,
                            'qty': qty,
                            'side': side,
                            'type': order_type,
                            'limit_price': limit_price,
                            'stop_price': stop_price,
                            'submitted_at': datetime.now(),
                            'status': 'pending'
                        }
                        self.orders[order_id] = order_info
                        return order_info
                    
                    executed_price = limit_price
                
                # Update cash and positions
                order_value = qty * executed_price
                
                if side == 'buy':
                    # Check if enough cash
                    if order_value > self.cash:
                        logger.warning(f"Paper trading: Not enough cash to buy {qty} shares of {symbol}")
                        return None
                    
                    # Update cash
                    self.cash -= order_value
                    
                    # Update position
                    if symbol in self.positions:
                        # Average down the position
                        current_qty = self.positions[symbol]['qty']
                        current_value = current_qty * self.positions[symbol]['avg_price']
                        new_value = current_value + order_value
                        new_qty = current_qty + qty
                        avg_price = new_value / new_qty
                        
                        self.positions[symbol]['qty'] = new_qty
                        self.positions[symbol]['avg_price'] = avg_price
                    else:
                        # New position
                        self.positions[symbol] = {
                            'qty': qty,
                            'avg_price': executed_price,
                            'current_price': executed_price
                        }
                
                elif side == 'sell':
                    # Check if position exists
                    if symbol not in self.positions or self.positions[symbol]['qty'] < qty:
                        logger.warning(f"Paper trading: Not enough shares to sell {qty} shares of {symbol}")
                        return None
                    
                    # Update cash
                    self.cash += order_value
                    
                    # Update position
                    current_qty = self.positions[symbol]['qty']
                    new_qty = current_qty - qty
                    
                    if new_qty <= 0:
                        # Close position
                        del self.positions[symbol]
                    else:
                        # Reduce position
                        self.positions[symbol]['qty'] = new_qty
                
                # Record transaction
                transaction = {
                    'id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'price': executed_price,
                    'value': order_value,
                    'type': order_type,
                    'timestamp': datetime.now()
                }
                self.transaction_history.append(transaction)
                
                # Update portfolio value
                self._update_portfolio_value()
                
                # Return order information
                order_info = {
                    'id': order_id,
                    'symbol': symbol,
                    'qty': qty,
                    'side': side,
                    'type': order_type,
                    'limit_price': limit_price,
                    'stop_price': stop_price,
                    'executed_price': executed_price,
                    'executed_at': datetime.now(),
                    'status': 'filled'
                }
                
                self.orders[order_id] = order_info
                return order_info
                
            except Exception as e:
                logger.error(f"Error simulating paper trade: {e}")
                return None
    
    def get_position(self, symbol):
        """
        Get current position information for a symbol.
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            dict: Position information
        """
        if self.alpaca_api is not None:
            try:
                position = self.alpaca_api.get_position(symbol)
                return {
                    'symbol': position.symbol,
                    'qty': int(position.qty),
                    'avg_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                }
            except Exception as e:
                # Position might not exist
                if 'position does not exist' in str(e).lower():
                    return None
                logger.error(f"Error fetching position for {symbol}: {e}")
                return None
        else:
            # Paper trading simulation
            return self.positions.get(symbol, None)
    
    def get_all_positions(self):
        """
        Get all current positions.
        
        Returns:
            list: List of position information dictionaries
        """
        if self.alpaca_api is not None:
            try:
                positions = self.alpaca_api.list_positions()
                return [
                    {
                        'symbol': position.symbol,
                        'qty': int(position.qty),
                        'avg_price': float(position.avg_entry_price),
                        'current_price': float(position.current_price),
                        'market_value': float(position.market_value),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc)
                    }
                    for position in positions
                ]
            except Exception as e:
                logger.error(f"Error fetching all positions: {e}")
                return []
        else:
            # Paper trading simulation
            return [
                {
                    'symbol': symbol,
                    'qty': details['qty'],
                    'avg_price': details['avg_price'],
                    'current_price': details.get('current_price', details['avg_price']),
                    'market_value': details['qty'] * details.get('current_price', details['avg_price']),
                    'unrealized_pl': details['qty'] * (details.get('current_price', details['avg_price']) - details['avg_price']),
                    'unrealized_plpc': (details.get('current_price', details['avg_price']) - details['avg_price']) / details['avg_price']
                }
                for symbol, details in self.positions.items()
            ]
    
    def cancel_order(self, order_id):
        """
        Cancel an open order.
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.alpaca_api is not None:
            try:
                self.alpaca_api.cancel_order(order_id)
                logger.info(f"Cancelled order: {order_id}")
                return True
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
                return False
        else:
            # Paper trading simulation
            if order_id in self.orders and self.orders[order_id]['status'] == 'pending':
                self.orders[order_id]['status'] = 'cancelled'
                logger.info(f"Paper trading: Cancelled order {order_id}")
                return True
            else:
                logger.warning(f"Paper trading: Order {order_id} not found or already executed")
                return False
    
    def get_order(self, order_id):
        """
        Get information about a specific order.
        
        Args:
            order_id (str): Order ID to query
            
        Returns:
            dict: Order information
        """
        if self.alpaca_api is not None:
            try:
                order = self.alpaca_api.get_order(order_id)
                return {
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': int(order.qty),
                    'side': order.side,
                    'type': order.type,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'status': order.status,
                    'filled_qty': int(order.filled_qty),
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
                }
            except Exception as e:
                logger.error(f"Error fetching order {order_id}: {e}")
                return None
        else:
            # Paper trading simulation
            return self.orders.get(order_id, None)
    
    def get_account(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        if self.alpaca_api is not None:
            try:
                account = self.alpaca_api.get_account()
                return {
                    'cash': float(account.cash),
                    'portfolio_value': float(account.portfolio_value),
                    'equity': float(account.equity),
                    'buying_power': float(account.buying_power),
                    'initial_margin': float(account.initial_margin),
                    'maintenance_margin': float(account.maintenance_margin),
                    'daytrade_count': int(account.daytrade_count),
                    'last_equity': float(account.last_equity),
                    'last_maintenance_margin': float(account.last_maintenance_margin)
                }
            except Exception as e:
                logger.error(f"Error fetching account information: {e}")
                return None
        else:
            # Paper trading simulation
            self._update_portfolio_value()  # Ensure portfolio value is up-to-date
            return {
                'cash': self.cash,
                'portfolio_value': self.portfolio_value,
                'equity': self.portfolio_value,
                'buying_power': self.cash * 2,  # Simplified 2x margin
                'initial_margin': self.portfolio_value - self.cash,
                'maintenance_margin': (self.portfolio_value - self.cash) * 0.25,  # Simplified 25% maintenance
                'daytrade_count': 0,
                'last_equity': self.initial_capital,
                'last_maintenance_margin': 0
            }
    
    def get_transaction_history(self):
        """
        Get transaction history.
        
        Returns:
            list: List of transactions
        """
        if self.alpaca_api is not None:
            try:
                # Get recent orders
                orders = self.alpaca_api.list_orders(status='closed', limit=100)
                return [
                    {
                        'id': order.id,
                        'symbol': order.symbol,
                        'side': order.side,
                        'qty': int(order.qty),
                        'price': float(order.filled_avg_price) if order.filled_avg_price else None,
                        'value': float(order.filled_avg_price) * int(order.qty) if order.filled_avg_price else None,
                        'type': order.type,
                        'timestamp': order.filled_at if order.filled_at else order.submitted_at
                    }
                    for order in orders if order.filled_qty > 0
                ]
            except Exception as e:
                logger.error(f"Error fetching transaction history: {e}")
                return self.transaction_history  # Fall back to local history
        else:
            # Paper trading simulation
            return self.transaction_history
    
    def update_paper_trading_prices(self, price_dict):
        """
        Update current prices for paper trading simulation.
        
        Args:
            price_dict (dict): Dictionary mapping symbols to current prices
        """
        if self.alpaca_api is None:  # Only for paper trading
            for symbol, price in price_dict.items():
                if symbol in self.positions:
                    self.positions[symbol]['current_price'] = price
            
            # Update portfolio value with new prices
            self._update_portfolio_value()
    
    def _get_paper_trading_price(self, symbol):
        """
        Get current price for paper trading simulation.
        In a real implementation, this would fetch from a data source.
        
        Args:
            symbol (str): Symbol to get price for
            
        Returns:
            float: Current price (simulated)
        """
        # If position exists, use the current price from there
        if symbol in self.positions and 'current_price' in self.positions[symbol]:
            return self.positions[symbol]['current_price']
        
        # Otherwise, return a placeholder price (in real implementation, this would fetch from API)
        # This is just a placeholder for simulation
        base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'AMZN': 3000.0,
            'META': 300.0,
            'TSLA': 800.0,
            'NVDA': 200.0,
            'JPM': 140.0,
            'V': 220.0,
            'WMT': 140.0
        }
        
        # Use base price if available, otherwise generate a random price
        base_price = base_prices.get(symbol, 100.0)
        variation = random.uniform(-0.02, 0.02)  # 2% random variation
        return base_price * (1 + variation)
    
    def _update_portfolio_value(self):
        """
        Update the portfolio value for paper trading simulation.
        """
        if self.alpaca_api is None:  # Only for paper trading
            positions_value = sum(
                pos['qty'] * pos.get('current_price', pos['avg_price'])
                for pos in self.positions.values()
            )
            self.portfolio_value = self.cash + positions_value

class PerformanceTracker:
    """Class for tracking trading performance metrics."""
    
    def __init__(self, config):
        """Initialize the PerformanceTracker."""
        self.config = config
        self.trades = []
        self.portfolio_values = []
        self.initial_capital = config.INITIAL_CAPITAL
        self.current_capital = config.INITIAL_CAPITAL
        
        # Record initial portfolio value
        self.portfolio_values.append({
            'timestamp': datetime.now().isoformat(),
            'value': self.current_capital
        })
    
    def record_trade(self, trade_result: Dict) -> None:
        """
        Record a trade and update performance metrics.
        
        Args:
            trade_result: Dictionary with trade execution details.
        """
        # Add trade to history
        self.trades.append(trade_result)
        
        # Update capital based on trade result
        if trade_result['success']:
            # For paper trading, we calculate P&L based on trade action
            if trade_result['action'] == 'buy' or trade_result['action'] == 'buy_to_cover':
                # Buying reduces capital
                self.current_capital -= trade_result['price'] * trade_result['quantity']
            elif trade_result['action'] == 'sell' or trade_result['action'] == 'sell_short':
                # Selling increases capital
                self.current_capital += trade_result['price'] * trade_result['quantity']
            
            # Deduct commission if applicable
            if hasattr(self.config, 'COMMISSION'):
                commission = trade_result['price'] * trade_result['quantity'] * self.config.COMMISSION
                self.current_capital -= commission
        
        # Record updated portfolio value
        self.portfolio_values.append({
            'timestamp': datetime.now().isoformat(),
            'value': self.current_capital
        })
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics.
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        
        # For a proper analysis, we need completed round trips (open and close)
        # This is a simplified calculation
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for trade in self.trades:
            if 'pnl' in trade:
                if trade['pnl'] > 0:
                    winning_trades += 1
                    total_profit += trade['pnl']
                else:
                    losing_trades += 1
                    total_loss += abs(trade['pnl'])
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        total_pnl = total_profit - total_loss
        
        # Calculate returns for Sharpe ratio
        if len(self.portfolio_values) > 1:
            # Get portfolio values
            values = [entry['value'] for entry in self.portfolio_values]
            
            # Calculate daily returns (simplified)
            returns = [(values[i] / values[i-1]) - 1 for i in range(1, len(values))]
            
            # Calculate Sharpe ratio
            avg_return = sum(returns) / len(returns) if returns else 0
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0
            
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,  # as percentage
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'return_pct': (self.current_capital / self.initial_capital - 1) * 100,
            'sharpe_ratio': sharpe_ratio
        }
    
    def save_to_csv(self, filename: str) -> None:
        """
        Save performance data to CSV files.
        
        Args:
            filename: Base name for the CSV files.
        """
        try:
            # Save trades
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(f"trades_{filename}", index=False)
            
            # Save portfolio values
            portfolio_df = pd.DataFrame(self.portfolio_values)
            portfolio_df.to_csv(f"portfolio_{filename}", index=False)
            
            # Save performance metrics
            metrics = self.get_performance_metrics()
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f"metrics_{filename}", index=False)
            
            logger.info(f"Performance data saved to CSV files with base name: {filename}")
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}") 