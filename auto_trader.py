#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated trading script that combines the Stock Trading Bot with Alpaca Markets trading.
"""

import os
import sys
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
from datetime import datetime
import argparse
from dotenv import load_dotenv
import json

from data_fetcher import DataFetcher
from strategy import (
    EnhancedMovingAverageCrossover, 
    EnhancedRSIStrategy, 
    EnhancedMomentumStrategy, 
    BreakoutStrategy,
    MeanReversionStrategy,
    DualStrategySystem,
    StrategyManager
)
from visualizer import Visualizer
from trader import AlpacaTrader, best_strategy_factory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
    logger.info("Created logs directory")

# Add file handler for logging
file_handler = logging.FileHandler(f'logs/auto_trader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Default settings file
SETTINGS_FILE = 'trading_bot_settings.json'

class AutoTrader:
    """Automated trading system that combines signal generation with Alpaca trading."""
    
    def __init__(self, symbols=None, interval='1d', period='3mo', update_interval=5, auto_trade=True, 
                 market_regime='auto', risk_pct=0.05, max_active_positions=10):
        """
        Initialize the AutoTrader with aggressive default settings.
        
        Args:
            symbols (list): List of stock symbols to trade
            interval (str): Data interval ('1d', '1h', etc.)
            period (str): Data period ('3mo', '6mo', etc.) - Default changed to 3mo for more data
            update_interval (int): Time in seconds between updates (default: 5 seconds)
            auto_trade (bool): Whether to automatically execute trades (default: True)
            market_regime (str): Market regime for strategy selection ('trending', 'ranging', 'auto')
            risk_pct (float): Percentage of portfolio to risk per trade (default: 5%)
            max_active_positions (int): Maximum number of active positions (default: 10)
        """
        # Setup logging
        self.setup_logging()
        
        # Load settings from file if available
        settings = self.load_settings()
        
        # Initialize variables with settings from file or defaults
        if symbols is None and settings and 'symbols' in settings:
            self.symbols = settings['symbols'].split(',')
        else:
            self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            
        self.interval = settings.get('interval', interval) 
        self.period = settings.get('period', period)
        self.update_interval = min(int(settings.get('update_interval', update_interval)), 10)  # Cap at 10 seconds max
        self.auto_trade = settings.get('auto_trade_enabled', auto_trade)
        self.market_regime = settings.get('market_regime', market_regime)
        self.risk_pct = float(settings.get('risk_percent', risk_pct))
        self.max_active_positions = int(settings.get('max_positions', max_active_positions))
        self.running = False
        self.thread = None
        self.save_charts = settings.get('save_charts', True)
        self.notify_on_signals = settings.get('notify_on_signals', True)
        
        # Track consecutive hold signals for each symbol
        self.consecutive_holds = {}
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.visualizer = Visualizer(figure_dir="figures")
        self.strategy_manager = StrategyManager()
        
        # Setup strategies
        self._setup_strategies(settings)
        
        logger.info(f"AutoTrader initialized with {len(self.symbols)} symbols")
        logger.info(f"Auto-trading is {'ENABLED' if self.auto_trade else 'DISABLED'}")
        logger.info(f"Market regime: {self.market_regime}")
        logger.info(f"Risk per trade: {self.risk_pct * 100}%")
        logger.info(f"Maximum active positions: {self.max_active_positions}")
        
        # Save settings to file
        self.save_settings()
        
        # Flag to track if we have a working trader (but maintain the auto_trade setting regardless)
        self.trader_available = False
        
        # Initialize the Alpaca trader
        try:
            self.trader = AlpacaTrader()
            self.trader_available = True
            logger.info("Alpaca trader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca trader: {str(e)}")
            logger.warning("Running in SIMULATION MODE (auto-trading will use simulated positions)")
            # Create a simulated trader for testing
            self.trader = self._create_simulated_trader()
            self.trader_available = True  # We can still use the simulated trader
        
        # Create strategy selector
        self.strategy_selector = best_strategy_factory(market_regime=self.market_regime)
        
        # Dictionary to store signals
        self.signals = {}
        
        # Set risk parameters from settings
        if settings:
            self.set_risk_parameters(
                stop_loss_pct=float(settings.get('stop_loss_pct', 5.0)),
                take_profit_pct=float(settings.get('take_profit_pct', 10.0))
            )
    
    def load_settings(self):
        """Load settings from the settings file."""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                logger.info(f"Loaded settings from {SETTINGS_FILE}")
                return settings
            else:
                logger.warning(f"Settings file {SETTINGS_FILE} not found. Using defaults.")
                return None
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            return None
    
    def save_settings(self):
        """Save current settings to the settings file."""
        try:
            settings = {
                'symbols': ','.join(self.symbols),
                'interval': self.interval,
                'period': self.period,
                'update_interval': self.update_interval,
                'auto_trade_enabled': self.auto_trade,
                'market_regime': self.market_regime,
                'risk_percent': self.risk_pct,
                'max_positions': self.max_active_positions,
                'save_charts': self.save_charts,
                'notify_on_signals': self.notify_on_signals
            }
            
            # Add strategy settings
            for strategy_name in ['ma_crossover', 'rsi_strategy', 'momentum_strategy', 'breakout_strategy', 'mean_reversion', 'dual_strategy']:
                settings[f'{strategy_name}_enabled'] = True  # Enable all strategies by default
            
            # Add risk parameters
            if hasattr(self, 'stop_loss_pct'):
                settings['stop_loss_pct'] = self.stop_loss_pct
            if hasattr(self, 'take_profit_pct'):
                settings['take_profit_pct'] = self.take_profit_pct
            
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            
            logger.info(f"Saved settings to {SETTINGS_FILE}")
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
    
    def _setup_strategies(self, settings=None):
        """Set up trading strategies with more aggressive settings."""
        # Add strategies to manager with more aggressive settings
        self.strategy_manager.add_strategy(EnhancedMovingAverageCrossover(short_window=10, long_window=30))
        self.strategy_manager.add_strategy(EnhancedRSIStrategy(period=7, overbought=60, oversold=40))
        self.strategy_manager.add_strategy(EnhancedMomentumStrategy(period=5, threshold=2.0))
        self.strategy_manager.add_strategy(BreakoutStrategy())
        self.strategy_manager.add_strategy(MeanReversionStrategy())
        self.strategy_manager.add_strategy(DualStrategySystem())
        
        logger.info("Trading strategies initialized with aggressive settings")
    
    def set_auto_trade(self, enabled):
        """Enable or disable auto-trading."""
        self.auto_trade = enabled
        logger.info(f"Auto-trading {'enabled' if enabled else 'disabled'}")
        # Save the setting
        self.save_settings()
    
    def start(self):
        """Start the auto trader."""
        if self.running:
            logger.warning("AutoTrader is already running")
            return
        
        self.running = True
        logger.info("Starting AutoTrader...")
        
        # Start in a separate thread
        self.trader_thread = threading.Thread(target=self._run_loop)
        self.trader_thread.daemon = True
        self.trader_thread.start()
    
    def stop(self):
        """Stop the auto trader."""
        if not self.running:
            logger.warning("AutoTrader is not running")
            return
        
        self.running = False
        logger.info("Stopping AutoTrader...")
    
    def _run_loop(self):
        """Main trading loop."""
        try:
            # Start trading immediately with no initial delay
            logger.info("Starting trading loop immediately")
            
            # Counter for cycle number
            cycle_count = 0
            
            while self.running:
                cycle_count += 1
                # Log the current auto_trade status at start of cycle
                logger.info(f"Trading cycle {cycle_count}: Auto trading is {'ENABLED' if self.auto_trade else 'DISABLED'}")
                
                # Process all symbols
                self._process_symbols()
                
                # Clean up old figures - do this less frequently to save processing time
                if cycle_count % 5 == 0:
                    self.visualizer.clean_old_figures(max_figures=30)
                
                # Wait for the next update - use a shorter interval for more active trading
                actual_interval = min(self.update_interval, 10)  # Cap at 10 seconds max
                logger.info(f"Waiting {actual_interval} seconds for next update")
                
                # Break this into smaller chunks to be more responsive to stop command
                chunk_size = 1  # 1 second
                for _ in range(actual_interval):
                    if not self.running:
                        break
                    time.sleep(chunk_size)
                
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
        finally:
            logger.info("Trading loop stopped")
    
    def _process_symbols(self):
        """Process all symbols to generate signals and execute trades."""
        signals_data = {}
        
        # Store current auto_trade setting before processing
        current_auto_trade_setting = self.auto_trade
        
        for symbol in self.symbols:
            try:
                # Fetch stock data
                logger.info(f"Fetching data for {symbol}")
                stock_data = self.data_fetcher.get_stock_data(symbol, period=self.period, interval=self.interval)
                
                if stock_data.empty:
                    logger.warning(f"No data retrieved for {symbol}")
                    continue
                
                # Generate signals for all strategies
                logger.info(f"Generating signals for {symbol}")
                all_signals = {}
                
                # Try each strategy individually to prevent one failure from affecting others
                for strategy_name, strategy in self.strategy_manager.strategies.items():
                    try:
                        # Apply the strategy to the data
                        signal_df = strategy.generate_signals(stock_data.copy())
                        
                        # Only add to signals if valid
                        if not signal_df.empty and 'signal' in signal_df.columns:
                            all_signals[strategy_name] = signal_df
                        else:
                            logger.warning(f"Strategy {strategy_name} did not generate valid signals for {symbol}")
                    except Exception as e:
                        logger.warning(f"Error with {strategy_name} for {symbol}: {str(e)}")
                        continue
                
                if not all_signals:
                    logger.warning(f"No signals generated for {symbol}")
                    continue
                
                # Store signals for this symbol
                signals_data[symbol] = all_signals
                
                # Extract and display the signal for each strategy
                for strategy_name, signal_df in all_signals.items():
                    try:
                        # Safely get the last signal
                        if 'signal' in signal_df.columns:
                            last_signal = signal_df['signal'].iloc[-1]
                            last_close = signal_df['close'].iloc[-1]
                            
                            # Convert signal to text
                            signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"
                            logger.info(f"{symbol} {strategy_name} signal: {signal_text} at ${last_close:.2f}")
                            
                            # Store the signal in the dictionary
                            if symbol not in self.signals:
                                self.signals[symbol] = {}
                            
                            self.signals[symbol][strategy_name] = {
                                'signal': last_signal,
                                'price': last_close,
                                'signal_text': signal_text,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                    except Exception as e:
                        logger.error(f"Error processing signal for {symbol} with {strategy_name}: {str(e)}")
                
                # Create visualizations
                self._create_visualizations(symbol, all_signals)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Ensure auto_trade setting hasn't been modified during processing
        if self.auto_trade != current_auto_trade_setting:
            logger.warning(f"Auto-trade setting was changed during processing. Restoring to original setting: {current_auto_trade_setting}")
            self.auto_trade = current_auto_trade_setting
        
        # Execute trades if auto-trading is enabled AND we have a working trader connection
        if self.auto_trade and self.trader_available and self.trader is not None and signals_data:
            logger.info(f"Auto-trading is ENABLED. Processing signals for trading.")
            self._execute_trades(signals_data)
        else:
            if not self.auto_trade:
                logger.info("Auto-trading is DISABLED. Not executing trades.")
            elif not self.trader_available or self.trader is None:
                logger.warning("Auto-trading is ENABLED but trader is not available. Cannot execute trades.")
            elif not signals_data:
                logger.warning("Auto-trading is ENABLED but no signals data available for trading.")
    
    def _create_visualizations(self, symbol, signals):
        """Create visualizations for the symbol."""
        try:
            # For each strategy, create appropriate visualization
            for strategy_name, signal_df in signals.items():
                if 'rsi' in signal_df.columns:
                    self.visualizer.plot_rsi(signal_df, f"{symbol}_{strategy_name}")
                else:
                    self.visualizer.plot_signals(signal_df, f"{symbol}_{strategy_name}")
        except Exception as e:
            logger.error(f"Error creating visualizations for {symbol}: {str(e)}")
    
    def _execute_trades(self, signals_data):
        """Execute trades based on signals."""
        try:
            logger.info("Processing signals for trading...")
            
            # Get current positions to check how many we already have
            self.trader.update_positions()
            current_position_count = len(self.trader.current_positions)
            
            logger.info(f"Current positions: {current_position_count}/{self.max_active_positions}")
            
            # Log current positions for better visibility
            if current_position_count > 0:
                for symbol, position in self.trader.current_positions.items():
                    logger.info(f"  Position: {symbol} - {position['qty']} shares @ ${position['avg_entry_price']:.2f}")
            else:
                logger.info("  No active positions")
            
            # Separate symbols into those we already own and those we don't
            owned_symbols = list(self.trader.current_positions.keys())
            unowned_symbols = [s for s in signals_data.keys() if s not in owned_symbols]
            
            logger.info(f"Owned symbols: {owned_symbols}")
            logger.info(f"Potential new positions: {unowned_symbols}")
            
            # Process existing positions first for potential sells
            for symbol in owned_symbols:
                if symbol in signals_data:
                    logger.info(f"Processing existing position in {symbol}")
                    # Use the strategy selector to decide what to do with this position
                    trade_info = self.trader.process_signals({symbol: signals_data[symbol]}, self.strategy_selector)
                    if trade_info and trade_info[0]['action'] == 'SELL' and trade_info[0]['status'] == 'submitted':
                        logger.info(f"Sold position in {symbol}")
                        current_position_count -= 1
            
            # Only look for new positions if we're below our maximum
            if current_position_count < self.max_active_positions and unowned_symbols:
                logger.info(f"Looking for new positions, can open {self.max_active_positions - current_position_count} more")
                
                # Identify buy candidates - only consider BUY signals for new positions
                buy_candidates = []
                
                for symbol in unowned_symbols:
                    has_buy_signal = False
                    # Check if any strategy is giving a buy signal
                    for strategy_name, signal_df in signals_data[symbol].items():
                        if not signal_df.empty and 'signal' in signal_df.columns and signal_df['signal'].iloc[-1] == 1:
                            buy_candidates.append((symbol, strategy_name))
                            has_buy_signal = True
                            break
                    
                    if not has_buy_signal:
                        logger.info(f"No buy signals for {symbol}, skipping")
                
                # Sort buy candidates by preferred strategies
                # Prioritize trend-following strategies in a bull market
                preferred_strategies = ['MA Crossover', 'Momentum Strategy', 'Dual Strategy System']
                buy_candidates.sort(key=lambda x: preferred_strategies.index(x[1]) if x[1] in preferred_strategies else 999)
                
                logger.info(f"Buy candidates: {buy_candidates}")
                
                # Process buy candidates until we reach max positions
                for symbol, strategy in buy_candidates:
                    if current_position_count >= self.max_active_positions:
                        logger.info("Maximum positions reached, stopping buy signal processing")
                        break
                    
                    logger.info(f"Attempting to buy {symbol} based on {strategy} signal")
                    # Execute buy order directly
                    trade_info = self.trader.execute_trade(
                        symbol=symbol,
                        signal=1,  # Force buy signal
                        strategy=strategy,
                        risk_pct=self.risk_pct
                    )
                    
                    if trade_info['status'] == 'submitted':
                        logger.info(f"Opened new position in {symbol}")
                        current_position_count += 1
                    else:
                        logger.warning(f"Failed to open position in {symbol}: {trade_info['status']}")
            
            # Update positions again to get the current state
            self.trader.update_positions()
            new_position_count = len(self.trader.current_positions)
            logger.info(f"Positions after trading: {new_position_count}/{self.max_active_positions}")
            
            # Check if there's any discrepancy in position count
            if new_position_count != current_position_count:
                logger.warning(f"Position count discrepancy: expected {current_position_count}, got {new_position_count}")
        
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_performance(self):
        """Get the trading performance metrics."""
        if self.trader:
            return self.trader.get_strategy_performance()
        return None
    
    def get_signals(self):
        """Get the current signals dictionary."""
        return self.signals
    
    def get_positions(self):
        """Get the current positions."""
        if self.trader:
            self.trader.update_positions()
            return self.trader.current_positions
        return {}

    def set_risk_parameters(self, stop_loss_pct=5.0, take_profit_pct=10.0):
        """Set risk management parameters."""
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        logger.info(f"Risk parameters set: stop loss={stop_loss_pct}%, take profit={take_profit_pct}%")

    def _process_symbol(self, symbol):
        """
        Process a single symbol: fetch data, generate signals, and execute trades.
        
        Args:
            symbol (str): Stock symbol to process
        """
        try:
            # Preserve auto_trade setting before processing
            auto_trade_setting = self.auto_trade
            
            # Fetch data
            df = self.data_fetcher.fetch_data(symbol, interval=self.interval, period=self.period)
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Check if market is open
            market_open = self.data_fetcher.is_market_open()
            
            # Generate signals using strategy manager
            signals_df = self.strategy_manager.generate_signals(df, market_regime=self.market_regime)
            
            # Get the latest signal
            latest_signal = signals_df['signal'].iloc[-1] if not signals_df.empty else 0
            
            # Log signal with more detail
            signal_text = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(latest_signal, f"UNKNOWN({latest_signal})")
            logger.info(f"Generated {signal_text} signal for {symbol}")
            
            # FORCE SIGNAL GENERATION: If we keep getting hold signals, randomly generate a buy or sell
            # This is for testing purposes only - remove in production
            if latest_signal == 0 and market_open:
                # Check if we've had consecutive hold signals
                if symbol in self.consecutive_holds:
                    self.consecutive_holds[symbol] += 1
                else:
                    self.consecutive_holds[symbol] = 1
                
                # After 3 consecutive holds, force a buy or sell signal
                if self.consecutive_holds[symbol] >= 3:
                    import random
                    latest_signal = random.choice([1, -1])  # Randomly choose buy or sell
                    logger.info(f"FORCED {signal_text} signal for {symbol} after {self.consecutive_holds[symbol]} consecutive holds")
                    self.consecutive_holds[symbol] = 0
            else:
                # Reset consecutive holds counter if we got a non-hold signal
                self.consecutive_holds[symbol] = 0
            
            # Create visualizations
            if self.save_charts:
                fig_path = self.visualizer.create_chart(
                    signals_df, 
                    symbol, 
                    title=f"{symbol} - {self.market_regime.upper()} Market",
                    save=True
                )
                logger.info(f"Chart saved to {fig_path}")
            
            # Execute trade if auto-trading is enabled and market is open
            if auto_trade_setting and market_open and latest_signal != 0:
                # Double check that auto_trade is still enabled
                if not self.auto_trade:
                    logger.warning("Auto-trade setting changed during processing. Restoring original setting.")
                    self.auto_trade = auto_trade_setting
                
                logger.info(f"Auto-trading enabled. Executing {signal_text} for {symbol}")
                trade_result = self.trader.execute_trade(
                    symbol=symbol,
                    signal=latest_signal,
                    strategy=self.market_regime,
                    risk_pct=self.risk_pct
                )
                
                # Log trade result
                logger.info(f"Trade result: {trade_result}")
                
                # Notify if configured
                if self.notify_on_signals:
                    self._send_notification(symbol, latest_signal, trade_result)
            else:
                if not auto_trade_setting:
                    logger.info(f"Auto-trading disabled. Not executing trade for {symbol}")
                elif not market_open:
                    logger.info(f"Market closed. Not executing trade for {symbol}")
                elif latest_signal == 0:
                    logger.info(f"Hold signal. No trade needed for {symbol}")
            
            # Ensure auto_trade setting is preserved
            if self.auto_trade != auto_trade_setting:
                logger.warning(f"Auto-trade setting was changed during processing from {auto_trade_setting} to {self.auto_trade}. Restoring to {auto_trade_setting}.")
                self.auto_trade = auto_trade_setting
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            
    def setup_logging(self):
        """Set up logging for the AutoTrader."""
        # Add file handler for logging
        file_handler = logging.FileHandler(f'logs/auto_trader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    def _create_simulated_trader(self):
        """Create a simulated trader for testing."""
        class SimulatedTrader:
            def __init__(self):
                self.current_positions = {}
                self.strategy_performance = {}
            
            def update_positions(self):
                """Simulate updating positions."""
                logger.info("SIMULATION: Updated positions")
                for symbol, position in self.current_positions.items():
                    logger.info(f"SIMULATION: Current position: {symbol} - {position['qty']} shares @ ${position['avg_entry_price']:.2f}")
                return
            
            def execute_trade(self, symbol, signal, quantity=None, strategy='auto', risk_pct=0.02):
                """Simulate executing a trade."""
                signal_types = {1: "BUY", -1: "SELL", 0: "HOLD"}
                signal_text = signal_types.get(signal, f"UNKNOWN({signal})")
                logger.info(f"SIMULATION: {signal_text} signal for {symbol} from {strategy}")
                
                # Check if we have an existing position for this symbol
                position_exists = symbol in self.current_positions
                
                # For SELL or HOLD signals, we must have an existing position
                if signal in [-1, 0] and not position_exists:
                    logger.warning(f"SIMULATION: Can't {signal_text} {symbol} - no position exists")
                    return {'symbol': symbol, 'action': signal_text, 'status': 'no_position', 'strategy': strategy}
                
                if signal == 0:  # Hold
                    return {'symbol': symbol, 'action': 'HOLD', 'status': 'no_action', 'strategy': strategy}
                
                elif signal == 1:  # Buy
                    # If we don't have a position, create one
                    if not position_exists:
                        price = 100.0  # Simulated price
                        qty = 10  # Default quantity
                        
                        # Add to positions
                        self.current_positions[symbol] = {
                            'qty': qty,
                            'avg_entry_price': price,
                            'current_price': price,
                            'market_value': price * qty,
                            'unrealized_pl': 0.0,
                            'unrealized_plpc': 0.0
                        }
                        
                        logger.info(f"SIMULATION: BUY order placed for {qty} shares of {symbol} at ~${price:.2f}")
                        
                        return {
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': qty,
                            'price': price,
                            'order_id': f'sim-buy-{symbol}',
                            'status': 'submitted',
                            'strategy': strategy
                        }
                    else:
                        # Add more to existing position
                        price = 100.0  # Simulated price
                        qty = 5  # Add fewer shares when adding to position
                        
                        # Update position
                        current_qty = self.current_positions[symbol]['qty']
                        current_value = current_qty * self.current_positions[symbol]['avg_entry_price']
                        new_value = qty * price
                        new_qty = current_qty + qty
                        new_avg_price = (current_value + new_value) / new_qty
                        
                        self.current_positions[symbol] = {
                            'qty': new_qty,
                            'avg_entry_price': new_avg_price,
                            'current_price': price,
                            'market_value': price * new_qty,
                            'unrealized_pl': 0.0,
                            'unrealized_plpc': 0.0
                        }
                        
                        logger.info(f"SIMULATION: Added {qty} shares to {symbol} position, now {new_qty} shares @ ${new_avg_price:.2f}")
                        
                        return {
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': qty,
                            'price': price,
                            'order_id': f'sim-add-{symbol}',
                            'status': 'submitted',
                            'strategy': strategy
                        }
                
                elif signal == -1:  # Sell
                    # We already checked that the position exists above
                    qty = self.current_positions[symbol]['qty']
                    price = 100.0  # Simulated price
                    
                    # Remove from positions
                    del self.current_positions[symbol]
                    
                    logger.info(f"SIMULATION: SELL order placed for {qty} shares of {symbol} at ~${price:.2f}")
                    
                    return {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': qty,
                        'price': price,
                        'order_id': f'sim-sell-{symbol}',
                        'status': 'submitted',
                        'strategy': strategy
                    }
            
            def process_signals(self, signals_data, strategy_selector=None):
                """Simulate processing signals."""
                # Just pass through to execute_trade for each symbol
                executed_trades = []
                
                for symbol, signals in signals_data.items():
                    # Get the last signal for each available strategy
                    strategy_signals = {}
                    
                    # Check if signals is a dictionary (multiple strategies) or DataFrame (single strategy)
                    if isinstance(signals, dict):
                        # Multiple strategies
                        for strategy_name, signal_df in signals.items():
                            if not signal_df.empty and 'signal' in signal_df.columns:
                                strategy_signals[strategy_name] = signal_df['signal'].iloc[-1]
                    else:
                        # Single strategy
                        if not signals.empty and 'signal' in signals.columns:
                            strategy_signals['default'] = signals['signal'].iloc[-1]
                    
                    if not strategy_signals:
                        logger.warning(f"SIMULATION: No valid signals found for {symbol}")
                        continue
                    
                    # Check if we have a position for this symbol
                    position_exists = symbol in self.current_positions
                    
                    # If no position exists, we can only BUY
                    if not position_exists:
                        # Look for BUY signals only
                        buy_strategies = {name: signal for name, signal in strategy_signals.items() if signal == 1}
                        if buy_strategies:
                            # Pick the first buy strategy
                            best_strategy = next(iter(buy_strategies.keys()))
                            trade_info = self.execute_trade(symbol, 1, strategy=best_strategy)
                            executed_trades.append(trade_info)
                        else:
                            logger.info(f"SIMULATION: No BUY signals for {symbol} and no existing position - skipping")
                    else:
                        # If we have a position, we can process any signal
                        if strategy_selector:
                            best_strategy = strategy_selector(symbol, strategy_signals, {})
                        else:
                            best_strategy = next(iter(strategy_signals.keys()))
                        
                        signal = strategy_signals.get(best_strategy, 0)
                        trade_info = self.execute_trade(symbol, signal, strategy=best_strategy)
                        executed_trades.append(trade_info)
                
                return executed_trades
            
            def get_strategy_performance(self):
                """Return empty performance data."""
                return {}
        
        return SimulatedTrader()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automated Stock Trading Bot with Alpaca")
    parser.add_argument("--symbols", nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                      help="List of stock symbols to trade")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "5m"],
                      help="Data interval")
    parser.add_argument("--period", default="1mo", choices=["1mo", "3mo"],
                      help="Data period")
    parser.add_argument("--update-interval", type=int, default=5,
                      help="Time in seconds between updates")
    parser.add_argument("--no-trade", action="store_true",
                      help="Disable auto-trading (signal generation only)")
    parser.add_argument("--market-regime", default="auto", choices=["auto", "trending", "ranging"],
                      help="Market regime for strategy selection")
    parser.add_argument("--risk-percent", type=float, default=0.05,
                      help="Percentage of portfolio to risk per trade (0.05 = 5%)")
    parser.add_argument("--max-positions", type=int, default=10,
                      help="Maximum number of active positions")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and start the auto trader
    trader = AutoTrader(
        symbols=args.symbols,
        interval=args.interval,
        period=args.period,
        update_interval=args.update_interval,
        auto_trade=not args.no_trade,
        market_regime=args.market_regime,
        risk_pct=args.risk_percent,
        max_active_positions=args.max_positions
    )
    
    try:
        # Start the auto trader
        trader.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop the auto trader when Ctrl+C is pressed
        trader.stop()
        logger.info("AutoTrader stopped by user") 