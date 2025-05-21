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

class AutoTrader:
    """Automated trading system that combines signal generation with Alpaca trading."""
    
    def __init__(self, symbols=None, interval='1d', period='1mo', update_interval=60, auto_trade=True, 
                 market_regime='auto', risk_pct=0.02, max_active_positions=5):
        """
        Initialize the AutoTrader.
        
        Args:
            symbols (list): List of stock symbols to trade
            interval (str): Data interval ('1d', '1h', etc.)
            period (str): Data period ('1mo', '3mo', etc.)
            update_interval (int): Time in seconds between updates
            auto_trade (bool): Whether to automatically execute trades
            market_regime (str): Market regime for strategy selection ('trending', 'ranging', 'auto')
            risk_pct (float): Percentage of portfolio to risk per trade
            max_active_positions (int): Maximum number of active positions
        """
        # Load environment variables
        load_dotenv()
        
        # Set configuration
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.interval = interval
        self.period = period
        self.update_interval = update_interval
        self.auto_trade = auto_trade
        self.market_regime = market_regime
        self.risk_pct = risk_pct
        self.max_active_positions = max_active_positions
        
        # Risk management parameters
        self.stop_loss_pct = 5.0
        self.take_profit_pct = 10.0
        
        # Flag for running state
        self.running = False
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.strategy_manager = StrategyManager()
        self.visualizer = Visualizer()
        
        # Initialize the Alpaca trader
        try:
            self.trader = AlpacaTrader()
            logger.info("Alpaca trader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca trader: {str(e)}")
            logger.warning("Running in signal-only mode (no automatic trading)")
            self.auto_trade = False
            self.trader = None
        
        # Set up strategies
        self._setup_strategies()
        
        # Create strategy selector
        self.strategy_selector = best_strategy_factory(market_regime=self.market_regime)
        
        # Dictionary to store signals
        self.signals = {}
        
        logger.info(f"AutoTrader initialized with symbols: {self.symbols}")
        logger.info(f"Auto-trading: {'ENABLED' if self.auto_trade else 'DISABLED'}")
        logger.info(f"Market regime: {self.market_regime}")
    
    def _setup_strategies(self):
        """Set up trading strategies."""
        # Add strategies to manager
        self.strategy_manager.add_strategy(EnhancedMovingAverageCrossover(short_window=20, long_window=50))
        self.strategy_manager.add_strategy(EnhancedRSIStrategy(period=14, overbought=70, oversold=30))
        self.strategy_manager.add_strategy(EnhancedMomentumStrategy(period=10, threshold=5.0))
        self.strategy_manager.add_strategy(BreakoutStrategy())
        self.strategy_manager.add_strategy(MeanReversionStrategy())
        self.strategy_manager.add_strategy(DualStrategySystem())
        
        logger.info("Trading strategies initialized")
    
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
            while self.running:
                # Process all symbols
                self._process_symbols()
                
                # Clean up old figures
                self.visualizer.clean_old_figures(max_figures=30)
                
                # Wait for the next update
                logger.info(f"Waiting {self.update_interval} seconds for next update")
                
                # Break this into smaller chunks to be more responsive to stop command
                chunk_size = 1  # 1 second
                for _ in range(self.update_interval):
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
                signals = self.strategy_manager.generate_signals(stock_data)
                
                if not signals:
                    logger.warning(f"No signals generated for {symbol}")
                    continue
                
                # Store signals for this symbol
                signals_data[symbol] = signals
                
                # Extract and display the signal for each strategy
                for strategy_name, signal_df in signals.items():
                    if signal_df.empty:
                        continue
                    
                    # Check if 'signal' column exists, otherwise default to 0 (hold)
                    if 'signal' not in signal_df.columns:
                        logger.warning(f"No signal column in {strategy_name} for {symbol}")
                        last_signal = 0
                    else:
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
                
                # Create visualizations
                self._create_visualizations(symbol, signals)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
        
        # Execute trades if auto-trading is enabled
        if self.auto_trade and self.trader is not None and signals_data:
            self._execute_trades(signals_data)
    
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
            
            # Only take new positions if we're below our max
            if current_position_count >= self.max_active_positions:
                logger.warning(f"Maximum positions ({self.max_active_positions}) reached. Not opening new positions.")
                
                # Still process existing positions for potential sells
                for symbol in self.trader.current_positions:
                    if symbol in signals_data:
                        # Check if any strategy is giving a sell signal
                        for strategy_name, signal_df in signals_data[symbol].items():
                            if not signal_df.empty and signal_df['signal'].iloc[-1] == -1:
                                logger.info(f"Processing potential sell signal for existing position in {symbol}")
                                self.trader.execute_trade(
                                    symbol=symbol,
                                    signal=-1,  # Force sell signal
                                    strategy=strategy_name,
                                    risk_pct=self.risk_pct
                                )
                                break  # Only need one sell signal
            else:
                # Process all symbols
                trades = self.trader.process_signals(signals_data, self.strategy_selector)
                
                # Log the trades
                for trade in trades:
                    if trade['action'] in ('BUY', 'SELL'):
                        logger.info(f"Trade executed: {trade['action']} {trade['symbol']} (Strategy: {trade['strategy']})")
        
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
    
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automated Stock Trading Bot with Alpaca")
    parser.add_argument("--symbols", nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                      help="List of stock symbols to trade")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "5m"],
                      help="Data interval")
    parser.add_argument("--period", default="1mo", choices=["1mo", "3mo"],
                      help="Data period")
    parser.add_argument("--update-interval", type=int, default=60,
                      help="Time in seconds between updates")
    parser.add_argument("--no-trade", action="store_true",
                      help="Disable auto-trading (signal generation only)")
    parser.add_argument("--market-regime", default="auto", choices=["auto", "trending", "ranging"],
                      help="Market regime for strategy selection")
    parser.add_argument("--risk-percent", type=float, default=0.02,
                      help="Percentage of portfolio to risk per trade (0.02 = 2%)")
    parser.add_argument("--max-positions", type=int, default=5,
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