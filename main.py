#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main module for the Stock Trading Bot.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime
import argparse

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
from auto_trader import AutoTrader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
    logger.info("Created logs directory")

# Add file handler for logging
file_handler = logging.FileHandler(f'logs/trading_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class StockTradingBot:
    """Main class for the Stock Trading Bot."""
    
    def __init__(self, symbols=None, strategy_name='MA Crossover', interval='1d', update_interval=60):
        """
        Initialize the Stock Trading Bot.
        
        Args:
            symbols (list): List of stock symbols to trade
            strategy_name (str): Name of the strategy to use
            interval (str): Data interval ('1d', '1h', etc.)
            update_interval (int): Time in seconds between updates
        """
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.strategy_name = strategy_name
        self.interval = interval
        self.update_interval = update_interval
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.strategy_manager = StrategyManager()
        self.visualizer = Visualizer()
        
        # Set up strategies
        self._setup_strategies()
        
        logger.info(f"Initialized StockTradingBot with symbols: {self.symbols}")
        logger.info(f"Using strategy: {self.strategy_name}")
    
    def _setup_strategies(self):
        """Set up trading strategies with enhanced versions."""
        # Add strategies to manager
        self.strategy_manager.add_strategy(EnhancedMovingAverageCrossover(short_window=10, long_window=30))
        self.strategy_manager.add_strategy(EnhancedRSIStrategy(period=7, overbought=60, oversold=40))
        self.strategy_manager.add_strategy(EnhancedMomentumStrategy(period=5, threshold=2.0))
        self.strategy_manager.add_strategy(BreakoutStrategy())
        self.strategy_manager.add_strategy(MeanReversionStrategy())
        self.strategy_manager.add_strategy(DualStrategySystem())
        
        logger.info("Set up enhanced trading strategies")
    
    def run(self, max_iterations=None):
        """
        Run the trading bot.
        
        Args:
            max_iterations (int, optional): Maximum number of iterations to run
        """
        logger.info("Starting StockTradingBot...")
        logger.info(f"Bot started with symbols: {self.symbols}")
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                logger.info(f"Iteration {iteration+1}")
                
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        # Fetch stock data
                        logger.info(f"Fetching data for {symbol}")
                        stock_data = self.data_fetcher.get_stock_data(symbol, period='1mo', interval=self.interval)
                        
                        if stock_data.empty:
                            logger.warning(f"No data retrieved for {symbol}")
                            continue
                            
                        # Generate signals
                        logger.info(f"Generating signals for {symbol}")
                        signals = self.strategy_manager.generate_signals(stock_data, self.strategy_name)
                        
                        if not signals:
                            logger.warning(f"No signals generated for {symbol}")
                            continue
                            
                        # Get the signals DataFrame for the selected strategy
                        signal_data = signals.get(self.strategy_name)
                        
                        if signal_data is None:
                            logger.warning(f"Strategy '{self.strategy_name}' not found")
                            continue
                        
                        # Get the last signal
                        last_signal = signal_data['signal'].iloc[-1]
                        last_close = signal_data['close'].iloc[-1]
                        
                        # Log the signal
                        signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"
                        logger.info(f"{symbol} signal: {signal_text} at ${last_close:.2f}")
                        
                        # Visualize the data
                        if 'rsi' in signal_data.columns:
                            self.visualizer.plot_rsi(signal_data, symbol)
                        else:
                            self.visualizer.plot_signals(signal_data, symbol)
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                
                # Clean up old figures to avoid filling up the disk and reduce lag
                self.visualizer.clean_old_figures(max_figures=5)
                
                # Increment iteration counter
                iteration += 1
                
                # Wait for the next update
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Waiting {self.update_interval} seconds for next update")
                    time.sleep(self.update_interval)
            
            logger.info("Trading bot completed all iterations")
            
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            logger.info("Trading bot shutting down")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    parser.add_argument("--symbols", nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                      help="List of stock symbols to trade")
    parser.add_argument("--strategy", default="MA Crossover", 
                      choices=["MA Crossover", "RSI Strategy", "Momentum Strategy", 
                              "Breakout Strategy", "Mean Reversion Strategy", "Dual Strategy System"],
                      help="Trading strategy to use")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "5m"],
                      help="Data interval")
    parser.add_argument("--update-interval", type=int, default=5,
                      help="Time in seconds between updates")
    parser.add_argument("--iterations", type=int, default=None,
                      help="Maximum number of iterations to run")
    parser.add_argument("--use-auto-trader", action="store_true",
                      help="Use the AutoTrader for more advanced trading features")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    if args.use_auto_trader:
        # Use the more advanced AutoTrader
        auto_trader = AutoTrader(
            symbols=args.symbols,
            interval=args.interval,
            period='3mo',  # Default to 3 months for more data
            update_interval=args.update_interval,
            auto_trade=True,  # Enable auto-trading by default
            market_regime='auto',
            risk_pct=0.05,  # 5% risk per trade
            max_active_positions=10
        )
        auto_trader.start()
        
        try:
            # Keep the main thread alive until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            auto_trader.stop()
            logger.info("Bot stopped")
    else:
        # Create and run the simple trading bot
        bot = StockTradingBot(
            symbols=args.symbols,
            strategy_name=args.strategy,
            interval=args.interval,
            update_interval=args.update_interval
        )
        
        bot.run(max_iterations=args.iterations) 