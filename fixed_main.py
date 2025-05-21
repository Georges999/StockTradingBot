#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed main entry point for the StockTradingBot.
"""

import os
import time
import logging
import shutil
from datetime import datetime
import pandas as pd

from dotenv import load_dotenv
from config import Config
from data_fetcher import DataFetcher
from fixed_indicators import TechnicalIndicators  # Import the fixed indicators
from fixed_strategies import StrategyManager  # Use fixed strategies
from risk_management import RiskManager
from trade_executor import TradeExecutor
from utils import setup_logger
from visualizer import Visualizer

# Load environment variables
load_dotenv()

def clean_old_figures(max_figures=20):
    """
    Clean old figures, keeping only the most recent ones.
    
    Args:
        max_figures (int): Maximum number of figures to keep
    """
    if not os.path.exists('figures'):
        os.makedirs('figures')
        return
        
    files = [os.path.join('figures', f) for f in os.listdir('figures') if f.endswith('.png')]
    files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first
    
    # Delete old files if we have more than max_figures
    if len(files) > max_figures:
        for file_to_delete in files[max_figures:]:
            try:
                os.remove(file_to_delete)
                logging.info(f"Deleted old figure: {file_to_delete}")
            except Exception as e:
                logging.error(f"Error deleting {file_to_delete}: {str(e)}")

def main():
    """Main function to run the trading bot."""
    # Set up logging
    log_file = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Note: The utils.setup_logger function already adds 'logs/' to the path,
    # so we just pass the filename directly
    logger = setup_logger(log_file)
    logger.info("Starting StockTradingBot with fixed indicators...")
    
    # Clean old figures
    clean_old_figures()
    
    # Initialize counter for figure saving
    plot_counter = 0
    # Only save a figure every N iterations to avoid too many figures
    PLOT_INTERVAL = 5
    
    try:
        # Initialize components
        config = Config()
        # Disable debug mode to prevent visualization issues
        config.DEBUG_MODE = False
        
        data_fetcher = DataFetcher(config)
        indicators = TechnicalIndicators()  # Use the fixed TechnicalIndicators
        strategy_manager = StrategyManager(config)
        risk_manager = RiskManager(config)
        trade_executor = TradeExecutor(config)
        visualizer = Visualizer()
        
        logger.info(f"Trading mode: {'Live' if config.LIVE_TRADING else 'Paper'}")
        logger.info(f"Selected strategy: {config.STRATEGY}")
        logger.info(f"Trading symbols: {config.SYMBOLS}")
        
        # Main trading loop
        while True:
            # Check if market is open
            if not data_fetcher.is_market_open() and config.LIVE_TRADING:
                next_open = data_fetcher.get_next_market_open()
                logger.info(f"Market is closed. Next open: {next_open}")
                sleep_time = min((next_open - datetime.now()).total_seconds(), 3600)
                time.sleep(max(1, sleep_time))
                continue
            
            # Increment plot counter for this iteration
            plot_counter += 1
            
            # Perform trading operations for each symbol
            for symbol in config.SYMBOLS:
                try:
                    # Get market data
                    logger.info(f"Fetching data for {symbol}")
                    data = data_fetcher.get_market_data(symbol, config.TIMEFRAME, config.LOOKBACK_PERIOD)
                    
                    # Instead of using TechnicalIndicators, we'll use simplified price-based signals
                    # Create a basic analysis data with price change percentages
                    analysis_data = data.copy()
                    
                    # Calculate simple price-based features
                    analysis_data['price_pct_change'] = analysis_data['close'].pct_change() * 100
                    analysis_data['price_5d_change'] = analysis_data['close'].pct_change(5) * 100
                    analysis_data['price_10d_change'] = analysis_data['close'].pct_change(10) * 100
                    analysis_data['volume_change'] = analysis_data['volume'].pct_change() * 100
                    
                    # Simple moving averages without using TechnicalIndicators
                    analysis_data['sma_20'] = analysis_data['close'].rolling(window=20).mean()
                    analysis_data['sma_50'] = analysis_data['close'].rolling(window=50).mean()
                    
                    # Generate trading signals
                    signals = strategy_manager.generate_signals(analysis_data, symbol)
                    
                    # Apply risk management
                    trade_decisions = risk_manager.evaluate_trades(signals, analysis_data, symbol)
                    
                    # Execute trades
                    if trade_decisions:
                        for decision in trade_decisions:
                            try:
                                # Use submit_order instead of execute_trade
                                symbol = decision.get('symbol')
                                side = decision.get('side', 'buy')
                                quantity = decision.get('quantity', 1)
                                order_type = decision.get('order_type', 'market')
                                
                                # Submit the order using the TradeExecutor's existing method
                                order_result = trade_executor.submit_order(
                                    symbol=symbol,
                                    qty=quantity,
                                    side=side,
                                    order_type=order_type
                                )
                                
                                if order_result:
                                    logger.info(f"Trade executed: {order_result}")
                                else:
                                    logger.warning(f"Failed to execute trade: {decision}")
                            except Exception as e:
                                logger.error(f"Error executing trade: {str(e)}")
                    
                    # Visualize current data (only periodically to avoid too many figures)
                    if plot_counter % PLOT_INTERVAL == 0:
                        visualizer.plot_data_with_signals(analysis_data, signals, symbol)
                        logger.info(f"Saved visualization for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
            
            # Clean old figures every 10 iterations
            if plot_counter % 10 == 0:
                clean_old_figures()
            
            # Wait for the next iteration
            logger.info(f"Waiting for {config.POLLING_INTERVAL} seconds before next update")
            time.sleep(config.POLLING_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
    finally:
        # Clean up and save performance metrics
        logger.info("Shutting down trading bot")
        if config.SAVE_PERFORMANCE and hasattr(trade_executor, 'performance'):
            trade_executor.performance.save_to_csv('performance_metrics.csv')

if __name__ == "__main__":
    main() 