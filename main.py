#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the StockTradingBot.
Orchestrates the entire trading workflow.
"""

import os
import time
import logging
from datetime import datetime
import pandas as pd

from dotenv import load_dotenv
from config import Config
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from strategies import StrategyManager
from risk_management import RiskManager
from trade_executor import TradeExecutor
from utils import setup_logger
from visualizer import Visualizer

# Load environment variables
load_dotenv()

def main():
    """Main function to run the trading bot."""
    # Set up logging
    log_file = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)
    logger.info("Starting StockTradingBot...")
    
    try:
        # Initialize components
        config = Config()
        data_fetcher = DataFetcher(config)
        indicators = TechnicalIndicators()
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
            
            # Perform trading operations for each symbol
            for symbol in config.SYMBOLS:
                try:
                    # Get market data
                    logger.info(f"Fetching data for {symbol}")
                    data = data_fetcher.get_market_data(symbol, config.TIMEFRAME, config.LOOKBACK_PERIOD)
                    
                    # Calculate technical indicators
                    analysis_data = indicators.calculate_indicators(data, config.INDICATORS)
                    
                    # Generate trading signals
                    signals = strategy_manager.generate_signals(analysis_data, symbol)
                    
                    # Apply risk management
                    trade_decisions = risk_manager.evaluate_trades(signals, analysis_data, symbol)
                    
                    # Execute trades
                    if trade_decisions:
                        for decision in trade_decisions:
                            trade_executor.execute_trade(decision)
                            logger.info(f"Trade executed: {decision}")
                    
                    # Visualize current data (if in debug mode)
                    if config.DEBUG_MODE:
                        visualizer.plot_data_with_signals(analysis_data, signals, symbol)
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
            
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