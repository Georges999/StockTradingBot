#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for using the StockTradingBot.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from config import Config
from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from strategies import StrategyManager
from risk_management import RiskManager
from trade_executor import TradeExecutor
from utils import setup_logger
from visualizer import Visualizer
from backtester import Backtester

def main():
    """Main function to demonstrate the StockTradingBot usage."""
    # Set up logging
    log_file = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)
    logger.info("Starting StockTradingBot example...")
    
    # Initialize configuration
    config = Config()
    
    # Example 1: Basic usage - Get data and generate signals
    print("\n=== Example 1: Basic Usage ===")
    data_fetcher = DataFetcher(config)
    indicators = TechnicalIndicators()
    strategy_manager = StrategyManager(config)
    
    # Get market data for a symbol
    symbol = "AAPL"
    timeframe = "1d"
    lookback_period = 100
    
    print(f"Fetching data for {symbol}...")
    data = data_fetcher.get_market_data(symbol, timeframe, lookback_period)
    
    if not data.empty:
        print(f"Data retrieved: {len(data)} candles")
        
        # Calculate indicators
        print("Calculating technical indicators...")
        analysis_data = indicators.calculate_indicators(data, config.INDICATORS)
        
        # Generate signals
        print("Generating trading signals...")
        signals = strategy_manager.generate_signals(analysis_data, symbol)
        
        # Display signal
        print(f"Signal for {symbol}: {signals['signal']} ({signals['direction']})")
        print(f"Signal strength: {signals['strength']:.2f}")
        print(f"Reason: {signals['reason']}")
        
        # Visualize data with signals
        print("Creating visualization...")
        visualizer = Visualizer()
        visualizer.plot_data_with_signals(analysis_data, signals, symbol)
    else:
        print(f"Failed to retrieve data for {symbol}")
    
    # Example 2: Running a backtest
    print("\n=== Example 2: Running a Backtest ===")
    
    # Initialize backtester
    backtester = Backtester(config)
    
    # Define backtest parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]
    strategy = "combined"
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Running backtest for {', '.join(symbols)} with {strategy} strategy from {start_date} to {end_date}...")
    
    # Run backtest
    backtest_result = backtester.run_backtest(symbols, strategy, start_date, end_date)
    
    if backtest_result.get('success', False):
        # Display backtest results
        metrics = backtest_result['metrics']
        print(f"Backtest completed successfully!")
        print(f"Total return: {backtest_result['total_return']:.2f}%")
        print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Win rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"Total trades: {metrics.get('total_trades', 0)}")
    else:
        print(f"Backtest failed: {backtest_result.get('error', 'Unknown error')}")
    
    # Example 3: Compare different strategies
    print("\n=== Example 3: Strategy Comparison ===")
    
    # Define strategies to compare
    strategies = ["momentum", "mean_reversion", "breakout", "combined"]
    symbol = "AAPL"  # Use a single symbol for simplicity
    
    print(f"Comparing strategies for {symbol}: {', '.join(strategies)}...")
    
    # Run multiple strategy backtest
    comparison_results = backtester.run_multiple_strategies([symbol], strategies, start_date, end_date)
    
    # Display comparison results
    print("Strategy comparison results:")
    for strategy, result in comparison_results.items():
        if result.get('success', False):
            print(f"- {strategy}: {result['total_return']:.2f}% return, " +
                  f"Sharpe: {result['metrics'].get('sharpe_ratio', 0):.2f}")
        else:
            print(f"- {strategy}: Failed - {result.get('error', 'Unknown error')}")
    
    # Example 4: Parameter optimization (simplified)
    print("\n=== Example 4: Parameter Optimization ===")
    
    # Define parameter grid for optimization
    parameter_grid = {
        'rsi_threshold': [50, 55, 60],
        'trend_strength': [0.3, 0.5, 0.7]
    }
    
    strategy_to_optimize = "momentum"
    optimization_symbol = "AAPL"
    
    print(f"Optimizing parameters for {strategy_to_optimize} strategy on {optimization_symbol}...")
    
    # Run optimization
    optimization_result = backtester.optimize_strategy_parameters(
        optimization_symbol, strategy_to_optimize, parameter_grid, start_date, end_date
    )
    
    if optimization_result.get('success', False):
        best_params = optimization_result['best_parameters']
        best_metrics = optimization_result['best_metrics']
        
        print("Optimization completed!")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"- {param}: {value}")
        
        print(f"Performance with best parameters:")
        print(f"- Return: {best_metrics.get('total_return', 0):.2f}%")
        print(f"- Sharpe: {best_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"- Win rate: {best_metrics.get('win_rate', 0):.2f}%")
    else:
        print(f"Optimization failed: {optimization_result.get('error', 'Unknown error')}")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main() 