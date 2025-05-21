#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration settings for the StockTradingBot.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')  # Default to paper trading

# General settings
LOG_LEVEL = 'INFO'
TIMEZONE = 'America/New_York'
MARKET_DATA_SOURCE = 'yfinance'  # Options: 'yfinance', 'alpaca'

# Trading parameters
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
DEFAULT_TIMEFRAME = '1d'  # Options: '1m', '5m', '15m', '30m', '1h', '1d'
LOOKBACK_PERIOD = 100  # Number of bars to retrieve for analysis

# Strategy parameters
STRATEGY = 'combined'  # Options: 'momentum', 'mean_reversion', 'breakout', 'combined'

# Momentum strategy
MOMENTUM_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Mean reversion strategy
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Breakout strategy
BREAKOUT_PERIOD = 20
ATR_PERIOD = 14
ATR_MULTIPLIER = 2

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Risk management
MAX_POSITION_SIZE = 0.05  # Maximum position size as percentage of portfolio
STOP_LOSS_PCT = 0.02  # Stop loss percentage
TAKE_PROFIT_PCT = 0.04  # Take profit percentage
MAX_OPEN_POSITIONS = 5  # Maximum number of open positions

# Backtesting parameters
BACKTEST_START_DATE = '2020-01-01'
BACKTEST_END_DATE = '2023-01-01'
INITIAL_CAPITAL = 100000
COMMISSION_PCT = 0.001  # 0.1% commission per trade

# Optimization parameters
OPTIMIZATION_METRIC = 'sharpe_ratio'  # Options: 'total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown'

class Config:
    """Configuration class for the StockTradingBot."""
    
    def __init__(self):
        # Trading parameters
        self.LIVE_TRADING = False  # Set to True for live trading, False for paper trading
        self.SYMBOLS = DEFAULT_SYMBOLS
        self.TIMEFRAME = DEFAULT_TIMEFRAME
        self.LOOKBACK_PERIOD = LOOKBACK_PERIOD
        self.STRATEGY = STRATEGY
        self.POLLING_INTERVAL = 60  # Seconds between updates
        
        # API credentials
        self.ALPACA_API_KEY = ALPACA_API_KEY
        self.ALPACA_SECRET_KEY = ALPACA_SECRET_KEY
        self.ALPACA_BASE_URL = ALPACA_BASE_URL
        
        # Risk management
        self.MAX_POSITION_SIZE = MAX_POSITION_SIZE
        self.STOP_LOSS_PCT = STOP_LOSS_PCT
        self.TAKE_PROFIT_PCT = TAKE_PROFIT_PCT
        self.MAX_OPEN_POSITIONS = MAX_OPEN_POSITIONS
        self.PORTFOLIO_STOP_LOSS = 0.15  # Close all positions if portfolio drops by this percentage
        
        # Technical indicators to use
        self.INDICATORS = {
            'sma': {'periods': [20, 50, 200]},
            'ema': {'periods': [9, 21]},
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14},
            'vwap': {},
            'support_resistance': {'periods': 30, 'tolerance': 0.02}
        }
        
        # Strategy parameters
        self.STRATEGY_PARAMS = {
            'momentum': {
                'rsi_threshold': 55,
                'macd_threshold': 0,
                'trend_strength': 0.5
            },
            'mean_reversion': {
                'bb_threshold': 0.85,
                'rsi_low': 30,
                'rsi_high': 70
            },
            'breakout': {
                'volume_factor': 2.0,
                'range_factor': 1.5,
                'breakout_periods': 20
            },
            'combined': {
                'strategy_weights': {
                    'momentum': 0.4,
                    'mean_reversion': 0.4,
                    'breakout': 0.2
                },
                'signal_threshold': 0.6
            }
        }
        
        # Backtesting parameters
        self.BACKTEST_START_DATE = BACKTEST_START_DATE
        self.BACKTEST_END_DATE = BACKTEST_END_DATE
        self.INITIAL_CAPITAL = INITIAL_CAPITAL
        self.COMMISSION = COMMISSION_PCT
        
        # Performance tracking
        self.SAVE_PERFORMANCE = True
        self.PERFORMANCE_METRICS = [
            'total_return', 'annualized_return', 'sharpe_ratio', 
            'max_drawdown', 'win_rate', 'profit_factor'
        ]
        
        # Debug and visualization
        self.DEBUG_MODE = True
        self.LOG_LEVEL = LOG_LEVEL
        self.PLOT_TRADES = True 