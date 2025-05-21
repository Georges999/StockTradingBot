#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading strategies module for the StockTradingBot.
Implements various technical and algorithmic trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Base class for all trading strategies.
    """
    def __init__(self, config=None):
        """
        Initialize strategy with optional configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.indicators = TechnicalIndicators()
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy.
        Must be implemented by subclasses.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def add_indicators(self, data):
        """
        Add required indicators for the strategy.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Default implementation adds all indicators
        return TechnicalIndicators.add_indicators(data)

class MomentumStrategy(Strategy):
    """
    Momentum strategy using RSI and MACD indicators.
    """
    def __init__(self, config=None, rsi_period=14, rsi_overbought=70, 
                 rsi_oversold=30, macd_fast=12, macd_slow=26, macd_signal=9):
        """
        Initialize momentum strategy.
        
        Args:
            config: Configuration object
            rsi_period (int): RSI period
            rsi_overbought (int): RSI overbought threshold
            rsi_oversold (int): RSI oversold threshold
            macd_fast (int): MACD fast period
            macd_slow (int): MACD slow period
            macd_signal (int): MACD signal period
        """
        super().__init__(config)
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def add_indicators(self, data):
        """
        Add momentum indicators to price data.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Add specific indicators needed for this strategy
        data = TechnicalIndicators.add_rsi(data, period=self.rsi_period)
        data = TechnicalIndicators.add_macd(data, 
                                           fast_period=self.macd_fast,
                                           slow_period=self.macd_slow,
                                           signal_period=self.macd_signal)
        data = TechnicalIndicators.add_ema(data, periods=[20, 50])
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on momentum indicators.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        # Ensure data has necessary indicators
        if 'rsi' not in data.columns or 'macd' not in data.columns:
            data = self.add_indicators(data)
        
        # Initialize signals column: 1 for buy, -1 for sell, 0 for hold
        data['signal'] = 0
        
        # RSI conditions
        rsi_buy_condition = (data['rsi'] < self.rsi_oversold) & (data['rsi'].shift(1) <= self.rsi_oversold)
        rsi_sell_condition = (data['rsi'] > self.rsi_overbought) & (data['rsi'].shift(1) >= self.rsi_overbought)
        
        # MACD conditions
        macd_buy_condition = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
        macd_sell_condition = (data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))
        
        # EMA conditions
        ema_buy_condition = data['close'] > data['ema_20'] and data['ema_20'] > data['ema_50']
        ema_sell_condition = data['close'] < data['ema_20'] and data['ema_20'] < data['ema_50']
        
        # Combined signals
        # Buy when RSI is oversold and MACD crosses above signal line
        data.loc[rsi_buy_condition & macd_buy_condition, 'signal'] = 1
        
        # Sell when RSI is overbought and MACD crosses below signal line
        data.loc[rsi_sell_condition & macd_sell_condition, 'signal'] = -1
        
        return data

class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands.
    """
    def __init__(self, config=None, bb_period=20, bb_std_dev=2):
        """
        Initialize mean reversion strategy.
        
        Args:
            config: Configuration object
            bb_period (int): Bollinger Bands period
            bb_std_dev (int): Bollinger Bands standard deviation multiplier
        """
        super().__init__(config)
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
    
    def add_indicators(self, data):
        """
        Add mean reversion indicators to price data.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Add specific indicators needed for this strategy
        data = TechnicalIndicators.add_bollinger_bands(data, 
                                                      period=self.bb_period, 
                                                      std_dev=self.bb_std_dev)
        data = TechnicalIndicators.add_rsi(data)  # Add RSI as a confirmation indicator
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on mean reversion strategy.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        # Ensure data has necessary indicators
        if 'bb_lower' not in data.columns:
            data = self.add_indicators(data)
        
        # Initialize signals column
        data['signal'] = 0
        
        # Bollinger Band signals
        # Buy when price touches the lower band and RSI < 30
        bb_buy_condition = (data['close'] <= data['bb_lower']) & (data['rsi'] < 30)
        
        # Sell when price touches the upper band and RSI > 70
        bb_sell_condition = (data['close'] >= data['bb_upper']) & (data['rsi'] > 70)
        
        # Additional middle band cross conditions
        middle_cross_up = (data['close'] > data['bb_middle']) & (data['close'].shift(1) <= data['bb_middle'].shift(1))
        middle_cross_down = (data['close'] < data['bb_middle']) & (data['close'].shift(1) >= data['bb_middle'].shift(1))
        
        # Set signals
        data.loc[bb_buy_condition, 'signal'] = 1
        data.loc[bb_sell_condition, 'signal'] = -1
        
        # Additional rules: Exit buy position when crossing middle band from below
        data.loc[middle_cross_up & (data['signal'].shift(1) == 1), 'signal'] = 0
        
        # Exit sell position when crossing middle band from above
        data.loc[middle_cross_down & (data['signal'].shift(1) == -1), 'signal'] = 0
        
        return data

class BreakoutStrategy(Strategy):
    """
    Breakout strategy using price channels and ATR.
    """
    def __init__(self, config=None, period=20, atr_period=14, atr_multiplier=2):
        """
        Initialize breakout strategy.
        
        Args:
            config: Configuration object
            period (int): Breakout period
            atr_period (int): ATR period
            atr_multiplier (int): ATR multiplier for stop levels
        """
        super().__init__(config)
        self.period = period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    def add_indicators(self, data):
        """
        Add breakout indicators to price data.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Calculate high and low price channels
        data['high_channel'] = data['high'].rolling(window=self.period).max()
        data['low_channel'] = data['low'].rolling(window=self.period).min()
        
        # Add ATR for stop loss calculation
        data = TechnicalIndicators.add_atr(data, period=self.atr_period)
        
        # Add volume for confirmation
        data['volume_sma'] = data['volume'].rolling(window=self.period).mean()
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on breakout strategy.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        # Ensure data has necessary indicators
        if 'high_channel' not in data.columns:
            data = self.add_indicators(data)
        
        # Initialize signals column
        data['signal'] = 0
        
        # Breakout signals with confirmation
        # Buy on breakout above the high channel with increasing volume
        breakout_up = (data['close'] > data['high_channel'].shift(1)) & (data['volume'] > data['volume_sma'])
        
        # Sell on breakout below the low channel with increasing volume
        breakout_down = (data['close'] < data['low_channel'].shift(1)) & (data['volume'] > data['volume_sma'])
        
        # Set signals
        data.loc[breakout_up, 'signal'] = 1
        data.loc[breakout_down, 'signal'] = -1
        
        # Calculate stop loss levels
        data['stop_loss_long'] = data['close'] - (data['atr'] * self.atr_multiplier)
        data['stop_loss_short'] = data['close'] + (data['atr'] * self.atr_multiplier)
        
        # Exit signals based on price crossing the stop loss
        exit_long = (data['close'] < data['stop_loss_long'].shift(1)) & (data['signal'].shift(1) == 1)
        exit_short = (data['close'] > data['stop_loss_short'].shift(1)) & (data['signal'].shift(1) == -1)
        
        # Apply exit signals
        data.loc[exit_long | exit_short, 'signal'] = 0
        
        return data

class CombinedStrategy(Strategy):
    """
    Combined strategy that uses multiple sub-strategies to generate signals.
    """
    def __init__(self, config=None):
        """
        Initialize combined strategy.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.momentum_strategy = MomentumStrategy(config)
        self.mean_reversion_strategy = MeanReversionStrategy(config)
        self.breakout_strategy = BreakoutStrategy(config)
        
        # Weights for each strategy
        self.weights = {
            'momentum': 0.4,
            'mean_reversion': 0.3,
            'breakout': 0.3
        }
    
    def add_indicators(self, data):
        """
        Add all necessary indicators from sub-strategies.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Add indicators from all strategies
        data = self.momentum_strategy.add_indicators(data)
        data = self.mean_reversion_strategy.add_indicators(data)
        data = self.breakout_strategy.add_indicators(data)
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals by combining signals from multiple strategies.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        # Ensure data has necessary indicators
        data = self.add_indicators(data)
        
        # Generate signals from each strategy
        momentum_data = self.momentum_strategy.generate_signals(data.copy())
        mean_reversion_data = self.mean_reversion_strategy.generate_signals(data.copy())
        breakout_data = self.breakout_strategy.generate_signals(data.copy())
        
        # Create a combined signal DataFrame
        data['momentum_signal'] = momentum_data['signal']
        data['mean_reversion_signal'] = mean_reversion_data['signal']
        data['breakout_signal'] = breakout_data['signal']
        
        # Calculate weighted signal
        data['weighted_signal'] = (
            data['momentum_signal'] * self.weights['momentum'] +
            data['mean_reversion_signal'] * self.weights['mean_reversion'] +
            data['breakout_signal'] * self.weights['breakout']
        )
        
        # Convert to -1, 0, 1 based on threshold
        data['signal'] = 0
        data.loc[data['weighted_signal'] >= 0.5, 'signal'] = 1
        data.loc[data['weighted_signal'] <= -0.5, 'signal'] = -1
        
        return data

class StrategyFactory:
    """
    Factory class to create strategy instances based on configuration.
    """
    @staticmethod
    def create_strategy(strategy_name, config=None):
        """
        Create and return a strategy instance based on the strategy name.
        
        Args:
            strategy_name (str): Name of the strategy to create
            config: Configuration object
            
        Returns:
            Strategy: Strategy instance
        """
        if strategy_name == 'momentum':
            return MomentumStrategy(config)
        elif strategy_name == 'mean_reversion':
            return MeanReversionStrategy(config)
        elif strategy_name == 'breakout':
            return BreakoutStrategy(config)
        elif strategy_name == 'combined':
            return CombinedStrategy(config)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

class StrategyManager:
    """Class for managing and selecting trading strategies."""
    
    def __init__(self, config):
        """Initialize the StrategyManager with configuration."""
        self.config = config
        
        # Create strategy instances
        self.strategies = {
            'momentum': MomentumStrategy(config),
            'mean_reversion': MeanReversionStrategy(config),
            'breakout': BreakoutStrategy(config),
            'combined': CombinedStrategy(config),
            'ml': MachineLearningStrategy(config)
        }
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Generate signals using the selected strategy.
        
        Args:
            data: DataFrame with price data and indicators.
            symbol: The ticker symbol.
            
        Returns:
            Dictionary with trading signals.
        """
        strategy_name = self.config.STRATEGY.lower()
        
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' not found. Using 'combined' instead.")
            strategy_name = 'combined'
        
        return self.strategies[strategy_name].generate_signals(data) 