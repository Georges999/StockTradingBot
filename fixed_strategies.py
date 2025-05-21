#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed strategies module for the StockTradingBot.
Implements various technical and algorithmic trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from fixed_indicators import TechnicalIndicators  # Import from fixed_indicators

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
        self.indicators = TechnicalIndicators()  # Use fixed TechnicalIndicators
    
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
        return self.indicators.calculate_indicators(data, {
            'sma': {'periods': [20, 50, 200]},
            'ema': {'periods': [9, 21]},
            'rsi': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14}
        })

class MomentumStrategy(Strategy):
    """
    Momentum strategy using simple price-based signals.
    """
    def __init__(self, config=None):
        """
        Initialize momentum strategy.
        """
        super().__init__(config)
    
    def add_indicators(self, data):
        """
        Add simple price-based indicators instead of using TechnicalIndicators.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate simple indicators
        df['price_pct_change'] = df['close'].pct_change() * 100  # Daily price change percentage
        df['price_5d_change'] = df['close'].pct_change(5) * 100  # 5-day price change percentage
        df['price_10d_change'] = df['close'].pct_change(10) * 100  # 10-day price change percentage
        df['volume_change'] = df['volume'].pct_change() * 100  # Volume change percentage
        
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate a simple momentum indicator (close price relative to N-day moving average)
        df['momentum'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        
        return df
    
    def generate_signals(self, data):
        """
        Generate trading signals based on momentum indicators.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        # Ensure data has necessary indicators
        if 'momentum' not in data.columns:
            data = self.add_indicators(data)
        
        # Initialize signals column: 1 for buy, -1 for sell, 0 for hold
        data['signal'] = 0
        
        # Price momentum conditions
        price_momentum_buy = (data['price_pct_change'] > 0) & (data['price_5d_change'] > 2) & (data['volume_change'] > 0)
        price_momentum_sell = (data['price_pct_change'] < 0) & (data['price_5d_change'] < -2) & (data['volume_change'] > 0)
        
        # Moving average conditions
        ma_buy_condition = (data['close'] > data['sma_20']) & (data['sma_20'] > data['sma_50'])
        ma_sell_condition = (data['close'] < data['sma_20']) & (data['sma_20'] < data['sma_50'])
        
        # Combined signals
        data.loc[price_momentum_buy & ma_buy_condition, 'signal'] = 1
        data.loc[price_momentum_sell & ma_sell_condition, 'signal'] = -1
        
        return data

class StrategyManager:
    """Class for managing and selecting trading strategies."""
    
    def __init__(self, config):
        """Initialize the StrategyManager with configuration."""
        self.config = config
        
        # Create strategy instances
        self.strategies = {
            'momentum': MomentumStrategy(config),
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
        strategy_name = 'momentum'  # Force to use momentum strategy for simplicity
        
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' not found. Using 'momentum' instead.")
            strategy_name = 'momentum'
        
        # Generate signals
        result_df = self.strategies[strategy_name].generate_signals(data.copy())
        
        # Convert the last row signal to a dictionary format
        if not result_df.empty:
            last_row = result_df.iloc[-1]
            signal_value = last_row['signal']
            
            # Convert numeric signal to string representation
            if signal_value > 0:
                signal = 'buy'
                direction = 'long'
            elif signal_value < 0:
                signal = 'sell'
                direction = 'short'
            else:
                signal = 'neutral'
                direction = 'neutral'
            
            return {
                'signal': signal,
                'direction': direction,
                'strength': abs(signal_value),
                'reason': f"Generated by {strategy_name} strategy",
                'timestamp': pd.Timestamp.now().isoformat()
            }
        
        # Default neutral signal if no data
        return {
            'signal': 'neutral',
            'direction': 'neutral',
            'strength': 0,
            'reason': 'No signal generated',
            'timestamp': pd.Timestamp.now().isoformat()
        } 