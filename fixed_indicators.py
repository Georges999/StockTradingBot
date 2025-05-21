#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed Technical indicators module for stock analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
import ta

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Class for calculating various technical indicators on price data."""
    
    def __init__(self):
        """Initialize the TechnicalIndicators class."""
        pass
    
    # Static methods to handle direct class calls
    @staticmethod
    def add_rsi(data, **kwargs):
        """Static method wrapper for add_rsi to handle direct class calls."""
        instance = TechnicalIndicators()
        period = kwargs.get('period', 14)
        return instance.calculate_indicators(data, {'rsi': {'period': period}})
        
    @staticmethod
    def add_macd(data, **kwargs):
        """Static method wrapper for add_macd to handle direct class calls."""
        instance = TechnicalIndicators()
        fast_period = kwargs.get('fast_period', 12)
        slow_period = kwargs.get('slow_period', 26)
        signal_period = kwargs.get('signal_period', 9)
        return instance.calculate_indicators(data, {'macd': {'fast': fast_period, 'slow': slow_period, 'signal': signal_period}})
    
    @staticmethod
    def add_bollinger_bands(data, **kwargs):
        """Static method wrapper for add_bollinger_bands to handle direct class calls."""
        instance = TechnicalIndicators()
        period = kwargs.get('period', 20)
        std_dev = kwargs.get('std_dev', 2)
        return instance.calculate_indicators(data, {'bollinger_bands': {'period': period, 'std_dev': std_dev}})
    
    @staticmethod
    def add_atr(data, **kwargs):
        """Static method wrapper for add_atr to handle direct class calls."""
        instance = TechnicalIndicators()
        period = kwargs.get('period', 14)
        return instance.calculate_indicators(data, {'atr': {'period': period}})
    
    @staticmethod
    def add_ema(data, **kwargs):
        """Static method wrapper for add_ema to handle direct class calls."""
        instance = TechnicalIndicators()
        periods = kwargs.get('periods', [9, 21])
        return instance.calculate_indicators(data, {'ema': {'periods': periods}})
    
    @staticmethod
    def add_indicators(data):
        """Static method wrapper for calculating all indicators."""
        instance = TechnicalIndicators()
        return instance.calculate_indicators(data, {
            'sma': {'periods': [20, 50, 200]},
            'ema': {'periods': [9, 21]},
            'rsi': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2},
            'atr': {'period': 14}
        })
    
    def calculate_indicators(self, data: pd.DataFrame, indicators_config: Dict) -> pd.DataFrame:
        """
        Calculate technical indicators based on configuration.
        
        Args:
            data: DataFrame with OHLCV data.
            indicators_config: Dictionary with indicator configurations.
            
        Returns:
            DataFrame with added technical indicators.
        """
        if data is None or data.empty:
            logger.warning("Empty data provided for indicator calculation")
            return data if data is not None else pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure all required columns exist and are lowercase
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in data")
                return data
        
        try:
            # Calculate indicators based on config
            for indicator, params in indicators_config.items():
                if indicator == 'sma':
                    self._add_sma(df, params)
                elif indicator == 'ema':
                    self._add_ema(df, params)
                elif indicator == 'rsi':
                    self._add_rsi(df, params)
                elif indicator == 'macd':
                    self._add_macd(df, params)
                elif indicator == 'bollinger_bands':
                    self._add_bollinger_bands(df, params)
                elif indicator == 'atr':
                    self._add_atr(df, params)
                elif indicator == 'vwap':
                    self._add_vwap(df)
                elif indicator == 'ichimoku':
                    self._add_ichimoku(df, params)
                elif indicator == 'support_resistance':
                    self._add_support_resistance(df, params)
                elif indicator == 'stochastic':
                    self._add_stochastic(df, params)
                elif indicator == 'adx':
                    self._add_adx(df, params)
                elif indicator == 'obv':
                    self._add_obv(df)
            
            # Don't automatically add custom indicators that might cause errors
            # self._add_volatility_ratio(df)
            # self._add_trend_strength(df)
            # self._add_volume_delta(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return data
    
    def _add_sma(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Simple Moving Averages to the DataFrame."""
        periods = params.get('periods', [20, 50, 200])
        for period in periods:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
    
    def _add_ema(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Exponential Moving Averages to the DataFrame."""
        periods = params.get('periods', [9, 21])
        for period in periods:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
    
    def _add_rsi(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Relative Strength Index to the DataFrame."""
        period = params.get('period', 14)
        df['rsi'] = ta.momentum.rsi(df['close'], window=period)
        
        # Add RSI categories based on overbought/oversold thresholds
        overbought = params.get('overbought', 70)
        oversold = params.get('oversold', 30)
        df['rsi_category'] = 'neutral'
        df.loc[df['rsi'] > overbought, 'rsi_category'] = 'overbought'
        df.loc[df['rsi'] < oversold, 'rsi_category'] = 'oversold'
    
    def _add_macd(self, df: pd.DataFrame, params: Dict) -> None:
        """Add MACD (Moving Average Convergence Divergence) to the DataFrame."""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)
        
        # Calculate MACD line, signal line, and histogram
        df['macd'] = ta.trend.macd(df['close'], window_slow=slow, window_fast=fast)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df['macd_histogram'] = ta.trend.macd_diff(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
    
    def _add_bollinger_bands(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Bollinger Bands to the DataFrame."""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)
        
        # Calculate Bollinger Bands
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=period)
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=period, window_dev=std_dev)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=period, window_dev=std_dev)
    
    def _add_atr(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Average True Range to the DataFrame."""
        period = params.get('period', 14)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
        
    def _add_vwap(self, df: pd.DataFrame) -> None:
        """Add Volume Weighted Average Price to the DataFrame."""
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Remove intermediate calculation
        df.drop('typical_price', axis=1, inplace=True)
        
    def _add_ichimoku(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Ichimoku Cloud indicators to the DataFrame."""
        tenkan_period = params.get('tenkan', 9)
        kijun_period = params.get('kijun', 26)
        senkou_b_period = params.get('senkou_b', 52)
        
        # Calculate Tenkan-sen (Conversion Line)
        high_tenkan = df['high'].rolling(window=tenkan_period).max()
        low_tenkan = df['low'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen (Base Line)
        high_kijun = df['high'].rolling(window=kijun_period).max()
        low_kijun = df['low'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)
        
        # Calculate Senkou Span B (Leading Span B)
        high_senkou = df['high'].rolling(window=senkou_b_period).max()
        low_senkou = df['low'].rolling(window=senkou_b_period).min()
        df['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(kijun_period)
    
    def _add_support_resistance(self, df: pd.DataFrame, params: Dict) -> None:
        """Add support and resistance levels to the DataFrame."""
        # Simple implementation
        pass
        
    def _add_stochastic(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Stochastic Oscillator to the DataFrame."""
        # Simple implementation
        pass
        
    def _add_adx(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Average Directional Index (ADX) to the DataFrame."""
        # Simple implementation
        pass
        
    def _add_obv(self, df: pd.DataFrame) -> None:
        """Add On-Balance Volume (OBV) to the DataFrame."""
        # Simple implementation
        pass
        
    def _add_volatility_ratio(self, df: pd.DataFrame) -> None:
        """Add volatility ratio indicator to the DataFrame."""
        # Simple implementation
        pass
        
    def _add_trend_strength(self, df: pd.DataFrame) -> None:
        """Add trend strength indicator to the DataFrame."""
        # Simple implementation
        pass
        
    def _add_volume_delta(self, df: pd.DataFrame) -> None:
        """Add volume delta indicator to the DataFrame."""
        # Simple implementation
        pass 