#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical indicators module for stock analysis.
Provides a comprehensive set of technical indicators for trading strategies.
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
    
    # Add static methods to handle direct class calls
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
        if data.empty:
            logger.warning("Empty data provided for indicator calculation")
            return data
        
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
            
            # Add custom indicators
            self._add_volatility_ratio(df)
            self._add_trend_strength(df)
            self._add_volume_delta(df)
            
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
        df['macd_line'] = ta.trend.macd(df['close'], window_slow=slow, window_fast=fast)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df['macd_histogram'] = ta.trend.macd_diff(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        
        # Add cross signals
        df['macd_cross'] = 'none'
        # Bullish cross (MACD line crosses above signal line)
        bullish_cross = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        df.loc[bullish_cross, 'macd_cross'] = 'bullish'
        # Bearish cross (MACD line crosses below signal line)
        bearish_cross = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        df.loc[bearish_cross, 'macd_cross'] = 'bearish'
    
    def _add_bollinger_bands(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Bollinger Bands to the DataFrame."""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)
        
        # Calculate Bollinger Bands
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=period)
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=period, window_dev=std_dev)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=period, window_dev=std_dev)
        
        # Calculate bandwidth and %B indicators
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Add Bollinger Band signals
        df['bb_signal'] = 'none'
        # Bullish signal when price crosses above lower band
        df.loc[(df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1)), 'bb_signal'] = 'bullish'
        # Bearish signal when price crosses below upper band
        df.loc[(df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'].shift(1)), 'bb_signal'] = 'bearish'
        # Overbought signal when price touches or exceeds upper band
        df.loc[df['close'] >= df['bb_upper'], 'bb_signal'] = 'overbought'
        # Oversold signal when price touches or falls below lower band
        df.loc[df['close'] <= df['bb_lower'], 'bb_signal'] = 'oversold'
    
    def _add_atr(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Average True Range to the DataFrame."""
        period = params.get('period', 14)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
        
        # Add ATR percentage (ATR relative to price)
        df['atr_percent'] = (df['atr'] / df['close']) * 100
    
    def _add_vwap(self, df: pd.DataFrame) -> None:
        """Add Volume Weighted Average Price to the DataFrame."""
        # VWAP is typically calculated from the day's open
        # For simplicity, we'll calculate a rolling VWAP over the available data
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Calculate distance from VWAP
        df['vwap_distance'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
        
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
        
        # Calculate Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-kijun_period)
    
    def _add_support_resistance(self, df: pd.DataFrame, params: Dict) -> None:
        """Add support and resistance levels to the DataFrame."""
        periods = params.get('periods', 30)
        tolerance = params.get('tolerance', 0.02)
        
        # We'll use a simple approach to find local minima and maxima
        df['local_min'] = df['low'].rolling(window=periods, center=True).min()
        df['local_max'] = df['high'].rolling(window=periods, center=True).max()
        
        # Identify potential support levels (local minima)
        df['is_support'] = (df['low'] <= df['low'].shift(1)) & (df['low'] <= df['low'].shift(-1)) & (df['low'] <= df['low'].shift(2)) & (df['low'] <= df['low'].shift(-2))
        
        # Identify potential resistance levels (local maxima)
        df['is_resistance'] = (df['high'] >= df['high'].shift(1)) & (df['high'] >= df['high'].shift(-1)) & (df['high'] >= df['high'].shift(2)) & (df['high'] >= df['high'].shift(-2))
        
        # Calculate nearest support and resistance levels
        df['nearest_support'] = np.nan
        df['nearest_resistance'] = np.nan
        
        # This is a simplified approach - a more sophisticated algorithm would cluster close levels
        # and track historically significant levels
        for i in range(periods, len(df)):
            # Find supports in lookback period
            supports = df.loc[i-periods:i, 'low'][df.loc[i-periods:i, 'is_support']]
            if not supports.empty:
                # Find closest support below current price
                lower_supports = supports[supports < df.loc[i, 'close']]
                if not lower_supports.empty:
                    df.loc[i, 'nearest_support'] = lower_supports.iloc[-1]
            
            # Find resistances in lookback period
            resistances = df.loc[i-periods:i, 'high'][df.loc[i-periods:i, 'is_resistance']]
            if not resistances.empty:
                # Find closest resistance above current price
                upper_resistances = resistances[resistances > df.loc[i, 'close']]
                if not upper_resistances.empty:
                    df.loc[i, 'nearest_resistance'] = upper_resistances.iloc[0]
        
        # Calculate distance to support and resistance as percentage
        df['support_distance'] = ((df['close'] - df['nearest_support']) / df['close']) * 100
        df['resistance_distance'] = ((df['nearest_resistance'] - df['close']) / df['close']) * 100
    
    def _add_stochastic(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Stochastic Oscillator to the DataFrame."""
        k_period = params.get('k_period', 14)
        d_period = params.get('d_period', 3)
        
        # Calculate %K (Fast Stochastic)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=k_period, smooth_window=3)
        
        # Calculate %D (Slow Stochastic)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=k_period, smooth_window=d_period)
        
        # Add stochastic signals
        df['stoch_signal'] = 'none'
        # Bullish when %K crosses above %D
        df.loc[(df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)), 'stoch_signal'] = 'bullish'
        # Bearish when %K crosses below %D
        df.loc[(df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)), 'stoch_signal'] = 'bearish'
        # Overbought when both %K and %D are above 80
        df.loc[(df['stoch_k'] > 80) & (df['stoch_d'] > 80), 'stoch_signal'] = 'overbought'
        # Oversold when both %K and %D are below 20
        df.loc[(df['stoch_k'] < 20) & (df['stoch_d'] < 20), 'stoch_signal'] = 'oversold'
    
    def _add_adx(self, df: pd.DataFrame, params: Dict) -> None:
        """Add Average Directional Index (ADX) to the DataFrame."""
        period = params.get('period', 14)
        
        # Calculate ADX, +DI, and -DI
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
        df['adx_pos_di'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=period)
        df['adx_neg_di'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=period)
        
        # Add ADX trend strength categorization
        df['adx_trend'] = 'weak'
        df.loc[df['adx'] > 25, 'adx_trend'] = 'moderate'
        df.loc[df['adx'] > 50, 'adx_trend'] = 'strong'
        df.loc[df['adx'] > 75, 'adx_trend'] = 'extreme'
        
        # Add ADX trend direction
        df['adx_direction'] = 'none'
        df.loc[df['adx_pos_di'] > df['adx_neg_di'], 'adx_direction'] = 'bullish'
        df.loc[df['adx_pos_di'] < df['adx_neg_di'], 'adx_direction'] = 'bearish'
        
        # Add ADX crossover signals
        df['adx_cross'] = 'none'
        # Bullish crossover
        df.loc[(df['adx_pos_di'] > df['adx_neg_di']) & (df['adx_pos_di'].shift(1) <= df['adx_neg_di'].shift(1)), 'adx_cross'] = 'bullish'
        # Bearish crossover
        df.loc[(df['adx_pos_di'] < df['adx_neg_di']) & (df['adx_pos_di'].shift(1) >= df['adx_neg_di'].shift(1)), 'adx_cross'] = 'bearish'
    
    def _add_obv(self, df: pd.DataFrame) -> None:
        """Add On-Balance Volume (OBV) to the DataFrame."""
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Calculate OBV momentum (rate of change)
        df['obv_momentum'] = df['obv'].pct_change(periods=5) * 100
        
        # Add OBV divergence signals (price/OBV divergence)
        df['obv_divergence'] = 'none'
        
        # Bullish divergence: price making lower lows but OBV making higher lows
        price_lower_low = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
        obv_higher_low = (df['obv'] > df['obv'].shift(1)) & (df['obv'].shift(1) > df['obv'].shift(2))
        df.loc[price_lower_low & obv_higher_low, 'obv_divergence'] = 'bullish'
        
        # Bearish divergence: price making higher highs but OBV making lower highs
        price_higher_high = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
        obv_lower_high = (df['obv'] < df['obv'].shift(1)) & (df['obv'].shift(1) < df['obv'].shift(2))
        df.loc[price_higher_high & obv_lower_high, 'obv_divergence'] = 'bearish'
    
    def _add_volatility_ratio(self, df: pd.DataFrame) -> None:
        """Add volatility ratio indicator to the DataFrame."""
        # Calculate average volatility over short and long term
        df['short_volatility'] = df['close'].pct_change().rolling(window=5).std() * 100
        df['long_volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        # Calculate volatility ratio
        df['volatility_ratio'] = df['short_volatility'] / df['long_volatility']
        
        # Add volatility regime
        df['volatility_regime'] = 'normal'
        df.loc[df['volatility_ratio'] > 1.5, 'volatility_regime'] = 'high'
        df.loc[df['volatility_ratio'] < 0.75, 'volatility_regime'] = 'low'
    
    def _add_trend_strength(self, df: pd.DataFrame) -> None:
        """Add trend strength indicator to the DataFrame."""
        if 'adx' not in df.columns:
            # If ADX not already calculated, use a simplified trend strength metric
            # Calculate a 20-day linear regression slope
            x = np.arange(20)
            
            def get_slope(y):
                if len(y) < 20:
                    return np.nan
                return np.polyfit(x, y, 1)[0]
            
            df['trend_slope'] = df['close'].rolling(window=20).apply(get_slope, raw=False)
            
            # Normalize slope to percentage
            mean_price = df['close'].rolling(window=20).mean()
            df['trend_strength'] = (df['trend_slope'] * 20) / mean_price * 100
        else:
            # Use ADX as trend strength
            df['trend_strength'] = df['adx'] / 100
            
        # Add trend direction
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['trend_direction'] = 'sideways'
            df.loc[(df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']), 'trend_direction'] = 'bullish'
            df.loc[(df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']), 'trend_direction'] = 'bearish'
    
    def _add_volume_delta(self, df: pd.DataFrame) -> None:
        """Add volume delta indicator to the DataFrame."""
        # Calculate volume delta (buying pressure vs selling pressure)
        df['volume_delta'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        
        # Calculate cumulative volume delta
        df['cum_volume_delta'] = df['volume_delta'].cumsum()
        
        # Volume change percentage (current vs n-day average)
        df['volume_change'] = df['volume'] / df['volume'].rolling(window=20).mean() - 1 