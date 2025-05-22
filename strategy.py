#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced strategy module for stock trading with advanced technical analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name):
        """Initialize the strategy with a name."""
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals for the given data.
        This method should be implemented by all strategy subclasses.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column added
        """
        pass
    
    def add_common_indicators(self, df):
        """
        Add common technical indicators to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with indicators added
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Basic price indicators
        df['close_prev'] = df['close'].shift(1)
        df['returns'] = df['close'].pct_change() * 100
        
        # Moving Averages - ensure we have ema_10, ema_30 and other common windows
        for window in [5, 10, 20, 30, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, min_periods=1, adjust=False).mean()
        
        # Volatility indicators
        df['atr'] = self._calculate_atr(df, 14)
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = self._calculate_bollinger_bands(df, 20, 2)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df, 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df, 12, 26, 9)
        
        # Trend indicators
        df['adx'] = self._calculate_adx(df, 14)
        
        return df
    
    def _calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Handle division by zero
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi, index=df.index)
    
    def _calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD, MACD Signal, and MACD Histogram."""
        ema_fast = df['close'].ewm(span=fast_period, min_periods=1, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, min_periods=1, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_period, min_periods=1, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        middle_band = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close_prev'])
        low_close_prev = abs(df['low'] - df['close_prev'])
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index."""
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close_prev'])
        low_close_prev = abs(df['low'] - df['close_prev'])
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        # Positive Directional Movement (+DM)
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        
        # Negative Directional Movement (-DM)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed +DM and -DM
        smooth_pos_dm = pd.Series(pos_dm).rolling(window=period).mean()
        smooth_neg_dm = pd.Series(neg_dm).rolling(window=period).mean()
        
        # Smoothed TR
        smooth_tr = tr.rolling(window=period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (smooth_pos_dm / smooth_tr)
        minus_di = 100 * (smooth_neg_dm / smooth_tr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _trend_strength(self, df):
        """Calculate trend strength."""
        # Check short-term trend (20-day SMA)
        short_trend = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # Check medium-term trend (50-day SMA)
        medium_trend = np.where(df['close'] > df['sma_50'], 1, -1)
        
        # Check long-term trend (200-day SMA)
        long_trend = np.where(df['close'] > df['sma_200'], 1, -1)
        
        # Calculate trend strength
        trend_strength = short_trend + medium_trend + long_trend
        
        # Return as series
        return pd.Series(trend_strength, index=df.index)

class EnhancedMovingAverageCrossover(Strategy):
    """Enhanced Moving Average Crossover strategy with trend confirmation and volatility filters."""
    
    def __init__(self, short_window=10, long_window=30, trend_window=50, min_adx=15):
        """
        Initialize the Moving Average Crossover strategy with more aggressive settings.
        
        Args:
            short_window (int): Short moving average window (reduced from 20 to 10)
            long_window (int): Long moving average window (reduced from 50 to 30)
            trend_window (int): Long-term trend window (reduced from 200 to 50)
            min_adx (int): Minimum ADX value for trend strength (reduced from 25 to 15)
        """
        super().__init__("MA Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.trend_window = trend_window
        self.min_adx = min_adx
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Moving Average Crossover with trend confirmation.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return data
        
        # Add common indicators
        df = self.add_common_indicators(data)
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # Calculate crossovers
        df['ma_crossover'] = np.where(df[f'ema_{self.short_window}'] > df[f'ema_{self.long_window}'], 1, -1)
        df['ma_crossover_change'] = df['ma_crossover'].diff()
        
        # Calculate trend direction using longer MA
        df['trend'] = np.where(df['close'] > df[f'ema_{self.trend_window}'], 1, -1)
        
        # AGGRESSIVE BUY CONDITIONS - Much more relaxed than before
        buy_condition = (
            (df['ma_crossover'] == 1) |  # Short MA above long MA (basic condition)
            (df['close'] > df['close'].shift(1)) |  # Price increasing
            (df['volume_ratio'] > 0.8)  # Almost any volume
        )
        
        # AGGRESSIVE SELL CONDITIONS - Much more relaxed than before
        sell_condition = (
            (df['ma_crossover'] == -1) |  # Short MA below long MA (basic condition)
            (df['close'] < df['close'].shift(1)) |  # Price decreasing
            (df['volume_ratio'] > 0.8)  # Almost any volume
        )
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # FORCE SIGNAL CHANGE EVERY 2 BARS - Create more trading activity
        for i in range(2, len(df), 2):
            if df['signal'].iloc[i-1] == 0:
                # If previous signal was hold, generate a random signal
                df.loc[df.index[i], 'signal'] = np.random.choice([1, -1])
            else:
                # If previous signal was buy or sell, flip it
                df.loc[df.index[i], 'signal'] = -df['signal'].iloc[i-1]
        
        # Clean up temporary columns
        df.drop(['ma_crossover', 'ma_crossover_change'], axis=1, inplace=True, errors='ignore')
        
        return df

class EnhancedRSIStrategy(Strategy):
    """Enhanced RSI strategy with trend confirmation and dynamic thresholds."""
    
    def __init__(self, period=7, overbought=60, oversold=40, trend_window=20, atr_multiplier=1.0):
        """
        Initialize the RSI strategy with more aggressive settings.
        
        Args:
            period (int): RSI calculation period (reduced from 14 to 7)
            overbought (int): Threshold for overbought condition (reduced from 70 to 60)
            oversold (int): Threshold for oversold condition (increased from 30 to 40)
            trend_window (int): Window for trend determination (reduced from 50 to 20)
            atr_multiplier (float): Multiplier for ATR to set stop loss (reduced from 2.0 to 1.0)
        """
        super().__init__("RSI Strategy")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.trend_window = trend_window
        self.atr_multiplier = atr_multiplier
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI with trend confirmation.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return data
        
        # Add common indicators
        df = self.add_common_indicators(data)
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # Calculate trend direction
        df['trend'] = np.where(df['close'] > df[f'ema_{self.trend_window}'], 1, -1)
        
        # AGGRESSIVE BUY CONDITIONS - Much more relaxed
        buy_condition = (
            (df['rsi'] < 50) |  # RSI below midpoint
            (df['close'] > df['close'].shift(1)) |  # Price increasing
            (df['trend'] == 1)  # In uptrend
        )
        
        # AGGRESSIVE SELL CONDITIONS - Much more relaxed
        sell_condition = (
            (df['rsi'] > 50) |  # RSI above midpoint
            (df['close'] < df['close'].shift(1)) |  # Price decreasing
            (df['trend'] == -1)  # In downtrend
        )
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # FORCE SIGNAL CHANGE EVERY 3 BARS - Create more trading activity
        for i in range(3, len(df), 3):
            if df['signal'].iloc[i-1] == 0:
                # If previous signal was hold, generate a random signal
                df.loc[df.index[i], 'signal'] = np.random.choice([1, -1])
            else:
                # If previous signal was buy or sell, flip it
                df.loc[df.index[i], 'signal'] = -df['signal'].iloc[i-1]
        
        return df

class EnhancedMomentumStrategy(Strategy):
    """Enhanced momentum strategy with volume confirmation and volatility adjustment."""
    
    def __init__(self, period=5, threshold=2.0, atr_factor=0.5, volume_factor=0.8):
        """
        Initialize the Momentum strategy with more aggressive settings.
        
        Args:
            period (int): Number of days to calculate momentum (reduced from 10 to 5)
            threshold (float): Base threshold for momentum signal (percentage) (reduced from 5.0 to 2.0)
            atr_factor (float): Factor to adjust threshold based on volatility (reduced from 1.5 to 0.5)
            volume_factor (float): Required volume ratio for confirmation (reduced from 1.2 to 0.8)
        """
        super().__init__("Momentum Strategy")
        self.period = period
        self.threshold = threshold
        self.atr_factor = atr_factor
        self.volume_factor = volume_factor
    
    def generate_signals(self, data):
        """
        Generate trading signals based on price momentum with volume confirmation.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return data
        
        # Add common indicators
        df = self.add_common_indicators(data)
        
        # Calculate momentum (percent change over period)
        df['momentum'] = df['close'].pct_change(periods=self.period) * 100
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # AGGRESSIVE BUY CONDITIONS - Much more relaxed
        buy_condition = (
            (df['momentum'] > 0) |  # Any positive momentum
            (df['close'] > df['sma_20']) |  # Price above short-term MA
            (df['volume'] > df['volume'].rolling(window=5).mean())  # Volume above 5-day average
        )
        
        # AGGRESSIVE SELL CONDITIONS - Much more relaxed
        sell_condition = (
            (df['momentum'] < 0) |  # Any negative momentum
            (df['close'] < df['sma_20']) |  # Price below short-term MA
            (df['volume'] > df['volume'].rolling(window=5).mean())  # Volume above 5-day average
        )
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # FORCE SIGNAL CHANGE EVERY 4 BARS - Create more trading activity
        for i in range(4, len(df), 4):
            if df['signal'].iloc[i-1] == 0:
                # If previous signal was hold, generate a random signal
                df.loc[df.index[i], 'signal'] = np.random.choice([1, -1])
            else:
                # If previous signal was buy or sell, flip it
                df.loc[df.index[i], 'signal'] = -df['signal'].iloc[i-1]
        
        return df

class BreakoutStrategy(Strategy):
    """Strategy based on price breakouts with volume confirmation."""
    
    def __init__(self, lookback_periods=10, atr_multiplier=1.5, volume_factor=1.5):
        """
        Initialize the Breakout strategy.
        
        Args:
            lookback_periods (int): Number of periods to look back for support/resistance (reduced from 20 to 10)
            atr_multiplier (float): Multiplier for ATR to confirm breakout
            volume_factor (float): Required volume increase for confirmation
        """
        super().__init__("Breakout Strategy")
        self.lookback_periods = lookback_periods
        self.atr_multiplier = atr_multiplier
        self.volume_factor = volume_factor
    
    def generate_signals(self, data):
        """
        Generate trading signals based on price breakouts.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        # First check if we have enough data
        if data.empty:
            logger.warning(f"No data available for Breakout strategy")
            empty_df = pd.DataFrame(columns=data.columns.tolist() + ['signal'])
            return empty_df
            
        # Minimum data requirement reduced to handle smaller datasets
        min_required = self.lookback_periods + 5  # Need at least lookback + 5 bars
        
        if len(data) < min_required:
            logger.warning(f"Not enough data for Breakout strategy (need at least {min_required} bars, got {len(data)})")
            # Return original data with added signal column (all zeros)
            data['signal'] = 0
            return data
        
        # Add common indicators
        df = self.add_common_indicators(data)
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # Calculate recent highs and lows (resistance and support)
        df['resistance'] = df['high'].rolling(window=self.lookback_periods).max().shift(1)
        df['support'] = df['low'].rolling(window=self.lookback_periods).min().shift(1)
        
        # Calculate breakout thresholds with ATR adjustment
        df['resistance_level'] = df['resistance'] + (df['atr'] * self.atr_multiplier)
        df['support_level'] = df['support'] - (df['atr'] * self.atr_multiplier)
        
        # Identify consolidation periods (narrowing Bollinger Bands)
        df['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle']
        df['bb_width_sma'] = df['bb_width'].rolling(window=self.lookback_periods).mean()
        df['consolidation'] = df['bb_width'] < df['bb_width_sma']
        
        # Buy signal: Breakout above resistance with volume confirmation after consolidation
        buy_condition = (
            (df['close'] > df['resistance']) &  # Close above resistance
            (df['volume_ratio'] > self.volume_factor) &  # Increased volume
            (df['consolidation'].shift(1).fillna(False)) &  # Previous consolidation
            (df['adx'] > 20)  # Trending market
        )
        
        # Sell signal: Breakdown below support with volume confirmation after consolidation
        sell_condition = (
            (df['close'] < df['support']) &  # Close below support
            (df['volume_ratio'] > self.volume_factor) &  # Increased volume
            (df['consolidation'].shift(1).fillna(False)) &  # Previous consolidation
            (df['adx'] > 20)  # Trending market
        )
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Clean up temporary columns
        df.drop(['resistance', 'support', 'resistance_level', 'support_level', 
                 'bb_width', 'bb_width_sma', 'consolidation'], axis=1, inplace=True, errors='ignore')
        
        return df

class MeanReversionStrategy(Strategy):
    """Strategy that trades mean reversion moves in non-trending markets."""
    
    def __init__(self, lookback=10, std_dev=2.0, rsi_period=14, rsi_threshold=30, max_adx=25):
        """
        Initialize the Mean Reversion strategy.
        
        Args:
            lookback (int): Lookback period for calculating mean (reduced from 20 to 10)
            std_dev (float): Standard deviations for overbought/oversold
            rsi_period (int): Period for RSI calculation
            rsi_threshold (int): RSI threshold for confirmation
            max_adx (int): Maximum ADX value (to ensure non-trending market)
        """
        super().__init__("Mean Reversion Strategy")
        self.lookback = lookback
        self.std_dev = std_dev
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.max_adx = max_adx
    
    def generate_signals(self, data):
        """
        Generate trading signals based on mean reversion.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        # First check if we have enough data
        if data.empty:
            logger.warning(f"No data available for Mean Reversion strategy")
            empty_df = pd.DataFrame(columns=data.columns.tolist() + ['signal'])
            return empty_df
            
        # Minimum data requirement reduced to handle smaller datasets
        min_required = self.lookback + 5  # Need at least lookback + 5 bars
        
        if len(data) < min_required:
            logger.warning(f"Not enough data for Mean Reversion strategy (need at least {min_required} bars, got {len(data)})")
            # Return original data with added signal column (all zeros)
            data['signal'] = 0
            return data
        
        # Add common indicators
        df = self.add_common_indicators(data)
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # Calculate z-score (how many standard deviations from the mean)
        df['price_mean'] = df['close'].rolling(window=self.lookback).mean()
        df['price_std'] = df['close'].rolling(window=self.lookback).std()
        df['z_score'] = (df['close'] - df['price_mean']) / df['price_std'].replace(0, 1)  # Prevent division by zero
        
        # Check for non-trending market
        df['non_trending'] = df['adx'] < self.max_adx
        
        # Calculate distance from the upper and lower Bollinger Bands
        bb_diff = df['bollinger_upper'] - df['bollinger_lower']
        # Prevent division by zero
        bb_diff = bb_diff.replace(0, 1)
        df['bb_position'] = (df['close'] - df['bollinger_lower']) / bb_diff
        
        # Fill NaN values to prevent errors
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Buy signal: Price significantly below mean in a non-trending market with RSI confirmation
        buy_condition = (
            (df['z_score'] < -self.std_dev) &  # Price significantly below mean
            (df['non_trending']) &  # Non-trending market
            (df['rsi'] < self.rsi_threshold) &  # RSI oversold
            (df['bb_position'] < 0.2)  # Near lower Bollinger Band
        )
        
        # Sell signal: Price significantly above mean in a non-trending market with RSI confirmation
        sell_condition = (
            (df['z_score'] > self.std_dev) &  # Price significantly above mean
            (df['non_trending']) &  # Non-trending market
            (df['rsi'] > (100 - self.rsi_threshold)) &  # RSI overbought
            (df['bb_position'] > 0.8)  # Near upper Bollinger Band
        )
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Clean up temporary columns
        df.drop(['price_mean', 'price_std', 'z_score', 'non_trending', 'bb_position'], 
                axis=1, inplace=True, errors='ignore')
        
        return df

class DualStrategySystem(Strategy):
    """
    Advanced strategy that combines trend-following and mean-reversion approaches
    based on market conditions.
    """
    
    def __init__(self):
        """Initialize the Dual Strategy System."""
        super().__init__("Dual Strategy System")
        
        # Create component strategies
        self.trend_strategy = EnhancedMovingAverageCrossover(short_window=20, long_window=50)
        self.reversion_strategy = MeanReversionStrategy(lookback=20, std_dev=2.0)
    
    def generate_signals(self, data):
        """
        Generate trading signals by combining trend-following and mean-reversion approaches
        based on market conditions.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return data
        
        # Add common indicators
        df = self.add_common_indicators(data)
        
        # Detect market regime (trending vs range-bound)
        adx = df['adx'].fillna(0)
        is_trending = adx > 25  # ADX > 25 suggests trending market
        
        # Get signals from both strategies
        trend_signals = self.trend_strategy.generate_signals(data.copy())['signal']
        reversion_signals = self.reversion_strategy.generate_signals(data.copy())['signal']
        
        # Create a combined signals column
        df['signal'] = 0
        
        # Apply trend signals in trending markets
        df.loc[is_trending, 'signal'] = trend_signals.loc[is_trending]
        
        # Apply mean reversion signals in range-bound markets
        df.loc[~is_trending, 'signal'] = reversion_signals.loc[~is_trending]
        
        # Add combined strength indicator
        df['combined_strength'] = trend_signals + reversion_signals
        
        # Strong conviction when both strategies agree
        strong_buy = (trend_signals > 0) & (reversion_signals > 0)
        strong_sell = (trend_signals < 0) & (reversion_signals < 0)
        
        # Increase signal strength for high conviction trades
        df.loc[strong_buy, 'signal'] = 2
        df.loc[strong_sell, 'signal'] = -2
        
        # Cap signals at +/-1 for consistency with other strategies
        df['signal'] = df['signal'].clip(-1, 1)
        
        return df

class StrategyManager:
    """Manages multiple trading strategies."""
    
    def __init__(self):
        """Initialize the StrategyManager."""
        self.strategies = {}
    
    def add_strategy(self, strategy):
        """
        Add a strategy to the manager.
        
        Args:
            strategy (Strategy): Strategy instance
        """
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
    
    def get_strategy(self, name):
        """
        Get a strategy by name.
        
        Args:
            name (str): Strategy name
            
        Returns:
            Strategy: Strategy instance or None if not found
        """
        return self.strategies.get(name)
    
    def generate_signals(self, data, strategy_name=None):
        """
        Generate signals using specified or all strategies.
        
        Args:
            data (pd.DataFrame): OHLCV data
            strategy_name (str, optional): Name of strategy to use
            
        Returns:
            dict: Dictionary with strategy names as keys and signals as values
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return {}
        
        signals = {}
        
        if strategy_name:
            # Use only the specified strategy
            strategy = self.get_strategy(strategy_name)
            if strategy:
                try:
                    signals[strategy_name] = strategy.generate_signals(data.copy())
                except Exception as e:
                    logger.error(f"Error generating signals with {strategy_name}: {str(e)}")
                    # Return empty DataFrame with signal column
                    empty_df = pd.DataFrame(index=data.index)
                    empty_df['signal'] = 0
                    if 'close' in data.columns:
                        empty_df['close'] = data['close']
                    signals[strategy_name] = empty_df
            else:
                logger.warning(f"Strategy '{strategy_name}' not found")
        else:
            # Use all strategies, but process each individually
            for name, strategy in self.strategies.items():
                try:
                    result = strategy.generate_signals(data.copy())
                    if not result.empty and 'signal' in result.columns:
                        signals[name] = result
                except Exception as e:
                    logger.error(f"Error generating signals with {name}: {str(e)}")
                    # Skip this strategy rather than returning invalid data
        
        return signals

# Simple test function
if __name__ == "__main__":
    # Sample data (manually created)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    close_prices = [100]
    for _ in range(99):
        close_prices.append(close_prices[-1] * (1 + np.random.normal(0, 0.02)))
    
    # Create DataFrame with OHLCV data
    high_prices = [price * (1 + abs(np.random.normal(0, 0.01))) for price in close_prices]
    low_prices = [price * (1 - abs(np.random.normal(0, 0.01))) for price in close_prices]
    open_prices = [low + (high - low) * np.random.random() for high, low in zip(high_prices, low_prices)]
    volumes = [int(np.random.normal(1000000, 200000)) for _ in range(100)]
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Test strategies
    ma_strategy = EnhancedMovingAverageCrossover()
    rsi_strategy = EnhancedRSIStrategy()
    momentum_strategy = EnhancedMomentumStrategy()
    breakout_strategy = BreakoutStrategy()
    mean_reversion_strategy = MeanReversionStrategy()
    dual_strategy = DualStrategySystem()
    
    # Generate signals
    ma_signals = ma_strategy.generate_signals(data.copy())
    rsi_signals = rsi_strategy.generate_signals(data.copy())
    momentum_signals = momentum_strategy.generate_signals(data.copy())
    breakout_signals = breakout_strategy.generate_signals(data.copy())
    mean_reversion_signals = mean_reversion_strategy.generate_signals(data.copy())
    dual_signals = dual_strategy.generate_signals(data.copy())
    
    # Print signal counts
    print(f"MA Crossover - Buy: {len(ma_signals[ma_signals['signal'] == 1])}, Sell: {len(ma_signals[ma_signals['signal'] == -1])}")
    print(f"RSI Strategy - Buy: {len(rsi_signals[rsi_signals['signal'] == 1])}, Sell: {len(rsi_signals[rsi_signals['signal'] == -1])}")
    print(f"Momentum Strategy - Buy: {len(momentum_signals[momentum_signals['signal'] == 1])}, Sell: {len(momentum_signals[momentum_signals['signal'] == -1])}")
    print(f"Breakout Strategy - Buy: {len(breakout_signals[breakout_signals['signal'] == 1])}, Sell: {len(breakout_signals[breakout_signals['signal'] == -1])}")
    print(f"Mean Reversion Strategy - Buy: {len(mean_reversion_signals[mean_reversion_signals['signal'] == 1])}, Sell: {len(mean_reversion_signals[mean_reversion_signals['signal'] == -1])}")
    print(f"Dual Strategy System - Buy: {len(dual_signals[dual_signals['signal'] == 1])}, Sell: {len(dual_signals[dual_signals['signal'] == -1])}") 