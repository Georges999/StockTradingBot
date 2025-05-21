#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple strategy module for stock trading.
"""

import pandas as pd
import numpy as np
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Strategy:
    """Base class for all trading strategies."""
    
    def __init__(self, name):
        """Initialize the strategy with a name."""
        self.name = name
    
    def generate_signals(self, data):
        """
        Generate trading signals for the given data.
        This method should be implemented by all strategy subclasses.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column added
        """
        raise NotImplementedError("Subclasses must implement this method")

class MovingAverageCrossover(Strategy):
    """Simple Moving Average Crossover strategy."""
    
    def __init__(self, short_window=20, long_window=50):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            short_window (int): Short moving average window
            long_window (int): Long moving average window
        """
        super().__init__("MA Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Moving Average Crossover.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate moving averages
        df[f'sma_{self.short_window}'] = df['close'].rolling(window=self.short_window, min_periods=1).mean()
        df[f'sma_{self.long_window}'] = df['close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # Buy signal: short-term MA crosses above long-term MA
        df.loc[df[f'sma_{self.short_window}'] > df[f'sma_{self.long_window}'], 'signal'] = 1
        
        # Sell signal: short-term MA crosses below long-term MA
        df.loc[df[f'sma_{self.short_window}'] < df[f'sma_{self.long_window}'], 'signal'] = -1
        
        return df

class RSIStrategy(Strategy):
    """Relative Strength Index (RSI) strategy."""
    
    def __init__(self, period=14, overbought=70, oversold=30):
        """
        Initialize the RSI strategy.
        
        Args:
            period (int): RSI calculation period
            overbought (int): Threshold for overbought condition
            oversold (int): Threshold for oversold condition
        """
        super().__init__("RSI Strategy")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate daily price changes
        df['price_change'] = df['close'].diff()
        
        # Calculate gains and losses
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Calculate average gains and losses
        avg_gain = df['gain'].rolling(window=self.period).mean()
        avg_loss = df['loss'].rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # Buy signal: RSI below oversold threshold
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1
        
        # Sell signal: RSI above overbought threshold
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1
        
        return df

class MomentumStrategy(Strategy):
    """Simple momentum strategy based on price changes."""
    
    def __init__(self, period=10, threshold=0.05):
        """
        Initialize the Momentum strategy.
        
        Args:
            period (int): Number of days to calculate momentum
            threshold (float): Threshold for momentum signal (as decimal, e.g., 0.05 = 5%)
        """
        super().__init__("Momentum Strategy")
        self.period = period
        self.threshold = threshold
    
    def generate_signals(self, data):
        """
        Generate trading signals based on momentum.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with signals column
        """
        if data.empty:
            logger.warning("Cannot generate signals for empty data")
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate momentum (percent change over period)
        df['momentum'] = df['close'].pct_change(periods=self.period) * 100
        
        # Create signals
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        
        # Buy signal: momentum above positive threshold
        df.loc[df['momentum'] > (self.threshold * 100), 'signal'] = 1
        
        # Sell signal: momentum below negative threshold
        df.loc[df['momentum'] < -(self.threshold * 100), 'signal'] = -1
        
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
                signals[strategy_name] = strategy.generate_signals(data)
            else:
                logger.warning(f"Strategy '{strategy_name}' not found")
        else:
            # Use all strategies
            for name, strategy in self.strategies.items():
                signals[name] = strategy.generate_signals(data)
        
        return signals

# Simple test function
if __name__ == "__main__":
    # Sample data (manually created)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    close_prices = [100]
    for _ in range(99):
        close_prices.append(close_prices[-1] * (1 + np.random.normal(0, 0.01)))
        
    data = pd.DataFrame({
        'date': dates,
        'close': close_prices,
        'open': [price * 0.99 for price in close_prices],
        'high': [price * 1.01 for price in close_prices],
        'low': [price * 0.98 for price in close_prices],
        'volume': np.random.randint(1000, 10000, size=100)
    })
    data.set_index('date', inplace=True)
    
    # Test strategies
    ma_strategy = MovingAverageCrossover(short_window=10, long_window=30)
    rsi_strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    momentum_strategy = MomentumStrategy(period=10, threshold=0.05)
    
    # Generate signals
    ma_signals = ma_strategy.generate_signals(data)
    rsi_signals = rsi_strategy.generate_signals(data)
    momentum_signals = momentum_strategy.generate_signals(data)
    
    print(f"MA Crossover Strategy - Buy Signals: {len(ma_signals[ma_signals['signal'] == 1])}")
    print(f"MA Crossover Strategy - Sell Signals: {len(ma_signals[ma_signals['signal'] == -1])}")
    
    print(f"RSI Strategy - Buy Signals: {len(rsi_signals[rsi_signals['signal'] == 1])}")
    print(f"RSI Strategy - Sell Signals: {len(rsi_signals[rsi_signals['signal'] == -1])}")
    
    print(f"Momentum Strategy - Buy Signals: {len(momentum_signals[momentum_signals['signal'] == 1])}")
    print(f"Momentum Strategy - Sell Signals: {len(momentum_signals[momentum_signals['signal'] == -1])}") 