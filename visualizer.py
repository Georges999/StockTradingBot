#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple visualizer module for stock trading charts and performance metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Visualizer:
    """Simple class for creating and saving stock charts."""
    
    def __init__(self, figure_dir='figures'):
        """
        Initialize the Visualizer.
        
        Args:
            figure_dir (str): Directory to save figures
        """
        self.figure_dir = figure_dir
        
        # Create figures directory if it doesn't exist
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
            logger.info(f"Created directory: {figure_dir}")
        
        # Configure matplotlib style
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_stock_data(self, data, symbol, save=True, filename=None):
        """
        Plot basic stock price data with volume.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            symbol (str): Stock symbol
            save (bool): Whether to save the figure
            filename (str, optional): Filename to save the figure
            
        Returns:
            tuple: Figure and axes objects
        """
        if data.empty:
            logger.warning("Cannot plot empty data")
            return None, None
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot price data
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        
        # Add moving averages if available
        for col in data.columns:
            if col.startswith('sma_') or col.startswith('ema_'):
                ax1.plot(data.index, data[col], label=col.upper(), linewidth=1, alpha=0.8)
        
        # Plot volume
        ax2.bar(data.index, data['volume'], color='gray', alpha=0.5)
        
        # Format axes
        ax1.set_title(f'{symbol} Stock Price', fontsize=14)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol}_price_{timestamp}.png"
            
            filepath = os.path.join(self.figure_dir, filename)
            plt.savefig(filepath, dpi=100)
            logger.info(f"Saved figure to {filepath}")
        
        return fig, (ax1, ax2)
    
    def plot_signals(self, data, symbol, save=True, filename=None):
        """
        Plot stock data with trading signals.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and signals
            symbol (str): Stock symbol
            save (bool): Whether to save the figure
            filename (str, optional): Filename to save the figure
            
        Returns:
            tuple: Figure and axes objects
        """
        if data.empty:
            logger.warning("Cannot plot empty data")
            return None, None
        
        if 'signal' not in data.columns:
            logger.warning("No signal column in data")
            return self.plot_stock_data(data, symbol, save, filename)
        
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot price data
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        
        # Add moving averages if available
        for col in data.columns:
            if col.startswith('sma_') or col.startswith('ema_'):
                ax1.plot(data.index, data[col], label=col.upper(), linewidth=1, alpha=0.8)
        
        # Plot buy signals
        buy_signals = data[data['signal'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
        
        # Plot sell signals
        sell_signals = data[data['signal'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
        
        # Plot volume
        ax2.bar(data.index, data['volume'], color='gray', alpha=0.5)
        
        # Format axes
        ax1.set_title(f'{symbol} Trading Signals', fontsize=14)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol}_signals_{timestamp}.png"
            
            filepath = os.path.join(self.figure_dir, filename)
            plt.savefig(filepath, dpi=100)
            logger.info(f"Saved signals chart to {filepath}")
        
        return fig, (ax1, ax2)
    
    def plot_rsi(self, data, symbol, save=True, filename=None):
        """
        Plot stock data with RSI indicator.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and RSI
            symbol (str): Stock symbol
            save (bool): Whether to save the figure
            filename (str, optional): Filename to save the figure
            
        Returns:
            tuple: Figure and axes objects
        """
        if data.empty:
            logger.warning("Cannot plot empty data")
            return None, None
        
        if 'rsi' not in data.columns:
            logger.warning("No RSI column in data")
            return self.plot_stock_data(data, symbol, save, filename)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
        
        # Plot price data
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        
        # Add moving averages if available
        for col in data.columns:
            if col.startswith('sma_') or col.startswith('ema_'):
                ax1.plot(data.index, data[col], label=col.upper(), linewidth=1, alpha=0.8)
        
        # Plot buy signals if available
        if 'signal' in data.columns:
            buy_signals = data[data['signal'] == 1]
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
            
            # Plot sell signals
            sell_signals = data[data['signal'] == -1]
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
        
        # Plot volume
        ax2.bar(data.index, data['volume'], color='gray', alpha=0.5)
        
        # Plot RSI
        ax3.plot(data.index, data['rsi'], label='RSI', color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.text(data.index[0], 70, 'Overbought', color='red', fontsize=10)
        ax3.text(data.index[0], 30, 'Oversold', color='green', fontsize=10)
        ax3.set_ylim(0, 100)
        
        # Format axes
        ax1.set_title(f'{symbol} with RSI', fontsize=14)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('RSI', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol}_rsi_{timestamp}.png"
            
            filepath = os.path.join(self.figure_dir, filename)
            plt.savefig(filepath, dpi=100)
            logger.info(f"Saved RSI chart to {filepath}")
        
        return fig, (ax1, ax2, ax3)
    
    def clean_old_figures(self, max_figures=20):
        """
        Clean old figures, keeping only the most recent ones.
        
        Args:
            max_figures (int): Maximum number of figures to keep
        """
        if not os.path.exists(self.figure_dir):
            logger.warning(f"Figure directory {self.figure_dir} does not exist")
            return
            
        files = [os.path.join(self.figure_dir, f) for f in os.listdir(self.figure_dir) if f.endswith('.png')]
        files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first
        
        # Delete old files if we have more than max_figures
        if len(files) > max_figures:
            for file_to_delete in files[max_figures:]:
                try:
                    os.remove(file_to_delete)
                    logger.info(f"Deleted old figure: {file_to_delete}")
                except Exception as e:
                    logger.error(f"Error deleting {file_to_delete}: {str(e)}")

# Simple test function
if __name__ == "__main__":
    # Generate sample data
    import numpy as np
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    close_prices = [100]
    for _ in range(99):
        close_prices.append(close_prices[-1] * (1 + np.random.normal(0, 0.01)))
    
    # Create DataFrame
    data = pd.DataFrame({
        'close': close_prices,
        'open': [price * 0.99 for price in close_prices],
        'high': [price * 1.01 for price in close_prices],
        'low': [price * 0.98 for price in close_prices],
        'volume': np.random.randint(1000, 10000, size=100)
    }, index=dates)
    
    # Calculate SMA
    data['sma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
    data['sma_50'] = data['close'].rolling(window=50, min_periods=1).mean()
    
    # Create synthetic signals
    data['signal'] = 0
    # Every 10 days, alternate between buy and sell signals
    data.iloc[10::20, data.columns.get_loc('signal')] = 1  # Buy
    data.iloc[20::20, data.columns.get_loc('signal')] = -1  # Sell
    
    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Test visualizer
    visualizer = Visualizer()
    
    # Plot basic stock data
    visualizer.plot_stock_data(data, 'EXAMPLE')
    
    # Plot with signals
    visualizer.plot_signals(data, 'EXAMPLE')
    
    # Plot with RSI
    visualizer.plot_rsi(data, 'EXAMPLE')
    
    # Clean old figures
    visualizer.clean_old_figures(2)  # Keep only the 2 most recent figures
    
    print("Test visualizations created. Check the 'figures' directory.") 