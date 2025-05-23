#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized visualizer module for stock trading charts with minimal lag.
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
    """Optimized class for creating and saving stock charts with minimal lag."""
    
    def __init__(self, figure_dir='figures'):
        """
        Initialize the Visualizer.
        
        Args:
            figure_dir (str): Directory to save figures
        """
        self.figure_dir = figure_dir
        self.max_figures = 10  # Reduced from 30 to 10 for less disk usage
        self.last_plot_time = {}  # Track when we last plotted each symbol to avoid excessive plotting
        self.plot_interval = 30  # Only create new plots every 30 seconds per symbol
        
        # Create figures directory if it doesn't exist
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
            logger.info(f"Created directory: {figure_dir}")
        
        # Configure matplotlib for better performance
        plt.style.use('default')  # Use default instead of ggplot for better performance
        plt.rcParams['figure.figsize'] = (8, 5)  # Smaller figure size
        plt.rcParams['figure.dpi'] = 60  # Lower DPI for faster rendering
        plt.rcParams['savefig.dpi'] = 60  # Lower save DPI
        plt.rcParams['axes.grid'] = False  # Disable grid by default for performance
    
    def _should_create_plot(self, symbol):
        """Check if we should create a new plot for this symbol."""
        current_time = datetime.now().timestamp()
        last_time = self.last_plot_time.get(symbol, 0)
        
        if current_time - last_time > self.plot_interval:
            self.last_plot_time[symbol] = current_time
            return True
        return False
    
    def plot_stock_data(self, data, symbol, save=True, filename=None):
        """
        Plot basic stock price data - SIMPLIFIED VERSION.
        
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
        
        # Skip if we plotted recently
        if not self._should_create_plot(symbol):
            return None, None
        
        try:
            # Limit data to last 30 points for performance
            if len(data) > 30:
                data = data.iloc[-30:]
            
            # Create simple single subplot (no volume for performance)
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            
            # Plot only close price - no moving averages for speed
            ax.plot(data.index, data['close'], label='Close', color='blue', linewidth=1)
            
            # Minimal formatting
            ax.set_title(f'{symbol}', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=10)
            
            plt.tight_layout()
            
            # Save if requested
            if save:
                if filename is None:
                    timestamp = datetime.now().strftime('%H%M%S')  # Only time, not date
                    filename = f"{symbol}_{timestamp}.png"
                
                filepath = os.path.join(self.figure_dir, filename)
                plt.savefig(filepath, dpi=60, bbox_inches='tight')
                logger.debug(f"Saved basic chart: {filepath}")
            
            # Close immediately to free memory
            plt.close(fig)
            
            return fig, ax
            
        except Exception as e:
            logger.error(f"Error plotting stock data: {str(e)}")
            plt.close()
            return None, None
    
    def plot_signals(self, data, symbol, save=True, filename=None):
        """
        Plot stock data with trading signals - OPTIMIZED VERSION.
        
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
        
        # Skip if we plotted recently
        if not self._should_create_plot(symbol):
            return None, None
        
        try:
            # Limit data to last 30 points for performance
            if len(data) > 30:
                data = data.iloc[-30:]
            
            # Single subplot only - no volume for performance
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            
            # Plot close price
            ax.plot(data.index, data['close'], label='Close', color='blue', linewidth=1.5)
            
            # Add only ONE moving average for performance
            if 'ema_20' in data.columns:
                ax.plot(data.index, data['ema_20'], label='EMA20', color='orange', linewidth=1, alpha=0.7)
            elif 'sma_20' in data.columns:
                ax.plot(data.index, data['sma_20'], label='SMA20', color='orange', linewidth=1, alpha=0.7)
            
            # Plot signals
            buy_signals = data[data['signal'] == 1]
            if not buy_signals.empty:
                ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=60, label='Buy', zorder=5)
            
            sell_signals = data[data['signal'] == -1]
            if not sell_signals.empty:
                ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=60, label='Sell', zorder=5)
            
            # Minimal formatting
            ax.set_title(f'{symbol} Signals', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=10)
            ax.legend(loc='upper left', fontsize=9)
            
            plt.tight_layout()
            
            # Save if requested
            if save:
                if filename is None:
                    timestamp = datetime.now().strftime('%H%M%S')
                    filename = f"{symbol}_signals_{timestamp}.png"
                
                filepath = os.path.join(self.figure_dir, filename)
                plt.savefig(filepath, dpi=60, bbox_inches='tight')
                logger.debug(f"Saved signals chart: {filepath}")
            
            # Close immediately to free memory
            plt.close(fig)
            
            return fig, ax
            
        except Exception as e:
            logger.error(f"Error plotting signals: {str(e)}")
            plt.close()
            return None, None
    
    def plot_rsi(self, data, symbol, save=True, filename=None):
        """
        Plot stock data with RSI - SIMPLIFIED VERSION.
        
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
            return self.plot_signals(data, symbol, save, filename)
        
        # Skip if we plotted recently
        if not self._should_create_plot(symbol):
            return None, None
        
        try:
            # Limit data to last 30 points for performance
            if len(data) > 30:
                data = data.iloc[-30:]
            
            # Only 2 subplots - price and RSI (no volume)
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 5), sharex=True)
            
            # Plot price data
            ax1.plot(data.index, data['close'], label='Close', color='blue', linewidth=1.5)
            
            # Add signals if available
            if 'signal' in data.columns:
                buy_signals = data[data['signal'] == 1]
                if not buy_signals.empty:
                    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=50, label='Buy')
                
                sell_signals = data[data['signal'] == -1]
                if not sell_signals.empty:
                    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=50, label='Sell')
            
            # Plot RSI
            ax2.plot(data.index, data['rsi'], label='RSI', color='purple', linewidth=1.5)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax2.set_ylim(0, 100)
            
            # Minimal formatting
            ax1.set_title(f'{symbol} with RSI', fontsize=12)
            ax1.set_ylabel('Price ($)', fontsize=10)
            ax1.legend(loc='upper left', fontsize=9)
            
            ax2.set_xlabel('Date', fontsize=10)
            ax2.set_ylabel('RSI', fontsize=10)
            
            plt.tight_layout()
            
            # Save if requested
            if save:
                if filename is None:
                    timestamp = datetime.now().strftime('%H%M%S')
                    filename = f"{symbol}_rsi_{timestamp}.png"
                
                filepath = os.path.join(self.figure_dir, filename)
                plt.savefig(filepath, dpi=60, bbox_inches='tight')
                logger.debug(f"Saved RSI chart: {filepath}")
            
            # Close immediately to free memory
            plt.close(fig)
            
            return fig, (ax1, ax2)
            
        except Exception as e:
            logger.error(f"Error plotting RSI: {str(e)}")
            plt.close()
            return None, None
    
    def clean_old_figures(self, max_figures=None):
        """
        Clean old figures more aggressively to prevent lag.
        
        Args:
            max_figures (int): Maximum number of figures to keep
        """
        if max_figures is None:
            max_figures = self.max_figures
        
        if not os.path.exists(self.figure_dir):
            logger.warning(f"Figure directory {self.figure_dir} does not exist")
            return
            
        try:
            files = [os.path.join(self.figure_dir, f) for f in os.listdir(self.figure_dir) if f.endswith('.png')]
            files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first
            
            # Keep only the most recent figures
            if len(files) > max_figures:
                files_to_delete = files[max_figures:]
                for file_to_delete in files_to_delete:
                    try:
                        os.remove(file_to_delete)
                        logger.debug(f"Deleted old figure: {os.path.basename(file_to_delete)}")
                    except Exception as e:
                        logger.error(f"Error deleting {file_to_delete}: {str(e)}")
                        
                logger.info(f"Cleaned {len(files_to_delete)} old figures, kept {max_figures}")
        except Exception as e:
            logger.error(f"Error cleaning figures: {str(e)}")

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