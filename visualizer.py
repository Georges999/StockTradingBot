#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizer module for the StockTradingBot.
Handles visualization of market data, indicators, and performance metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Visualizer:
    """Class for visualizing trading data and performance metrics."""
    
    def __init__(self):
        """Initialize the Visualizer."""
        # Create figures directory if it doesn't exist
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
        # Set the default style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Increase default figure size
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Use a higher DPI for better resolution
        plt.rcParams['figure.dpi'] = 100
        
        # Use a modern font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    def plot_data_with_signals(self, data: pd.DataFrame, signals: Dict, symbol: str) -> None:
        """
        Plot price data with trading signals.
        
        Args:
            data: DataFrame with price data and indicators.
            signals: Dictionary with trading signals.
            symbol: The ticker symbol.
        """
        if data.empty:
            logger.warning("Cannot plot empty data")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Main price chart
        ax_price = axes[0]
        
        # Plot price data
        ax_price.plot(data.index, data['close'], label='Close Price', color='black', linewidth=1.5)
        
        # Add moving averages if available
        if 'sma_20' in data.columns:
            ax_price.plot(data.index, data['sma_20'], label='SMA 20', color='blue', alpha=0.7)
        if 'sma_50' in data.columns:
            ax_price.plot(data.index, data['sma_50'], label='SMA 50', color='orange', alpha=0.7)
        if 'sma_200' in data.columns:
            ax_price.plot(data.index, data['sma_200'], label='SMA 200', color='red', alpha=0.7)
        
        # Add Bollinger Bands if available
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            ax_price.plot(data.index, data['bb_upper'], color='gray', linestyle='--', alpha=0.5)
            ax_price.plot(data.index, data['bb_lower'], color='gray', linestyle='--', alpha=0.5)
            ax_price.fill_between(data.index, data['bb_upper'], data['bb_lower'], color='gray', alpha=0.1)
        
        # Highlight the trading signal
        last_idx = data.index[-1]
        last_price = data['close'].iloc[-1]
        signal = signals.get('signal', 'neutral')
        direction = signals.get('direction', 'neutral')
        
        if signal == 'buy' and direction == 'long':
            ax_price.scatter(last_idx, last_price, color='green', s=100, marker='^', zorder=5)
            ax_price.annotate('BUY', (last_idx, last_price), xytext=(10, 10), 
                            textcoords='offset points', color='green', fontweight='bold')
        elif signal == 'sell' and direction == 'short':
            ax_price.scatter(last_idx, last_price, color='red', s=100, marker='v', zorder=5)
            ax_price.annotate('SELL', (last_idx, last_price), xytext=(10, 10), 
                            textcoords='offset points', color='red', fontweight='bold')
        
        # Format price axis
        ax_price.set_title(f'{symbol} Price Chart with Signals', fontsize=16)
        ax_price.set_ylabel('Price ($)', fontsize=12)
        ax_price.grid(True, alpha=0.3)
        ax_price.legend(loc='upper left')
        
        # RSI plot
        ax_rsi = axes[1]
        if 'rsi' in data.columns:
            ax_rsi.plot(data.index, data['rsi'], color='purple', linewidth=1.2)
            ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax_rsi.fill_between(data.index, data['rsi'], 70, where=(data['rsi'] >= 70), color='red', alpha=0.2)
            ax_rsi.fill_between(data.index, data['rsi'], 30, where=(data['rsi'] <= 30), color='green', alpha=0.2)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_ylabel('RSI', fontsize=12)
            ax_rsi.grid(True, alpha=0.3)
        else:
            ax_rsi.set_visible(False)
        
        # MACD plot
        ax_macd = axes[2]
        if 'macd_line' in data.columns and 'macd_signal' in data.columns and 'macd_histogram' in data.columns:
            ax_macd.plot(data.index, data['macd_line'], color='blue', linewidth=1.2, label='MACD')
            ax_macd.plot(data.index, data['macd_signal'], color='red', linewidth=1.2, label='Signal')
            
            # Plot histogram as bar chart
            for i, (idx, value) in enumerate(zip(data.index, data['macd_histogram'])):
                color = 'green' if value > 0 else 'red'
                ax_macd.bar(idx, value, color=color, alpha=0.5, width=0.7)
            
            ax_macd.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_macd.set_ylabel('MACD', fontsize=12)
            ax_macd.grid(True, alpha=0.3)
            ax_macd.legend(loc='upper left')
        else:
            ax_macd.set_visible(False)
        
        # Format x-axis for all subplots
        for ax in axes:
            if ax.get_visible():
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add a common xlabel
        fig.text(0.5, 0.02, 'Date', ha='center', fontsize=12)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        # Add signal information text box
        signal_text = f"Signal: {signal.upper()}\nDirection: {direction}\nStrength: {signals.get('strength', 0):.2f}"
        if 'reason' in signals:
            signal_text += f"\nReason: {signals['reason']}"
        
        # Position the text box in the first subplot
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax_price.text(0.02, 0.02, signal_text, transform=ax_price.transAxes, fontsize=11,
                    verticalalignment='bottom', bbox=props)
        
        # Save and show figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"figures/{symbol}_signal_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        logger.info(f"Saved signal chart to {filename}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_performance_dashboard(self, equity_curve: List[Dict], trades: List[Dict], 
                                metrics: Dict, filename: str = None) -> None:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            equity_curve: List of portfolio value dictionaries.
            trades: List of trade dictionaries.
            metrics: Dictionary of performance metrics.
            filename: Name of the file to save the dashboard.
        """
        # Convert to DataFrames
        if not equity_curve or not trades:
            logger.warning("Cannot plot performance dashboard with empty data")
            return
        
        # Create a 4-panel dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Equity curve (top-left)
        ax_equity = axes[0, 0]
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Plot equity curve
        ax_equity.plot(equity_df.index, equity_df['value'], color='blue', linewidth=2)
        ax_equity.set_title('Portfolio Equity Curve', fontsize=14)
        ax_equity.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax_equity.grid(True, alpha=0.3)
        ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Add metrics text box
        metrics_text = (
            f"Total Return: {metrics.get('return_pct', 0):.2f}%\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
            f"Win Rate: {metrics.get('win_rate', 0):.2f}%\n"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}"
        )
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax_equity.text(0.02, 0.02, metrics_text, transform=ax_equity.transAxes, fontsize=12,
                    verticalalignment='bottom', bbox=props)
        
        # Trade P&L distribution (top-right)
        ax_pnl = axes[0, 1]
        
        # Extract P&L from trades
        pnl_values = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
        
        if pnl_values:
            # Plot P&L distribution
            sns.histplot(pnl_values, bins=20, kde=True, ax=ax_pnl, color='skyblue')
            ax_pnl.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            ax_pnl.set_title('Trade P&L Distribution', fontsize=14)
            ax_pnl.set_xlabel('P&L ($)', fontsize=12)
            ax_pnl.set_ylabel('Frequency', fontsize=12)
            ax_pnl.grid(True, alpha=0.3)
        else:
            ax_pnl.text(0.5, 0.5, "No P&L data available", horizontalalignment='center',
                      verticalalignment='center', transform=ax_pnl.transAxes, fontsize=14)
        
        # Monthly returns (bottom-left)
        ax_monthly = axes[1, 0]
        
        if not equity_df.empty and len(equity_df) > 1:
            # Calculate monthly returns
            equity_df['monthly_return'] = equity_df['value'].pct_change(periods=30) * 100
            
            # Group by month and calculate average return
            monthly_returns = equity_df.resample('M')['monthly_return'].last()
            
            # Plot monthly returns
            colors = ['green' if r >= 0 else 'red' for r in monthly_returns]
            ax_monthly.bar(monthly_returns.index, monthly_returns, color=colors, alpha=0.7)
            ax_monthly.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_monthly.set_title('Monthly Returns', fontsize=14)
            ax_monthly.set_ylabel('Return (%)', fontsize=12)
            ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_monthly.grid(True, alpha=0.3)
            plt.setp(ax_monthly.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax_monthly.text(0.5, 0.5, "Insufficient data for monthly returns", horizontalalignment='center',
                          verticalalignment='center', transform=ax_monthly.transAxes, fontsize=14)
        
        # Drawdown chart (bottom-right)
        ax_drawdown = axes[1, 1]
        
        if not equity_df.empty and len(equity_df) > 1:
            # Calculate drawdown
            peak = equity_df['value'].cummax()
            drawdown = ((equity_df['value'] - peak) / peak) * 100
            
            # Plot drawdown
            ax_drawdown.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
            ax_drawdown.plot(drawdown.index, drawdown, color='red', linewidth=1)
            ax_drawdown.set_title('Portfolio Drawdown', fontsize=14)
            ax_drawdown.set_ylabel('Drawdown (%)', fontsize=12)
            ax_drawdown.grid(True, alpha=0.3)
            ax_drawdown.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax_drawdown.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Invert y-axis for better visualization (drawdown is negative)
            ax_drawdown.invert_yaxis()
        else:
            ax_drawdown.text(0.5, 0.5, "Insufficient data for drawdown calculation", horizontalalignment='center',
                           verticalalignment='center', transform=ax_drawdown.transAxes, fontsize=14)
        
        # Add a title for the whole dashboard
        fig.suptitle('Trading Performance Dashboard', fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure if filename provided
        if filename:
            plt.savefig(f"figures/{filename}", dpi=100, bbox_inches='tight')
            logger.info(f"Saved performance dashboard to figures/{filename}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f"figures/performance_dashboard_{timestamp}.png", dpi=100, bbox_inches='tight')
            logger.info(f"Saved performance dashboard to figures/performance_dashboard_{timestamp}.png")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_correlation_matrix(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame], 
                               period: int = 30, filename: str = None) -> None:
        """
        Plot a correlation matrix of multiple symbols.
        
        Args:
            symbols: List of symbols to include.
            data_dict: Dictionary mapping symbols to their price DataFrames.
            period: Number of days to use for correlation calculation.
            filename: Name of the file to save the correlation matrix.
        """
        # Check if we have data for all symbols
        if not all(symbol in data_dict for symbol in symbols):
            logger.warning("Not all symbols have data for correlation matrix")
            return
        
        # Extract closing prices for each symbol
        prices = pd.DataFrame()
        
        for symbol in symbols:
            if symbol in data_dict and 'close' in data_dict[symbol].columns:
                prices[symbol] = data_dict[symbol]['close'].values[-period:]
        
        # Check if we have enough data
        if prices.empty or prices.shape[1] < 2:
            logger.warning("Insufficient data for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = prices.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
        plt.title(f'Correlation Matrix ({period}-day)', fontsize=16)
        
        # Save figure if filename provided
        if filename:
            plt.savefig(f"figures/{filename}", dpi=100, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to figures/{filename}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f"figures/correlation_matrix_{timestamp}.png", dpi=100, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to figures/correlation_matrix_{timestamp}.png")
        
        # Close the figure to free memory
        plt.close()
    
    def plot_trading_activity(self, trades: List[Dict], price_data: pd.DataFrame, symbol: str, 
                             filename: str = None) -> None:
        """
        Plot trading activity overlaid on price chart.
        
        Args:
            trades: List of trade dictionaries.
            price_data: DataFrame with price data.
            symbol: The ticker symbol.
            filename: Name of the file to save the chart.
        """
        if price_data.empty or not trades:
            logger.warning("Cannot plot trading activity with empty data")
            return
        
        # Filter trades for the given symbol
        symbol_trades = [trade for trade in trades if trade.get('symbol') == symbol]
        
        if not symbol_trades:
            logger.warning(f"No trades found for symbol {symbol}")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(price_data.index, price_data['close'], color='black', linewidth=1.5)
        
        # Add entry and exit markers
        for trade in symbol_trades:
            # Extract trade data
            action = trade.get('action', '')
            timestamp = pd.to_datetime(trade.get('timestamp', ''))
            price = trade.get('price', 0)
            quantity = trade.get('quantity', 0)
            pnl = trade.get('pnl', 0)
            
            # Find nearest timestamp in price data
            if timestamp not in price_data.index:
                # Find closest date
                closest_idx = abs(price_data.index - timestamp).argmin()
                timestamp = price_data.index[closest_idx]
            
            # Add markers based on action
            if action == 'buy':
                ax.scatter(timestamp, price, color='green', s=100, marker='^', zorder=5)
                ax.annotate(f"BUY\n{quantity} @ ${price:.2f}", 
                          (timestamp, price), xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold', color='green')
            elif action == 'sell':
                ax.scatter(timestamp, price, color='red', s=100, marker='v', zorder=5)
                label = f"SELL\n{quantity} @ ${price:.2f}"
                if pnl != 0:
                    label += f"\nP&L: ${pnl:.2f}"
                ax.annotate(label, (timestamp, price), xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold', color='red')
            elif action == 'sell_short':
                ax.scatter(timestamp, price, color='purple', s=100, marker='v', zorder=5)
                ax.annotate(f"SHORT\n{quantity} @ ${price:.2f}", 
                          (timestamp, price), xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold', color='purple')
            elif action == 'buy_to_cover':
                ax.scatter(timestamp, price, color='blue', s=100, marker='^', zorder=5)
                label = f"COVER\n{quantity} @ ${price:.2f}"
                if pnl != 0:
                    label += f"\nP&L: ${pnl:.2f}"
                ax.annotate(label, (timestamp, price), xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold', color='blue')
        
        # Add indicators if available
        if 'sma_20' in price_data.columns:
            ax.plot(price_data.index, price_data['sma_20'], color='blue', alpha=0.7, linewidth=1, label='SMA 20')
        if 'sma_50' in price_data.columns:
            ax.plot(price_data.index, price_data['sma_50'], color='orange', alpha=0.7, linewidth=1, label='SMA 50')
        
        # Format chart
        ax.set_title(f'Trading Activity for {symbol}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.legend(loc='upper left')
        
        # Add trading summary text box
        total_trades = len(symbol_trades)
        total_pnl = sum(trade.get('pnl', 0) for trade in symbol_trades if 'pnl' in trade)
        win_trades = sum(1 for trade in symbol_trades if trade.get('pnl', 0) > 0)
        lose_trades = sum(1 for trade in symbol_trades if trade.get('pnl', 0) < 0)
        win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        
        summary_text = (
            f"Total Trades: {total_trades}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Total P&L: ${total_pnl:.2f}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=12,
              verticalalignment='top', bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if filename provided
        if filename:
            plt.savefig(f"figures/{filename}", dpi=100, bbox_inches='tight')
            logger.info(f"Saved trading activity chart to figures/{filename}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f"figures/{symbol}_trading_activity_{timestamp}.png", dpi=100, bbox_inches='tight')
            logger.info(f"Saved trading activity chart to figures/{symbol}_trading_activity_{timestamp}.png")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_strategy_comparison(self, strategy_results: Dict[str, Dict], filename: str = None) -> None:
        """
        Plot a comparison of different strategies.
        
        Args:
            strategy_results: Dictionary mapping strategy names to performance metrics.
            filename: Name of the file to save the chart.
        """
        if not strategy_results:
            logger.warning("Cannot plot strategy comparison with empty data")
            return
        
        # Extract metrics for comparison
        strategies = list(strategy_results.keys())
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Create a DataFrame for plotting
        comparison_data = pd.DataFrame(index=strategies, columns=metrics)
        
        for strategy, results in strategy_results.items():
            for metric in metrics:
                comparison_data.loc[strategy, metric] = results.get(metric, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            colors = ['green' if m >= 0 else 'red' for m in comparison_data[metric]]
            
            # Special case for max_drawdown (lower is better)
            if metric == 'max_drawdown':
                colors = ['green' if m <= 0 else 'red' for m in -comparison_data[metric]]
                comparison_data[metric] = -comparison_data[metric]  # Invert for visualization
            
            # Plot the metric
            ax.bar(comparison_data.index, comparison_data[metric], color=colors, alpha=0.7)
            
            # Format axes
            ax.set_title(f'Strategy Comparison: {metric.replace("_", " ").title()}', fontsize=14)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add a title for the whole figure
        fig.suptitle('Strategy Performance Comparison', fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if filename provided
        if filename:
            plt.savefig(f"figures/{filename}", dpi=100, bbox_inches='tight')
            logger.info(f"Saved strategy comparison chart to figures/{filename}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f"figures/strategy_comparison_{timestamp}.png", dpi=100, bbox_inches='tight')
            logger.info(f"Saved strategy comparison chart to figures/strategy_comparison_{timestamp}.png")
        
        # Close the figure to free memory
        plt.close(fig) 