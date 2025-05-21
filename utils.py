#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the StockTradingBot.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

def setup_logger(log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Set up the logger for the trading bot.
    
    Args:
        log_file: Path to the log file. If None, logs to console only.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Convert string level to logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(os.path.join('logs', log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def is_market_open() -> bool:
    """
    Check if the US stock market is currently open.
    This is a simplified check and doesn't account for holidays.
    
    Returns:
        Boolean indicating if the market is open.
    """
    now = datetime.now()
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if now.weekday() > 4:  # Saturday or Sunday
        return False
    
    # Check if it's between 9:30 AM and 4:00 PM Eastern Time
    # This is a simplified approach and assumes the local time is ET
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    
    return market_open <= now <= market_close

def get_next_market_open() -> datetime:
    """
    Get the next market open time.
    This is a simplified implementation and doesn't account for holidays.
    
    Returns:
        Datetime object representing the next market open time.
    """
    now = datetime.now()
    next_open = now.replace(hour=9, minute=30, second=0)
    
    # If we're past market open time today, move to tomorrow
    if now.hour > 9 or (now.hour == 9 and now.minute >= 30):
        next_open += timedelta(days=1)
    
    # If it's Friday after market open, move to Monday
    if next_open.weekday() == 4 and (now.hour > 9 or (now.hour == 9 and now.minute >= 30)):
        next_open += timedelta(days=3)
    
    # If it's Saturday, move to Monday
    elif next_open.weekday() == 5:
        next_open += timedelta(days=2)
    
    # If it's Sunday, move to Monday
    elif next_open.weekday() == 6:
        next_open += timedelta(days=1)
    
    return next_open

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio of a series of returns.
    
    Args:
        returns: List of period returns (not percentages).
        risk_free_rate: The risk-free rate for the period (not percentage).
        
    Returns:
        The Sharpe ratio.
    """
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # If all returns are identical, the standard deviation will be 0,
    # which would cause a division by zero error
    if len(returns) <= 1 or np.std(excess_returns, ddof=1) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252)

def calculate_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate the maximum drawdown and its duration from an equity curve.
    
    Args:
        equity_curve: Series of equity values over time.
        
    Returns:
        Tuple of (maximum drawdown percentage, start index, end index).
    """
    # Make a copy of the input
    equity = equity_curve.copy()
    
    # Initialize variables
    max_drawdown = 0
    max_drawdown_start = 0
    max_drawdown_end = 0
    peak = equity.iloc[0]
    peak_idx = 0
    
    for i, value in enumerate(equity):
        if value > peak:
            peak = value
            peak_idx = i
        
        # Calculate drawdown
        drawdown = (peak - value) / peak
        
        # Update max drawdown if current drawdown is greater
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_start = peak_idx
            max_drawdown_end = i
    
    return max_drawdown, max_drawdown_start, max_drawdown_end

def calculate_max_consecutive_losses(trades: List[Dict]) -> int:
    """
    Calculate the maximum number of consecutive losing trades.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key.
        
    Returns:
        Integer representing the maximum streak of consecutive losses.
    """
    if not trades:
        return 0
    
    max_streak = 0
    current_streak = 0
    
    for trade in trades:
        if trade.get('pnl', 0) < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def calculate_risk_of_ruin(win_rate: float, risk_reward_ratio: float) -> float:
    """
    Calculate the risk of ruin with fixed position sizing.
    
    Args:
        win_rate: Percentage of winning trades (0-1).
        risk_reward_ratio: Ratio of average win to average loss.
        
    Returns:
        Probability of ruin (0-1).
    """
    if win_rate >= 1 or win_rate <= 0 or risk_reward_ratio <= 0:
        return 0 if win_rate == 1 else 1
    
    # Expected value per trade
    expected_value = win_rate * risk_reward_ratio - (1 - win_rate)
    
    # If expected value is negative, ruin is certain given enough trades
    if expected_value <= 0:
        return 1
    
    # Calculate risk of ruin
    a = (1 - win_rate) / win_rate
    
    # If a^r = 1, then ruin is certain
    if a >= 1:
        return 1
    
    return a ** risk_reward_ratio

def normalize_data(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Normalize specified columns in a DataFrame to a 0-1 range.
    
    Args:
        data: DataFrame containing the data to normalize.
        cols: List of column names to normalize.
        
    Returns:
        DataFrame with normalized columns.
    """
    result = data.copy()
    
    for col in cols:
        if col in result.columns:
            min_val = result[col].min()
            max_val = result[col].max()
            
            # Avoid division by zero
            if max_val != min_val:
                result[col] = (result[col] - min_val) / (max_val - min_val)
            else:
                result[col] = 0.5  # If all values are the same, set to mid-point
    
    return result

def exponential_moving_correlation(series1: pd.Series, series2: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate the exponential moving correlation between two series.
    
    Args:
        series1: First time series.
        series2: Second time series.
        span: Span parameter for the exponential weighting.
        
    Returns:
        Series of exponential moving correlations.
    """
    # Make sure both series are the same length
    if len(series1) != len(series2):
        raise ValueError("Series must be of the same length")
    
    # Calculate returns
    returns1 = series1.pct_change().dropna()
    returns2 = series2.pct_change().dropna()
    
    # Initialize correlation series
    corr = pd.Series(index=returns1.index)
    
    # Calculate exponential moving correlation
    for i in range(span, len(returns1) + 1):
        window1 = returns1.iloc[i-span:i]
        window2 = returns2.iloc[i-span:i]
        corr.iloc[i-1] = window1.corr(window2)
    
    return corr

def calculate_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the Z-score of a series using a rolling window.
    
    Args:
        series: Time series data.
        window: Window size for rolling calculations.
        
    Returns:
        Series of Z-scores.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    # Calculate and return Z-score
    return (series - rolling_mean) / rolling_std

def format_currency(value: float) -> str:
    """
    Format a value as USD currency.
    
    Args:
        value: The numeric value to format.
        
    Returns:
        Formatted currency string.
    """
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: The numeric value to format (e.g., 0.1234 for 12.34%).
        
    Returns:
        Formatted percentage string.
    """
    return f"{value:.2%}" 