#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data fetching module for retrieving market data from various sources.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Tuple
import config
from alpaca_trade_api.rest import TimeFrame

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Class for fetching market data from different sources.
    """
    def __init__(self, config_obj=None):
        """
        Initialize the DataFetcher with configuration.
        
        Args:
            config_obj: Configuration object
        """
        if config_obj is None:
            self.config = config.Config()
        else:
            self.config = config_obj
            
        # Initialize Alpaca API if credentials are available
        if self.config.ALPACA_API_KEY and self.config.ALPACA_SECRET_KEY:
            self.alpaca_api = tradeapi.REST(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY,
                self.config.ALPACA_BASE_URL,
                api_version='v2'
            )
        else:
            self.alpaca_api = None
            logger.warning("Alpaca API credentials not found. Alpaca data source unavailable.")
    
    def get_historical_data(self, symbol, timeframe='1d', period=None, start_date=None, end_date=None, source='yfinance'):
        """
        Fetch historical market data for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            timeframe (str): Data timeframe (e.g., '1d', '1h', '5m')
            period (str, optional): Period to fetch (e.g., '1y', '6mo', '1mo')
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            source (str): Data source ('yfinance' or 'alpaca')
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if source == 'yfinance':
            return self._get_yfinance_data(symbol, timeframe, period, start_date, end_date)
        elif source == 'alpaca':
            return self._get_alpaca_data(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _get_yfinance_data(self, symbol, timeframe='1d', period=None, start_date=None, end_date=None):
        """
        Fetch data from Yahoo Finance.
        """
        logger.info(f"Fetching {timeframe} data for {symbol} from Yahoo Finance")
        
        # Convert timeframe to Yahoo Finance interval format
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d'
        }
        
        if timeframe not in interval_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        interval = interval_map[timeframe]
        
        # Determine period or date range
        if period is not None:
            df = yf.download(symbol, interval=interval, period=period, auto_adjust=True)
        elif start_date is not None and end_date is not None:
            df = yf.download(symbol, interval=interval, start=start_date, end=end_date, auto_adjust=True)
        else:
            # Default to last 100 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            df = yf.download(symbol, interval=interval, start=start_date, end=end_date, auto_adjust=True)
        
        # Check if data was retrieved
        if df.empty:
            logger.warning(f"No data retrieved for {symbol}")
            return None
        
        # Rename columns to standard format
        df.columns = [col.lower() for col in df.columns]
        if 'adj close' in df.columns:
            df.rename(columns={'adj close': 'adj_close'}, inplace=True)
        
        return df
    
    def _get_alpaca_data(self, symbol, timeframe='1d', start_date=None, end_date=None):
        """
        Fetch data from Alpaca.
        """
        if self.alpaca_api is None:
            raise ValueError("Alpaca API not initialized. Check your credentials.")
        
        logger.info(f"Fetching {timeframe} data for {symbol} from Alpaca")
        
        # Convert timeframe to Alpaca format
        timeframe_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame.Minute,
            '15m': TimeFrame.Minute,
            '30m': TimeFrame.Minute,
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        alpaca_timeframe = timeframe_map[timeframe]
        
        # Set multiplier for timeframes
        multiplier = 1
        if timeframe == '5m':
            multiplier = 5
        elif timeframe == '15m':
            multiplier = 15
        elif timeframe == '30m':
            multiplier = 30
        
        # Determine date range
        if end_date is None:
            end_date = datetime.now()
        else:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date is None:
            start_date = end_date - timedelta(days=100)
        else:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Fetch data
        try:
            bars = self.alpaca_api.get_bars(
                symbol,
                alpaca_timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                adjustment='raw',
                limit=10000,
                multiplier=multiplier
            ).df
            
            if bars.empty:
                logger.warning(f"No Alpaca data retrieved for {symbol}")
                return None
            
            # Rename columns to standard format
            bars.columns = [col.lower() for col in bars.columns]
            
            return bars
        
        except Exception as e:
            logger.error(f"Error fetching Alpaca data: {e}")
            return None
    
    def get_current_price(self, symbol, source='yfinance'):
        """
        Get the current market price for a symbol.
        
        Args:
            symbol (str): Stock symbol
            source (str): Data source ('yfinance' or 'alpaca')
            
        Returns:
            float: Current market price
        """
        if source == 'yfinance':
            ticker = yf.Ticker(symbol)
            todays_data = ticker.history(period='1d')
            if todays_data.empty:
                return None
            return todays_data['Close'].iloc[-1]
        
        elif source == 'alpaca':
            if self.alpaca_api is None:
                raise ValueError("Alpaca API not initialized. Check your credentials.")
            try:
                last_quote = self.alpaca_api.get_latest_quote(symbol)
                return (last_quote.ask_price + last_quote.bid_price) / 2
            except Exception as e:
                logger.error(f"Error fetching Alpaca price: {e}")
                return None
        
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def get_multiple_symbols_data(self, symbols, timeframe='1d', period=None, start_date=None, end_date=None, source='yfinance'):
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols (list): List of stock symbols
            timeframe (str): Data timeframe
            period (str, optional): Period to fetch
            start_date (str, optional): Start date
            end_date (str, optional): End date
            source (str): Data source
            
        Returns:
            dict: Dictionary of DataFrames with symbol as key
        """
        result = {}
        for symbol in symbols:
            try:
                data = self.get_historical_data(symbol, timeframe, period, start_date, end_date, source)
                if data is not None and not data.empty:
                    result[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return result
    
    def get_market_status(self):
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        if self.alpaca_api is None:
            # Fall back to a simple time check (US market hours)
            now = datetime.now()
            is_weekday = now.weekday() < 5  # 0-4 are Monday to Friday
            is_market_hours = 9 <= now.hour < 16  # 9:30 AM to 4:00 PM EST
            
            # This is a simplified check, doesn't account for holidays
            return is_weekday and is_market_hours
        
        try:
            clock = self.alpaca_api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Retrieve market data for a specific symbol and timeframe.
        
        Args:
            symbol: The ticker symbol.
            timeframe: Time interval (1m, 5m, 15m, 1h, 1d).
            limit: Number of data points to retrieve.
            
        Returns:
            DataFrame with OHLCV data.
        """
        try:
            # Try to get data from Alpaca if available
            if self.alpaca_api and timeframe in self.timeframe_map:
                return self._get_alpaca_data(symbol, timeframe, limit)
            
            # Fallback to Yahoo Finance
            return self._get_yfinance_data(symbol, timeframe, limit)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def _get_alpaca_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get data from Alpaca API."""
        try:
            # Convert timeframe to Alpaca format
            alpaca_timeframe = self.timeframe_map.get(timeframe, '1day')
            
            # Calculate start date based on limit and timeframe
            multiplier = 1
            if timeframe == '1h':
                multiplier = 60
            elif timeframe == '1d':
                multiplier = 24 * 60
            
            # Add some buffer to ensure we get enough data
            end = datetime.now()
            start = end - timedelta(minutes=multiplier * limit * 1.5)
            
            # Get data from Alpaca
            bars = self.alpaca_api.get_bars(
                symbol, 
                alpaca_timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=limit
            ).df
            
            if bars.empty:
                raise ValueError(f"No data returned from Alpaca for {symbol}")
            
            # Rename columns to lowercase for consistency
            bars.columns = [col.lower() for col in bars.columns]
            
            # Return only the most recent 'limit' rows
            return bars.iloc[-limit:].reset_index()
            
        except Exception as e:
            logger.error(f"Error fetching Alpaca data: {str(e)}")
            # Try Yahoo Finance as fallback
            return self._get_yfinance_data(symbol, timeframe, limit)
    
    def _get_yfinance_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get data from Yahoo Finance."""
        # Map timeframe to Yahoo Finance interval
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '1d': '1d'
        }
        interval = interval_map.get(timeframe, '1d')
        
        # Calculate period based on timeframe and limit
        period_days = 1
        if timeframe == '1m':
            period_days = max(7, int(limit / 390) + 1)  # 390 minutes in a trading day
        elif timeframe == '5m':
            period_days = max(60, int(limit / 78) + 1)  # 78 5-minute intervals in a trading day
        elif timeframe == '15m':
            period_days = max(60, int(limit / 26) + 1)  # 26 15-minute intervals in a trading day
        elif timeframe == '1h':
            period_days = max(60, int(limit / 7) + 1)   # ~7 hours in a trading day
        elif timeframe == '1d':
            period_days = max(limit, 7)  # At least a week of daily data
        
        # Get data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        data = ticker.history(interval=interval, period=f"{period_days}d")
        
        # Rename columns to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]
        
        # Reset index to make 'date' a column
        data = data.reset_index()
        
        # Return only the most recent 'limit' rows
        if len(data) > limit:
            return data.iloc[-limit:].reset_index(drop=True)
        return data
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        if self.alpaca_api:
            try:
                clock = self.alpaca_api.get_clock()
                return clock.is_open
            except Exception as e:
                logger.error(f"Error checking market status: {str(e)}")
        
        # Fallback: Check if current time is within market hours (9:30 AM - 4:00 PM ET, weekdays)
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Convert current time to ET
        # This is a simplistic approach, doesn't account for holidays
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        
        return market_open <= now <= market_close
    
    def get_next_market_open(self) -> datetime:
        """Get the next market open time."""
        if self.alpaca_api:
            try:
                clock = self.alpaca_api.get_clock()
                return clock.next_open
            except Exception as e:
                logger.error(f"Error getting next market open: {str(e)}")
        
        # Fallback: Calculate next market open (simplified, doesn't account for holidays)
        now = datetime.now()
        weekday = now.weekday()
        
        # If it's weekend, next open is Monday at 9:30 AM
        if weekday >= 5:  # Saturday or Sunday
            days_to_monday = (7 - weekday + 1) % 7
            next_open = now + timedelta(days=days_to_monday)
            next_open = next_open.replace(hour=9, minute=30, second=0)
        else:
            # If it's before market open today, open is at 9:30 AM
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                next_open = now.replace(hour=9, minute=30, second=0)
            # If it's after market close, next open is tomorrow at 9:30 AM
            elif now.hour >= 16:
                next_open = now + timedelta(days=1)
                next_open = next_open.replace(hour=9, minute=30, second=0)
            else:
                # Market should be open now
                next_open = now
        
        return next_open
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = '1d') -> pd.DataFrame:
        """Get historical data for backtesting."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            # Rename columns to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]
            
            return data.reset_index()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame() 