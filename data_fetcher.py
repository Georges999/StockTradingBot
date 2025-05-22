#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple data fetcher for retrieving stock market data using yfinance.
"""

import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
import pytz

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Simple class for fetching market data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the DataFetcher."""
        pass
    
    def get_stock_data(self, symbol, period='1mo', interval='1d'):
        """
        Get stock data using yfinance.
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', etc.)
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {interval} data for {symbol} with period {period}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Check if data was retrieved
            if data.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]
            
            # Rename 'adj close' to 'adj_close' if it exists
            if 'adj close' in data.columns:
                data.rename(columns={'adj close': 'adj_close'}, inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol):
        """
        Get the current market price for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Current market price or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            # Get the most recent price
            data = ticker.history(period='1d')
            
            if data.empty:
                logger.warning(f"No price data available for {symbol}")
                return None
                
            return data['Close'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return None
    
    def get_multiple_symbols_data(self, symbols, period='1mo', interval='1d'):
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols (list): List of stock symbols
            period (str): Time period to fetch
            interval (str): Data interval
            
        Returns:
            dict: Dictionary of DataFrames with symbol as key
        """
        result = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, period, interval)
            if not data.empty:
                result[symbol] = data
                
        return result
    
    def is_market_open(self):
        """
        Check if US market is likely open.
        For testing purposes, this function will assume market is open most of the time.
        
        Returns:
            bool: True if market is likely open, False otherwise
        """
        try:
            # TESTING MODE: Always return True for more active trading during development
            # Remove or comment this line in production
            return True
            
            # Current time in Eastern Time (US Market time)
            now = datetime.now(pytz.timezone('US/Eastern'))
            
            # Check if it's a weekday
            if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                logger.debug("Market closed: Weekend")
                return False
            
            # Check if it's during market hours (9:30 AM - 4:00 PM Eastern)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # FOR TESTING: Extended hours (8:00 AM - 6:00 PM Eastern)
            extended_open = now.replace(hour=8, minute=0, second=0, microsecond=0)
            extended_close = now.replace(hour=18, minute=0, second=0, microsecond=0)
            
            # Use extended hours for more trading opportunities
            if extended_open <= now <= extended_close:
                return True
            else:
                logger.debug(f"Market closed: Current time {now.strftime('%H:%M:%S')} is outside trading hours")
                return False
                
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            # In case of error, assume market is open for testing purposes
            return True


# Simple test function
if __name__ == "__main__":
    fetcher = DataFetcher()
    # Test with a single stock
    apple_data = fetcher.get_stock_data("AAPL", period="1mo")
    print(f"Retrieved {len(apple_data)} rows of Apple stock data")
    
    # Get current price
    current_price = fetcher.get_current_price("AAPL")
    print(f"Current Apple stock price: ${current_price:.2f}")
    
    # Check if market is open
    market_open = fetcher.is_market_open()
    print(f"Market is {'open' if market_open else 'closed'}") 