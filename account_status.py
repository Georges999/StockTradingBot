#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Account status diagnostic script to check buying power and positions.
"""

import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_account_status():
    """Check the current account status, buying power, and positions."""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    print("🔍 Account Status Diagnostic")
    print("=" * 50)
    
    # Check if API credentials are available
    if not api_key or not api_secret:
        print("❌ ERROR: Alpaca API credentials not found!")
        print("Please check your .env file and ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
        return
    
    if api_key == 'your_alpaca_api_key_here' or api_secret == 'your_alpaca_secret_key_here':
        print("❌ ERROR: Using example API credentials!")
        print("Please update your .env file with real Alpaca API credentials.")
        return
    
    try:
        # Import Alpaca API
        import alpaca_trade_api as tradeapi
        
        # Create API connection
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        
        print(f"✅ Connected to Alpaca API: {base_url}")
        
        # Get account information
        account = api.get_account()
        
        print(f"\n💰 Account Information:")
        print(f"   Account Status: {account.status}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
        print(f"   Pattern Day Trader: {account.pattern_day_trader}")
        
        # Calculate potential trade sizes
        buying_power = float(account.buying_power)
        
        print(f"\n📊 Trade Size Analysis:")
        risk_percentages = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%
        
        # Example stock prices
        example_stocks = [
            ("SPY", 580.0),
            ("AAPL", 190.0),
            ("MSFT", 420.0),
            ("QQQ", 480.0),
            ("NVDA", 132.0)
        ]
        
        print(f"   Available Buying Power: ${buying_power:,.2f}")
        print()
        
        for risk_pct in risk_percentages:
            trade_value = buying_power * risk_pct
            print(f"   Risk {risk_pct*100:.0f}% (${trade_value:,.2f}) can buy:")
            
            for symbol, price in example_stocks:
                max_shares = int(trade_value / price)
                if max_shares > 0:
                    total_cost = max_shares * price
                    print(f"     {symbol}: {max_shares} shares (${total_cost:,.2f})")
                else:
                    print(f"     {symbol}: 0 shares (price ${price:.2f} > trade value ${trade_value:.2f})")
            print()
        
        # Get current positions
        positions = api.list_positions()
        
        print(f"📈 Current Positions ({len(positions)} total):")
        if positions:
            total_market_value = 0
            for position in positions:
                market_value = float(position.market_value)
                total_market_value += market_value
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc) * 100
                
                print(f"   {position.symbol}: {position.qty} shares @ ${float(position.avg_entry_price):.2f}")
                print(f"     Market Value: ${market_value:,.2f}")
                print(f"     Unrealized P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:+.2f}%)")
                print()
            
            print(f"   Total Position Value: ${total_market_value:,.2f}")
        else:
            print("   No current positions")
        
        # Check market status
        try:
            clock = api.get_clock()
            print(f"\n🕐 Market Status:")
            print(f"   Market Open: {clock.is_open}")
            print(f"   Next Open: {clock.next_open}")
            print(f"   Next Close: {clock.next_close}")
        except Exception as e:
            print(f"   Could not get market status: {e}")
        
        print("\n✅ Account diagnostic complete!")
        
    except ImportError:
        print("❌ ERROR: alpaca-trade-api not installed!")
        print("Run: pip install alpaca-trade-api")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("Check your API credentials and network connection.")

if __name__ == "__main__":
    check_account_status() 