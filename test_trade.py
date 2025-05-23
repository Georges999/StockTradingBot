#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to debug trading issues.
"""

import logging
from trader import AlpacaTrader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_trade():
    """Test placing a single trade to debug issues."""
    
    print("🧪 Testing Single Trade")
    print("=" * 40)
    
    try:
        # Create trader
        trader = AlpacaTrader()
        
        # Get account info
        buying_power = trader.get_buying_power()
        print(f"Buying Power: ${buying_power:,.2f}")
        
        # Try to place a small trade
        print("\nAttempting to buy 1 share of AAPL...")
        
        result = trader.execute_trade(
            symbol='AAPL',
            signal=1,  # Buy signal
            quantity=1,  # Just 1 share
            strategy='test',
            risk_pct=0.01  # 1% risk
        )
        
        print(f"Trade result: {result}")
        
        if result['status'] == 'order_error':
            print(f"❌ Order failed with error: {result.get('error', 'Unknown error')}")
        elif result['status'] == 'submitted':
            print(f"✅ Order submitted successfully!")
            print(f"   Order ID: {result.get('order_id')}")
            print(f"   Quantity: {result.get('quantity')}")
            print(f"   Price: ${result.get('price', 0):.2f}")
        else:
            print(f"⚠️ Unexpected status: {result['status']}")
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_single_trade() 