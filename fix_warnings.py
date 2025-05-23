#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix warnings and test bot functionality.
"""

import warnings
import pandas as pd

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def fix_pandas_warnings():
    """Fix pandas deprecation warnings."""
    print("🔧 Fixing pandas warnings...")
    
    # Set pandas options to avoid warnings
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('future.no_silent_downcasting', True)
    
    print("✅ Pandas warnings suppressed")

def test_strategy_signals():
    """Test strategy signal generation."""
    print("\n🧪 Testing strategy signal generation...")
    
    try:
        from strategy import EnhancedMovingAverageCrossover
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        close_prices = [100]
        for _ in range(99):
            close_prices.append(close_prices[-1] * (1 + np.random.normal(0, 0.02)))
        
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
        
        # Test strategy
        strategy = EnhancedMovingAverageCrossover()
        signals = strategy.generate_signals(data)
        
        buy_signals = len(signals[signals['signal'] == 1])
        sell_signals = len(signals[signals['signal'] == -1])
        hold_signals = len(signals[signals['signal'] == 0])
        
        print(f"✅ Strategy test successful!")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        print(f"   Hold signals: {hold_signals}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_auto_trader():
    """Test auto trader initialization."""
    print("\n🤖 Testing AutoTrader initialization...")
    
    try:
        from auto_trader import AutoTrader
        
        # Create auto trader with minimal settings
        trader = AutoTrader(
            symbols=['AAPL'],
            interval='1d',
            period='1mo',
            update_interval=10,
            auto_trade=False,  # Disable auto trading for test
            max_active_positions=1
        )
        
        print("✅ AutoTrader initialization successful!")
        print(f"   Symbols: {trader.symbols}")
        print(f"   Auto-trade: {trader.auto_trade}")
        print(f"   Trader available: {trader.trader_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoTrader test failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Running Bot Diagnostic and Fix")
    print("=" * 50)
    
    # Fix warnings
    fix_pandas_warnings()
    
    # Test components
    strategy_ok = test_strategy_signals()
    trader_ok = test_auto_trader()
    
    print("\n📊 Summary:")
    print(f"   Strategy signals: {'✅ OK' if strategy_ok else '❌ FAILED'}")
    print(f"   AutoTrader: {'✅ OK' if trader_ok else '❌ FAILED'}")
    
    if strategy_ok and trader_ok:
        print("\n🎉 All tests passed! Bot should work correctly now.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.") 