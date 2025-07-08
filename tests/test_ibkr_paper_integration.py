#!/usr/bin/env python3
"""
Test IBKR integration with paper trading system.
This test verifies that IBKR data source works with the paper trader.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from data.ibkr_sync_wrapper import create_ibkr_sync_source
from config.ibkr_config import create_ibkr_config
from strategies.test_strategy import TestStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_ibkr_sync_wrapper():
    """Test the IBKR synchronous wrapper"""
    print("🔌 Testing IBKR Synchronous Wrapper...")
    
    try:
        # Create IBKR data source
        ibkr_source = create_ibkr_sync_source()
        
        # Test connection
        print("🔗 Testing connection...")
        if not ibkr_source.connect():
            print("❌ Failed to connect to IBKR")
            return False
        
        print("✅ Connected to IBKR successfully")
        
        # Test current price
        print("💰 Testing current price...")
        price = ibkr_source.get_current_price('SPY')
        if price:
            print(f"✅ SPY current price: ${price['price']:.2f}")
        else:
            print("❌ Failed to get current price")
            return False
        
        # Test historical data
        print("📊 Testing historical data...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        df = ibkr_source.get_historical_data('SPY', start_date, end_date, '1h')
        if df is not None and not df.empty:
            print(f"✅ Retrieved {len(df)} bars for SPY")
            print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
            print("📋 Sample data:")
            print(df.tail(3).round(2))
        else:
            print("❌ Failed to get historical data")
            return False
        
        # Test with strategy
        print("🎯 Testing with strategy...")
        strategy = TestStrategy()
        signals = strategy.generate_signals(df)
        print(f"✅ Generated {len(signals)} signals")
        print(f"📈 Signal values: {signals.unique()}")
        
        # Disconnect
        ibkr_source.disconnect()
        print("🔌 Disconnected from IBKR")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_paper_trading_compatibility():
    """Test compatibility with paper trading framework"""
    print("\n📄 Testing Paper Trading Compatibility...")
    
    try:
        # Import paper trading components
        from execution.paper.paper_trader import PaperTrader
        from config.settings import TradingConfig
        from strategies.test_strategy import TestStrategy
        
        # Create configuration
        config = TradingConfig()
        config.data_source = 'ibkr'
        config.initial_capital = 10000.0
        config.timeframe = '1h'
        
        # Create IBKR data source
        ibkr_source = create_ibkr_sync_source()
        
        # Create strategy
        strategy = TestStrategy()
        
        # Create paper trader
        paper_trader = PaperTrader(
            config=config,
            data_source=ibkr_source,
            strategy=strategy,
            position_size_pct=0.1,
            risk_pct=0.02
        )
        
        print("✅ Paper trader created successfully with IBKR source")
        print(f"📊 Data source: {ibkr_source.get_name()}")
        print(f"🎯 Strategy: {strategy.get_strategy_name()}")
        print(f"💰 Initial capital: ${config.initial_capital:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Paper trading compatibility test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 IBKR Paper Trading Integration Test")
    print("=" * 50)
    
    # Load IBKR configuration
    env_files = ['.env.ibkr.paper', '.env']
    env_loaded = False
    
    for env_file in env_files:
        if Path(env_file).exists():
            load_dotenv(env_file)
            print(f"📝 Loaded configuration from {env_file}")
            env_loaded = True
            break
    
    if not env_loaded:
        print("⚠️  No IBKR configuration found. Using defaults...")
        os.environ['IBKR_ACCOUNT_TYPE'] = 'paper'
        os.environ['IBKR_PORT'] = '7497'
        os.environ['IBKR_HOST'] = '127.0.0.1'
        os.environ['IBKR_CLIENT_ID'] = '1'
        os.environ['IBKR_MARKET_DATA_TYPE'] = '3'
    
    # Test 1: IBKR sync wrapper
    wrapper_ok = test_ibkr_sync_wrapper()
    if not wrapper_ok:
        print("❌ IBKR sync wrapper test failed")
        print("\n💡 Make sure TWS or IB Gateway is running and properly configured:")
        print("   - Enable API connections")
        print("   - Set correct port (7497 for paper)")
        print("   - Configure trusted IPs if needed")
        return
    
    # Test 2: Paper trading compatibility
    paper_ok = test_paper_trading_compatibility()
    if not paper_ok:
        print("❌ Paper trading compatibility test failed")
        return
    
    print("\n🎉 All IBKR paper trading integration tests passed!")
    print("✅ IBKR is ready for paper trading")
    print("\n🚀 You can now run:")
    print("   python scripts/run_paper_trading.py --source ibkr --symbols SPY --timeframe 1h --strategy test")


if __name__ == "__main__":
    main()