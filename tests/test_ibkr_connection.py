#!/usr/bin/env python3
"""
Test script for IBKR connection and data retrieval.
This script tests the IBKR implementation without requiring actual trading.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.ibkr_config import create_ibkr_config
from data.ibkr_connection import test_connection
from data.ibkr_source import create_ibkr_data_source
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_ibkr_setup():
    """Test IBKR setup and configuration"""
    print("🔧 Testing IBKR Configuration...")
    
    try:
        config = create_ibkr_config()
        print(f"✅ Configuration loaded: {config}")
        print(f"📊 Account type: {config.account_type.value}")
        print(f"🔌 Connection: {config.host}:{config.port}")
        print(f"📈 Market data type: {config.market_data_type.value}")
        print(f"📝 Paper trading: {config.is_paper_trading}")
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return None


async def test_ibkr_connection_basic():
    """Test basic IBKR connection"""
    print("\n🔌 Testing IBKR Connection...")
    
    try:
        config = create_ibkr_config()
        success = await test_connection(config)
        
        if success:
            print("✅ IBKR connection successful!")
            return True
        else:
            print("❌ IBKR connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False


async def test_ibkr_data_source():
    """Test IBKR data source functionality"""
    print("\n📊 Testing IBKR Data Source...")
    
    try:
        data_source = create_ibkr_data_source()
        
        # Test connection
        print("🔌 Connecting to IBKR...")
        if not await data_source.connect():
            print("❌ Failed to connect to IBKR data source")
            return False
        
        print("✅ Connected to IBKR data source")
        
        # Test account access
        accounts = data_source.connection.get_managed_accounts()
        print(f"📋 Managed accounts: {accounts}")
        
        # Test current price
        print("\n💰 Testing current price retrieval...")
        test_symbols = ['SPY', 'AAPL', 'MSFT']
        
        for symbol in test_symbols:
            try:
                price_data = await data_source.get_current_price(symbol)
                if price_data:
                    print(f"✅ {symbol}: ${price_data['price']:.2f} (bid: ${price_data['bid']:.2f}, ask: ${price_data['ask']:.2f})")
                else:
                    print(f"❌ Failed to get price for {symbol}")
            except Exception as e:
                print(f"❌ Error getting price for {symbol}: {e}")
        
        # Test historical data
        print("\n📈 Testing historical data retrieval...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        for timeframe in ['1h', '1d']:
            try:
                print(f"📊 Getting {timeframe} data for SPY from {start_date} to {end_date}...")
                df = await data_source.get_historical_data('SPY', start_date, end_date, timeframe)
                
                if df is not None and not df.empty:
                    print(f"✅ Retrieved {len(df)} {timeframe} bars for SPY")
                    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
                    print(f"💲 Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
                    print("📋 Sample data:")
                    print(df.tail(3).round(2))
                else:
                    print(f"❌ No historical data received for SPY ({timeframe})")
                    
            except Exception as e:
                print(f"❌ Error getting historical data for SPY ({timeframe}): {e}")
        
        # Test supported timeframes
        print(f"\n⏰ Supported timeframes: {data_source.get_supported_timeframes()}")
        
        # Disconnect
        await data_source.disconnect()
        print("🔌 Disconnected from IBKR")
        
        return True
        
    except Exception as e:
        print(f"❌ Data source test error: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 IBKR Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration
    config = await test_ibkr_setup()
    if not config:
        print("❌ Configuration test failed - stopping tests")
        return
    
    # Test 2: Basic connection
    connection_ok = await test_ibkr_connection_basic()
    if not connection_ok:
        print("❌ Connection test failed - stopping tests")
        print("\n💡 Make sure TWS or IB Gateway is running and properly configured:")
        print("   - Enable API connections")
        print("   - Set correct port (7497 for paper, 7496 for live)")
        print("   - Configure trusted IPs if needed")
        return
    
    # Test 3: Data source functionality
    data_source_ok = await test_ibkr_data_source()
    if not data_source_ok:
        print("❌ Data source test failed")
        return
    
    print("\n🎉 All IBKR tests passed!")
    print("✅ IBKR integration is working correctly")
    print("\n📝 Next steps:")
    print("   1. Configure your TWS/IB Gateway for production use")
    print("   2. Set up appropriate market data subscriptions")
    print("   3. Test with your specific trading symbols")
    print("   4. Integrate with your trading strategies")


if __name__ == "__main__":
    # Check if we have the required configuration
    import os
    from dotenv import load_dotenv
    
    # Try to load IBKR configuration
    env_files = ['.env.ibkr.paper', '.env.ibkr.live', '.env']
    env_loaded = False
    
    for env_file in env_files:
        if Path(env_file).exists():
            load_dotenv(env_file)
            print(f"📝 Loaded configuration from {env_file}")
            env_loaded = True
            break
    
    if not env_loaded:
        print("⚠️  No IBKR configuration found. Creating default configuration...")
        os.environ['IBKR_ACCOUNT_TYPE'] = 'paper'
        os.environ['IBKR_PORT'] = '7497'
        os.environ['IBKR_HOST'] = '127.0.0.1'
        os.environ['IBKR_CLIENT_ID'] = '1'
        os.environ['IBKR_MARKET_DATA_TYPE'] = '3'
    
    # Run tests
    asyncio.run(main())