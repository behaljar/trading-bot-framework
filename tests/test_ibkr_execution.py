#!/usr/bin/env python3
"""
Test script for IBKR execution engine
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from execution.ibkr.ibkr_trader import IBKRTrader
from config.ibkr_config import create_ibkr_config


class TestConfig:
    """Simple test configuration"""
    def __init__(self):
        self.initial_capital = 10000.0
        self.max_position_size = 0.1


async def test_ibkr_execution():
    """Test IBKR execution functionality"""
    print("🧪 IBKR Execution Engine Test")
    print("=" * 50)
    
    try:
        # Load configuration
        print("📝 Loading IBKR configuration...")
        
        # Check if .env is configured for IBKR
        if not os.path.exists('.env'):
            print("❌ .env file not found")
            print("💡 Make sure to configure .env for IBKR testing")
            return False
        
        # Create configurations
        test_config = TestConfig()
        ibkr_config = create_ibkr_config()
        
        print(f"🔧 IBKR Config: {ibkr_config.host}:{ibkr_config.port}")
        print(f"📊 Account Type: {ibkr_config.account_type.value}")
        print(f"💰 Initial Capital: ${test_config.initial_capital:,.2f}")
        
        # Initialize trader
        print("\n🚀 Initializing IBKR trader...")
        trader = IBKRTrader(config=test_config, ibkr_config=ibkr_config)
        
        # Connect and initialize
        if not await trader.initialize():
            print("❌ Failed to initialize IBKR trader")
            return False
        
        print("✅ IBKR trader initialized successfully")
        
        # Test account information
        print("\n💰 Testing account information...")
        balance_info = await trader.get_account_balance()
        if balance_info:
            print("✅ Account balance retrieved:")
            for key, value in balance_info.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: ${value:,.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ Failed to get account balance")
        
        # Test position sync
        print("\n📊 Testing position synchronization...")
        positions = await trader.get_positions()
        print(f"✅ Retrieved {len(positions)} positions")
        
        if positions:
            print("📈 Current positions:")
            for symbol, pos in positions.items():
                size = pos.get('size', 0)
                market_value = pos.get('market_value', 0)
                print(f"   {symbol}: {size} shares (${market_value:,.2f})")
        else:
            print("   No positions found")
        
        # Test order placement (paper trading only)
        if ibkr_config.account_type.value == 'paper':
            print("\n📋 Testing order placement (paper trading)...")
            
            # Test small market order
            test_symbol = 'AAPL'
            test_size = 1  # 1 share for testing
            
            print(f"🛒 Placing test market order: BUY {test_size} {test_symbol}")
            
            order = await trader.place_market_order(
                symbol=test_symbol,
                side='buy',
                size=test_size,
                stop_loss=None,  # No stop loss for test
                take_profit=None  # No take profit for test
            )
            
            if order:
                print(f"✅ Order placed successfully:")
                print(f"   Order ID: {order.id}")
                print(f"   IBKR Order ID: {order.ibkr_order_id}")
                print(f"   Status: {order.status.value}")
                print(f"   Symbol: {order.symbol}")
                print(f"   Side: {order.side}")
                print(f"   Size: {order.size}")
                
                # Wait a moment for execution
                print("⏳ Waiting for order execution...")
                await asyncio.sleep(5)
                
                # Check order status
                if order.id in trader.orders:
                    updated_order = trader.orders[order.id]
                    print(f"📊 Updated order status: {updated_order.status.value}")
                    if updated_order.filled_size > 0:
                        print(f"✅ Filled: {updated_order.filled_size} @ ${updated_order.average_price:.2f}")
                
                # Cancel order if still pending
                if order.status.value in ['pending', 'placed']:
                    print("🚫 Cancelling test order...")
                    cancel_success = await trader.cancel_order(order.id)
                    if cancel_success:
                        print("✅ Order cancelled successfully")
                    else:
                        print("❌ Failed to cancel order")
            else:
                print("❌ Failed to place order")
        else:
            print("\n⚠️  Skipping order placement test (live account)")
            print("💡 Switch to paper trading to test order placement")
        
        # Test emergency stop
        print("\n🛑 Testing emergency stop...")
        trader.set_emergency_stop(True)
        print(f"✅ Emergency stop status: {trader.is_emergency_stopped()}")
        
        # Try to place order with emergency stop
        if ibkr_config.account_type.value == 'paper':
            test_order = await trader.place_market_order('SPY', 'buy', 1)
            if test_order is None:
                print("✅ Emergency stop prevented order placement")
            else:
                print("❌ Emergency stop did not prevent order placement")
        
        # Reset emergency stop
        trader.set_emergency_stop(False)
        print(f"✅ Emergency stop reset: {trader.is_emergency_stopped()}")
        
        # Test state persistence
        print("\n💾 Testing state persistence...")
        trader._save_state()
        print("✅ State saved successfully")
        
        # Shutdown
        print("\n🔌 Shutting down trader...")
        await trader.shutdown()
        print("✅ Trader shutdown completed")
        
        print("\n🎉 All IBKR execution tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("🚀 IBKR Execution Engine Test Suite")
    print("==================================")
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    # Check if TWS/Gateway is likely running
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 7497))  # Paper trading port
            if result == 0:
                print("✅ TWS/Gateway appears to be running on port 7497 (paper)")
            else:
                print("⚠️  TWS/Gateway not detected on port 7497")
                print("💡 Make sure TWS or IB Gateway is running")
    except Exception as e:
        print(f"⚠️  Could not check TWS/Gateway status: {e}")
    
    # Run tests
    success = await test_ibkr_execution()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n🚀 IBKR execution engine is ready for use!")
        print("\n💡 Next steps:")
        print("   1. Test with your preferred symbols")
        print("   2. Configure risk management parameters")
        print("   3. Integrate with your trading strategies")
    else:
        print("\n❌ Some tests failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure TWS/IB Gateway is running")
        print("   2. Check API settings are enabled")
        print("   3. Verify .env configuration")
        print("   4. Check network connectivity")


if __name__ == "__main__":
    asyncio.run(main())