#!/usr/bin/env python3
"""
Test script to verify position synchronization
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_config
from execution.ccxt import CCXTTrader
from utils.logger import setup_logger


def main():
    """Test position synchronization"""
    print("=" * 60)
    print("POSITION SYNCHRONIZATION TEST")
    print("=" * 60)
    
    try:
        # Load config
        config = load_config()
        logger = setup_logger(config.log_level)
        
        print(f"Exchange: {config.exchange_name}")
        print(f"Trading Type: {getattr(config, 'trading_type', 'spot')}")
        print(f"Symbols: {config.symbols}")
        print(f"Sandbox: {config.use_sandbox}")
        
        # Initialize trader (this will sync positions)
        print("\n🔄 Initializing trader and syncing positions...")
        trader = CCXTTrader(config)
        
        print("\n📊 Position Sync Results:")
        print("-" * 40)
        
        if trader.positions:
            for symbol, position in trader.positions.items():
                size = position.get('size', 0)
                if abs(size) > 0.0001:
                    side = position.get('side', 'unknown')
                    entry_price = position.get('entry_price', 'N/A')
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    
                    print(f"Symbol: {symbol}")
                    print(f"  Side: {side}")
                    print(f"  Size: {size}")
                    print(f"  Entry Price: {entry_price}")
                    print(f"  Unrealized P&L: {unrealized_pnl}")
                    print(f"  Synchronized: {position.get('synchronized', False)}")
                    print()
        else:
            print("No positions found")
            
        # Test account info
        print("\n💰 Account Information:")
        print("-" * 40)
        account_info = trader.position_sync.get_account_info()
        
        total_balance = account_info.get('total_balance_usdt', 0)
        print(f"Total Balance: {total_balance} USDT")
        
        balances = account_info.get('balances', {})
        if balances:
            print("Non-zero balances:")
            for currency, balance_info in balances.items():
                if balance_info.get('total', 0) > 0.001:
                    print(f"  {currency}: {balance_info}")
        
        print("\n✅ Position synchronization test completed!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()