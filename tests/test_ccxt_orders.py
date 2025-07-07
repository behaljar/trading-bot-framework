#!/usr/bin/env python3
"""
Quick test script for order handling
Run this to test the order execution system with the test strategy
"""
import os
import sys
import time
import signal
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_config
from strategies.test_strategy import TestStrategy
from execution.ccxt import CCXTTrader
from utils.logger import setup_logger


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nShutting down gracefully...')
    sys.exit(0)


def main():
    """Main test function"""
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("CCXT ORDER HANDLING TEST")
    print("=" * 60)
    print("This script tests basic order execution with a simple strategy")
    print("that alternates between buying and selling every minute.")
    print("\nMAKE SURE YOU'RE USING SANDBOX/TESTNET MODE!")
    print("=" * 60)
    
    # Load config
    try:
        config = load_config()
        
        # Verify sandbox mode
        if not config.use_sandbox:
            print("\n⚠️  WARNING: NOT IN SANDBOX MODE!")
            response = input("Are you sure you want to use REAL money? (type 'YES' to continue): ")
            if response != 'YES':
                print("Aborting for safety. Set USE_SANDBOX=true in .env")
                return
                
        # Setup logging
        logger = setup_logger(config.log_level)
        
        print(f"\n✅ Configuration loaded:")
        print(f"   Exchange: {config.exchange_name}")
        print(f"   Sandbox: {config.use_sandbox}")
        print(f"   Symbols: {config.symbols}")
        print(f"   Timeframe: {config.timeframe}")
        print(f"   Strategy: {config.strategy_name}")
        
        # Initialize strategy
        strategy = TestStrategy(config.strategy_params)
        
        # Initialize trader
        print(f"\n🔄 Initializing trader...")
        trader = CCXTTrader(config)
        
        print(f"✅ Trader initialized successfully")
        
        # Get initial account info
        performance = trader.get_performance_summary()
        print(f"\n📊 Initial State:")
        print(f"   Balance: {performance.get('current_balance', 'Unknown')} USDT")
        print(f"   Open Positions: {performance.get('open_positions', 0)}")
        print(f"   Mode: {performance.get('mode', 'Unknown')}")
        
        print(f"\n🚀 Starting test trading loop...")
        print(f"   Will run for max 10 cycles or until manually stopped")
        print(f"   Press Ctrl+C to stop\n")
        
        cycle_count = 0
        max_cycles = 10
        
        while cycle_count < max_cycles:
            try:
                cycle_count += 1
                print(f"\n--- Cycle {cycle_count}/{max_cycles} ---")
                
                # Process each symbol
                for symbol in config.symbols:
                    print(f"Processing {symbol}...")
                    trader.run_trading_cycle(symbol, strategy)
                    
                # Get updated performance
                performance = trader.get_performance_summary()
                print(f"Daily P&L: {performance.get('daily_pnl', 0):.2f} USDT")
                print(f"Open Positions: {performance.get('open_positions', 0)}")
                
                # Calculate sleep time until next candle
                timeframe_seconds = trader.data_manager.get_timeframe_seconds(config.timeframe)
                print(f"Waiting {timeframe_seconds}s for next {config.timeframe} candle...")
                
                # Sleep until next candle (but allow interruption)
                for i in range(timeframe_seconds):
                    time.sleep(1)
                    if i % 10 == 0 and i > 0:
                        print(f"  {timeframe_seconds - i}s remaining...")
                        
            except KeyboardInterrupt:
                print("\n⏹️  Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in test cycle: {e}", exc_info=True)
                time.sleep(5)
                
        # Final summary
        print(f"\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        
        performance = trader.get_performance_summary()
        print(f"Final Performance:")
        for key, value in performance.items():
            print(f"  {key}: {value}")
            
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Check logs in logs/ directory for detailed information")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()