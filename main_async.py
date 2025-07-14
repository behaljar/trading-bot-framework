"""
Async version of main.py that can use IBKRTrader directly
"""
import os
import sys
import asyncio
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import load_config
from strategies.trend_following import SMAStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.mean_reversion import MeanReversion
from strategies.test_strategy import TestStrategy
from execution.ccxt import CCXTTrader
from execution.ibkr import IBKRTrader
from utils.logger import setup_logger


async def run_ibkr_trading_cycle(trader: IBKRTrader, symbol: str, strategy):
    """Run a single trading cycle with IBKR trader"""
    try:
        # Check emergency stop
        if trader.is_emergency_stopped():
            trader.logger.error("Emergency stop active - no trading")
            return
        
        # Check daily loss limit
        daily_pnl = trader.get_daily_pnl()
        daily_loss_limit = 1000  # Configure as needed
        if abs(daily_pnl) > daily_loss_limit:
            trader.logger.error(f"Daily loss limit reached: {daily_pnl}")
            trader.set_emergency_stop(True)
            return
        
        # For now, just log - you'd need to implement data fetching
        trader.logger.info(f"Would process trading cycle for {symbol}")
        
        # Get positions
        positions = await trader.get_positions()
        trader.logger.info(f"Current positions: {len(positions)}")
        
        # Example: Place a small test order (commented out for safety)
        # order = await trader.place_market_order(symbol, 'buy', 1)
        # if order:
        #     trader.logger.info(f"Placed test order: {order.id}")
        
    except Exception as e:
        trader.logger.error(f"Error in trading cycle for {symbol}: {e}")


async def get_ibkr_performance_summary(trader: IBKRTrader):
    """Get performance summary from IBKR trader"""
    try:
        daily_pnl = trader.get_daily_pnl()
        positions = await trader.get_positions()
        
        return {
            'daily_pnl': daily_pnl,
            'open_positions': len(positions),
            'positions': positions,
            'emergency_stop': trader.is_emergency_stopped()
        }
    except Exception as e:
        trader.logger.error(f"Error getting performance summary: {e}")
        return {
            'daily_pnl': 0.0,
            'open_positions': 0,
            'positions': {},
            'emergency_stop': False
        }


async def main_async():
    """Async main trading bot function"""
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logger(config.log_level)
    logger.info("Starting async trading bot...")

    # Validate data source for live trading
    if config.data_source not in ["ccxt", "ibkr"]:
        raise ValueError("Live trading only supported with CCXT or IBKR data source. Set DATA_SOURCE=ccxt or DATA_SOURCE=ibkr")
    
    # Select strategy based on configuration
    if config.strategy_name.lower() in ["sma", "sma_crossover", "trend_following"]:
        strategy = SMAStrategy(config.strategy_params)
    elif config.strategy_name.lower() in ["mean_reversion", "zscore", "z_score"]:
        strategy = MeanReversion(config.strategy_params)
    elif config.strategy_name.lower() in ["breakout", "breakout_strategy"]:
        strategy = BreakoutStrategy(config.strategy_params)
    elif config.strategy_name.lower() in ["test", "test_strategy"]:
        strategy = TestStrategy(config.strategy_params)
        logger.warning("Using TEST strategy - This is for testing only!")
    else:
        # Default to SMA if unknown strategy
        logger.warning(f"Unknown strategy {config.strategy_name}, defaulting to SMA")
        strategy = SMAStrategy(config.strategy_params)
    
    # Initialize trader based on data source
    if config.data_source == "ccxt":
        trader = CCXTTrader(config)
        # Use existing sync methods for CCXT
        logger.info(f"Using strategy: {strategy.get_strategy_name()}")
        
        # Announce trading mode
        mode = "SANDBOX/PAPER" if config.use_sandbox else "LIVE"
        logger.warning(f"Starting in {mode} mode - {'Test' if config.use_sandbox else 'REAL'} money!")
        
        # Main trading loop (sync for CCXT)
        while True:
            try:
                # Process each symbol
                for symbol in config.symbols:
                    trader.run_trading_cycle(symbol, strategy)

                # Get performance summary
                performance = trader.get_performance_summary()
                logger.info(
                    f"Performance: Daily P&L={performance.get('daily_pnl', 0):.2f}, Open Positions={performance.get('open_positions', 0)}")

                # Calculate sleep time until next candle
                timeframe_seconds = trader.data_manager.get_timeframe_seconds(config.timeframe)
                sleep_time = timeframe_seconds - (time.time() % timeframe_seconds)

                if sleep_time > 5:
                    logger.info(f"Waiting {sleep_time:.0f}s until next {config.timeframe} candle...")
                    await asyncio.sleep(sleep_time)
                else:
                    # If too close to next candle, wait for the one after
                    await asyncio.sleep(sleep_time + timeframe_seconds)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry
    
    elif config.data_source == "ibkr":
        # Use IBKRTrader directly (async)
        trader = IBKRTrader(config)
        
        # Initialize trader
        logger.info("Initializing IBKR trader...")
        if not await trader.initialize():
            logger.error("Failed to initialize IBKR trader")
            return
        
        logger.info(f"Using strategy: {strategy.get_strategy_name()}")
        
        # Announce trading mode
        mode = "PAPER" if trader.ibkr_config.is_paper_trading else "LIVE"
        logger.warning(f"Starting IBKR in {mode} mode - {'Test' if trader.ibkr_config.is_paper_trading else 'REAL'} money!")
        
        try:
            # Main trading loop (async for IBKR)
            while True:
                try:
                    # Process each symbol
                    for symbol in config.symbols:
                        await run_ibkr_trading_cycle(trader, symbol, strategy)

                    # Get performance summary
                    performance = await get_ibkr_performance_summary(trader)
                    logger.info(
                        f"Performance: Daily P&L={performance.get('daily_pnl', 0):.2f}, Open Positions={performance.get('open_positions', 0)}")

                    # Calculate sleep time until next candle
                    timeframe_map = {
                        '1m': 60, '5m': 300, '15m': 900, '30m': 1800, 
                        '1h': 3600, '4h': 14400, '1d': 86400
                    }
                    timeframe_seconds = timeframe_map.get(config.timeframe, 3600)
                    sleep_time = timeframe_seconds - (time.time() % timeframe_seconds)

                    if sleep_time > 5:
                        logger.info(f"Waiting {sleep_time:.0f}s until next {config.timeframe} candle...")
                        await asyncio.sleep(sleep_time)
                    else:
                        # If too close to next candle, wait for the one after
                        await asyncio.sleep(sleep_time + timeframe_seconds)

                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(60)  # Wait before retry
        
        finally:
            # Shutdown trader
            logger.info("Shutting down IBKR trader...")
            await trader.shutdown()
    
    else:
        raise ValueError(f"Unsupported data source: {config.data_source}")


def main():
    """Sync entry point that runs the async main"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()