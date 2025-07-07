"""
Main entry point for trading bot
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import load_config
from strategies.trend_following import SMAStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.mean_reversion import MeanReversion
from strategies.test_strategy import TestStrategy
from execution.ccxt import CCXTTrader
from utils.logger import setup_logger


def main() -> None:
    """Main trading bot function"""
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logger(config.log_level)
    logger.info("Starting trading bot...")

    # Only CCXT is supported for live trading
    if config.data_source != "ccxt":
        raise ValueError("Live trading only supported with CCXT data source. Set DATA_SOURCE=ccxt")
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
    # Initialize trader - works for both sandbox and live
    trader = CCXTTrader(config)

    logger.info(f"Using strategy: {strategy.get_strategy_name()}")

    # Announce trading mode
    mode = "SANDBOX/PAPER" if config.use_sandbox else "LIVE"
    logger.warning(f"Starting in {mode} mode - {'Test' if config.use_sandbox else 'REAL'} money!")

    # Main trading loop
    import time

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
                time.sleep(sleep_time)
            else:
                # If too close to next candle, wait for the one after
                time.sleep(sleep_time + timeframe_seconds)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            # Final performance summary
            performance = trader.get_performance_summary()
            logger.info("=== FINAL PERFORMANCE SUMMARY ===")
            for key, value in performance.items():
                logger.info(f"{key}: {value}")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(60)  # Wait before retry


if __name__ == "__main__":
    main()
