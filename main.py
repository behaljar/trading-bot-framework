"""
Main entry point for trading bot
"""
import os
import sys
import signal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import load_config
from strategies.trend_following import SMAStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.mean_reversion import MeanReversion
from strategies.test_strategy import TestStrategy
from execution.ccxt import CCXTTrader
from execution.ibkr import IBKRTrader
from utils.logger import setup_logger


def main() -> None:
    """Main trading bot function"""
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logger(config.log_level)
    logger.info("Starting trading bot...")
    
    # Global trader reference for signal handler
    trader = None
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        if trader:
            try:
                if config.data_source == "ibkr":
                    trader.shutdown_sync()
                elif hasattr(trader, 'shutdown'):
                    trader.shutdown()
            except Exception as e:
                logger.error(f"Error during signal shutdown: {e}")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
    elif config.data_source == "ibkr":
        trader = IBKRTrader(config)
        # Initialize the IBKR trader
        logger.info("Initializing IBKR trader...")
        if not trader.initialize_sync():
            logger.error("Failed to initialize IBKR trader")
            return
    else:
        raise ValueError(f"Unsupported data source: {config.data_source}")

    logger.info(f"Using strategy: {strategy.get_strategy_name()}")

    # Announce trading mode
    if config.data_source == "ccxt":
        mode = "SANDBOX/PAPER" if config.use_sandbox else "LIVE"
        logger.warning(f"Starting in {mode} mode - {'Test' if config.use_sandbox else 'REAL'} money!")
    elif config.data_source == "ibkr":
        # For IBKR, mode is determined by IBKR configuration
        mode = "PAPER" if trader.ibkr_config.is_paper_trading else "LIVE"
        logger.warning(f"Starting IBKR in {mode} mode - {'Test' if trader.ibkr_config.is_paper_trading else 'REAL'} money!")

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
            if hasattr(trader, 'data_manager'):
                timeframe_seconds = trader.data_manager.get_timeframe_seconds(config.timeframe)
            else:
                # Default timeframe mappings for IBKR
                timeframe_map = {
                    '1m': 60, '5m': 300, '15m': 900, '30m': 1800, 
                    '1h': 3600, '4h': 14400, '1d': 86400
                }
                timeframe_seconds = timeframe_map.get(config.timeframe, 3600)  # Default to 1 hour
            
            sleep_time = timeframe_seconds - (time.time() % timeframe_seconds)

            if sleep_time > 5:
                logger.info(f"Waiting {sleep_time:.0f}s until next {config.timeframe} candle...")
                time.sleep(sleep_time)
            else:
                # If too close to next candle, wait for the one after
                time.sleep(sleep_time + timeframe_seconds)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            # Shutdown trader based on type
            if config.data_source == "ibkr":
                trader.shutdown_sync()
            elif hasattr(trader, 'shutdown'):
                trader.shutdown()
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
