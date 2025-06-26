"""
Main entry point for trading bot
"""
import sys
import os
from typing import Dict, Any
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import load_config, TradingConfig
from data import YahooFinanceSource, CCXTSource, CSVDataSource
from strategies.trend_following import SMAStrategy
from strategies.mean_reversion import RSIStrategy
from strategies.breakout_strategy import BreakoutStrategy
from risk.risk_manager import RiskManager
from execution.paper_trader import PaperTrader
from utils.logger import setup_logger
import pandas as pd
import logging

def main() -> None:
    """Main trading bot function"""
    # Load configuration
    config = load_config()

    # Setup logging
    logger = setup_logger(config.log_level)
    logger.info("Starting trading bot...")

    # Initialize components
    if config.data_source == "yahoo":
        data_source = YahooFinanceSource()
    elif config.data_source == "ccxt":
        data_source = CCXTSource(
            exchange_name=config.exchange_name,
            api_key=config.api_key,
            api_secret=config.api_secret,
            sandbox=config.use_sandbox
        )
    elif config.data_source == "csv":
        data_source = CSVDataSource(data_directory=config.csv_data_directory)
    else:
        raise ValueError(f"Unknown data source: {config.data_source}")
    # Select strategy based on configuration
    if config.strategy_name.lower() in ["sma", "sma_crossover", "trend_following"]:
        strategy = SMAStrategy(config.strategy_params)
    elif config.strategy_name.lower() in ["rsi", "mean_reversion"]:
        strategy = RSIStrategy(config.strategy_params)
    elif config.strategy_name.lower() in ["breakout", "breakout_strategy"]:
        strategy = BreakoutStrategy(config.strategy_params)
    else:
        # Default to SMA if unknown strategy
        logger.warning(f"Unknown strategy {config.strategy_name}, defaulting to SMA")
        strategy = SMAStrategy(config.strategy_params)
    risk_manager = RiskManager(config)
    paper_trader = PaperTrader(config)

    logger.info(f"Using strategy: {strategy.get_strategy_name()}")

    # Main trading loop
    for symbol in config.symbols:
        try:
            logger.info(f"Processing symbol: {symbol}")

            # Get historical data
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

            data = data_source.get_historical_data(symbol, start_date, end_date)

            if data.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Generate signals
            signals = strategy.generate_signals(data)
            latest_signal = signals.iloc[-1]

            if latest_signal != 0:  # Some signal
                current_price = data['Close'].iloc[-1]
                position_size = strategy.get_position_size(symbol, latest_signal, config.initial_capital)

                # Apply risk management
                adjusted_size = risk_manager.check_position_size(symbol, position_size, current_price)

                # Execute trade
                success = paper_trader.place_order(symbol, adjusted_size)

                if success:
                    logger.info(f"Successfully executed trade: {symbol}, size: {adjusted_size}")
                else:
                    logger.warning(f"Failed to execute trade for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    # Print performance summary
    performance = paper_trader.get_performance_summary()
    logger.info("=== PERFORMANCE SUMMARY ===")
    for key, value in performance.items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()