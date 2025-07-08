#!/usr/bin/env python3
"""
Run paper trading with any data source and strategy.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import TradingConfig
from utils.logger import setup_logger
from data.yahoo_finance import YahooFinanceSource
from data.ccxt_source import CCXTSource
from data.csv_source import CSVDataSource
from data.ibkr_sync_wrapper import IBKRSyncWrapper
from strategies.trend_following import SMAStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.mean_reversion import MeanReversion
from strategies.test_strategy import TestStrategy
from execution.paper import PaperTrader, PerformanceTracker


def main():
    parser = argparse.ArgumentParser(description='Run paper trading simulation')
    parser.add_argument('--source', choices=['yahoo', 'ccxt', 'csv', 'ibkr'], 
                       default='yahoo', help='Data source to use')
    parser.add_argument('--strategy', default='sma', 
                       help='Strategy name (sma, zscore, etc.)')
    parser.add_argument('--symbols', nargs='+', 
                       help='Symbols to trade (e.g., AAPL MSFT)')
    parser.add_argument('--interval', default='1h', 
                       help='Bar interval (1m, 5m, 15m, 1h, 1d)')
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (0.001 = 0.1%%)')
    parser.add_argument('--spread', type=float, default=10,
                       help='Spread in basis points')
    parser.add_argument('--slippage', type=float, default=5,
                       help='Slippage in basis points')
    parser.add_argument('--position-size', type=float, default=0.1,
                       help='Position size as percentage of portfolio (0.1 = 10%%)')
    parser.add_argument('--risk-pct', type=float, default=0.02,
                       help='Risk percentage when using stop loss (0.02 = 2%%)')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger('INFO')
    
    try:
        # Load configuration
        config = TradingConfig()
        
        # Override config with command line args
        config.initial_capital = args.initial_capital
        config.paper_commission_rate = args.commission
        config.paper_spread_bps = args.spread
        config.paper_slippage_bps = args.slippage
        
        # Set data source
        config.data_source = args.source
        if args.source == 'ccxt':
            config.use_sandbox = True  # Always use sandbox for paper trading
            
        # Get symbols
        symbols = args.symbols
        if not symbols:
            # Use default symbols based on data source
            if args.source == 'ccxt':
                symbols = ['BTC/USDT', 'ETH/USDT']
            elif args.source == 'ibkr':
                symbols = ['SPY', 'AAPL', 'MSFT']
            else:
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                
        logger.info(f"Starting paper trading with {args.source} data source")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Interval: {args.interval}")
        logger.info(f"Initial Capital: ${args.initial_capital:,.2f}")
        
        # Create data source
        if args.source == 'yahoo':
            data_source = YahooFinanceSource()
        elif args.source == 'ccxt':
            data_source = CCXTSource(
                exchange_name=config.exchange_name,
                api_key=config.api_key,
                api_secret=config.api_secret,
                sandbox=config.use_sandbox
            )
        elif args.source == 'csv':
            data_source = CSVDataSource(config.csv_data_directory)
        elif args.source == 'ibkr':
            from config.ibkr_config import create_ibkr_config
            ibkr_config = create_ibkr_config()
            data_source = IBKRSyncWrapper(ibkr_config)
        else:
            raise ValueError(f"Unknown data source: {args.source}")
        
        # Create strategy
        if args.strategy.lower() in ["sma", "sma_crossover", "trend_following"]:
            strategy = SMAStrategy(config.strategy_params)
        elif args.strategy.lower() in ["mean_reversion", "zscore", "z_score"]:
            strategy = MeanReversion(config.strategy_params)
        elif args.strategy.lower() in ["breakout", "breakout_strategy"]:
            strategy = BreakoutStrategy(config.strategy_params)
        elif args.strategy.lower() in ["test", "test_strategy"]:
            strategy = TestStrategy(config.strategy_params)
        else:
            logger.warning(f"Unknown strategy {args.strategy}, defaulting to SMA")
            strategy = SMAStrategy(config.strategy_params)
        
        # Create paper trader
        paper_trader = PaperTrader(config, data_source, strategy, args.position_size, args.risk_pct)
        
        # Run paper trading
        logger.info("Starting paper trading... Press Ctrl+C to stop")
        paper_trader.run_paper_trading(
            symbols=symbols,
            interval=args.interval
        )
        
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    except Exception as e:
        logger.error(f"Paper trading error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()