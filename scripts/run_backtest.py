"""
Universal backtest runner - completely generic
"""
import sys
import os
import argparse
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.backtesting.backtest_engine import BacktestEngine
from framework.utils.logger import setup_logger


def get_strategy(strategy_name: str, params: dict = None):
    """Get strategy instance by name"""
    if params is None:
        params = {}
    
    # Import strategies
    from framework.strategies.trend_following import SMAStrategy
    from framework.strategies.mean_reversion import MeanReversion
    from framework.strategies.breakout_strategy import BreakoutStrategy
    
    strategy_map = {
        'sma': SMAStrategy,
        'trend_following': SMAStrategy,
        'mean_reversion': MeanReversion,
        'breakout': BreakoutStrategy
    }
    
    strategy_class = strategy_map.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_class(params)


def load_data(file_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load CSV data from file path"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load CSV file directly
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Filter by date range
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    return df


def main():
    """Main backtest runner"""
    parser = argparse.ArgumentParser(description='Run backtests on CSV data')
    parser.add_argument('--strategy', required=True, help='Strategy name (sma, breakout, mean_reversion)')
    parser.add_argument('--data-file', required=True, help='Path to CSV data file')
    parser.add_argument('--symbol', default='UNKNOWN', help='Symbol name for results')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("DEBUG" if args.debug else "INFO")
    
    # Parse strategy parameters from environment if available
    strategy_params = {}
    if 'STRATEGY_PARAMS' in os.environ:
        import json
        try:
            strategy_params = json.loads(os.environ['STRATEGY_PARAMS'])
        except json.JSONDecodeError:
            logger.warning("Invalid STRATEGY_PARAMS environment variable")
    
    try:
        # Load data
        logger.info(f"Loading data from: {args.data_file}")
        data = load_data(args.data_file, args.start, args.end)
        logger.info(f"Loaded {len(data)} data points")
        
        # Get strategy
        strategy = get_strategy(args.strategy, strategy_params)
        logger.info(f"Using strategy: {strategy.get_strategy_name()}")
        
        # Initialize backtest engine
        engine = BacktestEngine()
        
        # Run backtest
        results = engine.run_backtest(
            strategy=strategy,
            data=data,
            symbol=args.symbol
        )
        
        logger.info("Backtest completed successfully")
        print(results.summary())
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()