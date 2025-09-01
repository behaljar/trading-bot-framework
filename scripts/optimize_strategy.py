#!/usr/bin/env python3
"""
Strategy Optimization Script

Simple wrapper for the grid search optimizer.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import json
import os
from datetime import datetime

# Add framework to path
script_dir = Path(__file__).parent
framework_dir = script_dir.parent
sys.path.append(str(framework_dir))

from framework.optimization.grid_search import GridSearchOptimizer
from framework.strategies.sma_strategy import SMAStrategy
from framework.strategies.fvg_strategy import FVGStrategy
from framework.strategies.breakout_strategy import BreakoutStrategy
from framework.risk.fixed_position_size_manager import FixedPositionSizeManager
from framework.risk.fixed_risk_manager import FixedRiskManager
from framework.utils.logger import setup_logger


def get_default_param_config(strategy: str) -> dict:
    """Get default parameter configuration for each strategy."""
    configs = {
        'sma': {
            'short_window': {'type': 'int', 'min': 5, 'max': 50, 'step': 5},
            'long_window': {'type': 'int', 'min': 20, 'max': 200, 'step': 20},
            'stop_loss_pct': {'type': 'float', 'min': 0.01, 'max': 0.05, 'step': 0.01},
            'take_profit_pct': {'type': 'float', 'min': 0.02, 'max': 0.10, 'step': 0.02}
        },
        'fvg': {
            'h1_lookback_candles': {'type': 'int', 'min': 12, 'max': 48, 'step': 12},
            'risk_reward_ratio': {'type': 'float', 'min': 1.5, 'max': 4.0, 'step': 0.5},
            'max_hold_hours': {'type': 'int', 'min': 2, 'max': 8, 'step': 2},
            'position_size': {'type': 'float', 'min': 0.02, 'max': 0.10, 'step': 0.02}
        },
        'breakout': {
            'entry_lookback': {'type': 'int', 'min': 15, 'max': 60, 'step': 15},
            'exit_lookback': {'type': 'int', 'min': 5, 'max': 30, 'step': 5},
            'atr_multiplier': {'type': 'float', 'min': 1.0, 'max': 3.5, 'step': 0.5},
            'medium_trend_threshold': {'type': 'float', 'min': 0.02, 'max': 0.10, 'step': 0.02},
            'relative_volume_threshold': {'type': 'float', 'min': 1.2, 'max': 3.0, 'step': 0.4},
            'cooldown_periods': {'type': 'int', 'min': 2, 'max': 20, 'step': 6}
        }
    }
    return configs.get(strategy, {})


def load_data(file_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load and prepare data for optimization."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    
    # Load CSV
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Filter by date range
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    
    # Strategy selection
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['sma', 'fvg', 'breakout'],
                       help='Strategy to optimize')
    
    # Data parameters
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD)')
    
    # Optimization parameters
    parser.add_argument('--metric', type=str, default='win_rate',
                       choices=['return_pct', 'sharpe_ratio', 'win_rate', 
                               'max_drawdown', 'profit_factor', 'calmar_ratio'],
                       help='Metric to optimize and visualize')
    parser.add_argument('--minimize', action='store_true',
                       help='Minimize the metric instead of maximizing')
    parser.add_argument('--param-config', type=str,
                       help='Custom parameter configuration as JSON string or file path')
    
    # Risk management
    parser.add_argument('--risk-manager', type=str,
                       choices=['fixed_position', 'fixed_risk'],
                       help='Risk manager type')
    
    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate')
    parser.add_argument('--margin', type=float, default=0.01,
                       help='Margin requirement (0.01 = 100x leverage)')
    
    # Performance
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 = all cores)')
    
    # Output
    parser.add_argument('--output', type=str,
                       help='Output base path for results (default: auto-generated)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("DEBUG" if args.debug else "INFO")
    
    # Load data
    data = load_data(args.data_file, args.start, args.end)
    
    # Get parameter configuration
    if args.param_config:
        # Try to load from file or parse as JSON
        if os.path.exists(args.param_config):
            with open(args.param_config, 'r') as f:
                param_config = json.load(f)
        else:
            try:
                param_config = json.loads(args.param_config)
            except json.JSONDecodeError:
                # Try as environment variable
                env_value = os.getenv(args.param_config)
                if env_value:
                    param_config = json.loads(env_value)
                else:
                    raise ValueError(f"Invalid parameter configuration: {args.param_config}")
    else:
        param_config = get_default_param_config(args.strategy)
    
    print(f"\nParameter configuration: {json.dumps(param_config, indent=2)}")
    
    # Get strategy class
    strategy_map = {
        'sma': SMAStrategy,
        'fvg': FVGStrategy,
        'breakout': BreakoutStrategy
    }
    strategy_class = strategy_map[args.strategy]
    
    # Create risk manager (default to fixed_risk if not specified)
    if args.risk_manager == 'fixed_position':
        risk_manager = FixedPositionSizeManager(position_size=0.05)
    else:
        # Default to fixed_risk (even if None or 'fixed_risk' specified)
        risk_manager = FixedRiskManager(risk_percent=0.01)
    
    # Create optimizer
    optimizer = GridSearchOptimizer(
        strategy_class=strategy_class,
        parameter_config=param_config,
        data=data,
        initial_capital=args.initial_capital,
        commission=args.commission,
        margin=args.margin,
        risk_manager=risk_manager,
        n_jobs=args.n_jobs,
        debug=args.debug
    )
    
    print(f"\nGenerated {len(optimizer.combinations)} parameter combinations")
    
    # Run optimization
    results_df = optimizer.optimize(metric=args.metric, maximize=not args.minimize)
    
    # Generate output path with separate directory for each run
    if args.output:
        output_base = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file_name = Path(args.data_file).stem
        parts = data_file_name.split('_')
        symbol = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"
        
        # Create directory for this optimization run
        run_dir = f"output/optimizations/{args.strategy}_{symbol}_{timestamp}"
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        output_base = f"{run_dir}/results"
    
    # Save results
    optimizer.save_results(results_df, output_base)
    
    # Create visualizations
    optimizer.create_visualization(results_df, output_base, args.metric)
    
    # Print top results
    print("\n" + "=" * 80)
    print("TOP 10 RESULTS")
    print("=" * 80)
    
    top_10 = results_df.head(10)
    param_cols = [col for col in results_df.columns if col not in 
                  ['return_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                   'num_trades', 'exposure_time', 'profit_factor', 'avg_trade',
                   'best_trade', 'worst_trade', 'calmar_ratio', 'sortino_ratio', 'error']]
    
    for idx, (i, row) in enumerate(top_10.iterrows(), 1):
        print(f"\n#{idx}")
        print("Parameters:")
        for col in param_cols:
            print(f"  {col}: {row[col]}")
        print("Metrics:")
        print(f"  Return: {row['return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {row['win_rate']:.2f}%")
        print(f"  Max Drawdown: {row['max_drawdown']:.2f}%")
        print(f"  Profit Factor: {row['profit_factor']:.2f}")
        print(f"  Num Trades: {row['num_trades']}")


if __name__ == '__main__':
    main()