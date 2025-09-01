#!/usr/bin/env python3
"""
Strategy Optimization Script

Run grid search optimization on trading strategies to find optimal parameters.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import json
import os

# Add framework to path
script_dir = Path(__file__).parent
framework_dir = script_dir.parent
sys.path.append(str(framework_dir))

from framework.optimization.simple_grid_search import SimpleGridSearchOptimizer
from framework.optimization.parallel_grid_search import ParallelGridSearchOptimizer
from framework.optimization.parameter_space import ParameterSpace
from framework.strategies.sma_strategy import SMAStrategy
from framework.strategies.fvg_strategy import FVGStrategy
from framework.risk.fixed_position_size_manager import FixedPositionSizeManager
from framework.risk.fixed_risk_manager import FixedRiskManager
from framework.utils.logger import setup_logger


def parse_custom_parameter_space(param_space_arg: str) -> ParameterSpace:
    """
    Parse custom parameter space from JSON string or environment variable.
    
    Args:
        param_space_arg: JSON string or environment variable name
        
    Returns:
        ParameterSpace object
        
    Example JSON format:
    {
        "short_window": {"type": "int", "min": 5, "max": 50, "step": 5},
        "long_window": {"type": "int", "min": 20, "max": 200, "step": 10},
        "stop_loss_pct": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01},
        "strategy_type": {"type": "choice", "choices": ["aggressive", "conservative"]}
    }
    """
    # Try to parse as JSON first
    try:
        param_config = json.loads(param_space_arg)
    except json.JSONDecodeError:
        # Try as environment variable
        env_value = os.getenv(param_space_arg)
        if env_value is None:
            raise ValueError(f"Invalid JSON and environment variable '{param_space_arg}' not found")
        try:
            param_config = json.loads(env_value)
        except json.JSONDecodeError:
            raise ValueError(f"Environment variable '{param_space_arg}' does not contain valid JSON")
    
    # Create parameter space from config
    space = ParameterSpace()
    
    for param_name, config in param_config.items():
        param_type = config.get('type', 'float')
        
        if param_type == 'choice':
            choices = config.get('choices')
            if not choices:
                raise ValueError(f"Parameter '{param_name}' of type 'choice' must have 'choices' defined")
            space.add_parameter(param_name, param_type='choice', choices=choices)
            
        elif param_type in ['int', 'float']:
            min_val = config.get('min')
            max_val = config.get('max')
            step = config.get('step')
            
            if min_val is None or max_val is None:
                raise ValueError(f"Parameter '{param_name}' must have 'min' and 'max' values")
            
            space.add_parameter(
                param_name,
                min_value=min_val,
                max_value=max_val,
                step=step,
                param_type=param_type
            )
        else:
            raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{param_name}'")
    
    return space


def create_sma_parameter_space(args) -> ParameterSpace:
    """Create parameter space for SMA strategy."""
    space = ParameterSpace()
    
    # Window parameters
    space.add_parameter('short_window', 
                       min_value=5, 
                       max_value=50, 
                       step=5, 
                       param_type='int')
    
    space.add_parameter('long_window',
                       min_value=20,
                       max_value=200, 
                       step=20,
                       param_type='int')
    
    # Risk parameters
    if args.optimize_risk:
        space.add_parameter('stop_loss_pct',
                           min_value=0.01,
                           max_value=0.05,
                           step=0.01,
                           param_type='float')
        
        space.add_parameter('take_profit_pct',
                           min_value=0.02,
                           max_value=0.10,
                           step=0.02,
                           param_type='float')
    
    return space


def create_fvg_parameter_space(args) -> ParameterSpace:
    """Create parameter space for FVG strategy."""
    space = ParameterSpace()
    
    # H1 lookback
    space.add_parameter('h1_lookback_candles',
                       min_value=12,
                       max_value=48,
                       step=12,
                       param_type='int')
    
    # Risk reward ratio
    space.add_parameter('risk_reward_ratio',
                       min_value=1.5,
                       max_value=4.0,
                       step=0.5,
                       param_type='float')
    
    # Max hold hours
    space.add_parameter('max_hold_hours',
                       min_value=2,
                       max_value=8,
                       step=2,
                       param_type='int')
    
    # Position size (if not using risk manager)
    if not args.risk_manager or args.risk_manager == 'fixed_position':
        space.add_parameter('position_size',
                           min_value=0.02,
                           max_value=0.10,
                           step=0.02,
                           param_type='float')
    
    return space


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
                       choices=['sma', 'fvg'],
                       help='Strategy to optimize')
    
    # Data parameters
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD)')
    
    # Optimization parameters
    parser.add_argument('--metric', type=str, default='Return [%]',
                       choices=['Return [%]', 'Sharpe Ratio', 'Win Rate [%]', 
                               'Max. Drawdown [%]', 'Profit Factor'],
                       help='Metric to optimize')
    parser.add_argument('--minimize', action='store_true',
                       help='Minimize the metric instead of maximizing')
    parser.add_argument('--param-space', type=str,
                       help='Custom parameter space as JSON string or environment variable name')
    
    # Risk management
    parser.add_argument('--risk-manager', type=str,
                       choices=['fixed_position', 'fixed_risk'],
                       help='Risk manager type')
    parser.add_argument('--optimize-risk', action='store_true',
                       help='Include risk parameters in optimization')
    
    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate')
    parser.add_argument('--margin', type=float, default=0.01,
                       help='Margin requirement (0.01 = 100x leverage)')
    
    # Performance options
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 = all cores, 1 = sequential)')
    parser.add_argument('--use-simple', action='store_true',
                       help='Use simple sequential optimizer instead of parallel')
    
    # Output
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON, default: auto-generated in output/optimizations/)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("DEBUG" if args.debug else "INFO")
    
    # Load data
    data = load_data(args.data_file, args.start, args.end)
    
    # Create parameter space
    if args.param_space:
        # Use custom parameter space from JSON/environment variable
        param_space = parse_custom_parameter_space(args.param_space)
        logger.info("Using custom parameter space from JSON")
    else:
        # Use default parameter space based on strategy
        if args.strategy == 'sma':
            param_space = create_sma_parameter_space(args)
        elif args.strategy == 'fvg':
            param_space = create_fvg_parameter_space(args)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        logger.info(f"Using default parameter space for {args.strategy}")
    
    # Get strategy class
    if args.strategy == 'sma':
        strategy_class = SMAStrategy
    elif args.strategy == 'fvg':
        strategy_class = FVGStrategy
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Create risk manager if specified
    risk_manager = None
    if args.risk_manager == 'fixed_position':
        risk_manager = FixedPositionSizeManager(position_size=0.05)
    elif args.risk_manager == 'fixed_risk':
        risk_manager = FixedRiskManager(risk_percent=0.01)
    
    # Log parameter space
    logger.info(f"Parameter space: {param_space.get_total_combinations()} combinations")
    logger.info(str(param_space))
    
    # Create optimizer - choose between parallel and simple based on options
    if args.use_simple or args.n_jobs == 1:
        # Use simple sequential optimizer
        optimizer = SimpleGridSearchOptimizer(
            strategy_class=strategy_class,
            parameter_space=param_space,
            data=data,
            initial_capital=args.initial_capital,
            commission=args.commission,
            margin=args.margin,
            risk_manager=risk_manager,
            n_jobs=1,
            debug=args.debug
        )
        logger.info("Using SimpleGridSearchOptimizer (sequential)")
    else:
        # Use parallel optimizer for better performance
        optimizer = ParallelGridSearchOptimizer(
            strategy_class=strategy_class,
            parameter_space=param_space,
            data=data,
            initial_capital=args.initial_capital,
            commission=args.commission,
            margin=args.margin,
            risk_manager=risk_manager,
            n_jobs=args.n_jobs,
            debug=args.debug
        )
        logger.info(f"Using ParallelGridSearchOptimizer ({args.n_jobs} jobs)")
    
    # Run optimization
    logger.info("Starting optimization...")
    results = optimizer.optimize()
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best {args.metric}: {results['metric_value']:.4f}")
    print("\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    print("\nPerformance Metrics:")
    for metric, value in results['best_stats'].items():
        if value is not None:
            print(f"  {metric}: {value}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        # Generate default output path with backtest-like naming
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Extract symbol from data file path if possible
        data_file_name = Path(args.data_file).stem
        if '_' in data_file_name:
            # Try to extract symbol from filename like "BTC_USDT_binance_15m_..."
            parts = data_file_name.split('_')
            if len(parts) >= 2:
                symbol = f"{parts[0]}_{parts[1]}"
            else:
                symbol = "unknown"
        else:
            symbol = "unknown"
        
        output_path = f"output/optimizations/{args.strategy}_{symbol}_{timestamp}_optimization.json"
    
    optimizer.save_results(output_path, results, save_heatmap=True)
    print(f"\nResults saved to: {output_path}")
    
    # Print CSV path
    csv_path = Path(output_path).with_suffix('.csv')
    print(f"All results saved to: {csv_path}")
    
    # Also print chart paths if they were generated
    if len(param_space.parameters) == 2:
        heatmap_path = Path(output_path).with_suffix('.png')
        print(f"Heatmap saved to: {heatmap_path}")
    elif len(param_space.parameters) > 2:
        charts_base = Path(output_path).with_suffix('')
        print(f"Charts saved to: {charts_base}_*.png and {charts_base}_*.pdf")


if __name__ == '__main__':
    main()