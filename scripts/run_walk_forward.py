#!/usr/bin/env python3
"""
Walk-Forward Analysis Runner

Performs walk-forward analysis using GridSearchOptimizer for consistent
parameter optimization across IS/OOS periods.

Example usage:
    # Basic walk-forward analysis with SMA strategy
    PARAM_CONFIG='{"short_window": {"values": [10, 15, 20]}, "long_window": {"values": [30, 40, 50]}}' python scripts/run_walk_forward.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

    # Walk-forward with custom windows and risk management
    PARAM_CONFIG='{"h1_lookback_candles": {"min": 20, "max": 28, "step": 2}, "risk_reward_ratio": {"choices": [2.5, 3.0, 3.5]}}' RISK_PARAMS='{"risk_percent": 0.01}' python scripts/run_walk_forward.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --is-window-months 3 --oos-window-months 1
"""

import argparse
import pandas as pd
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add framework to path
script_dir = Path(__file__).parent
framework_dir = script_dir.parent
sys.path.append(str(framework_dir))

from framework.utils.logger import setup_logger
from framework.strategies.sma_strategy import SMAStrategy
from framework.strategies.fvg_strategy import FVGStrategy
from framework.strategies.breakout_strategy import BreakoutStrategy
from framework.strategies.silver_bullet_fvg_strategy import SilverBulletFVGStrategy
from framework.optimization.walk_forward_analyzer import WalkForwardAnalyzer




def load_data(file_path: str, start_date: Optional[str] = None, 
              end_date: Optional[str] = None) -> pd.DataFrame:
    """Load and prepare data for walk-forward analysis."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    
    # Load CSV file
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Filter by date range
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    # Ensure we have the required columns (lowercase for framework)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} data points")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def print_results_summary(result: Dict[str, Any]):
    """Print a formatted summary of walk-forward analysis results."""
    summary = result['summary']
    info = result['analysis_info']
    
    print("\n" + "="*80)
    print(f"WALK-FORWARD ANALYSIS RESULTS - {info['strategy']}")
    print("="*80)
    
    # Print basic info
    print(f"Symbol: {info['symbol']}")
    print(f"Total Periods: {info['total_periods']}")
    print(f"IS Window: {info['is_window_months']} months, OOS Window: {info['oos_window_months']} months")
    print(f"Window Mode: {info['window_mode']}, Step: {info['step_months']} months")
    print(f"Optimization Metric: {info['optimization_metric']}")
    
    # Print WFE metrics
    print("\nWALK-FORWARD EFFICIENCY (WFE):")
    print("-" * 40)
    print(f"Average WFE:           {summary.get('avg_wfe', 0):.1f}%")
    print(f"Median WFE:            {summary.get('median_wfe', 0):.1f}%")
    print(f"WFE Std Dev:           {summary.get('std_wfe', 0):.1f}%")
    print(f"Positive WFE Periods:  {summary.get('positive_wfe_periods', 0)}/{summary.get('total_periods', 0)} ({summary.get('positive_wfe_pct', 0):.1f}%)")
    print(f"WFE Consistency:       {summary.get('wfe_consistency', 0):.3f}")
    
    # Print performance metrics
    print("\nPERFORMANCE METRICS:")
    print("-" * 40)
    print(f"Avg IS Return:         {summary.get('avg_is_return_pct', 0):.2f}%")
    print(f"Avg OOS Return:        {summary.get('avg_oos_return_pct', 0):.2f}%")
    print(f"Median OOS Return:     {summary.get('median_oos_return_pct', 0):.2f}%")
    print(f"OOS Return Std Dev:    {summary.get('std_oos_return_pct', 0):.2f}%")
    print(f"Positive OOS Periods:  {summary.get('positive_oos_periods', 0)}/{summary.get('total_periods', 0)} ({summary.get('positive_oos_pct', 0):.1f}%)")
    print(f"IS-OOS Correlation:    {summary.get('is_oos_correlation', 0):.3f}")
    print(f"Avg IS Sharpe:         {summary.get('avg_is_sharpe', 0):.3f}")
    print(f"Avg OOS Sharpe:        {summary.get('avg_oos_sharpe', 0):.3f}")
    print(f"Avg IS Trades:         {summary.get('avg_is_trades', 0):.1f}")
    print(f"Avg OOS Trades:        {summary.get('avg_oos_trades', 0):.1f}")
    
    # WFE Interpretation
    avg_wfe = summary.get('avg_wfe', 0)
    if avg_wfe >= 100:
        print(f"\n🟢 Excellent: Strategy maintains or improves performance out-of-sample")
    elif avg_wfe >= 80:
        print(f"\n🟡 Good: Strategy shows reasonable out-of-sample robustness")
    elif avg_wfe >= 60:
        print(f"\n🟠 Moderate: Strategy shows some degradation out-of-sample")
    else:
        print(f"\n🔴 Poor: Strategy appears overfitted to training data")
    
    # Show best and worst periods
    periods = result['periods']
    if periods:
        periods_sorted = sorted(periods, key=lambda x: x['wfe_metrics']['oos_return_pct'], reverse=True)
        
        print("\nTOP 3 OOS PERFORMING PERIODS:")
        print("-" * 40)
        for i, period in enumerate(periods_sorted[:3], 1):
            oos_return = period['wfe_metrics']['oos_return_pct']
            wfe = period['wfe_metrics']['wfe']
            start_date = period['oos_start'].strftime('%Y-%m-%d')
            end_date = period['oos_end'].strftime('%Y-%m-%d')
            print(f"  {i}. {oos_return:.2f}% OOS (WFE: {wfe:.0f}%) - {start_date} to {end_date}")
        
        print("\nWORST 3 OOS PERFORMING PERIODS:")
        print("-" * 40)
        for i, period in enumerate(periods_sorted[-3:], 1):
            oos_return = period['wfe_metrics']['oos_return_pct']
            wfe = period['wfe_metrics']['wfe']
            start_date = period['oos_start'].strftime('%Y-%m-%d')
            end_date = period['oos_end'].strftime('%Y-%m-%d')
            print(f"  {i}. {oos_return:.2f}% OOS (WFE: {wfe:.0f}%) - {start_date} to {end_date}")


def main():
    """Main function for running walk-forward analysis."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward analysis using GridSearchOptimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic walk-forward with SMA strategy
  PARAM_CONFIG='{"short_window": {"values": [10, 15, 20]}, "long_window": {"values": [30, 40, 50]}}' python scripts/run_walk_forward.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

  # Walk-forward with range parameters
  PARAM_CONFIG='{"short_window": {"min": 8, "max": 12, "step": 2}, "long_window": {"min": 25, "max": 35, "step": 5}}' python scripts/run_walk_forward.py --strategy sma --data-file data.csv --symbol BTC_USDT

  # FVG strategy with custom risk management
  PARAM_CONFIG='{"h1_lookback_candles": {"choices": [20, 24, 28]}, "risk_reward_ratio": {"choices": [2.5, 3.0, 3.5]}}' RISK_PARAMS='{"risk_percent": 0.01}' python scripts/run_walk_forward.py --strategy fvg --data-file data.csv --symbol BTC_USDT
        """
    )
    
    parser.add_argument("--strategy", type=str, required=True, 
                       choices=['sma', 'fvg', 'breakout', 'silver_bullet_fvg'],
                       help="Strategy to analyze")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to CSV data file")
    parser.add_argument("--symbol", type=str, default="UNKNOWN",
                       help="Symbol identifier")
    parser.add_argument("--start", type=str,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--is-window-months", type=int, default=3,
                       help="Months of in-sample data for optimization (default: 3)")
    parser.add_argument("--oos-window-months", type=int, default=1,
                       help="Months of out-of-sample data for testing (default: 1)")
    parser.add_argument("--step-months", type=int, default=1,
                       help="Months to step forward between periods (default: 1)")
    parser.add_argument("--window-mode", type=str, default="rolling",
                       choices=['rolling', 'anchored'],
                       help="Window mode: 'rolling' (fixed IS window) or 'anchored' (expanding IS window) (default: rolling)")
    parser.add_argument("--initial-capital", type=float, default=10000,
                       help="Initial capital (default: 10000)")
    parser.add_argument("--commission", type=float, default=0.001,
                       help="Commission rate (default: 0.001 = 0.1%)")
    parser.add_argument("--margin", type=float, default=0.01,
                       help="Margin requirement (default: 0.01 = 100x leverage)")
    parser.add_argument("--risk-manager", type=str, default="fixed_risk",
                       choices=['fixed_position', 'fixed_risk'],
                       help="Risk manager type (default: fixed_risk)")
    parser.add_argument("--optimization-metric", type=str, default="return_pct",
                       choices=["return_pct", "sharpe_ratio", "sortino_ratio"],
                       help="Metric to optimize for (default: return_pct)")
    parser.add_argument("--minimize", action="store_true",
                       help="Minimize metric instead of maximize")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs (default: auto-detect)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(log_level)
    
    try:
        # Map strategy names to classes
        strategy_classes = {
            'sma': SMAStrategy,
            'fvg': FVGStrategy,
            'breakout': BreakoutStrategy,
            'silver_bullet_fvg': SilverBulletFVGStrategy
        }
        
        if args.strategy not in strategy_classes:
            raise ValueError(f"Unknown strategy: {args.strategy}. Available: {list(strategy_classes.keys())}")
        
        strategy_class = strategy_classes[args.strategy]
        
        # Get parameter configuration from environment variable
        parameter_config = {}
        if 'PARAM_CONFIG' in os.environ:
            parameter_config = json.loads(os.environ['PARAM_CONFIG'])
            logger.info(f"Loaded parameter config: {parameter_config}")
        else:
            # Use default parameter configs for the strategy
            if args.strategy == 'sma':
                parameter_config = {
                    'short_window': {'min': 8, 'max': 20, 'step': 2, 'type': 'int'},
                    'long_window': {'min': 25, 'max': 50, 'step': 5, 'type': 'int'},
                    'stop_loss_pct': {'choices': [0.015, 0.02, 0.025]},
                    'take_profit_pct': {'choices': [0.03, 0.04, 0.05]}
                }
            elif args.strategy == 'fvg':
                parameter_config = {
                    'h1_lookback_candles': {'choices': [20, 24, 28]},
                    'risk_reward_ratio': {'choices': [2.5, 3.0, 3.5]},
                    'max_hold_hours': {'choices': [3, 4, 5]}
                }
            elif args.strategy == 'breakout':
                parameter_config = {
                    'entry_lookback': {'min': 20, 'max': 40, 'step': 5, 'type': 'int'},
                    'exit_lookback': {'min': 10, 'max': 20, 'step': 2, 'type': 'int'},
                    'atr_multiplier': {'choices': [2.0, 2.5, 3.0]}
                }
            logger.warning(f"No PARAM_CONFIG environment variable found, using defaults: {parameter_config}")
        
        # Get risk manager parameters from environment variable
        risk_manager_params = {}
        if 'RISK_PARAMS' in os.environ:
            risk_manager_params = json.loads(os.environ['RISK_PARAMS'])
            logger.info(f"Loaded risk manager parameters: {risk_manager_params}")
        
        # Load data
        data = load_data(args.data_file, args.start, args.end)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_clean = args.symbol.replace('/', '_')
        output_dir = f"output/walk_forward/{args.strategy}_{symbol_clean}_{timestamp}"
        
        # Create walk-forward analyzer
        analyzer = WalkForwardAnalyzer(
            strategy_class=strategy_class,
            parameter_config=parameter_config,
            is_window_months=args.is_window_months,
            oos_window_months=args.oos_window_months,
            step_months=args.step_months,
            window_mode=args.window_mode,
            initial_capital=args.initial_capital,
            commission=args.commission,
            margin=args.margin,
            risk_manager_type=args.risk_manager,
            risk_manager_params=risk_manager_params,
            optimization_metric=args.optimization_metric,
            maximize=not args.minimize,
            n_jobs=args.n_jobs
        )
        
        print(f"\nStarting walk-forward analysis...")
        print(f"Strategy: {args.strategy}")
        print(f"Symbol: {args.symbol}")
        print(f"IS Window: {args.is_window_months} months, OOS Window: {args.oos_window_months} months")
        print(f"Window Mode: {args.window_mode}")
        print(f"Step Size: {args.step_months} months")
        print(f"Risk Manager: {args.risk_manager}")
        print(f"Output Directory: {output_dir}")
        print(f"Parallel Jobs: {analyzer.n_jobs}")
        
        # Run walk-forward analysis with interrupt handling
        try:
            result = analyzer.analyze(data, args.symbol, output_dir)
            
            
            # Print results summary
            print_results_summary(result)
            
            logger.info("Walk-forward analysis completed successfully")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Analysis interrupted by user")
            print(f"📁 Partial results may be available in: {output_dir}")
            # Force kill any remaining processes
            os._exit(0)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Analysis interrupted by user")
        # Force exit to kill all child processes
        os._exit(0)
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()