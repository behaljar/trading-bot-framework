#!/usr/bin/env python3
"""
Walk-Forward Analysis Runner

This script performs walk-forward analysis on trading strategies to validate
their robustness across different time periods. It uses pre-optimized parameters
to test strategy performance on rolling time windows.

Example usage:
    # Basic walk-forward analysis with SMA strategy
    STRATEGY_PARAMS='{"short_window": 10, "long_window": 20}' python scripts/run_walk_forward.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

    # Walk-forward with custom windows and risk management
    STRATEGY_PARAMS='{"h1_lookback_candles": 24, "risk_reward_ratio": 3.0}' RISK_PARAMS='{"risk_percent": 0.01}' python scripts/run_walk_forward.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --test-window-months 3 --step-months 1 --risk-manager fixed_risk
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add framework to path
script_dir = Path(__file__).parent
framework_dir = script_dir.parent
sys.path.append(str(framework_dir))

from framework.utils.logger import setup_logger
from framework.strategies.sma_strategy import SMAStrategy
from framework.strategies.fvg_strategy import FVGStrategy
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


def save_results(analysis_result, output_dir: Path, strategy_name: str, symbol: str):
    """Save walk-forward analysis results to files."""
    # Create timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    clean_symbol = symbol.replace('/', '_')
    
    # Save summary results to JSON
    summary_file = output_dir / f"walk_forward_{strategy_name}_{clean_symbol}_{timestamp}_summary.json"
    
    summary_data = {
        'analysis_summary': analysis_result.get_summary(),
        'combined_metrics': analysis_result.combined_metrics,
        'stability_metrics': analysis_result.stability_metrics,
        'analysis_parameters': analysis_result.analysis_parameters,
        'metadata': analysis_result.metadata
    }
    
    # Convert datetime objects to strings for JSON serialization
    def convert_datetime(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return str(obj)
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=convert_datetime)
    
    print(f"Summary results saved to: {summary_file}")
    
    # Save detailed period results to CSV
    try:
        periods_df = analysis_result.to_dataframe()
        if not periods_df.empty:
            periods_file = output_dir / f"walk_forward_{strategy_name}_{clean_symbol}_{timestamp}_periods.csv"
            periods_df.to_csv(periods_file, index=False)
            print(f"Period-by-period results saved to: {periods_file}")
    except Exception as e:
        print(f"Could not save period results to CSV: {e}")


def print_results_summary(analysis_result):
    """Print a formatted summary of walk-forward analysis results."""
    print("\n" + "="*80)
    print(f"WALK-FORWARD ANALYSIS RESULTS - {analysis_result.strategy_name}")
    print("="*80)
    
    # Print basic info
    print(f"Symbol: {analysis_result.symbol}")
    print(f"Analysis Period: {analysis_result.analysis_start_date.date()} to {analysis_result.analysis_end_date.date()}")
    print(f"Total Test Periods: {len(analysis_result.individual_results)}")
    print(f"Test Window: {analysis_result.analysis_parameters['test_window_months']} months")
    print(f"Step Size: {analysis_result.analysis_parameters['step_months']} months")
    
    # Print combined metrics
    print("\nCOMBINED METRICS:")
    print("-" * 40)
    metrics = analysis_result.combined_metrics
    
    print(f"Average OOS Return:    {metrics.get('avg_return_pct', 0):.2f}%")
    print(f"Median OOS Return:     {metrics.get('median_return_pct', 0):.2f}%")
    print(f"OOS Return Std Dev:    {metrics.get('std_return_pct', 0):.2f}%")
    print(f"Positive OOS Periods:  {metrics.get('positive_return_periods', 0)}/{metrics.get('total_periods', 0)} ({metrics.get('positive_return_pct', 0):.1f}%)")
    print(f"Average Sharpe:        {metrics.get('avg_sharpe_ratio', 0):.3f}")
    print(f"Average Max DD:        {metrics.get('avg_max_drawdown', 0):.2f}%")
    print(f"Worst Max DD:          {metrics.get('worst_max_drawdown', 0):.2f}%")
    print(f"Average Win Rate:      {metrics.get('avg_win_rate', 0):.2f}%")
    print(f"Avg Trades/Period:     {metrics.get('avg_trades_per_period', 0):.1f}")
    
    # WFE specific metrics
    print("\nWALK-FORWARD EFFICIENCY (WFE):")
    print("-" * 40)
    print(f"Average WFE:           {metrics.get('avg_wfe', 0):.1f}%")
    print(f"Median WFE:            {metrics.get('median_wfe', 0):.1f}%")
    print(f"WFE Std Dev:           {metrics.get('wfe_std', 0):.1f}%")
    print(f"Positive WFE Periods:  {metrics.get('positive_wfe_periods', 0)}/{metrics.get('total_periods', 0)} ({metrics.get('positive_wfe_pct', 0):.1f}%)")
    print(f"IS-OOS Correlation:    {metrics.get('is_oos_correlation', 0):.3f}")
    print(f"Avg IS Return:         {metrics.get('avg_is_return_pct', 0):.2f}%")
    print(f"Avg IS Annual Return:  {metrics.get('avg_is_annual_return_pct', 0):.2f}%")
    print(f"Avg OOS Annual Return: {metrics.get('avg_oos_annual_return_pct', 0):.2f}%")
    
    # WFE Interpretation
    avg_wfe = metrics.get('avg_wfe', 0)
    if avg_wfe >= 100:
        print(f"🟢 Excellent: Strategy maintains or improves performance out-of-sample")
    elif avg_wfe >= 80:
        print(f"🟡 Good: Strategy shows reasonable out-of-sample robustness")
    elif avg_wfe >= 60:
        print(f"🟠 Moderate: Strategy shows some degradation out-of-sample")
    else:
        print(f"🔴 Poor: Strategy appears overfitted to training data")
    
    # Print stability metrics
    if analysis_result.stability_metrics:
        print("\nSTABILITY METRICS:")
        print("-" * 40)
        stability = analysis_result.stability_metrics
        
        print(f"Return Volatility:     {stability.get('return_pct_volatility', 0):.2f}%")
        print(f"Return Consistency:    {stability.get('return_consistency', 0):.3f}")
        print(f"Temporal Stability:    {stability.get('temporal_stability', 0):.3f}")
        
        if 'trend_consistency' in stability:
            print(f"Trend Consistency:     {stability.get('trend_consistency', 0):.3f}")
    
    # Show best and worst periods
    best_periods = analysis_result.get_best_periods(3)
    if best_periods:
        print("\nTOP 3 PERFORMING PERIODS:")
        print("-" * 40)
        for i, result in enumerate(best_periods, 1):
            return_pct = getattr(result, 'Return [%]', getattr(result, 'return_pct', 0))
            start_date = result.metadata.get('test_start', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
            end_date = result.metadata.get('test_end', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
            print(f"  {i}. {return_pct:.2f}% ({start_date} to {end_date})")
    
    worst_periods = analysis_result.get_worst_periods(3)
    if worst_periods:
        print("\nWORST 3 PERFORMING PERIODS:")
        print("-" * 40)
        for i, result in enumerate(worst_periods, 1):
            return_pct = getattr(result, 'Return [%]', getattr(result, 'return_pct', 0))
            start_date = result.metadata.get('test_start', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
            end_date = result.metadata.get('test_end', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
            print(f"  {i}. {return_pct:.2f}% ({start_date} to {end_date})")


def main():
    """Main function for running walk-forward analysis."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward analysis on trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic walk-forward with SMA strategy
  STRATEGY_PARAMS='{"short_window": 10, "long_window": 20}' python scripts/run_walk_forward.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

  # Walk-forward with FVG strategy and custom risk management
  STRATEGY_PARAMS='{"h1_lookback_candles": 24}' RISK_PARAMS='{"risk_percent": 0.01}' python scripts/run_walk_forward.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --risk-manager fixed_risk

  # Custom time windows
  python scripts/run_walk_forward.py --strategy sma --data-file data.csv --symbol BTC_USDT --test-window-months 3 --step-months 1
        """
    )
    
    parser.add_argument("--strategy", type=str, required=True, 
                       choices=['sma', 'fvg'],
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
    parser.add_argument("--use-standard", action="store_true",
                       help="Use standard Backtest instead of FractionalBacktest")
    parser.add_argument("--optimization-metric", type=str, default="Return [%]",
                       choices=["Return [%]", "Sharpe Ratio", "Sortino Ratio"],
                       help="Metric to optimize for (default: Return [%])")
    parser.add_argument("--minimize", action="store_true",
                       help="Minimize metric instead of maximize")
    parser.add_argument("--n-jobs", type=int, default=None,
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
        }
        
        if args.strategy not in strategy_classes:
            raise ValueError(f"Unknown strategy: {args.strategy}. Available: {list(strategy_classes.keys())}")
        
        strategy_class = strategy_classes[args.strategy]
        
        # Get parameter space from environment variable
        parameter_space = {}
        if 'PARAMETER_SPACE' in os.environ:
            parameter_space = json.loads(os.environ['PARAMETER_SPACE'])
            logger.info(f"Loaded parameter space: {parameter_space}")
        else:
            # Use default parameter space for the strategy
            if args.strategy == 'sma':
                parameter_space = {
                    'short_window': (5, 30),
                    'long_window': (20, 100),
                    'stop_loss_pct': [0.01, 0.02, 0.03],
                    'take_profit_pct': [0.04, 0.06, 0.08]
                }
            elif args.strategy == 'fvg':
                parameter_space = {
                    'h1_lookback_candles': (12, 48),
                    'risk_reward_ratio': [2.0, 2.5, 3.0, 3.5],
                    'max_hold_hours': [2, 4, 6, 8]
                }
            logger.warning(f"No PARAMETER_SPACE environment variable found, using defaults: {parameter_space}")
        
        # Get risk manager parameters from environment variable
        risk_manager_params = {}
        if 'RISK_PARAMS' in os.environ:
            risk_manager_params = json.loads(os.environ['RISK_PARAMS'])
            logger.info(f"Loaded risk manager parameters: {risk_manager_params}")
        
        # Load data
        data = load_data(args.data_file, args.start, args.end)
        
        # Create walk-forward analyzer
        analyzer = WalkForwardAnalyzer(
            strategy_class=strategy_class,
            parameter_space=parameter_space,
            is_window_months=args.is_window_months,
            oos_window_months=args.oos_window_months,
            step_months=args.step_months,
            window_mode=args.window_mode,
            initial_capital=args.initial_capital,
            commission=args.commission,
            margin=args.margin,
            risk_manager_type=args.risk_manager,
            risk_manager_params=risk_manager_params,
            use_fractional=not args.use_standard,
            optimization_metric=args.optimization_metric,
            maximize=not args.minimize,
            n_jobs=args.n_jobs,
            generate_charts=True
        )
        
        print(f"\nStarting walk-forward analysis...")
        print(f"Strategy: {args.strategy}")
        print(f"Symbol: {args.symbol}")
        print(f"IS Window: {args.is_window_months} months, OOS Window: {args.oos_window_months} months")
        print(f"Window Mode: {args.window_mode}")
        print(f"Step Size: {args.step_months} months")
        print(f"Risk Manager: {args.risk_manager}")
        print(f"Generate Charts: {analyzer.generate_charts}")
        print(f"Parallel Jobs: {analyzer.n_jobs}")
        
        # Run walk-forward analysis
        result = analyzer.analyze(data, args.symbol)
        
        # Print results summary
        print_results_summary(result)
        
        # Save results to files
        output_dir = Path("output/walk_forward")
        output_dir.mkdir(exist_ok=True)
        save_results(result, output_dir, args.strategy, args.symbol)
        
        logger.info("Walk-forward analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()