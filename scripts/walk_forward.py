"""
Walk-forward optimization runner supporting multiple data sources and strategies

Walk-forward analysis optimizes parameters on a training period, then tests those parameters
on a subsequent test period. This is repeated over multiple windows to assess robustness.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Backtest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from config.settings import load_config
from data import YahooFinanceSource, CCXTSource, CSVDataSource
from strategies.trend_following import SMAStrategy
from strategies.mean_reversion import RSIStrategy
from strategies.sma_position import SMAPositionStrategy
from utils.logger import setup_logger
from scripts.optimize import OptimizationWrapper, optimize_strategy, _strategy_class, _data, _optimization_params, _symbol

def run_walk_forward_optimization(strategy_name: str, symbol: str, start_date: str, end_date: str,
                                 train_period: int, test_period: int, optimization_params: dict,
                                 data_source: str = None, timeframe: str = None,
                                 metric: str = 'Return [%]', maximize: bool = True, max_tries: int = None):
    """
    Run walk-forward optimization
    
    Args:
        strategy_name: Name of strategy to optimize
        symbol: Symbol to optimize on
        start_date: Start date for entire analysis
        end_date: End date for entire analysis  
        train_period: Training period in days
        test_period: Testing period in days
        optimization_params: Dict of parameters to optimize with ranges
        data_source: Data source override
        timeframe: Timeframe override  
        metric: Metric to optimize for
        maximize: Whether to maximize or minimize the metric
        max_tries: Maximum optimization attempts
    """
    
    config = load_config()
    logger = setup_logger()

    # Use specified data source or config default
    if data_source:
        config.data_source = data_source
    
    # Use specified timeframe or config default
    if timeframe:
        config.timeframe = timeframe

    logger.info(f"Starting walk-forward optimization for {strategy_name} on {symbol}")
    logger.info(f"Training period: {train_period} days, Test period: {test_period} days")
    logger.info(f"Data source: {config.data_source} ({config.timeframe})")
    logger.info(f"Optimization metric: {metric} ({'maximize' if maximize else 'minimize'})")

    # Initialize data source
    if config.data_source == "yahoo":
        source = YahooFinanceSource()
    elif config.data_source == "ccxt":
        source = CCXTSource(
            exchange_name=config.exchange_name,
            api_key=config.api_key,
            api_secret=config.api_secret,
            sandbox=config.use_sandbox
        )
    elif config.data_source == "csv":
        source = CSVDataSource(data_directory=config.csv_data_directory)
    elif config.data_source == "csv_processed":
        source = CSVDataSource(data_directory=config.csv_data_directory, use_processed=True)
    else:
        logger.error(f"Unknown data source: {config.data_source}")
        return None

    # Load full dataset
    full_data = source.get_historical_data(symbol, start_date, end_date, config.timeframe)
    
    if full_data.empty:
        logger.error(f"Failed to load data for {symbol} from {config.data_source}")
        return None

    logger.info(f"Loaded {len(full_data)} data points from {full_data.index[0]} to {full_data.index[-1]}")

    # Generate walk-forward windows
    windows = generate_walk_forward_windows(full_data, train_period, test_period)
    
    if not windows:
        logger.error("No valid walk-forward windows generated")
        return None
    
    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Run optimization for each window
    walk_forward_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        logger.info(f"\n=== Window {i+1}/{len(windows)} ===")
        logger.info(f"Train: {train_start} to {train_end}")
        logger.info(f"Test: {test_start} to {test_end}")
        
        # Extract training data
        train_data = full_data.loc[train_start:train_end]
        test_data = full_data.loc[test_start:test_end]
        
        if len(train_data) < 30 or len(test_data) < 10:
            logger.warning(f"Insufficient data for window {i+1}, skipping")
            continue
        
        # Optimize on training period
        logger.info(f"Optimizing on training data ({len(train_data)} points)...")
        
        # Use the optimize_strategy function for training optimization
        train_start_str = train_start.strftime('%Y-%m-%d')
        train_end_str = train_end.strftime('%Y-%m-%d')
        
        opt_result = optimize_strategy(
            strategy_name, symbol, train_start_str, train_end_str,
            optimization_params, data_source, timeframe, metric, maximize, max_tries
        )
        
        if opt_result is None:
            logger.warning(f"Optimization failed for window {i+1}")
            continue
        
        # Extract optimized parameters
        best_params = opt_result._strategy.__dict__
        best_params = {k: v for k, v in best_params.items() if k in optimization_params.keys()}
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Training {metric}: {opt_result[metric]:.2f}")
        
        # Test on out-of-sample data
        logger.info(f"Testing on out-of-sample data ({len(test_data)} points)...")
        
        test_result = run_backtest_with_params(
            strategy_name, test_data, best_params, config
        )
        
        if test_result is None:
            logger.warning(f"Backtest failed for window {i+1}")
            continue
            
        logger.info(f"Test {metric}: {test_result[metric]:.2f}")
        
        # Store results
        window_result = {
            'window': i + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_points': len(train_data),
            'test_points': len(test_data),
            'best_params': best_params,
            'train_result': dict(opt_result),
            'test_result': dict(test_result),
            'train_metric': opt_result[metric],
            'test_metric': test_result[metric]
        }
        
        walk_forward_results.append(window_result)
    
    if not walk_forward_results:
        logger.error("No successful walk-forward windows")
        return None
    
    # Calculate walk-forward efficiency and statistics
    efficiency_stats = calculate_walk_forward_efficiency(walk_forward_results, metric)
    
    # Save results
    results_dict = save_walk_forward_results(
        walk_forward_results, efficiency_stats, strategy_name, symbol,
        train_period, test_period, optimization_params, config
    )
    
    # Generate plots
    generate_walk_forward_plots(walk_forward_results, efficiency_stats, metric)
    
    # Print summary
    print_walk_forward_summary(efficiency_stats, metric)
    
    return results_dict

def generate_walk_forward_windows(data, train_period, test_period):
    """Generate walk-forward windows from the data"""
    windows = []
    
    # Convert to business days for more accurate period calculation
    start_date = data.index[0]
    end_date = data.index[-1]
    
    current_train_start = start_date
    
    while True:
        # Calculate train end (train_period days from train start)
        train_end = current_train_start + timedelta(days=train_period)
        
        # Calculate test start (day after train end)
        test_start = train_end + timedelta(days=1)
        
        # Calculate test end (test_period days from test start)
        test_end = test_start + timedelta(days=test_period)
        
        # Check if we have enough data for this window
        if test_end > end_date:
            break
            
        # Ensure we have actual data for these periods
        train_data = data.loc[current_train_start:train_end]
        test_data = data.loc[test_start:test_end]
        
        if len(train_data) > 0 and len(test_data) > 0:
            windows.append((
                train_data.index[0],   # Actual train start
                train_data.index[-1],  # Actual train end
                test_data.index[0],    # Actual test start  
                test_data.index[-1]    # Actual test end
            ))
        
        # Move to next window (advance by test_period)
        current_train_start = test_start
    
    return windows

def run_backtest_with_params(strategy_name, data, params, config):
    """Run backtest with specific parameters"""
    from backtesting import Strategy
    
    # Create a custom strategy class for walk-forward testing
    class WalkForwardTestStrategy(Strategy):
        # Add parameters dynamically
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        stop_loss_pct = params.get('stop_loss_pct', 2.0)
        take_profit_pct = params.get('take_profit_pct', 4.0)
        
        def init(self):
            # Calculate indicators
            self.sma_short = self.I(lambda x: pd.Series(x).rolling(self.short_window).mean(), self.data.Close)
            self.sma_long = self.I(lambda x: pd.Series(x).rolling(self.long_window).mean(), self.data.Close)
            
        def next(self):
            # Skip if we don't have enough data
            if len(self.data.Close) < self.long_window:
                return
                
            # Current values
            current_short = self.sma_short[-1]
            current_long = self.sma_long[-1]
            
            # Skip if indicators are not ready
            if pd.isna(current_short) or pd.isna(current_long):
                return
            
            # Position logic
            if current_short > current_long:
                # Should be long
                if not self.position:
                    self.buy()
                elif self.position.is_short:
                    self.position.close()
                    self.buy()
            elif current_short < current_long:
                # Should be short
                if not self.position:
                    self.sell()
                elif self.position.is_long:
                    self.position.close()
                    self.sell()
    
    # Run backtest with the position-based strategy
    bt = Backtest(
        data,
        WalkForwardTestStrategy,
        cash=config.initial_capital,
        commission=config.commission,
        trade_on_close=True,
        hedging=False,
        exclusive_orders=True
    )
    
    try:
        result = bt.run()
        return result
    except Exception as e:
        setup_logger().error(f"Backtest failed: {e}")
        return None

def calculate_walk_forward_efficiency(results, metric):
    """Calculate walk-forward efficiency statistics"""
    train_metrics = [r['train_metric'] for r in results]
    test_metrics = [r['test_metric'] for r in results]
    
    # Walk-forward efficiency (average test performance / average train performance)
    avg_train = np.mean(train_metrics)
    avg_test = np.mean(test_metrics)
    efficiency = (avg_test / avg_train) * 100 if avg_train != 0 else 0
    
    # Additional statistics
    stats = {
        'efficiency': efficiency,
        'avg_train_metric': avg_train,
        'avg_test_metric': avg_test,
        'train_std': np.std(train_metrics),
        'test_std': np.std(test_metrics),
        'train_min': np.min(train_metrics),
        'train_max': np.max(train_metrics),
        'test_min': np.min(test_metrics),
        'test_max': np.max(test_metrics),
        'windows_count': len(results),
        'positive_test_windows': sum(1 for x in test_metrics if x > 0),
        'positive_test_ratio': sum(1 for x in test_metrics if x > 0) / len(test_metrics) * 100,
        'correlation': np.corrcoef(train_metrics, test_metrics)[0, 1] if len(train_metrics) > 1 else 0
    }
    
    return stats

def save_walk_forward_results(results, efficiency_stats, strategy_name, symbol,
                             train_period, test_period, optimization_params, config):
    """Save walk-forward results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_filename = f"walk_forward_{strategy_name}_{symbol}_{timestamp}"
    
    # Clean results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = {}
        for key, value in result.items():
            if hasattr(value, 'isoformat'):  # datetime objects
                clean_result[key] = value.isoformat()
            elif isinstance(value, dict):
                # Clean nested dict
                clean_dict = {}
                for k, v in value.items():
                    if hasattr(v, 'isoformat'):
                        clean_dict[k] = v.isoformat()
                    elif not isinstance(v, (int, float, str, bool, type(None))):
                        clean_dict[k] = str(v)
                    else:
                        clean_dict[k] = v
                clean_result[key] = clean_dict
            elif not isinstance(value, (int, float, str, bool, type(None))):
                clean_result[key] = str(value)
            else:
                clean_result[key] = value
        clean_results.append(clean_result)
    
    results_dict = {
        'walk_forward_results': clean_results,
        'efficiency_stats': efficiency_stats,
        'metadata': {
            'strategy': strategy_name,
            'symbol': symbol,
            'train_period_days': train_period,
            'test_period_days': test_period,
            'data_source': config.data_source,
            'timeframe': config.timeframe,
            'optimization_params': optimization_params,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    json_file = output_dir / f"{output_filename}_results.json"
    with open(json_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    setup_logger().info(f"Walk-forward results saved to: {json_file}")
    return results_dict

def generate_walk_forward_plots(results, efficiency_stats, metric):
    """Generate walk-forward analysis plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "output"
    
    # Extract data for plotting
    windows = [r['window'] for r in results]
    train_metrics = [r['train_metric'] for r in results]
    test_metrics = [r['test_metric'] for r in results]
    
    # Plot 1: Train vs Test Performance
    plt.figure(figsize=(12, 6))
    plt.plot(windows, train_metrics, 'bo-', label=f'Training {metric}', alpha=0.7)
    plt.plot(windows, test_metrics, 'ro-', label=f'Test {metric}', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Window')
    plt.ylabel(metric)
    plt.title('Walk-Forward Analysis: Train vs Test Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot1_file = output_dir / f"walk_forward_performance_{timestamp}.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Efficiency and Statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Efficiency bar
    ax1.bar(['Walk-Forward Efficiency'], [efficiency_stats['efficiency']], 
            color='green' if efficiency_stats['efficiency'] > 80 else 'orange' if efficiency_stats['efficiency'] > 60 else 'red')
    ax1.set_ylabel('Efficiency (%)')
    ax1.set_title(f'Walk-Forward Efficiency: {efficiency_stats["efficiency"]:.1f}%')
    ax1.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% (Perfect)')
    ax1.legend()
    
    # Distribution comparison
    ax2.hist(train_metrics, alpha=0.5, label='Training', bins=10)
    ax2.hist(test_metrics, alpha=0.5, label='Test', bins=10)
    ax2.set_xlabel(metric)
    ax2.set_ylabel('Frequency')
    ax2.set_title('Performance Distribution')
    ax2.legend()
    
    # Scatter plot
    ax3.scatter(train_metrics, test_metrics, alpha=0.7)
    ax3.plot([min(train_metrics), max(train_metrics)], [min(train_metrics), max(train_metrics)], 
             'r--', alpha=0.5, label='Perfect correlation')
    ax3.set_xlabel(f'Training {metric}')
    ax3.set_ylabel(f'Test {metric}')
    ax3.set_title(f'Correlation: {efficiency_stats["correlation"]:.3f}')
    ax3.legend()
    
    # Statistics text
    stats_text = f"""
    Avg Train: {efficiency_stats['avg_train_metric']:.2f}
    Avg Test: {efficiency_stats['avg_test_metric']:.2f}
    Test Std: {efficiency_stats['test_std']:.2f}
    Positive Windows: {efficiency_stats['positive_test_windows']}/{efficiency_stats['windows_count']}
    Win Rate: {efficiency_stats['positive_test_ratio']:.1f}%
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plot2_file = output_dir / f"walk_forward_analysis_{timestamp}.png"
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    setup_logger().info(f"Walk-forward plots saved to: {plot1_file} and {plot2_file}")

def print_walk_forward_summary(efficiency_stats, metric):
    """Print walk-forward analysis summary"""
    logger = setup_logger()
    
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Walk-Forward Efficiency: {efficiency_stats['efficiency']:.1f}%")
    logger.info(f"Average Training {metric}: {efficiency_stats['avg_train_metric']:.2f}")
    logger.info(f"Average Test {metric}: {efficiency_stats['avg_test_metric']:.2f}")
    logger.info(f"Test Performance Std Dev: {efficiency_stats['test_std']:.2f}")
    logger.info(f"Train-Test Correlation: {efficiency_stats['correlation']:.3f}")
    logger.info(f"Positive Test Windows: {efficiency_stats['positive_test_windows']}/{efficiency_stats['windows_count']} ({efficiency_stats['positive_test_ratio']:.1f}%)")
    
    # Interpretation
    if efficiency_stats['efficiency'] >= 80:
        logger.info("🟢 Excellent: Strategy shows good out-of-sample robustness")
    elif efficiency_stats['efficiency'] >= 60:
        logger.info("🟡 Good: Strategy shows reasonable out-of-sample performance")
    elif efficiency_stats['efficiency'] >= 40:
        logger.info("🟠 Moderate: Strategy shows some overfitting concerns")
    else:
        logger.info("🔴 Poor: Strategy appears to be overfitted")
    
    logger.info("="*60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run walk-forward optimization analysis')
    parser.add_argument('--strategy', required=True, help='Strategy name (sma, rsi)')
    parser.add_argument('--symbol', help='Symbol to analyze')
    parser.add_argument('--start', default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--train-period', type=int, default=180, help='Training period in days')
    parser.add_argument('--test-period', type=int, default=60, help='Test period in days')
    parser.add_argument('--data-source', help='Data source (yahoo, ccxt, csv, csv_processed)')
    parser.add_argument('--timeframe', help='Timeframe (1m, 5m, 15m, 30m, 1h, 1d)')
    parser.add_argument('--metric', default='Return [%]', help='Optimization metric')
    parser.add_argument('--minimize', action='store_true', help='Minimize metric instead of maximize')
    parser.add_argument('--max-tries', type=int, help='Maximum optimization attempts')
    
    # Strategy-specific parameters (same as optimize.py)
    parser.add_argument('--short-window', help='SMA short window range (e.g., "5,50" or "10,20,30")')
    parser.add_argument('--long-window', help='SMA long window range (e.g., "20,100" or "50,60,70")')
    parser.add_argument('--stop-loss', help='Stop loss percentage range (e.g., "1,5" or "2,3,4")')
    parser.add_argument('--take-profit', help='Take profit percentage range (e.g., "2,10" or "4,6,8")')

    args = parser.parse_args()

    # Load config to get defaults
    config = load_config()
    symbol = args.symbol or (config.symbols[0] if config.symbols else "AAPL")
    
    # Build optimization parameters (same logic as optimize.py)
    optimization_params = {}
    
    if args.strategy.lower() in ["sma", "sma_crossover", "trend_following"]:
        if args.short_window:
            values = args.short_window.split(',')
            if len(values) == 2:
                optimization_params['short_window'] = (int(values[0]), int(values[1]))
            else:
                optimization_params['short_window'] = [int(v) for v in values]
        else:
            optimization_params['short_window'] = (5, 50)
            
        if args.long_window:
            values = args.long_window.split(',')
            if len(values) == 2:
                optimization_params['long_window'] = (int(values[0]), int(values[1]))
            else:
                optimization_params['long_window'] = [int(v) for v in values]
        else:
            optimization_params['long_window'] = (20, 100)
            
        if args.stop_loss:
            values = args.stop_loss.split(',')
            if len(values) == 2:
                optimization_params['stop_loss_pct'] = (float(values[0]), float(values[1]))
            else:
                optimization_params['stop_loss_pct'] = [float(v) for v in values]
            
        if args.take_profit:
            values = args.take_profit.split(',')
            if len(values) == 2:
                optimization_params['take_profit_pct'] = (float(values[0]), float(values[1]))
            else:
                optimization_params['take_profit_pct'] = [float(v) for v in values]
    
    if not optimization_params:
        print(f"No optimization parameters defined for strategy: {args.strategy}")
        sys.exit(1)
    
    run_walk_forward_optimization(
        args.strategy,
        symbol,
        args.start,
        args.end,
        args.train_period,
        args.test_period,
        optimization_params,
        args.data_source,
        args.timeframe,
        args.metric,
        not args.minimize,
        args.max_tries
    )