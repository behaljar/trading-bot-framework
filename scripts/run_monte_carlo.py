#!/usr/bin/env python3
"""
Monte Carlo Simulation for Backtest Results

Performs Monte Carlo simulation on trading strategy results by randomizing trade order
to assess strategy robustness and potential drawdown scenarios.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse
from typing import List, Dict, Tuple, Optional
import logging

# Add framework to path
script_dir = Path(__file__).parent
framework_dir = script_dir.parent
sys.path.append(str(framework_dir))

from framework.utils.logger import setup_logger


def load_trades_from_backtest(backtest_dir: str) -> Optional[pd.DataFrame]:
    """
    Load trades from backtest output directory.
    
    Args:
        backtest_dir: Path to backtest output directory
        
    Returns:
        DataFrame with trade data or None if not found
    """
    backtest_path = Path(backtest_dir)
    
    # Look for trades CSV file
    trades_files = list(backtest_path.glob("*_trades.csv"))
    if not trades_files:
        # Try looking in parent directory
        trades_files = list(backtest_path.parent.glob("*_trades.csv"))
    
    if trades_files:
        trades_df = pd.read_csv(trades_files[0])
        return trades_df
    
    # If no trades CSV, try to extract from results JSON
    results_files = list(backtest_path.glob("*_results.json"))
    if not results_files:
        results_files = list(backtest_path.parent.glob("*_results.json"))
    
    if results_files:
        with open(results_files[0], 'r') as f:
            results = json.load(f)
            if 'trades' in results:
                return pd.DataFrame(results['trades'])
    
    return None


def calculate_equity_curve(trades: pd.DataFrame, initial_capital: float = 10000) -> pd.Series:
    """
    Calculate equity curve from trades.
    
    Args:
        trades: DataFrame with trade results
        initial_capital: Starting capital
        
    Returns:
        Series with equity values
    """
    equity = initial_capital
    equity_curve = [equity]
    
    # Determine return column name
    return_col = None
    for col in ['ReturnPct', 'return_pct', 'Return', 'PnL']:
        if col in trades.columns:
            return_col = col
            break
    
    if return_col is None:
        raise ValueError("No return column found in trades DataFrame")
    
    for _, trade in trades.iterrows():
        if 'Pct' in return_col or 'pct' in return_col:
            # Percentage return
            ret = trade[return_col] / 100
        else:
            # Absolute PnL
            ret = trade[return_col] / equity
        
        equity = equity * (1 + ret)
        equity_curve.append(equity)
    
    return pd.Series(equity_curve)


def calculate_drawdown(equity_curve: pd.Series) -> Tuple[pd.Series, float, float, float]:
    """
    Calculate drawdown statistics from equity curve.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, avg_drawdown, max_drawdown_duration)
    """
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown percentage
    drawdown = ((equity_curve - running_max) / running_max) * 100
    
    # Calculate statistics
    max_drawdown = drawdown.min()
    
    # Average drawdown (only negative values)
    negative_dd = drawdown[drawdown < 0]
    avg_drawdown = negative_dd.mean() if len(negative_dd) > 0 else 0
    
    # Maximum drawdown duration
    in_drawdown = drawdown < 0
    drawdown_periods = []
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                drawdown_periods.append(current_duration)
            current_duration = 0
    
    if current_duration > 0:
        drawdown_periods.append(current_duration)
    
    max_duration = max(drawdown_periods) if drawdown_periods else 0
    
    return drawdown, max_drawdown, avg_drawdown, max_duration


def run_monte_carlo_simulation(trades: pd.DataFrame, 
                             num_simulations: int = 1000,
                             initial_capital: float = 10000,
                             confidence_levels: List[float] = [5, 25, 50, 75, 95]) -> Dict:
    """
    Run Monte Carlo simulation by randomizing trade order.
    
    Args:
        trades: DataFrame with trade results
        num_simulations: Number of simulations to run
        initial_capital: Starting capital
        confidence_levels: Percentile levels for statistics
        
    Returns:
        Dictionary with simulation results
    """
    logger = logging.getLogger("TradingBot")
    logger.info(f"Running {num_simulations} Monte Carlo simulations on {len(trades)} trades...")
    
    # Extract returns
    return_col = None
    for col in ['ReturnPct', 'return_pct', 'Return', 'PnL']:
        if col in trades.columns:
            return_col = col
            break
    
    if return_col is None:
        raise ValueError("No return column found in trades DataFrame")
    
    # Convert to decimal returns
    if 'Pct' in return_col or 'pct' in return_col:
        trade_returns = trades[return_col].values / 100
    else:
        # Assume absolute PnL, need to convert to percentage
        # This is approximate without position sizes
        trade_returns = trades[return_col].values / initial_capital
    
    # Storage for results
    max_drawdowns = []
    avg_drawdowns = []
    max_dd_durations = []
    final_returns = []
    final_equities = []
    sharpe_ratios = []
    sortino_ratios = []
    win_rates = []
    
    # Sample equity curves for plotting
    sample_curves = []
    sample_size = min(100, num_simulations)
    
    # Run simulations
    for sim in range(num_simulations):
        # Randomize trade order
        shuffled_indices = np.random.permutation(len(trade_returns))
        shuffled_returns = trade_returns[shuffled_indices]
        
        # Calculate equity curve
        equity = initial_capital
        equity_values = [equity]
        
        for ret in shuffled_returns:
            equity = equity * (1 + ret)
            equity_values.append(equity)
        
        equity_curve = pd.Series(equity_values)
        
        # Store sample curves
        if sim < sample_size:
            sample_curves.append(equity_values)
        
        # Calculate metrics
        _, max_dd, avg_dd, max_duration = calculate_drawdown(equity_curve)
        
        # Final return
        final_return = ((equity_values[-1] / initial_capital) - 1) * 100
        
        # Win rate
        wins = sum(shuffled_returns > 0)
        win_rate = (wins / len(shuffled_returns)) * 100 if len(shuffled_returns) > 0 else 0
        
        # Sharpe ratio (simplified - assuming daily returns)
        if np.std(shuffled_returns) > 0:
            sharpe = (np.mean(shuffled_returns) / np.std(shuffled_returns)) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = shuffled_returns[shuffled_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = (np.mean(shuffled_returns) / np.std(downside_returns)) * np.sqrt(252)
        else:
            sortino = sharpe
        
        # Store results
        max_drawdowns.append(max_dd)
        avg_drawdowns.append(avg_dd)
        max_dd_durations.append(max_duration)
        final_returns.append(final_return)
        final_equities.append(equity_values[-1])
        sharpe_ratios.append(sharpe)
        sortino_ratios.append(sortino)
        win_rates.append(win_rate)
    
    # Calculate statistics
    results = {
        'num_simulations': num_simulations,
        'num_trades': len(trades),
        'initial_capital': initial_capital,
        'sample_curves': sample_curves,
        'metrics': {
            'max_drawdowns': max_drawdowns,
            'avg_drawdowns': avg_drawdowns,
            'max_dd_durations': max_dd_durations,
            'final_returns': final_returns,
            'final_equities': final_equities,
            'sharpe_ratios': sharpe_ratios,
            'sortino_ratios': sortino_ratios,
            'win_rates': win_rates
        },
        'statistics': {}
    }
    
    # Calculate percentile statistics for each metric
    for metric_name, metric_values in results['metrics'].items():
        results['statistics'][metric_name] = {
            'mean': float(np.mean(metric_values)),
            'std': float(np.std(metric_values)),
            'min': float(np.min(metric_values)),
            'max': float(np.max(metric_values)),
            'percentiles': {}
        }
        
        for level in confidence_levels:
            results['statistics'][metric_name]['percentiles'][f'p{level}'] = float(
                np.percentile(metric_values, level)
            )
    
    return results


def create_monte_carlo_report(results: Dict, output_dir: Path, base_name: str):
    """
    Create comprehensive Monte Carlo simulation report with visualizations.
    
    Args:
        results: Monte Carlo simulation results
        output_dir: Directory to save outputs
        base_name: Base name for output files
    """
    logger = logging.getLogger("TradingBot")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Sample Equity Curves
    ax1 = fig.add_subplot(gs[0, :2])
    for curve in results['sample_curves']:
        ax1.plot(curve, alpha=0.3, linewidth=0.5)
    
    # Plot mean curve
    mean_curve = np.mean(results['sample_curves'], axis=0)
    ax1.plot(mean_curve, color='red', linewidth=2, label='Mean')
    
    # Plot percentile bands
    p5_curve = np.percentile(results['sample_curves'], 5, axis=0)
    p95_curve = np.percentile(results['sample_curves'], 95, axis=0)
    ax1.fill_between(range(len(mean_curve)), p5_curve, p95_curve, 
                      alpha=0.2, color='blue', label='5-95% Range')
    
    ax1.set_title('Sample Equity Curves', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Account Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Maximum Drawdown Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(results['metrics']['max_drawdowns'], bins=50, 
             edgecolor='black', alpha=0.7, color='red')
    ax2.axvline(results['statistics']['max_drawdowns']['mean'], 
                color='black', linestyle='--', linewidth=2,
                label=f"Mean: {results['statistics']['max_drawdowns']['mean']:.1f}%")
    ax2.axvline(results['statistics']['max_drawdowns']['percentiles']['p5'],
                color='green', linestyle='--',
                label=f"5%: {results['statistics']['max_drawdowns']['percentiles']['p5']:.1f}%")
    ax2.axvline(results['statistics']['max_drawdowns']['percentiles']['p95'],
                color='orange', linestyle='--',
                label=f"95%: {results['statistics']['max_drawdowns']['percentiles']['p95']:.1f}%")
    ax2.set_title('Maximum Drawdown Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Max Drawdown (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Final Returns Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(results['metrics']['final_returns'], bins=50,
             edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(results['statistics']['final_returns']['mean'],
                color='black', linestyle='--', linewidth=2,
                label=f"Mean: {results['statistics']['final_returns']['mean']:.1f}%")
    ax3.axvline(0, color='red', linestyle='-', linewidth=1)
    ax3.set_title('Final Return Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Final Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Sharpe Ratio Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(results['metrics']['sharpe_ratios'], bins=50,
             edgecolor='black', alpha=0.7, color='blue')
    ax4.axvline(results['statistics']['sharpe_ratios']['mean'],
                color='black', linestyle='--', linewidth=2,
                label=f"Mean: {results['statistics']['sharpe_ratios']['mean']:.2f}")
    ax4.set_title('Sharpe Ratio Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sharpe Ratio')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Risk-Return Scatter
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(results['metrics']['max_drawdowns'],
                         results['metrics']['final_returns'],
                         alpha=0.5, s=10, c=results['metrics']['sharpe_ratios'],
                         cmap='viridis')
    plt.colorbar(scatter, ax=ax5, label='Sharpe Ratio')
    ax5.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Max Drawdown (%)')
    ax5.set_ylabel('Final Return (%)')
    ax5.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax5.grid(True, alpha=0.3)
    
    # 6. Drawdown Duration Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(results['metrics']['max_dd_durations'], bins=30,
             edgecolor='black', alpha=0.7, color='purple')
    ax6.axvline(results['statistics']['max_dd_durations']['mean'],
                color='black', linestyle='--', linewidth=2,
                label=f"Mean: {results['statistics']['max_dd_durations']['mean']:.0f}")
    ax6.set_title('Max Drawdown Duration Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Duration (trades)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Statistics Summary Table
    ax7 = fig.add_subplot(gs[:, 2])
    ax7.axis('off')
    
    # Create summary statistics text
    stats_text = f"""
MONTE CARLO SIMULATION SUMMARY
{'='*35}
Simulations: {results['num_simulations']:,}
Total Trades: {results['num_trades']}
Initial Capital: ${results['initial_capital']:,.0f}

RETURN STATISTICS
{'-'*35}
Mean Return: {results['statistics']['final_returns']['mean']:.1f}%
Std Dev: {results['statistics']['final_returns']['std']:.1f}%
5th Percentile: {results['statistics']['final_returns']['percentiles']['p5']:.1f}%
25th Percentile: {results['statistics']['final_returns']['percentiles']['p25']:.1f}%
Median: {results['statistics']['final_returns']['percentiles']['p50']:.1f}%
75th Percentile: {results['statistics']['final_returns']['percentiles']['p75']:.1f}%
95th Percentile: {results['statistics']['final_returns']['percentiles']['p95']:.1f}%

DRAWDOWN STATISTICS
{'-'*35}
Mean Max DD: {results['statistics']['max_drawdowns']['mean']:.1f}%
Std Dev: {results['statistics']['max_drawdowns']['std']:.1f}%
Best Case (5%): {results['statistics']['max_drawdowns']['percentiles']['p5']:.1f}%
Median: {results['statistics']['max_drawdowns']['percentiles']['p50']:.1f}%
Worst Case (95%): {results['statistics']['max_drawdowns']['percentiles']['p95']:.1f}%

Mean DD Duration: {results['statistics']['max_dd_durations']['mean']:.0f} trades
Max DD Duration: {results['statistics']['max_dd_durations']['max']:.0f} trades

RISK METRICS
{'-'*35}
Mean Sharpe: {results['statistics']['sharpe_ratios']['mean']:.2f}
Mean Sortino: {results['statistics']['sortino_ratios']['mean']:.2f}
Mean Win Rate: {results['statistics']['win_rates']['mean']:.1f}%

RISK ASSESSMENT
{'-'*35}
Prob(Loss): {100 * sum(r < 0 for r in results['metrics']['final_returns']) / len(results['metrics']['final_returns']):.1f}%
Prob(DD > 20%): {100 * sum(abs(d) > 20 for d in results['metrics']['max_drawdowns']) / len(results['metrics']['max_drawdowns']):.1f}%
Prob(DD > 30%): {100 * sum(abs(d) > 30 for d in results['metrics']['max_drawdowns']) / len(results['metrics']['max_drawdowns']):.1f}%
"""
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add title
    fig.suptitle(f'Monte Carlo Simulation Results - {base_name}', 
                 fontsize=16, fontweight='bold')
    
    # Save plot
    plot_path = output_dir / f"{base_name}_monte_carlo.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Monte Carlo plot to: {plot_path}")
    
    # Save detailed results to JSON
    results_json = {
        'num_simulations': results['num_simulations'],
        'num_trades': results['num_trades'],
        'initial_capital': results['initial_capital'],
        'statistics': results['statistics'],
        'timestamp': datetime.now().isoformat()
    }
    
    json_path = output_dir / f"{base_name}_monte_carlo_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Saved Monte Carlo results to: {json_path}")
    
    # Save detailed metrics to CSV
    metrics_df = pd.DataFrame(results['metrics'])
    csv_path = output_dir / f"{base_name}_monte_carlo_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved detailed metrics to: {csv_path}")


def main():
    """Main function for Monte Carlo simulation."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation on backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on most recent backtest
  python scripts/run_monte_carlo.py
  
  # Run on specific backtest directory
  python scripts/run_monte_carlo.py --backtest-dir output/backtests/sma_BTC_USDT_20240101
  
  # Run with more simulations
  python scripts/run_monte_carlo.py --simulations 10000
  
  # Run on trades CSV file directly
  python scripts/run_monte_carlo.py --trades-file output/backtests/my_trades.csv
        """
    )
    
    parser.add_argument("--backtest-dir", type=str,
                       help="Path to backtest output directory")
    parser.add_argument("--trades-file", type=str,
                       help="Path to trades CSV file")
    parser.add_argument("--simulations", type=int, default=1000,
                       help="Number of Monte Carlo simulations (default: 1000)")
    parser.add_argument("--initial-capital", type=float, default=10000,
                       help="Initial capital (default: 10000)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(log_level)
    
    # Load trades
    if args.trades_file:
        logger.info(f"Loading trades from: {args.trades_file}")
        trades = pd.read_csv(args.trades_file)
        base_name = Path(args.trades_file).stem.replace('_trades', '')
    elif args.backtest_dir:
        logger.info(f"Loading trades from backtest directory: {args.backtest_dir}")
        trades = load_trades_from_backtest(args.backtest_dir)
        if trades is None:
            logger.error("Could not find trades in backtest directory")
            return
        base_name = Path(args.backtest_dir).name
    else:
        # Find most recent backtest
        backtest_dir = Path("output/backtests")
        if not backtest_dir.exists():
            logger.error("No backtest output directory found")
            return
        
        # Find most recent trades CSV file
        trades_files = list(backtest_dir.glob("*_trades.csv"))
        if not trades_files:
            logger.error("No backtest results found")
            return
        
        recent_trades_file = max(trades_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using most recent backtest trades: {recent_trades_file}")
        
        trades = pd.read_csv(recent_trades_file)
        base_name = recent_trades_file.stem.replace('_trades', '')
    
    logger.info(f"Loaded {len(trades)} trades")
    
    if len(trades) == 0:
        logger.error("No trades found to simulate")
        return
    
    # Run Monte Carlo simulation
    results = run_monte_carlo_simulation(
        trades,
        num_simulations=args.simulations,
        initial_capital=args.initial_capital
    )
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output/monte_carlo/{base_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report
    create_monte_carlo_report(results, output_dir, base_name)
    
    # Print summary
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION COMPLETE")
    print("="*60)
    print(f"Simulations: {results['num_simulations']:,}")
    print(f"Trades: {results['num_trades']}")
    print(f"\nRETURN STATISTICS:")
    print(f"  Mean: {results['statistics']['final_returns']['mean']:.1f}%")
    print(f"  5th Percentile: {results['statistics']['final_returns']['percentiles']['p5']:.1f}%")
    print(f"  Median: {results['statistics']['final_returns']['percentiles']['p50']:.1f}%")
    print(f"  95th Percentile: {results['statistics']['final_returns']['percentiles']['p95']:.1f}%")
    print(f"\nDRAWDOWN STATISTICS:")
    print(f"  Mean Max DD: {results['statistics']['max_drawdowns']['mean']:.1f}%")
    print(f"  5th Percentile (Best): {results['statistics']['max_drawdowns']['percentiles']['p5']:.1f}%")
    print(f"  Median: {results['statistics']['max_drawdowns']['percentiles']['p50']:.1f}%")
    print(f"  95th Percentile (Worst): {results['statistics']['max_drawdowns']['percentiles']['p95']:.1f}%")
    print(f"\nRISK METRICS:")
    print(f"  Mean Sharpe: {results['statistics']['sharpe_ratios']['mean']:.2f}")
    print(f"  Mean Sortino: {results['statistics']['sortino_ratios']['mean']:.2f}")
    print(f"  Probability of Loss: {100 * sum(r < 0 for r in results['metrics']['final_returns']) / len(results['metrics']['final_returns']):.1f}%")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()