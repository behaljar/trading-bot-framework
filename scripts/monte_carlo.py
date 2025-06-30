"""
Monte Carlo simulation for trading strategy drawdown analysis
Runs backtest, extracts trades, and performs Monte Carlo simulation with randomized trade order
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import List, Dict, Tuple
import argparse
from run_backtest import run_backtest
from utils.logger import setup_logger

def calculate_drawdown_series(returns: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Calculate drawdown series and statistics from returns
    
    Returns:
        - drawdown_series: Series of drawdowns over time
        - max_drawdown: Maximum drawdown percentage
        - avg_drawdown: Average drawdown percentage
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max * 100
    
    # Calculate statistics
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
    
    return drawdown, max_drawdown, avg_drawdown

def extract_trades_from_backtest(backtest_results_path: str) -> pd.DataFrame:
    """Extract trades from backtest results"""
    trades_path = backtest_results_path.replace('_results.json', '_trades.csv')
    
    if os.path.exists(trades_path):
        trades = pd.read_csv(trades_path)
        return trades
    else:
        raise FileNotFoundError(f"Trades file not found: {trades_path}")

def run_monte_carlo_simulation(trades: pd.DataFrame, num_simulations: int = 1000, 
                             initial_capital: float = 10000) -> Dict:
    """
    Run Monte Carlo simulation by randomizing trade order
    
    Returns dictionary with simulation results including drawdown statistics
    """
    logger = logging.getLogger("TradingBot")
    logger.info(f"Running {num_simulations} Monte Carlo simulations...")
    
    # Extract trade returns
    trade_returns = trades['ReturnPct'].values / 100  # Convert percentage to decimal
    num_trades = len(trade_returns)
    
    # Storage for simulation results
    max_drawdowns = []
    avg_drawdowns = []
    final_returns = []
    equity_curves = []
    
    # Run simulations
    for sim in range(num_simulations):
        # Randomize trade order
        shuffled_returns = np.random.permutation(trade_returns)
        
        # Calculate equity curve
        equity = initial_capital
        equity_curve = [equity]
        
        for ret in shuffled_returns:
            equity = equity * (1 + ret)
            equity_curve.append(equity)
        
        # Convert to returns series for drawdown calculation
        equity_series = pd.Series(equity_curve)
        returns_series = equity_series.pct_change().fillna(0)
        
        # Calculate drawdowns
        drawdown_series, max_dd, avg_dd = calculate_drawdown_series(returns_series)
        
        # Store results
        max_drawdowns.append(max_dd)
        avg_drawdowns.append(avg_dd)
        final_returns.append((equity_curve[-1] / initial_capital - 1) * 100)
        equity_curves.append(equity_curve)
    
    # Calculate statistics
    results = {
        'num_simulations': num_simulations,
        'num_trades': num_trades,
        'original_return': (trades['ReturnPct'].sum()),  # Approximate - compound effects ignored
        'max_drawdowns': max_drawdowns,
        'avg_drawdowns': avg_drawdowns,
        'final_returns': final_returns,
        'equity_curves': equity_curves,
        'stats': {
            'max_dd_mean': np.mean(max_drawdowns),
            'max_dd_std': np.std(max_drawdowns),
            'max_dd_worst': np.min(max_drawdowns),
            'max_dd_best': np.max(max_drawdowns),
            'max_dd_percentiles': {
                '5%': np.percentile(max_drawdowns, 5),
                '25%': np.percentile(max_drawdowns, 25),
                '50%': np.percentile(max_drawdowns, 50),
                '75%': np.percentile(max_drawdowns, 75),
                '95%': np.percentile(max_drawdowns, 95)
            },
            'avg_dd_mean': np.mean(avg_drawdowns),
            'avg_dd_std': np.std(avg_drawdowns),
            'final_return_mean': np.mean(final_returns),
            'final_return_std': np.std(final_returns),
            'final_return_percentiles': {
                '5%': np.percentile(final_returns, 5),
                '25%': np.percentile(final_returns, 25),
                '50%': np.percentile(final_returns, 50),
                '75%': np.percentile(final_returns, 75),
                '95%': np.percentile(final_returns, 95)
            }
        }
    }
    
    return results

def plot_monte_carlo_results(results: Dict, output_path: str):
    """Create visualization of Monte Carlo simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Monte Carlo Simulation Results - Trade Randomization', fontsize=16)
    
    # Plot 1: Equity curves (sample of simulations)
    ax1 = axes[0, 0]
    num_curves_to_plot = min(100, len(results['equity_curves']))
    for i in range(num_curves_to_plot):
        ax1.plot(results['equity_curves'][i], alpha=0.1, color='blue')
    
    # Plot average equity curve
    avg_equity = np.mean(results['equity_curves'], axis=0)
    ax1.plot(avg_equity, color='red', linewidth=2, label='Average')
    ax1.set_title('Sample Equity Curves (100 simulations)')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Account Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Maximum drawdown distribution
    ax2 = axes[0, 1]
    ax2.hist(results['max_drawdowns'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(results['stats']['max_dd_mean'], color='red', linestyle='--', 
                label=f"Mean: {results['stats']['max_dd_mean']:.2f}%")
    ax2.axvline(results['stats']['max_dd_percentiles']['5%'], color='orange', linestyle='--',
                label=f"5th %ile: {results['stats']['max_dd_percentiles']['5%']:.2f}%")
    ax2.axvline(results['stats']['max_dd_percentiles']['95%'], color='orange', linestyle='--',
                label=f"95th %ile: {results['stats']['max_dd_percentiles']['95%']:.2f}%")
    ax2.set_title('Maximum Drawdown Distribution')
    ax2.set_xlabel('Maximum Drawdown (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final returns distribution
    ax3 = axes[1, 0]
    ax3.hist(results['final_returns'], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(results['stats']['final_return_mean'], color='red', linestyle='--',
                label=f"Mean: {results['stats']['final_return_mean']:.2f}%")
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Final Return Distribution')
    ax3.set_xlabel('Final Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Simulation Statistics ({results['num_simulations']} simulations, {results['num_trades']} trades)
    
    Maximum Drawdown:
    - Mean: {results['stats']['max_dd_mean']:.2f}%
    - Std Dev: {results['stats']['max_dd_std']:.2f}%
    - Worst Case: {results['stats']['max_dd_worst']:.2f}%
    - Best Case: {results['stats']['max_dd_best']:.2f}%
    
    Percentiles:
    - 5th: {results['stats']['max_dd_percentiles']['5%']:.2f}%
    - 25th: {results['stats']['max_dd_percentiles']['25%']:.2f}%
    - 50th: {results['stats']['max_dd_percentiles']['50%']:.2f}%
    - 75th: {results['stats']['max_dd_percentiles']['75%']:.2f}%
    - 95th: {results['stats']['max_dd_percentiles']['95%']:.2f}%
    
    Final Returns:
    - Mean: {results['stats']['final_return_mean']:.2f}%
    - Std Dev: {results['stats']['final_return_std']:.2f}%
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulation on backtest trades')
    parser.add_argument('--strategy', help='Strategy name (sma, rsi)')
    parser.add_argument('--symbol', help='Symbol to test')
    parser.add_argument('--start', default='2024-06-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-source', help='Data source (yahoo, ccxt, csv, csv_processed)')
    parser.add_argument('--timeframe', help='Timeframe (1m, 5m, 15m, 30m, 1h, 1d)')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of Monte Carlo simulations')
    parser.add_argument('--no-sl-tp', action='store_true', help='Disable stop loss and take profit')
    parser.add_argument('--use-existing', help='Path to existing backtest results JSON file')
    
    args = parser.parse_args()
    
    logger = logging.getLogger("TradingBot")
    
    # Step 1: Run backtest or use existing results
    if args.use_existing:
        logger.info(f"Using existing backtest results: {args.use_existing}")
        with open(args.use_existing, 'r') as f:
            backtest_results = json.load(f)
        results_path = args.use_existing
    else:
        logger.info("Running backtest...")
        from config.settings import load_config
        config = load_config()
        
        strategy = args.strategy or config.strategy_name
        symbol = args.symbol or (config.symbols[0] if config.symbols else "AAPL")
        
        # Run backtest
        backtest_results = run_backtest(
            strategy, symbol, args.start, args.end, 
            args.data_source, args.timeframe, False, args.no_sl_tp
        )
        
        # Find the most recent results file
        base_output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "output" / "backtests"
        results_files = []
        # Search through all subdirectories
        for subdir in base_output_dir.glob("*"):
            if subdir.is_dir():
                results_files.extend(subdir.glob(f"{strategy}_{symbol}_*_results.json"))
        results_path = str(max(results_files, key=os.path.getctime))
    
    # Step 2: Extract trades
    try:
        trades = extract_trades_from_backtest(results_path)
        logger.info(f"Extracted {len(trades)} trades from backtest")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if len(trades) == 0:
        logger.error("No trades found in backtest results")
        return
    
    # Step 3: Run Monte Carlo simulation
    mc_results = run_monte_carlo_simulation(trades, num_simulations=args.simulations)
    
    # Step 4: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"monte_carlo_{strategy}_{symbol}_{timestamp}"
    output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "output" / "monte_carlo" / run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract strategy and symbol from results path if using existing
    if args.use_existing:
        base_name = Path(results_path).stem.replace('_results', '')
    else:
        base_name = f"{strategy}_{symbol}_{timestamp}"
    
    # Save Monte Carlo results
    mc_results_file = output_dir / f"{base_name}_monte_carlo.json"
    mc_results_save = {
        'num_simulations': mc_results['num_simulations'],
        'num_trades': mc_results['num_trades'],
        'original_return': mc_results['original_return'],
        'stats': mc_results['stats'],
        'backtest_results_file': results_path
    }
    
    with open(mc_results_file, 'w') as f:
        json.dump(mc_results_save, f, indent=2)
    
    # Create visualization
    plot_file = output_dir / f"{base_name}_monte_carlo_plot.png"
    plot_monte_carlo_results(mc_results, str(plot_file))
    
    # Print summary
    logger.info("\n=== MONTE CARLO SIMULATION RESULTS ===")
    logger.info(f"Simulations: {mc_results['num_simulations']}")
    logger.info(f"Trades: {mc_results['num_trades']}")
    logger.info(f"\nMaximum Drawdown Statistics:")
    logger.info(f"  Mean: {mc_results['stats']['max_dd_mean']:.2f}%")
    logger.info(f"  Worst Case (5%): {mc_results['stats']['max_dd_percentiles']['5%']:.2f}%")
    logger.info(f"  Median: {mc_results['stats']['max_dd_percentiles']['50%']:.2f}%")
    logger.info(f"  Best Case (95%): {mc_results['stats']['max_dd_percentiles']['95%']:.2f}%")
    logger.info(f"\nFinal Return Statistics:")
    logger.info(f"  Mean: {mc_results['stats']['final_return_mean']:.2f}%")
    logger.info(f"  5th Percentile: {mc_results['stats']['final_return_percentiles']['5%']:.2f}%")
    logger.info(f"  95th Percentile: {mc_results['stats']['final_return_percentiles']['95%']:.2f}%")
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - {mc_results_file}")
    logger.info(f"  - {plot_file}")

if __name__ == "__main__":
    main()