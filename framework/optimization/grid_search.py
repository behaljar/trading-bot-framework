#!/usr/bin/env python3
"""
Grid Search Optimizer

Direct backtest execution with parallel processing for optimal performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Type, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import sys
import os

from backtesting import Backtest
from backtesting.lib import FractionalBacktest
from framework.strategies.base_strategy import BaseStrategy
from framework.backtesting.strategy_wrapper import StrategyWrapper
from framework.risk.base_risk_manager import BaseRiskManager


def generate_parameter_combinations(param_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from configuration.
    
    Args:
        param_config: Dictionary defining parameter ranges
        
    Returns:
        List of parameter dictionaries
    """
    param_names = []
    param_values = []
    
    for name, config in param_config.items():
        param_names.append(name)
        
        if 'values' in config:
            # Direct list of values
            values = config['values']
        elif 'choices' in config:
            # Choice parameter
            values = config['choices']
        else:
            # Range parameter
            min_val = config['min']
            max_val = config['max']
            step = config.get('step', 1)
            param_type = config.get('type', 'float')
            
            if param_type == 'int':
                values = list(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                # Generate float range
                values = []
                current = min_val
                while current <= max_val:
                    values.append(round(current, 6))
                    current += step
        
        param_values.append(values)
    
    # Generate all combinations
    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
    
    return combinations


def run_single_backtest(params: Dict[str, Any], 
                        strategy_class: Type,
                        data: pd.DataFrame,
                        initial_capital: float,
                        commission: float,
                        margin: float,
                        risk_manager: Any) -> Dict[str, Any]:
    """
    Run a single backtest with given parameters.
    
    Returns:
        Dictionary with parameters and performance metrics
    """
    try:
        # Create strategy wrapper
        class TestStrategy(StrategyWrapper):
            framework_strategy_class = strategy_class
            strategy_params = params
            debug = False
        
        # Set risk manager as class attribute
        TestStrategy.risk_manager = risk_manager
        
        # Run backtest
        bt = FractionalBacktest(
            data=data,
            strategy=TestStrategy,
            cash=initial_capital,
            commission=commission,
            margin=margin,
            exclusive_orders=True
        )
        
        results = bt.run()
        
        # Extract key metrics
        metrics = {
            'params': params,
            'return_pct': results.get('Return [%]', 0),
            'sharpe_ratio': results.get('Sharpe Ratio', 0) if results.get('Sharpe Ratio') is not None else 0,
            'max_drawdown': abs(results.get('Max. Drawdown [%]', 0)),
            'win_rate': results.get('Win Rate [%]', 0) if results.get('Win Rate [%]') is not None else 0,
            'num_trades': results.get('# Trades', 0),
            'exposure_time': results.get('Exposure Time [%]', 0),
            'profit_factor': results.get('Profit Factor', 0) if results.get('Profit Factor') is not None else 0,
            'avg_trade': results.get('Avg. Trade [%]', 0) if results.get('Avg. Trade [%]') is not None else 0,
            'best_trade': results.get('Best Trade [%]', 0) if results.get('Best Trade [%]') is not None else 0,
            'worst_trade': results.get('Worst Trade [%]', 0) if results.get('Worst Trade [%]') is not None else 0,
            'calmar_ratio': results.get('Calmar Ratio', 0) if results.get('Calmar Ratio') is not None else 0,
            'sortino_ratio': results.get('Sortino Ratio', 0) if results.get('Sortino Ratio') is not None else 0,
        }
        
        return metrics
        
    except Exception as e:
        # Print error for debugging
        print(f"Error in backtest with params {params}: {e}")
        # Return neutral result (0% return when error/no trades)
        return {
            'params': params,
            'return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'num_trades': 0,
            'exposure_time': 0,
            'profit_factor': 0,
            'avg_trade': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'calmar_ratio': 0,
            'sortino_ratio': 0,
            'error': str(e)
        }


# Global variables for multiprocessing
_worker_strategy_class = None
_worker_data = None
_worker_initial_capital = None
_worker_commission = None
_worker_margin = None
_worker_risk_manager = None


def worker_init(strategy_class, data, initial_capital, commission, margin, risk_manager):
    """Initialize worker process with shared data."""
    global _worker_strategy_class, _worker_data, _worker_initial_capital
    global _worker_commission, _worker_margin, _worker_risk_manager
    
    _worker_strategy_class = strategy_class
    _worker_data = data
    _worker_initial_capital = initial_capital
    _worker_commission = commission
    _worker_margin = margin
    _worker_risk_manager = risk_manager


def worker_run_backtest(params):
    """Worker function that uses global variables with error handling."""
    try:
        return run_single_backtest(
            params,
            _worker_strategy_class,
            _worker_data,
            _worker_initial_capital,
            _worker_commission,
            _worker_margin,
            _worker_risk_manager
        )
    except Exception as e:
        import traceback
        # Return neutral result (0% return when error/no trades)
        return {
            'params': params,
            'return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'num_trades': 0,
            'exposure_time': 0,
            'profit_factor': 0,
            'avg_trade': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'calmar_ratio': 0,
            'sortino_ratio': 0,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


class GridSearchOptimizer:
    """
    Grid search optimizer using direct bt.run() calls with parallel processing.
    
    Provides full control over parameter optimization with multiprocessing support.
    """
    
    def __init__(self,
                 strategy_class: Type[BaseStrategy],
                 parameter_config: Dict[str, Any],
                 data: pd.DataFrame,
                 initial_capital: float = 10000,
                 commission: float = 0.0005,
                 margin: float = 0.01,
                 risk_manager: Optional[BaseRiskManager] = None,
                 n_jobs: int = 4,
                 debug: bool = False):
        """
        Initialize the optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_config: Parameter configuration dictionary
            data: OHLCV data DataFrame
            initial_capital: Initial capital for backtesting
            commission: Commission rate
            margin: Margin requirement (0.01 = 100x leverage)
            risk_manager: Risk manager instance
            n_jobs: Number of parallel jobs (-1 = all cores)
            debug: Enable debug output
        """
        self.strategy_class = strategy_class
        self.parameter_config = parameter_config
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.margin = margin
        self.risk_manager = risk_manager
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.debug = debug
        
        # Generate parameter combinations
        self.combinations = generate_parameter_combinations(parameter_config)
    
    def _print_progress_bar(self, completed: int, total: int, start_time: float, 
                           bar_length: int = 50):
        """Print a progress bar with time estimation."""
        percent = completed / total
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate time estimates
        elapsed = time.time() - start_time
        if completed > 0:
            avg_time_per_item = elapsed / completed
            remaining_items = total - completed
            eta_seconds = avg_time_per_item * remaining_items
            
            # Format time strings
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            # Build progress string
            progress_str = f"\r|{bar}| {completed}/{total} ({percent*100:.1f}%) "
            progress_str += f"| Elapsed: {elapsed_str} | ETA: {eta_str}"
        else:
            progress_str = f"\r|{bar}| {completed}/{total} ({percent*100:.1f}%) | Starting..."
        
        # Print and flush
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        
        if completed == total:
            print()  # New line when complete
    
    def optimize(self, metric: str = 'return_pct', maximize: bool = True) -> pd.DataFrame:
        """
        Run optimization and return results DataFrame.
        
        Args:
            metric: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False)
            
        Returns:
            DataFrame with all optimization results
        """
        total = len(self.combinations)
        print(f"\nStarting optimization of {total} parameter combinations")
        print(f"Using {self.n_jobs} parallel workers")
        print(f"Optimizing for: {metric} ({'maximize' if maximize else 'minimize'})")
        print("-" * 80)
        
        results = []
        start_time = time.time()
        
        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=worker_init,
            initargs=(self.strategy_class, self.data, self.initial_capital,
                     self.commission, self.margin, self.risk_manager)
        ) as executor:
            
            # Submit all tasks
            futures = {executor.submit(worker_run_backtest, params): i 
                      for i, params in enumerate(self.combinations)}
            
            # Process results with progress bar
            completed = 0
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    self._print_progress_bar(completed, total, start_time)
            except KeyboardInterrupt:
                print(f"\n\n🛑 Optimization interrupted after {completed}/{total} combinations")
                print("🔄 Cancelling remaining tasks...")
                # Cancel remaining futures
                for future in futures:
                    future.cancel()
                # Shutdown executor immediately
                executor.shutdown(wait=False, cancel_futures=True)
                print("✅ Tasks cancelled. Processing completed results...")
        
        # Print completion summary
        total_time = time.time() - start_time
        avg_time = total_time / total if total > 0 else 0
        print(f"\nOptimization completed in {timedelta(seconds=int(total_time))}")
        print(f"Average time per backtest: {avg_time:.2f} seconds")
        print("-" * 80)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Expand params dict into columns
        param_df = pd.DataFrame(results_df['params'].tolist())
        results_df = pd.concat([param_df, results_df.drop('params', axis=1)], axis=1)
        
        # Sort by metric
        results_df = results_df.sort_values(metric, ascending=not maximize)
        
        return results_df
    
    def create_visualization(self, results_df: pd.DataFrame, output_path: str, 
                           metric: str = 'win_rate'):
        """
        Create focused parameter optimization visualization.
        
        Args:
            results_df: DataFrame with optimization results
            output_path: Base path for saving files
            metric: Metric to visualize
        """
        # Sort by metric
        results_df = results_df.sort_values(metric, ascending=False)
        
        # Get parameter columns
        param_cols = [col for col in results_df.columns if col not in 
                      ['return_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                       'num_trades', 'exposure_time', 'profit_factor', 'avg_trade',
                       'best_trade', 'worst_trade', 'calmar_ratio', 'sortino_ratio', 'error']]
        
        # Create figure with focused parameter analysis
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Top parameter combinations (larger, more prominent)
        ax1 = plt.subplot(2, 2, 1)
        top_n = min(20, len(results_df))
        top_results = results_df.head(top_n)
        
        # Create parameter strings for y-axis labels
        labels = []
        for _, row in top_results.iterrows():
            label_parts = []
            for col in param_cols:
                val = row[col]
                if isinstance(val, float):
                    label_parts.append(f"{col[:8]}:{val:.2f}")
                else:
                    label_parts.append(f"{col[:8]}:{val}")
            labels.append(' | '.join(label_parts))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))
        bars = ax1.barh(range(top_n), top_results[metric].values, color=colors)
        ax1.set_yticks(range(top_n))
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_title(f'Top {top_n} Parameter Combinations', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_results[metric].values)):
            if not pd.isna(val):
                ax1.text(val + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.01, 
                        bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                        va='center', ha='left', fontsize=8, fontweight='bold')
        
        # 2. Parameter importance
        ax2 = plt.subplot(2, 2, 2)
        if len(param_cols) > 0 and len(results_df) > 3:
            # Calculate correlation of each parameter with the target metric
            importances = {}
            for col in param_cols:
                try:
                    numeric_vals = pd.to_numeric(results_df[col], errors='coerce')
                    if numeric_vals.notna().sum() > 1:
                        corr = numeric_vals.corr(results_df[metric])
                        if not pd.isna(corr):
                            importances[col] = abs(corr)
                except:
                    pass
            
            if importances:
                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                params_names = [x[0].replace('_', '\n') for x in sorted_imp]
                params_values = [x[1] for x in sorted_imp]
                
                bars = ax2.bar(range(len(params_names)), params_values, 
                              color='steelblue', alpha=0.7, edgecolor='black')
                ax2.set_xticks(range(len(params_names)))
                ax2.set_xticklabels(params_names, fontsize=10, ha='center')
                ax2.set_ylabel(f'Impact on {metric.replace("_", " ").title()}')
                ax2.set_title('Parameter Importance', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, val in zip(bars, params_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, val + max(params_values) * 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Individual parameter impact plots
        ax3 = plt.subplot(2, 2, 3)
        if len(param_cols) > 0:
            # Show how each parameter value affects the metric
            param_to_plot = param_cols[0]  # Use first parameter
            param_impact = results_df.groupby(param_to_plot)[metric].agg(['mean', 'std', 'count']).reset_index()
            param_impact = param_impact[param_impact['count'] > 0].sort_values(param_to_plot)
            
            if len(param_impact) > 1:
                ax3.errorbar(param_impact[param_to_plot], param_impact['mean'], 
                            yerr=param_impact['std'], fmt='o-', capsize=5, 
                            color='darkgreen', linewidth=2, markersize=8)
                ax3.set_xlabel(param_to_plot.replace('_', ' ').title())
                ax3.set_ylabel(f'Average {metric.replace("_", " ").title()}')
                ax3.set_title(f'{param_to_plot.replace("_", " ").title()} Impact', 
                             fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Add count labels
                for _, row in param_impact.iterrows():
                    ax3.text(row[param_to_plot], row['mean'], f'n={int(row["count"])}', 
                            ha='center', va='bottom', fontsize=8)
        
        # 4. Best vs worst comparison
        ax4 = plt.subplot(2, 2, 4)
        if len(results_df) >= 6:
            # Compare top 3 vs bottom 3 parameter sets
            top_3 = results_df.head(3)
            bottom_3 = results_df.tail(3)
            
            # Create comparison metrics
            comparison_metrics = ['return_pct', 'win_rate', 'profit_factor', 'max_drawdown']
            comparison_metrics = [m for m in comparison_metrics if m in results_df.columns]
            
            if comparison_metrics:
                x_pos = np.arange(len(comparison_metrics))
                width = 0.35
                
                top_values = [top_3[m].mean() for m in comparison_metrics]
                bottom_values = [bottom_3[m].mean() for m in comparison_metrics]
                
                bars1 = ax4.bar(x_pos - width/2, top_values, width, 
                               label='Top 3', color='green', alpha=0.7)
                bars2 = ax4.bar(x_pos + width/2, bottom_values, width,
                               label='Bottom 3', color='red', alpha=0.7)
                
                ax4.set_xlabel('Metrics')
                ax4.set_ylabel('Average Value')
                ax4.set_title('Best vs Worst Parameters', fontsize=14, fontweight='bold')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels([m.replace('_', ' ').title() for m in comparison_metrics])
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if not pd.isna(height):
                            ax4.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'Parameter Optimization Analysis - {metric.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_path = f"{output_path}_analysis_{metric}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Parameter analysis saved to: {fig_path}")
        
        plt.close()
        
        # Create parameter interaction heatmaps
        if len(param_cols) >= 2:
            self._create_parameter_heatmaps(results_df, param_cols, metric, output_path)
    
    def _create_parameter_heatmaps(self, results_df: pd.DataFrame, param_cols: List[str], 
                                  metric: str, output_path: str):
        """Create parameter interaction heatmaps."""
        
        # Create heatmap for the first two most important parameters
        if len(param_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Pivot data for heatmap
            pivot = results_df.pivot_table(
                index=param_cols[1], 
                columns=param_cols[0], 
                values=metric,
                aggfunc='mean'
            )
            
            # Create heatmap with better formatting
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                        center=pivot.mean().mean(), ax=ax, 
                        cbar_kws={'label': metric.replace('_', ' ').title()},
                        square=True, linewidths=0.5)
            
            ax.set_xlabel(param_cols[0].replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(param_cols[1].replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'Parameter Interaction: {metric.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            heatmap_path = f"{output_path}_heatmap_{metric}.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"Parameter interaction heatmap saved to: {heatmap_path}")
            plt.close()
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Save optimization results to CSV and JSON.
        
        Args:
            results_df: DataFrame with optimization results
            output_path: Base path for saving files
        """
        # Save CSV with all results
        csv_path = f"{output_path}_all_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"All results saved to: {csv_path}")
        
        # Get parameter columns
        param_cols = [col for col in results_df.columns if col not in 
                      ['return_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                       'num_trades', 'exposure_time', 'profit_factor', 'avg_trade',
                       'best_trade', 'worst_trade', 'calmar_ratio', 'sortino_ratio', 'error']]
        
        # Save best parameters as JSON
        best_row = results_df.iloc[0]
        best_params = {}
        for col in param_cols:
            val = best_row[col]
            # Convert numpy types to Python types for JSON serialization
            if hasattr(val, 'item'):
                best_params[col] = val.item()
            else:
                best_params[col] = val
        
        best_results = {
            'best_params': best_params,
            'best_metrics': {
                'return_pct': float(best_row['return_pct']),
                'sharpe_ratio': float(best_row['sharpe_ratio']),
                'win_rate': float(best_row['win_rate']),
                'max_drawdown': float(best_row['max_drawdown']),
                'profit_factor': float(best_row['profit_factor']),
                'num_trades': int(best_row['num_trades'])
            },
            'optimization_info': {
                'total_combinations': len(results_df),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        json_path = f"{output_path}_best_params.json"
        with open(json_path, 'w') as f:
            json.dump(best_results, f, indent=2)
        print(f"Best parameters saved to: {json_path}")