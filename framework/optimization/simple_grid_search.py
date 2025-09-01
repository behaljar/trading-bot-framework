#!/usr/bin/env python3
"""
Simple Grid Search implementation using backtesting.py's built-in optimization.

This is a simpler approach that directly uses backtesting.py's optimize method
without complex wrapper classes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import json
from datetime import datetime
import logging
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import FractionalBacktest
except ImportError:
    raise ImportError("backtesting library not found. Install it with: uv add backtesting")

from framework.strategies.base_strategy import BaseStrategy
from framework.risk.base_risk_manager import BaseRiskManager
from .parameter_space import ParameterSpace


class SimpleGridSearchOptimizer:
    """
    Simple grid search optimizer that directly uses backtesting.py's optimization.
    
    This version creates strategy classes on-the-fly for each parameter combination,
    avoiding complex pickling issues with multiprocessing.
    """
    
    def __init__(self,
                 strategy_class: Type[BaseStrategy],
                 parameter_space: ParameterSpace,
                 data: pd.DataFrame,
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 margin: float = 0.01,
                 use_fractional: bool = True,
                 risk_manager: Optional[BaseRiskManager] = None,
                 n_jobs: int = 1,  # Number of parallel jobs
                 debug: bool = False):
        """
        Initialize grid search optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Parameter search space
            data: Historical price data
            initial_capital: Starting capital
            commission: Commission rate
            margin: Margin requirement
            use_fractional: Use fractional shares
            risk_manager: Risk manager (optional)
            n_jobs: Number of parallel jobs (1 = sequential)
            debug: Enable debug output
        """
        self.strategy_class = strategy_class
        self.parameter_space = parameter_space
        self.data = self._prepare_data(data)
        self.initial_capital = initial_capital
        self.commission = commission
        self.margin = margin
        self.use_fractional = use_fractional
        self.risk_manager = risk_manager
        self.n_jobs = n_jobs
        self.debug = debug
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting.py format."""
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        prepared_data = data.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in prepared_data.columns and new_name not in prepared_data.columns:
                prepared_data = prepared_data.rename(columns={old_name: new_name})
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in prepared_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if not isinstance(prepared_data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
        
        return prepared_data
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Returns:
            Dictionary with optimization results
        """
        # Get all parameter combinations
        combinations = self.parameter_space.get_grid_combinations()
        total_combinations = len(combinations)
        
        self.logger.info(f"Starting grid search with {total_combinations} combinations")
        
        if total_combinations == 0:
            raise ValueError("No parameter combinations to test")
        
        # Run backtests for each combination
        results = []
        best_result = None
        best_return = float('-inf')
        
        for i, params in enumerate(combinations):
            self.logger.debug(f"Testing combination {i+1}/{total_combinations}: {params}")
            
            # Run single backtest
            result = self._run_single_backtest(params)
            
            if result is not None:
                results.append(result)
                
                # Track best result
                if result['stats']['Return [%]'] > best_return:
                    best_return = result['stats']['Return [%]']
                    best_result = result
        
        if best_result is None:
            raise ValueError("No valid results obtained from optimization")
        
        # Format final results
        optimization_results = {
            'best_params': best_result['params'],
            'best_stats': best_result['stats'],
            'metric_value': best_return,
            'total_combinations': total_combinations,
            'all_results': results
        }
        
        # Log results
        self._log_results(optimization_results)
        
        return optimization_results
    
    def _run_single_backtest(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run a single backtest with given parameters."""
        try:
            # Create strategy wrapper for these specific parameters
            from framework.backtesting.strategy_wrapper import StrategyWrapper
            
            class TestStrategy(StrategyWrapper):
                framework_strategy_class = self.strategy_class
                strategy_params = params
                risk_manager = self.risk_manager
                debug = self.debug
            
            # Choose backtest class
            BacktestClass = FractionalBacktest if self.use_fractional else Backtest
            
            # Run backtest
            bt = BacktestClass(
                data=self.data,
                strategy=TestStrategy,
                cash=self.initial_capital,
                commission=self.commission,
                margin=self.margin,
                exclusive_orders=True,
                hedging=False,
                trade_on_close=False
            )
            
            stats = bt.run()
            
            # Extract key metrics
            result = {
                'params': params,
                'stats': self._format_stats(stats)
            }
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate params {params}: {e}")
            return None
    
    def _format_stats(self, stats: pd.Series) -> Dict[str, Any]:
        """Format statistics for display."""
        formatted = {}
        
        # Key metrics to extract
        metrics = [
            'Return [%]',
            'Buy & Hold Return [%]',
            'Max. Drawdown [%]',
            'Sharpe Ratio',
            'Win Rate [%]',
            '# Trades',
            'Avg. Trade [%]',
            'Profit Factor',
            'Expectancy [%]'
        ]
        
        for metric in metrics:
            if metric in stats.index:
                value = stats[metric]
                if pd.notna(value):
                    if isinstance(value, (int, np.integer)):
                        formatted[metric] = int(value)
                    elif isinstance(value, (float, np.floating)):
                        formatted[metric] = round(float(value), 4)
                    else:
                        formatted[metric] = value
                else:
                    formatted[metric] = None
        
        return formatted
    
    def _log_results(self, results: Dict[str, Any]):
        """Log optimization results."""
        self.logger.info("=" * 60)
        self.logger.info("Grid Search Optimization Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"Total combinations tested: {results['total_combinations']}")
        self.logger.info(f"Best Return [%]: {results['metric_value']:.4f}")
        self.logger.info("Best parameters:")
        for param, value in results['best_params'].items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info("Best performance:")
        for metric, value in results['best_stats'].items():
            if value is not None:
                self.logger.info(f"  {metric}: {value}")
    
    def save_results(self, filepath: str, results: Dict[str, Any] = None, save_heatmap: bool = True):
        """
        Save optimization results to JSON file and optionally generate heatmap.
        
        Args:
            filepath: Path to save JSON results
            results: Optimization results dictionary
            save_heatmap: Whether to generate and save heatmap (default: True)
        """
        if results is None:
            raise ValueError("No results to save")
        
        output = {
            'metadata': {
                'strategy': self.strategy_class.__name__,
                'timestamp': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'margin': self.margin,
                'data_points': len(self.data),
                'data_start': str(self.data.index[0]),
                'data_end': str(self.data.index[-1])
            },
            'parameter_space': {
                name: {
                    'type': param.param_type,
                    'min': param.min_value if param.param_type != 'choice' else None,
                    'max': param.max_value if param.param_type != 'choice' else None,
                    'step': param.step,
                    'choices': param.choices
                }
                for name, param in self.parameter_space.parameters.items()
            },
            'best_params': results['best_params'],
            'best_stats': results['best_stats'],
            'all_results': results.get('all_results', [])
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
        
        # Save CSV with all results
        csv_path = filepath.with_suffix('.csv')
        self._save_csv_results(results, csv_path)
        
        # Generate heatmap(s) if requested
        if save_heatmap:
            if len(self.parameter_space.parameters) == 2:
                # Single heatmap for 2 parameters
                heatmap_path = filepath.with_suffix('.png')
                self._generate_heatmap(results, heatmap_path)
            elif len(self.parameter_space.parameters) > 2:
                # Multiple visualizations for >2 parameters
                self._generate_multi_parameter_plots(results, filepath)
    
    def _generate_heatmap(self, results: Dict[str, Any], filepath: Path):
        """Generate heatmap for 2-parameter optimization results."""
        all_results = results.get('all_results', [])
        if not all_results:
            self.logger.warning("No results available for heatmap generation")
            return
        
        # Get parameter names
        param_names = list(self.parameter_space.parameters.keys())
        if len(param_names) != 2:
            self.logger.warning(f"Heatmap requires exactly 2 parameters, got {len(param_names)}")
            return
        
        param1_name, param2_name = param_names
        
        # Extract parameter values and returns
        param1_values = []
        param2_values = []
        returns = []
        
        for result in all_results:
            params = result['params']
            stats = result['stats']
            param1_values.append(params[param1_name])
            param2_values.append(params[param2_name])
            returns.append(stats.get('Return [%]', 0))
        
        # Create pivot table for heatmap
        df = pd.DataFrame({
            param1_name: param1_values,
            param2_name: param2_values,
            'Return [%]': returns
        })
        
        pivot_table = df.pivot_table(
            values='Return [%]',
            index=param2_name,
            columns=param1_name,
            aggfunc='mean'  # In case of duplicates
        )
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Return [%]'},
            square=False
        )
        
        plt.title(f'{self.strategy_class.__name__} Optimization Heatmap\nReturn [%]')
        plt.xlabel(param1_name)
        plt.ylabel(param2_name)
        
        # Mark best parameters
        best_params = results['best_params']
        best_x = best_params[param1_name]
        best_y = best_params[param2_name]
        
        # Find position in pivot table
        x_idx = list(pivot_table.columns).index(best_x)
        y_idx = list(pivot_table.index).index(best_y)
        
        # Add star marker for best parameters
        plt.plot(x_idx + 0.5, y_idx + 0.5, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Heatmap saved to {filepath}")
    
    def _generate_multi_parameter_plots(self, results: Dict[str, Any], base_filepath: Path):
        """Generate multiple visualization plots for >2 parameter optimization."""
        all_results = results.get('all_results', [])
        if not all_results:
            self.logger.warning("No results available for multi-parameter visualization")
            return
        
        param_names = list(self.parameter_space.parameters.keys())
        n_params = len(param_names)
        
        self.logger.info(f"Generating multi-parameter visualizations for {n_params} parameters")
        
        # 1. Generate pairwise heatmaps for all parameter combinations
        self._generate_pairwise_heatmaps(all_results, param_names, base_filepath, results)
        
        # 2. Generate parameter importance plot
        self._generate_parameter_importance_plot(all_results, param_names, base_filepath)
        
        # 3. Generate parameter correlation matrix
        self._generate_parameter_correlation_matrix(all_results, param_names, base_filepath)
        
        # 4. If exactly 3 parameters, generate 3D scatter plot
        if n_params == 3:
            self._generate_3d_scatter_plot(all_results, param_names, base_filepath, results)
    
    def _save_csv_results(self, results: Dict[str, Any], csv_path: Path):
        """Save all optimization results to CSV file."""
        all_results = results.get('all_results', [])
        if not all_results:
            self.logger.warning("No detailed results to save to CSV")
            return
        
        # Convert results to DataFrame
        rows = []
        for result in all_results:
            row = {}
            # Add parameters
            row.update(result['params'])
            # Add performance metrics
            row.update(result['stats'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by optimization metric (descending)
        metric_col = self.optimize_metric
        if metric_col in df.columns:
            df = df.sort_values(metric_col, ascending=not self.maximize)
        
        # Save to CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"All results saved to CSV: {csv_path}")
    
    def _generate_pairwise_heatmaps(self, all_results: List[Dict], param_names: List[str], base_filepath: Path, results: Dict[str, Any]):
        """Generate heatmaps for all pairs of parameters."""
        import itertools
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Create PDF with all pairwise heatmaps
        pdf_path = base_filepath.with_name(f"{base_filepath.stem}_pairwise_heatmaps.pdf")
        
        with PdfPages(pdf_path) as pdf:
            for i, (param1, param2) in enumerate(itertools.combinations(param_names, 2)):
                # Extract data for this parameter pair
                param1_values = []
                param2_values = []
                returns = []
                
                for result in all_results:
                    params = result['params']
                    stats = result['stats']
                    param1_values.append(params[param1])
                    param2_values.append(params[param2])
                    returns.append(stats.get('Return [%]', 0))
                
                # Create pivot table
                df = pd.DataFrame({
                    param1: param1_values,
                    param2: param2_values,
                    'Return [%]': returns
                })
                
                try:
                    pivot_table = df.pivot_table(
                        values='Return [%]',
                        index=param2,
                        columns=param1,
                        aggfunc='mean'
                    )
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        pivot_table,
                        annot=True,
                        fmt='.1f',
                        cmap='RdYlGn',
                        center=0,
                        cbar_kws={'label': 'Return [%]'},
                        square=False
                    )
                    
                    plt.title(f'{self.strategy_class.__name__} Optimization\n{param1} vs {param2}')
                    plt.xlabel(param1)
                    plt.ylabel(param2)
                    
                    # Mark best parameters
                    best_params = results['best_params']
                    if param1 in best_params and param2 in best_params:
                        best_x = best_params[param1]
                        best_y = best_params[param2]
                        
                        try:
                            x_idx = list(pivot_table.columns).index(best_x)
                            y_idx = list(pivot_table.index).index(best_y)
                            plt.plot(x_idx + 0.5, y_idx + 0.5, 'r*', markersize=20, 
                                    markeredgecolor='white', markeredgewidth=2)
                        except (ValueError, IndexError):
                            pass
                    
                    plt.tight_layout()
                    pdf.savefig(bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create heatmap for {param1} vs {param2}: {e}")
                    plt.close()
        
        self.logger.info(f"Pairwise heatmaps saved to {pdf_path}")
    
    def _generate_parameter_importance_plot(self, all_results: List[Dict], param_names: List[str], base_filepath: Path):
        """Generate parameter importance plot showing which parameters have most impact."""
        # Calculate parameter importance using variance analysis
        param_importance = {}
        
        for param_name in param_names:
            # Group results by parameter value and calculate variance
            param_groups = {}
            for result in all_results:
                param_value = result['params'][param_name]
                return_val = result['stats'].get('Return [%]', 0)
                
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(return_val)
            
            # Calculate variance between groups (higher = more important)
            group_means = [np.mean(returns) for returns in param_groups.values()]
            param_importance[param_name] = np.var(group_means) if len(group_means) > 1 else 0
        
        # Create importance plot
        plt.figure(figsize=(10, 6))
        params = list(param_importance.keys())
        importances = list(param_importance.values())
        
        bars = plt.bar(params, importances, color='steelblue', alpha=0.7)
        plt.title(f'{self.strategy_class.__name__} Parameter Importance\n(Variance in Returns by Parameter)')
        plt.xlabel('Parameters')
        plt.ylabel('Importance (Variance in Returns)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(importances)*0.01,
                    f'{importance:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        importance_path = base_filepath.with_name(f"{base_filepath.stem}_parameter_importance.png")
        plt.savefig(importance_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Parameter importance plot saved to {importance_path}")
    
    def _generate_parameter_correlation_matrix(self, all_results: List[Dict], param_names: List[str], base_filepath: Path):
        """Generate correlation matrix between parameters and performance."""
        # Create DataFrame with all parameters and returns
        data_for_corr = []
        for result in all_results:
            row = result['params'].copy()
            row['Return [%]'] = result['stats'].get('Return [%]', 0)
            row['Sharpe Ratio'] = result['stats'].get('Sharpe Ratio', 0) or 0
            row['Win Rate [%]'] = result['stats'].get('Win Rate [%]', 0) or 0
            data_for_corr.append(row)
        
        df_corr = pd.DataFrame(data_for_corr)
        
        # Calculate correlation matrix
        corr_matrix = df_corr.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title(f'{self.strategy_class.__name__} Parameter Correlation Matrix')
        plt.tight_layout()
        
        corr_path = base_filepath.with_name(f"{base_filepath.stem}_correlation_matrix.png")
        plt.savefig(corr_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Correlation matrix saved to {corr_path}")
    
    def _generate_3d_scatter_plot(self, all_results: List[Dict], param_names: List[str], base_filepath: Path, results: Dict[str, Any]):
        """Generate 3D scatter plot for exactly 3 parameters."""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Extract data
        param1_values = [result['params'][param_names[0]] for result in all_results]
        param2_values = [result['params'][param_names[1]] for result in all_results]
        param3_values = [result['params'][param_names[2]] for result in all_results]
        returns = [result['stats'].get('Return [%]', 0) for result in all_results]
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color points by return value
        scatter = ax.scatter(param1_values, param2_values, param3_values, 
                           c=returns, cmap='RdYlGn', s=50, alpha=0.7)
        
        # Highlight best result
        best_params = results['best_params']
        best_x = best_params[param_names[0]]
        best_y = best_params[param_names[1]] 
        best_z = best_params[param_names[2]]
        
        ax.scatter([best_x], [best_y], [best_z], c='red', s=200, marker='*', 
                  edgecolors='white', linewidth=2, label='Best Parameters')
        
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_zlabel(param_names[2])
        ax.set_title(f'{self.strategy_class.__name__} 3D Parameter Space\nColored by Return [%]')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Return [%]', shrink=0.8)
        ax.legend()
        
        plt.tight_layout()
        scatter3d_path = base_filepath.with_name(f"{base_filepath.stem}_3d_scatter.png")
        plt.savefig(scatter3d_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"3D scatter plot saved to {scatter3d_path}")