#!/usr/bin/env python3
"""
Parallel Grid Search using backtesting.py's built-in parallel optimization.

This version leverages backtesting.py's native parallel optimization capabilities
for better performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import FractionalBacktest
except ImportError:
    raise ImportError("backtesting library not found. Install it with: uv add backtesting")

from framework.strategies.base_strategy import BaseStrategy
from framework.backtesting.strategy_wrapper import StrategyWrapper
from framework.risk.base_risk_manager import BaseRiskManager
from .parameter_space import ParameterSpace


# Global variables for multiprocessing
_global_strategy_class = None
_global_risk_manager = None
_global_debug = False


def _init_worker(strategy_class, risk_manager, debug):
    """Initialize worker process with global variables."""
    global _global_strategy_class, _global_risk_manager, _global_debug
    _global_strategy_class = strategy_class
    _global_risk_manager = risk_manager
    _global_debug = debug


class OptimizableStrategy(StrategyWrapper):
    """Strategy wrapper that can accept dynamic parameters."""
    
    # Define parameter placeholders that will be optimized
    # These will be overridden by backtesting.py during optimization
    short_window = 10
    long_window = 30
    stop_loss_pct = 0.02
    take_profit_pct = 0.04
    h1_lookback_candles = 24
    risk_reward_ratio = 2.0
    max_hold_hours = 4
    position_size = 0.05
    
    # Store parameter space for filtering
    _parameter_space_keys = set()
    
    @classmethod
    def set_parameter_space_keys(cls, keys):
        """Set which parameters are being optimized."""
        cls._parameter_space_keys = set(keys)
    
    def init(self):
        """Initialize with dynamic parameters."""
        # Get parameters from instance attributes set by backtesting.py
        strategy_params = {}
        
        # Only include parameters that are in the parameter space being optimized
        for param_name in self._parameter_space_keys:
            if hasattr(self, param_name):
                strategy_params[param_name] = getattr(self, param_name)
        
        # Use global variables set in worker process
        self.__class__.framework_strategy_class = _global_strategy_class
        self.__class__.strategy_params = strategy_params
        self.__class__.risk_manager = _global_risk_manager
        self.__class__.debug = _global_debug
        
        # Call parent init
        super().init()


class ParallelGridSearchOptimizer:
    """
    Parallel grid search optimizer using backtesting.py's native optimization.
    
    This version properly leverages backtesting.py's built-in parallel capabilities
    for significant performance improvements on multi-core systems.
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
                 optimize_metric: str = 'Return [%]',
                 maximize: bool = True,
                 n_jobs: int = -1,  # -1 = use all cores
                 debug: bool = False):
        """
        Initialize parallel grid search optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Parameter search space
            data: Historical price data
            initial_capital: Starting capital
            commission: Commission rate
            margin: Margin requirement
            use_fractional: Use fractional shares
            risk_manager: Risk manager (optional)
            optimize_metric: Metric to optimize
            maximize: Whether to maximize the metric
            n_jobs: Number of parallel jobs (-1 = all cores)
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
        self.optimize_metric = optimize_metric
        self.maximize = maximize
        self.n_jobs = n_jobs
        self.debug = debug
        
        # Set global variables for multiprocessing
        global _global_strategy_class, _global_risk_manager, _global_debug
        _global_strategy_class = strategy_class
        _global_risk_manager = risk_manager
        _global_debug = debug
        
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
        Run parallel grid search optimization.
        
        Returns:
            Dictionary with optimization results
        """
        # Get parameter ranges for backtesting.py
        param_ranges = self._prepare_param_ranges()
        
        total_combinations = self.parameter_space.get_total_combinations()
        self.logger.info(f"Starting parallel grid search with {total_combinations} combinations")
        self.logger.info(f"Using {self.n_jobs} parallel jobs")
        
        # Choose backtest class
        BacktestClass = FractionalBacktest if self.use_fractional else Backtest
        
        # Create backtest instance
        bt = BacktestClass(
            data=self.data,
            strategy=OptimizableStrategy,
            cash=self.initial_capital,
            commission=self.commission,
            margin=self.margin,
            exclusive_orders=True,
            hedging=False,
            trade_on_close=False
        )
        
        # Set parameter space keys for filtering in OptimizableStrategy
        OptimizableStrategy.set_parameter_space_keys(param_ranges.keys())
        
        # Run parallel optimization
        try:
            # Use backtesting.py's optimize method with grid search
            # Don't request heatmap from backtesting.py as it can be unreliable
            # We'll generate our own if needed
            
            # Try sambo method first to get all optimization results
            try:
                stats, optimization_result = bt.optimize(
                    **param_ranges,
                    maximize=self.optimize_metric,
                    method='sambo',
                    max_tries=total_combinations,  # Try all combinations
                    return_heatmap=False,
                    return_optimization=True,
                    random_state=42
                )
            except Exception as e:
                self.logger.warning(f"SAMBO method failed ({e}), falling back to grid search")
                # Fallback to grid method without optimization results
                stats = bt.optimize(
                    **param_ranges,
                    maximize=self.optimize_metric,
                    method='grid',
                    max_tries=None,
                    return_heatmap=False,
                    random_state=None
                )
                optimization_result = None
            
            heatmap = None  # Don't use backtesting.py's heatmap
            
            # Extract best parameters
            best_params = self._extract_best_params(stats, param_ranges)
            
            # Convert optimization result to our format
            all_results = []
            if optimization_result is not None:
                # Convert backtesting.py optimization result to our format
                for result in optimization_result:
                    all_results.append({
                        'params': dict(result[0]),  # Parameter combination
                        'stats': self._format_stats(result[1])  # Performance stats
                    })
            
            # Format results
            optimization_results = {
                'best_params': best_params,
                'best_stats': self._format_stats(stats),
                'metric_value': stats[self.optimize_metric],
                'total_combinations': total_combinations,
                'all_results': all_results,
                'heatmap': heatmap  # Store heatmap data if available
            }
            
            # Log results
            self._log_results(optimization_results)
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _prepare_param_ranges(self) -> Dict[str, List[Any]]:
        """Prepare parameter ranges for backtesting.py optimize method."""
        param_ranges = {}
        
        for name, param in self.parameter_space.parameters.items():
            if param.param_type == 'choice':
                param_ranges[name] = param.choices
            else:
                values = param.get_grid_values()
                param_ranges[name] = values
        
        return param_ranges
    
    def _extract_best_params(self, stats: pd.Series, param_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Extract best parameters from optimization stats."""
        best_params = {}
        
        # Extract parameter values from stats
        for param_name in param_ranges.keys():
            if hasattr(stats, '_strategy'):
                # Try to get from strategy instance
                strategy = stats._strategy
                if hasattr(strategy, param_name):
                    best_params[param_name] = getattr(strategy, param_name)
            elif param_name in stats.index:
                # Fallback to stats index
                best_params[param_name] = stats[param_name]
        
        return best_params
    
    def _format_stats(self, stats: pd.Series) -> Dict[str, Any]:
        """Format statistics for display."""
        formatted = {}
        
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
        self.logger.info("Parallel Grid Search Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"Total combinations tested: {results['total_combinations']}")
        self.logger.info(f"Best {self.optimize_metric}: {results['metric_value']:.4f}")
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
                'optimize_metric': self.optimize_metric,
                'maximize': self.maximize,
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'margin': self.margin,
                'n_jobs': self.n_jobs,
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
        
        # Generate heatmap if available
        if save_heatmap and results.get('heatmap') is not None:
            heatmap_path = filepath.with_suffix('.png')
            self._save_heatmap(results['heatmap'], results, heatmap_path)
    
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
    
    def _save_heatmap(self, heatmap_data: pd.DataFrame, results: Dict[str, Any], filepath: Path):
        """Save heatmap visualization from backtesting.py's heatmap data."""
        if heatmap_data is None:
            self.logger.warning("No heatmap data available")
            return
        
        # Handle different heatmap data formats from backtesting.py
        if hasattr(heatmap_data, 'empty') and heatmap_data.empty:
            self.logger.warning("Empty heatmap data")
            return
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        
        try:
            # Use seaborn for better heatmap
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': self.optimize_metric},
                square=False
            )
        except (IndexError, ValueError) as e:
            # Fallback to matplotlib if seaborn fails
            self.logger.warning(f"Seaborn heatmap failed ({e}), using matplotlib")
            plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
            plt.colorbar(label=self.optimize_metric)
            
            # Add text annotations manually
            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    value = heatmap_data.iloc[i, j] if hasattr(heatmap_data, 'iloc') else heatmap_data[i, j]
                    plt.text(j, i, f'{value:.1f}', ha='center', va='center')
        
        plt.title(f'{self.strategy_class.__name__} Optimization Heatmap\n{self.optimize_metric}')
        
        # Get parameter names from heatmap axes
        if hasattr(heatmap_data, 'columns') and hasattr(heatmap_data, 'index'):
            plt.xlabel(heatmap_data.columns.name or 'Parameter 1')
            plt.ylabel(heatmap_data.index.name or 'Parameter 2')
            
            # Mark best parameters if available
            best_params = results['best_params']
            param_names = list(self.parameter_space.parameters.keys())
            
            if len(param_names) == 2 and all(p in best_params for p in param_names):
                best_x = best_params[param_names[0]]
                best_y = best_params[param_names[1]]
                
                # Find position in heatmap
                try:
                    x_idx = list(heatmap_data.columns).index(best_x)
                    y_idx = list(heatmap_data.index).index(best_y)
                    # Add star marker for best parameters
                    plt.plot(x_idx + 0.5, y_idx + 0.5, 'r*', markersize=20, 
                            markeredgecolor='white', markeredgewidth=2)
                except (ValueError, IndexError):
                    pass  # Best params not in heatmap grid
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Heatmap saved to {filepath}")