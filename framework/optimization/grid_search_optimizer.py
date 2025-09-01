"""
Grid Search Optimizer for systematic parameter optimization.

This optimizer tests all possible parameter combinations from the
defined parameter space using backtesting.py's optimization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type, Tuple
from pathlib import Path
import json
from datetime import datetime
import logging

try:
    from backtesting import Backtest
    from backtesting.lib import FractionalBacktest
except ImportError:
    raise ImportError("backtesting library not found. Install it with: uv add backtesting")

from framework.strategies.base_strategy import BaseStrategy
from framework.backtesting.strategy_wrapper import create_wrapper_class, StrategyWrapper
from framework.risk.base_risk_manager import BaseRiskManager
from .parameter_space import ParameterSpace


def _create_optimizable_wrapper(strategy_class, risk_manager, debug):
    """Create an optimizable wrapper class for backtesting.py.
    
    This function must be at module level for pickling to work with multiprocessing.
    """
    class OptimizableWrapper(StrategyWrapper):
        framework_strategy_class = strategy_class
        
        def init(self):
            # Extract strategy parameters from class attributes
            strategy_params = {}
            for key in dir(self):
                if key.startswith('strategy_params__'):
                    param_name = key.replace('strategy_params__', '')
                    strategy_params[param_name] = getattr(self, key)
            
            # Set class attributes
            self.__class__.strategy_params = strategy_params
            self.__class__.risk_manager = risk_manager
            self.__class__.debug = debug
            
            # Call parent init
            super().init()
    
    return OptimizableWrapper


class GridSearchOptimizer:
    """
    Grid search optimizer using backtesting.py's optimization engine.
    
    This optimizer systematically tests all parameter combinations defined
    in the parameter space and returns the best performing configurations.
    
    Example:
        # Define parameter space
        space = ParameterSpace()
        space.add_parameter('short_window', 5, 50, step=5, param_type='int')
        space.add_parameter('long_window', 20, 200, step=10, param_type='int')
        
        # Create optimizer
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_space=space,
            data=price_data
        )
        
        # Run optimization
        results = optimizer.optimize()
        
        # Get best parameters
        best_params = results['best_params']
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
                 constraint: Optional[callable] = None,
                 debug: bool = False):
        """
        Initialize grid search optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Parameter search space
            data: Historical price data (must have Open, High, Low, Close, Volume columns)
            initial_capital: Starting capital for backtests
            commission: Commission rate per trade
            margin: Margin requirement (1.0 = no leverage, 0.01 = 100x leverage)
            use_fractional: Use fractional shares/units
            risk_manager: Risk manager for position sizing (optional)
            optimize_metric: Metric to optimize ('Return [%]', 'Sharpe Ratio', 'Win Rate [%]', etc.)
            maximize: Whether to maximize (True) or minimize (False) the metric
            constraint: Optional constraint function for filtering results
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
        self.constraint = constraint
        self.debug = debug
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Store optimization results
        self.optimization_results = None
        self.best_params = None
        self.best_stats = None
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting.py format."""
        # Ensure proper column names for backtesting.py
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Create a copy and rename columns if needed
        prepared_data = data.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in prepared_data.columns and new_name not in prepared_data.columns:
                prepared_data = prepared_data.rename(columns={old_name: new_name})
        
        # Ensure we have all required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in prepared_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure datetime index
        if not isinstance(prepared_data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
        
        return prepared_data
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Returns:
            Dictionary containing:
            - 'best_params': Best parameter combination
            - 'best_stats': Statistics for best parameters
            - 'all_results': DataFrame with all tested combinations
            - 'total_combinations': Total number of combinations tested
        """
        # Get all parameter combinations
        total_combinations = self.parameter_space.get_total_combinations()
        self.logger.info(f"Starting grid search with {total_combinations} parameter combinations")
        
        if total_combinations == 0:
            raise ValueError("No parameters defined in parameter space")
        
        # Prepare parameter ranges for backtesting.py
        param_ranges = self._prepare_param_ranges()
        
        # Create wrapper strategy class - must be global for pickling
        wrapper_class = _create_optimizable_wrapper(
            self.strategy_class,
            self.risk_manager,
            self.debug
        )
        
        # Choose backtest class
        BacktestClass = FractionalBacktest if self.use_fractional else Backtest
        
        # Create backtest instance
        bt = BacktestClass(
            data=self.data,
            strategy=wrapper_class,
            cash=self.initial_capital,
            commission=self.commission,
            margin=self.margin,
            exclusive_orders=True,
            hedging=False,
            trade_on_close=False
        )
        
        # Run optimization
        self.logger.info(f"Running optimization on metric: {self.optimize_metric}")
        
        try:
            # Run the grid search optimization
            stats, heatmap = bt.optimize(
                **param_ranges,
                maximize=self.optimize_metric if self.maximize else f'-{self.optimize_metric}',
                constraint=self.constraint,
                return_heatmap=True,
                method='grid'  # Force grid search
            )
            
            # Store results
            self.optimization_results = stats
            self.best_stats = stats
            
            # Extract best parameters from the stats
            self.best_params = self._extract_best_params(stats, param_ranges)
            
            # Process and return results
            results = {
                'best_params': self.best_params,
                'best_stats': self._format_stats(stats),
                'metric_value': stats[self.optimize_metric],
                'total_combinations': total_combinations,
                'heatmap': heatmap if heatmap is not None else None
            }
            
            # Log results
            self._log_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _prepare_param_ranges(self) -> Dict[str, Any]:
        """Prepare parameter ranges for backtesting.py optimize method."""
        param_ranges = {}
        
        for name, param in self.parameter_space.parameters.items():
            # Convert to format expected by backtesting.py
            # Parameters passed to strategy need 'strategy_params' prefix
            param_name = f'strategy_params__{name}'
            
            if param.param_type == 'choice':
                param_ranges[param_name] = param.choices
            else:
                values = param.get_grid_values()
                param_ranges[param_name] = values
        
        return param_ranges
    
    def _extract_best_params(self, stats: pd.Series, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best parameters from optimization stats."""
        best_params = {}
        
        # The stats._strategy attribute contains the optimal strategy instance
        if hasattr(stats, '_strategy'):
            # Get parameters from the strategy instance
            strategy_params = stats._strategy.strategy_params
            best_params = strategy_params.copy()
        else:
            # Fallback: extract from stats index if available
            for key in param_ranges.keys():
                # Remove 'strategy_params__' prefix
                param_name = key.replace('strategy_params__', '')
                if key in stats.index:
                    best_params[param_name] = stats[key]
        
        return best_params
    
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
                # Format based on type
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
        self.logger.info(f"Best {self.optimize_metric}: {results['metric_value']:.4f}")
        self.logger.info("Best parameters:")
        for param, value in results['best_params'].items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info("Best performance:")
        for metric, value in results['best_stats'].items():
            if value is not None:
                self.logger.info(f"  {metric}: {value}")
    
    def save_results(self, filepath: str):
        """
        Save optimization results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        if self.best_params is None:
            raise ValueError("No optimization results to save. Run optimize() first.")
        
        # Prepare data for JSON serialization
        output = {
            'metadata': {
                'strategy': self.strategy_class.__name__,
                'timestamp': datetime.now().isoformat(),
                'optimize_metric': self.optimize_metric,
                'maximize': self.maximize,
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
            'best_params': self.best_params,
            'best_stats': self._format_stats(self.best_stats) if self.best_stats is not None else None
        }
        
        # Save to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def get_top_n_results(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N parameter combinations.
        
        Args:
            n: Number of top results to return
            
        Returns:
            List of dictionaries with parameters and metrics
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # For now, return just the best result
        # In a full implementation, we would store all results during optimization
        return [{
            'params': self.best_params,
            'stats': self._format_stats(self.best_stats)
        }]