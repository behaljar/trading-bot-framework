"""
Walk-forward analysis implementation for strategy validation.

Walk-forward analysis tests strategy robustness by testing pre-optimized
parameters on rolling time windows to validate out-of-sample performance
across different market conditions and time periods.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from pathlib import Path

# Add framework to path  
framework_dir = Path(__file__).parent.parent.parent
sys.path.append(str(framework_dir))

try:
    from backtesting import Backtest
    from backtesting.lib import FractionalBacktest
except ImportError:
    raise ImportError("backtesting library not found. Install it with: uv add backtesting")

from framework.strategies.base_strategy import BaseStrategy
from framework.backtesting.strategy_wrapper import create_wrapper_class
from framework.risk.base_risk_manager import BaseRiskManager
from framework.risk.fixed_position_size_manager import FixedPositionSizeManager
from framework.risk.fixed_risk_manager import FixedRiskManager
from framework.utils.logger import setup_logger
from .analysis_result import AnalysisResult


class WalkForwardAnalyzer:
    """
    Walk-forward analysis implementation for strategy validation.
    
    Walk-forward analysis tests strategy robustness by testing pre-optimized
    parameters on rolling time windows to validate out-of-sample performance
    across different market conditions and time periods.
    
    This analyzer assumes you have already run optimization to find optimal
    parameters, and now want to test those parameters across different time
    periods to assess temporal robustness.
    """
    
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 parameter_space: Dict[str, Any],
                 is_window_months: int = 3,
                 oos_window_months: int = 1,
                 step_months: int = 1,
                 window_mode: str = "rolling",
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 margin: float = 0.01,
                 risk_manager_type: str = "fixed_risk",
                 risk_manager_params: Optional[Dict[str, Any]] = None,
                 use_fractional: bool = True,
                 optimization_metric: str = "Return [%]",
                 maximize: bool = True,
                 n_jobs: Optional[int] = None,
                 max_results_in_memory: int = 1000,
                 generate_charts: bool = True):
        """
        Initialize Walk-Forward Analyzer.
        
        Args:
            strategy_class: Strategy class to analyze
            parameter_space: Parameter ranges to optimize (e.g., {'short_window': [5, 10, 15], 'long_window': [20, 30, 40]})
            is_window_months: Months of in-sample data for optimization
            oos_window_months: Months of out-of-sample data for testing
            step_months: Months to step forward between periods
            window_mode: 'rolling' (fixed IS window) or 'anchored' (expanding IS window)
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            margin: Margin requirement (0.01 = 100x leverage)
            risk_manager_type: Type of risk manager ('fixed_risk' or 'fixed_position')
            risk_manager_params: Parameters for risk manager
            use_fractional: Use FractionalBacktest for precise position sizing
            optimization_metric: Metric to optimize for ('Return [%]', 'Sharpe Ratio', etc.)
            maximize: Whether to maximize or minimize the optimization metric
            n_jobs: Number of parallel jobs (None = auto-detect)
            max_results_in_memory: Maximum number of results to keep in memory
            generate_charts: Generate charts for each period
        """
        self.strategy_class = strategy_class
        self.parameter_space = parameter_space
        self.is_window_months = is_window_months
        self.oos_window_months = oos_window_months
        self.step_months = step_months
        self.window_mode = window_mode
        self.initial_capital = initial_capital
        self.commission = commission
        self.margin = margin
        self.risk_manager_type = risk_manager_type
        self.risk_manager_params = risk_manager_params or {}
        self.use_fractional = use_fractional
        self.optimization_metric = optimization_metric
        self.maximize = maximize
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)  # Default to max 4 processes
        self.max_results_in_memory = max_results_in_memory
        self.generate_charts = generate_charts
        
        self.logger = setup_logger("INFO")
        
        self._validate_parameters()
        self._validate_parameter_space()
        
        # Create risk manager
        self.risk_manager = self._create_risk_manager()
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not self.parameter_space:
            raise ValueError("parameter_space cannot be empty")
        if self.oos_window_months <= 0:
            raise ValueError("oos_window_months must be positive")
        if self.step_months <= 0:
            raise ValueError("step_months must be positive")
        if self.is_window_months <= 0:
            raise ValueError("is_window_months must be positive")
        if self.window_mode not in ['rolling', 'anchored']:
            raise ValueError("window_mode must be 'rolling' or 'anchored'")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission < 0:
            raise ValueError("commission cannot be negative")
        if self.margin <= 0 or self.margin > 1:
            raise ValueError("margin must be between 0 and 1")
    
    def _validate_parameter_space(self) -> None:
        """Validate that parameter space is properly formatted."""
        for param_name, param_range in self.parameter_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Range format (min, max)
                continue
            elif isinstance(param_range, list):
                # Discrete values format
                continue
            else:
                raise ValueError(f"Invalid parameter range for {param_name}: {param_range}. Use (min, max) or [val1, val2, ...]")
    
    def _create_risk_manager(self) -> BaseRiskManager:
        """Create risk manager instance based on configuration."""
        if self.risk_manager_type == "fixed_position":
            return FixedPositionSizeManager(
                position_size=self.risk_manager_params.get('position_size', 0.1)
            )
        elif self.risk_manager_type == "fixed_risk":
            return FixedRiskManager(
                risk_percent=self.risk_manager_params.get('risk_percent', 0.01),
                default_stop_distance=self.risk_manager_params.get('default_stop_distance', 0.02)
            )
        else:
            raise ValueError(f"Unknown risk manager type: {self.risk_manager_type}")
    
    def analyze(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> AnalysisResult:
        """
        Perform walk-forward analysis on the given data.
        
        Args:
            data: Historical price data with OHLCV columns
            symbol: Symbol identifier
            
        Returns:
            AnalysisResult containing all walk-forward test results
        """
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data for walk-forward analysis")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Validate minimum data for meaningful analysis
        min_required_periods = max(100, ((self.is_window_months + self.oos_window_months) * 30))  # Rough estimate
        if len(data) < min_required_periods:
            raise ValueError(f"Need at least {min_required_periods} data points for {self.is_window_months + self.oos_window_months}-month periods")
        
        self.logger.info(f"Starting walk-forward analysis for {symbol}")
        self.logger.info(f"Parameter space: {self.parameter_space}")
        self.logger.info(f"Optimization metric: {self.optimization_metric} ({'maximize' if self.maximize else 'minimize'})")
        self.logger.info(f"IS window: {self.is_window_months} months, OOS window: {self.oos_window_months} months")
        self.logger.info(f"Window mode: {self.window_mode}, Step: {self.step_months} months")
        
        # Generate walk-forward periods with IS/OOS splits
        periods = self._generate_walk_forward_periods(data)
        self.logger.info(f"Generated {len(periods)} walk-forward periods")
        
        if not periods:
            raise ValueError("No valid walk-forward periods found")
        
        # Run analysis for each period (use sequential due to multiprocessing issues)
        results = self._run_periods_sequential(data, symbol, periods)
        
        if not results:
            raise ValueError("No successful walk-forward periods")
        
        # Calculate combined metrics including WFE
        combined_metrics = self._calculate_combined_metrics(results)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(results)
        
        # Generate overall analysis charts
        if self.generate_charts:
            self._generate_overall_charts(results, symbol)
        
        return AnalysisResult(
            analysis_type='walk_forward',
            strategy_name=self.strategy_class.__name__,
            symbol=symbol,
            analysis_start_date=data.index[0],
            analysis_end_date=data.index[-1],
            individual_results=results,
            combined_metrics=combined_metrics,
            analysis_parameters={
                'parameter_space': self.parameter_space,
                'optimization_metric': self.optimization_metric,
                'maximize': self.maximize,
                'is_window_months': self.is_window_months,
                'oos_window_months': self.oos_window_months,
                'step_months': self.step_months,
                'window_mode': self.window_mode,
                'n_periods': len(results),
                'risk_manager_type': self.risk_manager_type,
                'risk_manager_params': self.risk_manager_params
            },
            stability_metrics=stability_metrics,
            metadata={
                'test_periods': [(r.metadata.get('test_start'), r.metadata.get('test_end')) 
                               for r in results if hasattr(r, 'metadata')]
            }
        )
    
    def _run_periods_sequential(self, data: pd.DataFrame, symbol: str, periods: List[tuple]) -> List[Any]:
        """Run walk-forward periods sequentially with memory management."""
        results = []
        
        for i, (is_start, is_end, oos_start, oos_end) in enumerate(periods):
            if i % 10 == 0:
                self.logger.info(f"Testing period {i+1}/{len(periods)}: IS {is_start.date()}-{is_end.date()}, OOS {oos_start.date()}-{oos_end.date()}")
                
                # Memory management: keep only the best results if we have too many
                if len(results) > self.max_results_in_memory:
                    results = self._manage_memory_efficient_results(results)
            
            try:
                period_result = self._run_single_period_with_wfe(data, is_start, is_end, oos_start, oos_end, symbol, i+1)
                
                if period_result:
                    results.append(period_result)
                
            except Exception as e:
                self.logger.warning(f"Error in period {i+1}: {e}")
                continue
        
        return results
    
    def _run_periods_parallel(self, data: pd.DataFrame, symbol: str, periods: List[tuple]) -> List[Any]:
        """Run walk-forward periods in parallel."""
        results = []
        failed_count = 0
        
        self.logger.info(f"Running {len(periods)} periods using {self.n_jobs} parallel jobs")
        
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs
                future_to_period = {}
                for i, (test_start, test_end) in enumerate(periods):
                    try:
                        future = executor.submit(
                            self._run_single_period_static, 
                            data, test_start, test_end, symbol, i+1,
                            self.strategy_class, self.optimal_parameters,
                            self.initial_capital, self.commission, self.margin,
                            self.risk_manager_type, self.risk_manager_params,
                            self.use_fractional
                        )
                        future_to_period[future] = (i+1, test_start, test_end)
                    except Exception as e:
                        self.logger.warning(f"Failed to submit period {i+1}: {e}")
                        failed_count += 1
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_period):
                    period_num, test_start, test_end = future_to_period[future]
                    completed += 1
                    
                    if completed % 10 == 0:
                        self.logger.info(f"Completed {completed}/{len(periods)} periods")
                    
                    try:
                        result = future.result(timeout=120)  # 2 minute timeout per period
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Period {period_num} ({test_start.date()}-{test_end.date()}) failed: {e}")
                        failed_count += 1
                        
        except Exception as e:
            self.logger.error(f"Critical error in parallel walk-forward: {e}")
            raise
        
        if failed_count > 0:
            success_rate = (len(results) / len(periods)) * 100
            self.logger.warning(f"Parallel walk-forward completed with {failed_count} failures ({success_rate:.1f}% success rate)")
        
        return results
    
    def _manage_memory_efficient_results(self, results: List[Any]) -> List[Any]:
        """Manage results list to prevent excessive memory usage by keeping only the best results."""
        if len(results) <= self.max_results_in_memory:
            return results
        
        self.logger.info(f"Memory management: trimming {len(results)} results to {self.max_results_in_memory // 2}")
        
        # Sort by return percentage (descending - best first)
        try:
            results.sort(key=lambda r: getattr(r, 'return_pct', 0), reverse=True)
        except:
            # Fallback sorting if return_pct attribute doesn't exist
            results.sort(key=lambda r: getattr(r, 'Return [%]', 0), reverse=True)
        
        # Keep only the best half
        keep_count = self.max_results_in_memory // 2
        return results[:keep_count]
    
    @staticmethod
    def _run_single_period_static(data: pd.DataFrame, test_start: datetime, test_end: datetime,
                                 symbol: str, period_number: int, strategy_class: Type[BaseStrategy],
                                 optimal_parameters: Dict[str, Any], initial_capital: float,
                                 commission: float, margin: float, risk_manager_type: str,
                                 risk_manager_params: Dict[str, Any], use_fractional: bool) -> Optional[Any]:
        """Static method for parallel processing (must be picklable)."""
        # Note: This method is disabled due to multiprocessing pickle issues
        # with dynamic wrapper classes. Use sequential processing instead.
        return None
    
    def _generate_walk_forward_periods(self, data: pd.DataFrame) -> List[tuple]:
        """Generate periods for walk-forward analysis with IS/OOS splits."""
        periods = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Start after we have enough data for initial IS window
        if self.window_mode == "anchored":
            # Anchored mode: IS window starts at beginning and grows
            is_start_base = start_date
            current_date = start_date + relativedelta(months=self.is_window_months)
        else:
            # Rolling mode: IS window has fixed length
            current_date = start_date + relativedelta(months=self.is_window_months)
        
        period_number = 1
        
        while True:
            if self.window_mode == "anchored":
                # Anchored: IS window grows from start_date
                is_start = is_start_base
                is_end = start_date + relativedelta(months=self.is_window_months + (period_number - 1) * self.step_months)
            else:
                # Rolling: IS window has fixed length
                is_start = current_date - relativedelta(months=self.is_window_months)
                is_end = current_date
            
            # OOS window always follows IS window
            oos_start = is_end
            oos_end = oos_start + relativedelta(months=self.oos_window_months)
            
            # Check if we have enough data for both IS and OOS
            if oos_end > end_date:
                break
            
            # Ensure we have actual data in both windows
            is_data = data[(data.index >= is_start) & (data.index < is_end)]
            oos_data = data[(data.index >= oos_start) & (data.index < oos_end)]
            
            if len(is_data) >= 100 and len(oos_data) >= 50:  # Minimum data requirements
                periods.append((is_start, is_end, oos_start, oos_end))
            
            # Step forward
            current_date += relativedelta(months=self.step_months)
            period_number += 1
        
        return periods
    
    def _run_single_period_with_wfe(self, data: pd.DataFrame, 
                                   is_start: datetime, is_end: datetime,
                                   oos_start: datetime, oos_end: datetime,
                                   symbol: str, period_number: int) -> Optional[Any]:
        """Run testing for a single walk-forward period with WFE calculation."""
        
        # Extract IS and OOS data
        is_data = data[(data.index >= is_start) & (data.index < is_end)].copy()
        oos_data = data[(data.index >= oos_start) & (data.index < oos_end)].copy()
        
        if len(is_data) < 100 or len(oos_data) < 50:
            self.logger.warning(f"Insufficient data: IS={len(is_data)}, OOS={len(oos_data)} rows")
            return None
        
        try:
            # Step 1: Optimize parameters on IS data
            self.logger.info(f"Period {period_number}: Optimizing parameters on IS data ({len(is_data)} points)")
            
            optimal_params, is_result = self._optimize_on_data(is_data, symbol, f"IS_P{period_number}")
            
            if optimal_params is None or is_result is None:
                self.logger.warning(f"Period {period_number}: Optimization failed")
                return None
            
            self.logger.info(f"Period {period_number}: Best params: {optimal_params}")
            self.logger.info(f"Period {period_number}: IS {self.optimization_metric}: {getattr(is_result, self.optimization_metric, 'N/A')}")
            
            # Step 2: Test optimized parameters on OOS data
            self.logger.info(f"Period {period_number}: Testing on OOS data ({len(oos_data)} points)")
            
            oos_result = self._run_backtest_with_params(oos_data, optimal_params, symbol, f"OOS_P{period_number}")
            
            if oos_result is None:
                self.logger.warning(f"Period {period_number}: OOS testing failed")
                return None
            
            self.logger.info(f"Period {period_number}: OOS {self.optimization_metric}: {getattr(oos_result, self.optimization_metric, 'N/A')}")
            
            # Calculate WFE
            wfe = self._calculate_wfe(is_result, oos_result, is_data, oos_data)
            
            # Generate charts if requested
            chart_paths = []
            if self.generate_charts:
                chart_paths = self._generate_period_charts(is_data, oos_data, is_result, oos_result, 
                                                         symbol, period_number)
            
            # Create combined result with WFE metrics
            combined_result = self._create_combined_result(is_result, oos_result, wfe, {
                'period_number': period_number,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'is_data_points': len(is_data),
                'oos_data_points': len(oos_data),
                'chart_paths': chart_paths,
                'parameters_used': optimal_params
            })
            
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward period {period_number}: {e}")
            return None
    
    def _run_backtest_on_data(self, data: pd.DataFrame, symbol: str, period_label: str) -> Optional[Any]:
        """Run backtest on specific data segment."""
        try:
            # Prepare data for backtesting.py (needs capitalized columns)
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            test_data = data.rename(columns=column_mapping)
            
            # Create wrapper strategy class
            WrapperClass = create_wrapper_class(
                self.strategy_class, 
                self.optimal_parameters,
                risk_manager=self.risk_manager,
                debug=False
            )
            
            # Run backtest
            if self.use_fractional:
                bt = FractionalBacktest(
                    data=test_data,
                    strategy=WrapperClass,
                    cash=self.initial_capital,
                    commission=self.commission,
                    margin=self.margin,
                    exclusive_orders=True
                )
            else:
                bt = Backtest(
                    data=test_data,
                    strategy=WrapperClass,
                    cash=self.initial_capital,
                    commission=self.commission,
                    margin=self.margin,
                    exclusive_orders=True
                )
            
            result = bt.run()
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {period_label}: {e}")
            return None
    
    def _calculate_wfe(self, is_result: Any, oos_result: Any, is_data: pd.DataFrame, oos_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Walk-Forward Efficiency metrics."""
        # Extract returns (try different attribute names)
        def safe_get_return(result):
            for attr in ['Return [%]', 'return_pct', 'total_return_pct']:
                if hasattr(result, attr):
                    return getattr(result, attr)
            return 0.0
        
        is_return = safe_get_return(is_result)
        oos_return = safe_get_return(oos_result)
        
        # Calculate period lengths in days for annualization
        is_days = (is_data.index[-1] - is_data.index[0]).days
        oos_days = (oos_data.index[-1] - oos_data.index[0]).days
        
        # Annualize returns (assuming 365 days per year)
        is_annual_return = (1 + is_return / 100) ** (365 / is_days) - 1 if is_days > 0 else 0
        oos_annual_return = (1 + oos_return / 100) ** (365 / oos_days) - 1 if oos_days > 0 else 0
        
        # Convert back to percentage
        is_annual_return_pct = is_annual_return * 100
        oos_annual_return_pct = oos_annual_return * 100
        
        # Calculate WFE = (Annualized OOS Return) / (Annualized IS Return) × 100%
        wfe = (oos_annual_return_pct / is_annual_return_pct * 100) if is_annual_return_pct != 0 else 0
        
        return {
            'is_return_pct': is_return,
            'oos_return_pct': oos_return,
            'is_annual_return_pct': is_annual_return_pct,
            'oos_annual_return_pct': oos_annual_return_pct,
            'wfe': wfe,
            'is_period_days': is_days,
            'oos_period_days': oos_days
        }
    
    def _generate_period_charts(self, is_data: pd.DataFrame, oos_data: pd.DataFrame, 
                               is_result: Any, oos_result: Any, symbol: str, period_number: int) -> List[str]:
        """Generate charts for the walk-forward period."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Create output directory
            output_dir = Path("output/walk_forward/charts")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            chart_paths = []
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            clean_symbol = symbol.replace('/', '_')
            
            # Chart 1: Price action with IS/OOS split
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot price data
            combined_data = pd.concat([is_data, oos_data])
            ax1.plot(is_data.index, is_data['close'], 'b-', label='In-Sample (IS)', alpha=0.7)
            ax1.plot(oos_data.index, oos_data['close'], 'r-', label='Out-of-Sample (OOS)', alpha=0.7)
            ax1.axvline(x=oos_data.index[0], color='black', linestyle='--', alpha=0.5, label='IS/OOS Split')
            ax1.set_title(f'{symbol} - Period {period_number} - Price Action')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot returns comparison
            is_return = getattr(is_result, 'Return [%]', 0)
            oos_return = getattr(oos_result, 'Return [%]', 0)
            
            returns_data = ['IS Return', 'OOS Return']
            returns_values = [is_return, oos_return]
            colors = ['blue', 'red']
            
            bars = ax2.bar(returns_data, returns_values, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title(f'Period {period_number} - Performance Comparison')
            ax2.set_ylabel('Return (%)')
            
            # Add value labels on bars
            for bar, value in zip(bars, returns_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                        f'{value:.2f}%', ha='center', va='bottom' if height > 0 else 'top')
            
            plt.tight_layout()
            chart1_path = output_dir / f"period_{period_number}_{clean_symbol}_{timestamp}.png"
            plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(str(chart1_path))
            
            return chart_paths
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping chart generation")
            return []
        except Exception as e:
            self.logger.warning(f"Chart generation failed for period {period_number}: {e}")
            return []
    
    def _create_combined_result(self, is_result: Any, oos_result: Any, wfe_metrics: Dict[str, float], 
                               metadata: Dict[str, Any]) -> Any:
        """Create combined result object with IS, OOS, and WFE data."""
        
        class WalkForwardResult:
            def __init__(self, is_res, oos_res, wfe_data, meta):
                # Copy OOS result attributes (primary result)
                oos_dict = oos_res.to_dict() if hasattr(oos_res, 'to_dict') else dict(oos_res)
                for key, value in oos_dict.items():
                    setattr(self, key, value)
                
                # Add IS result data with prefix
                is_dict = is_res.to_dict() if hasattr(is_res, 'to_dict') else dict(is_res)
                for key, value in is_dict.items():
                    setattr(self, f"is_{key.lower().replace(' ', '_').replace('[%]', '_pct').replace('.', '').replace('#', 'num')}", value)
                
                # Add WFE metrics
                for key, value in wfe_data.items():
                    setattr(self, key, value)
                
                # Add metadata
                self.metadata = meta
        
        return WalkForwardResult(is_result, oos_result, wfe_metrics, metadata)
    
    def _optimize_on_data(self, data: pd.DataFrame, symbol: str, period_label: str) -> tuple:
        """Optimize parameters on given data using grid search."""
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        
        if not param_combinations:
            self.logger.error(f"No parameter combinations generated for {period_label}")
            return None, None
        
        self.logger.info(f"{period_label}: Testing {len(param_combinations)} parameter combinations")
        
        best_result = None
        best_params = None
        best_metric_value = float('-inf') if self.maximize else float('inf')
        
        for i, params in enumerate(param_combinations):
            try:
                result = self._run_backtest_with_params(data, params, symbol, f"{period_label}_opt_{i}")
                
                if result is None:
                    continue
                
                # Get metric value
                metric_value = getattr(result, self.optimization_metric, None)
                
                if metric_value is None:
                    continue
                
                # Check if this is the best result so far
                is_better = (
                    (self.maximize and metric_value > best_metric_value) or
                    (not self.maximize and metric_value < best_metric_value)
                )
                
                if is_better:
                    best_metric_value = metric_value
                    best_result = result
                    best_params = params.copy()
                
            except Exception as e:
                self.logger.debug(f"Parameter combination {i} failed: {e}")
                continue
        
        if best_result is None:
            self.logger.error(f"No successful parameter combinations for {period_label}")
            return None, None
        
        return best_params, best_result
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from parameter space."""
        from itertools import product
        
        param_names = list(self.parameter_space.keys())
        param_values = []
        
        for param_name in param_names:
            param_range = self.parameter_space[param_name]
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Range format: generate 10 values between min and max
                min_val, max_val = param_range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    values = list(range(min_val, max_val + 1, max(1, (max_val - min_val) // 10)))
                else:
                    values = [min_val + i * (max_val - min_val) / 10 for i in range(11)]
            elif isinstance(param_range, list):
                # Discrete values
                values = param_range
            else:
                raise ValueError(f"Invalid parameter range format for {param_name}")
            
            param_values.append(values)
        
        # Generate all combinations
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _run_backtest_with_params(self, data: pd.DataFrame, params: Dict[str, Any], 
                                  symbol: str, period_label: str) -> Optional[Any]:
        """Run backtest with specific parameters."""
        try:
            # Prepare data for backtesting.py (needs capitalized columns)
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            test_data = data.rename(columns=column_mapping)
            
            # Create wrapper strategy class
            WrapperClass = create_wrapper_class(
                self.strategy_class, 
                params,
                risk_manager=self.risk_manager,
                debug=False
            )
            
            # Run backtest
            if self.use_fractional:
                bt = FractionalBacktest(
                    data=test_data,
                    strategy=WrapperClass,
                    cash=self.initial_capital,
                    commission=self.commission,
                    margin=self.margin,
                    exclusive_orders=True
                )
            else:
                bt = Backtest(
                    data=test_data,
                    strategy=WrapperClass,
                    cash=self.initial_capital,
                    commission=self.commission,
                    margin=self.margin,
                    exclusive_orders=True
                )
            
            result = bt.run()
            return result
            
        except Exception as e:
            self.logger.debug(f"Backtest failed for {period_label}: {e}")
            return None
    
    def _generate_overall_charts(self, results: List[Any], symbol: str) -> None:
        """Generate overall walk-forward analysis charts."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create output directory
            output_dir = Path("output/walk_forward/charts")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            clean_symbol = symbol.replace('/', '_')
            
            # Extract data for plotting
            periods = [r.metadata['period_number'] for r in results]
            is_returns = [getattr(r, 'is_return_pct', 0) for r in results]
            oos_returns = [getattr(r, 'oos_return_pct', 0) for r in results]
            wfe_values = [getattr(r, 'wfe', 0) for r in results]
            
            # Chart 1: WFE Analysis Overview
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # IS vs OOS Returns
            ax1.plot(periods, is_returns, 'bo-', label='In-Sample Returns', alpha=0.7)
            ax1.plot(periods, oos_returns, 'ro-', label='Out-of-Sample Returns', alpha=0.7)
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Return (%)')
            ax1.set_title('In-Sample vs Out-of-Sample Returns')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # WFE by Period
            colors = ['green' if w > 100 else 'orange' if w > 80 else 'red' for w in wfe_values]
            bars = ax2.bar(periods, wfe_values, color=colors, alpha=0.7)
            ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% (Perfect)')
            ax2.set_xlabel('Period')
            ax2.set_ylabel('WFE (%)')
            ax2.set_title('Walk-Forward Efficiency by Period')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # IS vs OOS Scatter Plot
            ax3.scatter(is_returns, oos_returns, alpha=0.7, s=60)
            
            # Add diagonal line for perfect correlation
            min_val = min(min(is_returns), min(oos_returns))
            max_val = max(max(is_returns), max(oos_returns))
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
            
            ax3.set_xlabel('In-Sample Return (%)')
            ax3.set_ylabel('Out-of-Sample Return (%)')
            correlation = np.corrcoef(is_returns, oos_returns)[0, 1] if len(is_returns) > 1 else 0
            ax3.set_title(f'IS vs OOS Correlation: {correlation:.3f}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # WFE Distribution
            ax4.hist(wfe_values, bins=min(10, len(wfe_values)), alpha=0.7, edgecolor='black')
            ax4.axvline(x=100, color='k', linestyle='--', alpha=0.5, label='100% (Perfect)')
            ax4.axvline(x=np.mean(wfe_values), color='r', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(wfe_values):.1f}%')
            ax4.set_xlabel('WFE (%)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('WFE Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            chart_path = output_dir / f"walk_forward_analysis_{clean_symbol}_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Overall analysis chart saved to: {chart_path}")
            
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available, skipping chart generation")
        except Exception as e:
            self.logger.warning(f"Overall chart generation failed: {e}")
    
    def _run_single_period(self, data: pd.DataFrame, 
                          test_start: datetime, test_end: datetime,
                          symbol: str, period_number: int) -> Optional[Any]:
        """Run testing for a single walk-forward period."""
        
        # Extract test data  
        test_data = data[(data.index >= test_start) & (data.index < test_end)].copy()
        if len(test_data) < 50:
            self.logger.warning(f"Insufficient test data: {len(test_data)} rows")
            return None
        
        try:
            # Prepare data for backtesting.py (needs capitalized columns)
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            test_data = test_data.rename(columns=column_mapping)
            
            # Create wrapper strategy class
            WrapperClass = create_wrapper_class(
                self.strategy_class, 
                self.optimal_parameters,
                risk_manager=self.risk_manager,
                debug=False
            )
            
            # Run backtest
            if self.use_fractional:
                bt = FractionalBacktest(
                    data=test_data,
                    strategy=WrapperClass,
                    cash=self.initial_capital,
                    commission=self.commission,
                    margin=self.margin,
                    exclusive_orders=True
                )
            else:
                bt = Backtest(
                    data=test_data,
                    strategy=WrapperClass,
                    cash=self.initial_capital,
                    commission=self.commission,
                    margin=self.margin,
                    exclusive_orders=True
                )
            
            result = bt.run()
            
            # Add metadata about the test period (create a new attribute)
            # Note: result is a pandas Series-like object, so we need to handle it carefully
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else dict(result)
            result_dict['metadata'] = {
                'period_number': period_number,
                'parameters_used': self.optimal_parameters,
                'test_start': test_start,
                'test_end': test_end,
                'test_data_points': len(test_data)
            }
            
            # Create a simple object to hold both result data and metadata
            class ResultWithMetadata:
                def __init__(self, result_data, metadata):
                    # Copy all attributes from original result
                    for key, value in result_data.items():
                        setattr(self, key, value)
                    self.metadata = metadata
            
            return ResultWithMetadata(result_dict, result_dict['metadata'])
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward period {period_number}: {e}")
            return None
    
    def _calculate_combined_metrics(self, results: List[Any]) -> Dict[str, float]:
        """Calculate combined metrics across all walk-forward periods."""
        if not results:
            return {}
        
        # Extract metrics from results (handling different attribute names)
        def safe_get_metric(result, metric_names):
            for name in metric_names:
                if hasattr(result, name):
                    value = getattr(result, name)
                    return value if pd.notna(value) else 0.0
            return 0.0
        
        # Try different attribute names for common metrics
        returns_pct = [safe_get_metric(r, ['Return [%]', 'return_pct', 'total_return_pct']) for r in results]
        sharpe_ratios = [safe_get_metric(r, ['Sharpe Ratio', 'sharpe_ratio']) for r in results]
        max_drawdowns = [safe_get_metric(r, ['Max. Drawdown [%]', 'max_drawdown', 'max_drawdown_pct']) for r in results]
        win_rates = [safe_get_metric(r, ['Win Rate [%]', 'win_rate', 'win_rate_pct']) for r in results]
        total_trades = [safe_get_metric(r, ['# Trades', 'num_trades', 'total_trades']) for r in results]
        
        # WFE specific metrics
        wfe_values = [safe_get_metric(r, ['wfe']) for r in results]
        is_returns = [safe_get_metric(r, ['is_return_pct']) for r in results]
        oos_returns = [safe_get_metric(r, ['oos_return_pct']) for r in results]
        is_annual_returns = [safe_get_metric(r, ['is_annual_return_pct']) for r in results]
        oos_annual_returns = [safe_get_metric(r, ['oos_annual_return_pct']) for r in results]
        
        # Calculate metrics including WFE
        metrics = {
            'total_periods': len(results),
            'avg_return_pct': np.mean(returns_pct),
            'median_return_pct': np.median(returns_pct),
            'std_return_pct': np.std(returns_pct),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': max(max_drawdowns) if max_drawdowns else 0.0,
            'best_max_drawdown': min(max_drawdowns) if max_drawdowns else 0.0,
            'avg_win_rate': np.mean(win_rates),
            'avg_trades_per_period': np.mean(total_trades),
            'positive_return_periods': sum(1 for r in returns_pct if r > 0),
            'positive_return_pct': (sum(1 for r in returns_pct if r > 0) / len(returns_pct)) * 100,
            # WFE specific metrics
            'avg_wfe': np.mean(wfe_values),
            'median_wfe': np.median(wfe_values),
            'wfe_std': np.std(wfe_values),
            'avg_is_return_pct': np.mean(is_returns),
            'avg_oos_return_pct': np.mean(oos_returns),
            'avg_is_annual_return_pct': np.mean(is_annual_returns),
            'avg_oos_annual_return_pct': np.mean(oos_annual_returns),
            'positive_wfe_periods': sum(1 for w in wfe_values if w > 100),
            'positive_wfe_pct': (sum(1 for w in wfe_values if w > 100) / len(wfe_values)) * 100,
            'is_oos_correlation': np.corrcoef(is_returns, oos_returns)[0, 1] if len(is_returns) > 1 else 0
        }
        
        return metrics
    
    def _calculate_stability_metrics(self, results: List[Any]) -> Dict[str, float]:
        """Calculate metrics measuring performance stability across periods."""
        if not results:
            return {}
        
        # Extract return percentages
        def safe_get_metric(result, metric_names):
            for name in metric_names:
                if hasattr(result, name):
                    value = getattr(result, name)
                    return value if pd.notna(value) else 0.0
            return 0.0
        
        returns_pct = [safe_get_metric(r, ['Return [%]', 'return_pct', 'total_return_pct']) for r in results]
        sharpe_ratios = [safe_get_metric(r, ['Sharpe Ratio', 'sharpe_ratio']) for r in results]
        
        stability_metrics = {
            'return_pct_volatility': np.std(returns_pct),
            'sharpe_volatility': np.std(sharpe_ratios),
            'return_consistency': np.mean(returns_pct) / np.std(returns_pct) if np.std(returns_pct) > 0 else 0.0,
            'temporal_stability': self._calculate_temporal_stability(returns_pct)
        }
        
        # Calculate rolling correlations if we have enough periods
        if len(results) >= 4:
            stability_metrics.update(self._calculate_rolling_stability(returns_pct))
        
        return stability_metrics
    
    def _calculate_temporal_stability(self, returns: List[float]) -> float:
        """Calculate temporal stability - how consistent performance is over time."""
        if len(returns) < 3:
            return 1.0
        
        # Calculate period-to-period changes
        changes = [returns[i+1] - returns[i] for i in range(len(returns) - 1)]
        
        # Count direction changes (positive to negative or vice versa)
        direction_changes = 0
        for i in range(len(changes) - 1):
            if (changes[i] > 0) != (changes[i+1] > 0):
                direction_changes += 1
        
        # Calculate temporal stability (lower direction changes = higher stability)
        max_possible_changes = len(changes) - 1
        if max_possible_changes > 0:
            stability = 1.0 - (direction_changes / max_possible_changes)
        else:
            stability = 1.0
        
        return stability
    
    def _calculate_rolling_stability(self, returns: List[float]) -> Dict[str, float]:
        """Calculate rolling performance stability metrics."""
        # Calculate rolling average stability
        window_size = min(4, len(returns) // 2)
        rolling_means = []
        
        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i:i + window_size]
            rolling_means.append(np.mean(window_returns))
        
        rolling_stability = {
            'rolling_mean_stability': 1.0 / (1.0 + np.std(rolling_means)) if len(rolling_means) > 1 else 1.0,
            'trend_consistency': self._calculate_trend_consistency(returns)
        }
        
        return rolling_stability
    
    def _calculate_trend_consistency(self, returns: List[float]) -> float:
        """Calculate how consistent the performance trend is across periods."""
        if len(returns) < 3:
            return 1.0
        
        # Calculate period-to-period changes
        changes = [returns[i+1] - returns[i] for i in range(len(returns) - 1)]
        
        # Count direction changes (positive to negative or vice versa)
        direction_changes = 0
        for i in range(len(changes) - 1):
            if (changes[i] > 0) != (changes[i+1] > 0):
                direction_changes += 1
        
        # Calculate trend consistency (lower direction changes = higher consistency)
        max_possible_changes = len(changes) - 1
        if max_possible_changes > 0:
            consistency = 1.0 - (direction_changes / max_possible_changes)
        else:
            consistency = 1.0
        
        return consistency