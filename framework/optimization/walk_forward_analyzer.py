"""
Walk-forward analysis implementation for strategy validation.

Walk-forward analysis tests strategy robustness by optimizing parameters on 
in-sample (IS) data and testing on out-of-sample (OOS) data across rolling 
time windows. Uses GridSearchOptimizer for consistent optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
from pathlib import Path
import time

from framework.strategies.base_strategy import BaseStrategy
from framework.risk.base_risk_manager import BaseRiskManager
from framework.risk.fixed_position_size_manager import FixedPositionSizeManager
from framework.risk.fixed_risk_manager import FixedRiskManager
from framework.utils.logger import setup_logger
from .grid_search import GridSearchOptimizer


class WalkForwardAnalyzer:
    """
    Walk-forward analysis using GridSearchOptimizer.
    
    Optimizes parameters on in-sample data and tests on out-of-sample data
    across rolling time windows. Calculates Walk-Forward Efficiency (WFE)
    to measure parameter robustness.
    """
    
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 parameter_config: Dict[str, Any],
                 is_window_months: int = 3,
                 oos_window_months: int = 1,
                 step_months: int = 1,
                 window_mode: str = "rolling",
                 initial_capital: float = 10000.0,
                 commission: float = 0.0005,
                 margin: float = 0.01,
                 risk_manager_type: str = "fixed_risk",
                 risk_manager_params: Optional[Dict[str, Any]] = None,
                 optimization_metric: str = "return_pct",
                 maximize: bool = True,
                 n_jobs: int = 4):
        """
        Initialize Walk-Forward Analyzer.
        
        Args:
            strategy_class: Strategy class to analyze
            parameter_config: Parameter configuration for GridSearchOptimizer
            is_window_months: Months of in-sample data for optimization
            oos_window_months: Months of out-of-sample data for testing
            step_months: Months to step forward between periods
            window_mode: 'rolling' (fixed IS window) or 'anchored' (expanding IS window)
            initial_capital: Starting capital for backtests
            commission: Commission rate for trades
            margin: Margin requirement (0.01 = 100x leverage)
            risk_manager_type: Type of risk manager ('fixed_risk' or 'fixed_position')
            risk_manager_params: Parameters for risk manager
            optimization_metric: Metric to optimize for ('return_pct', 'sharpe_ratio', etc.)
            maximize: Whether to maximize or minimize the optimization metric
            n_jobs: Number of parallel jobs (-1 = auto-detect)
        """
        self.strategy_class = strategy_class
        self.parameter_config = parameter_config
        self.is_window_months = is_window_months
        self.oos_window_months = oos_window_months
        self.step_months = step_months
        self.window_mode = window_mode
        self.initial_capital = initial_capital
        self.commission = commission
        self.margin = margin
        self.risk_manager_type = risk_manager_type
        self.risk_manager_params = risk_manager_params or {}
        self.optimization_metric = optimization_metric
        self.maximize = maximize
        self.n_jobs = n_jobs
        
        self.logger = setup_logger("INFO")
        self._validate_parameters()
        
        # Create risk manager
        self.risk_manager = self._create_risk_manager()
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not self.parameter_config:
            raise ValueError("parameter_config cannot be empty")
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
    
    def analyze(self, data: pd.DataFrame, symbol: str = "UNKNOWN", output_dir: str = None) -> Dict[str, Any]:
        """
        Perform walk-forward analysis on the given data.
        
        Args:
            data: Historical price data with OHLCV columns
            symbol: Symbol identifier
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing walk-forward analysis results
        """
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data for walk-forward analysis")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Validate minimum data for meaningful analysis
        min_required_periods = max(100, ((self.is_window_months + self.oos_window_months) * 30))
        if len(data) < min_required_periods:
            raise ValueError(f"Need at least {min_required_periods} data points for {self.is_window_months + self.oos_window_months}-month periods")
        
        self.logger.info(f"Starting walk-forward analysis for {symbol}")
        self.logger.info(f"Parameter config: {self.parameter_config}")
        self.logger.info(f"Optimization metric: {self.optimization_metric} ({'maximize' if self.maximize else 'minimize'})")
        self.logger.info(f"IS window: {self.is_window_months} months, OOS window: {self.oos_window_months} months")
        self.logger.info(f"Window mode: {self.window_mode}, Step: {self.step_months} months")
        
        # Generate walk-forward periods with IS/OOS splits
        periods = self._generate_walk_forward_periods(data)
        self.logger.info(f"Generated {len(periods)} walk-forward periods")
        
        if not periods:
            raise ValueError("No valid walk-forward periods found")
        
        # Run analysis for each period
        results = []
        start_time = time.time()
        
        for i, (is_start, is_end, oos_start, oos_end) in enumerate(periods):
            period_num = i + 1
            elapsed = time.time() - start_time
            if i > 0:
                eta = (elapsed / i) * (len(periods) - i)
                eta_str = str(timedelta(seconds=int(eta)))
                self.logger.info(f"Period {period_num}/{len(periods)} (ETA: {eta_str})")
            else:
                self.logger.info(f"Period {period_num}/{len(periods)}")
            
            try:
                result = self._run_walk_forward_period(data, is_start, is_end, oos_start, oos_end, symbol, period_num)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Period {period_num} failed: {e}")
                continue
        
        if not results:
            raise ValueError("No successful walk-forward periods")
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(results)
        
        # Generate visualizations and save results
        if output_dir:
            self._save_results(results, summary_metrics, output_dir, symbol)
        
        return {
            'periods': results,
            'summary': summary_metrics,
            'analysis_info': {
                'strategy': self.strategy_class.__name__,
                'symbol': symbol,
                'total_periods': len(results),
                'is_window_months': self.is_window_months,
                'oos_window_months': self.oos_window_months,
                'step_months': self.step_months,
                'window_mode': self.window_mode,
                'optimization_metric': self.optimization_metric
            }
        }
    
    def _run_walk_forward_period(self, data: pd.DataFrame, 
                                   is_start: datetime, is_end: datetime,
                                   oos_start: datetime, oos_end: datetime,
                                   symbol: str, period_num: int) -> Optional[Dict[str, Any]]:
        """Run a single walk-forward period: optimize on IS, test on OOS."""
        
        # Extract IS and OOS data
        is_data = data[(data.index >= is_start) & (data.index < is_end)].copy()
        oos_data = data[(data.index >= oos_start) & (data.index < oos_end)].copy()
        
        if len(is_data) < 100 or len(oos_data) < 50:
            self.logger.warning(f"Period {period_num}: Insufficient data (IS={len(is_data)}, OOS={len(oos_data)})")
            return None
        
        try:
            # Step 1: Optimize on IS data using GridSearchOptimizer
            self.logger.info(f"Period {period_num}: Optimizing on IS data ({len(is_data)} points)")
            
            # Get default strategy parameters to merge with optimization parameters
            import inspect
            strategy_init_sig = inspect.signature(self.strategy_class.__init__)
            default_strategy_params = {}
            for param_name, param in strategy_init_sig.parameters.items():
                if param_name != 'self' and param.default != inspect.Parameter.empty:
                    default_strategy_params[param_name] = param.default
            
            # Create parameter config that includes defaults for non-optimized parameters
            complete_param_config = {}
            # First add all parameters being optimized
            complete_param_config.update(self.parameter_config)
            # Then add any missing default parameters as fixed values
            for param_name, default_value in default_strategy_params.items():
                if param_name not in complete_param_config:
                    complete_param_config[param_name] = {'values': [default_value]}
            
            optimizer = GridSearchOptimizer(
                strategy_class=self.strategy_class,
                parameter_config=complete_param_config,
                data=is_data,
                initial_capital=self.initial_capital,
                commission=self.commission,
                margin=self.margin,
                risk_manager=self.risk_manager,
                n_jobs=self.n_jobs,
                debug=False
            )
            
            # Run optimization
            is_results_df = optimizer.optimize(metric=self.optimization_metric, maximize=self.maximize)
            
            if is_results_df.empty:
                self.logger.warning(f"Period {period_num}: No optimization results")
                return None
            
            # Get best parameters
            best_row = is_results_df.iloc[0]
            
            # Extract parameter columns (exclude metric columns)
            metric_cols = ['return_pct', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                          'num_trades', 'exposure_time', 'profit_factor', 'avg_trade',
                          'best_trade', 'worst_trade', 'calmar_ratio', 'sortino_ratio']
            param_cols = [col for col in best_row.index if col not in metric_cols]
            best_params = {col: best_row[col] for col in param_cols}
            
            # Convert parameters to correct types (fix float -> int conversion issue)
            best_params = self._convert_parameter_types(best_params)
            is_return = best_row['return_pct']
            
            self.logger.info(f"Period {period_num}: Best IS {self.optimization_metric}: {best_row[self.optimization_metric]:.2f}")
            self.logger.info(f"Period {period_num}: IS trades: {best_row['num_trades']}, Best params: {best_params}")
            
            # Step 2: Test on OOS data
            self.logger.info(f"Period {period_num}: Testing on OOS data ({len(oos_data)} points)")
            
            # Create single-value parameter config for OOS testing
            # IMPORTANT: We need to include ALL strategy parameters, not just the optimized ones
            # Get default strategy parameters first
            import inspect
            strategy_init_sig = inspect.signature(self.strategy_class.__init__)
            default_strategy_params = {}
            for param_name, param in strategy_init_sig.parameters.items():
                if param_name != 'self' and param.default != inspect.Parameter.empty:
                    default_strategy_params[param_name] = param.default
            
            # Merge optimized parameters with defaults
            complete_params = default_strategy_params.copy()
            complete_params.update(best_params)
            
            # Create single-value parameter config for OOS testing with complete parameters
            oos_param_config = {}
            for param_name, param_value in complete_params.items():
                oos_param_config[param_name] = {'values': [param_value]}
            
            oos_optimizer = GridSearchOptimizer(
                strategy_class=self.strategy_class,
                parameter_config=oos_param_config,
                data=oos_data,
                initial_capital=self.initial_capital,
                commission=self.commission,
                margin=self.margin,
                risk_manager=self.risk_manager,
                n_jobs=1,
                debug=False
            )
            
            # Test on OOS (single run with optimized params)
            oos_results_df = oos_optimizer.optimize(metric=self.optimization_metric, maximize=self.maximize)
            
            if oos_results_df.empty:
                self.logger.warning(f"Period {period_num}: OOS test failed")
                return None
            
            oos_row = oos_results_df.iloc[0]
            oos_return = oos_row['return_pct']
            
            self.logger.info(f"Period {period_num}: OOS {self.optimization_metric}: {oos_row[self.optimization_metric]:.2f}")
            self.logger.info(f"Period {period_num}: OOS trades: {oos_row['num_trades']}")
            
            # Warn if OOS has 0 trades
            if oos_row['num_trades'] == 0:
                self.logger.warning(f"Period {period_num}: OOS period had 0 trades! This may indicate overfitting or parameter issues.")
            
            # Calculate WFE
            wfe_metrics = self._calculate_wfe(best_row, oos_row, is_data, oos_data)
            
            # Create period result
            result = {
                'period_num': period_num,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'best_params': best_params,
                'is_metrics': dict(best_row),
                'oos_metrics': dict(oos_row),
                'wfe_metrics': wfe_metrics,
                'is_data_points': len(is_data),
                'oos_data_points': len(oos_data)
            }
            
            return result
            
        except Exception as e:
            import traceback
            self.logger.error(f"Period {period_num} failed: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_wfe(self, is_row: pd.Series, oos_row: pd.Series, 
                       is_data: pd.DataFrame, oos_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Walk-Forward Efficiency metrics."""
        is_return = is_row['return_pct']
        oos_return = oos_row['return_pct']
        
        # Calculate period lengths in days for annualization
        is_days = (is_data.index[-1] - is_data.index[0]).days
        oos_days = (oos_data.index[-1] - oos_data.index[0]).days
        
        # Annualize returns (assuming 365 days per year)
        if is_days > 0 and is_return > -100:
            is_annual_return = (1 + is_return / 100) ** (365 / is_days) - 1
        else:
            is_annual_return = 0
            
        if oos_days > 0 and oos_return > -100:
            oos_annual_return = (1 + oos_return / 100) ** (365 / oos_days) - 1
        else:
            oos_annual_return = 0
        
        # Convert back to percentage
        is_annual_return_pct = is_annual_return * 100
        oos_annual_return_pct = oos_annual_return * 100
        
        # Calculate WFE = (Annualized OOS Return) / (Annualized IS Return) × 100%
        # Handle edge cases properly
        if abs(is_annual_return_pct) < 0.1:  # IS annual return near zero
            wfe = 0.0  # Cannot calculate meaningful WFE with near-zero IS return
        else:
            # Standard WFE calculation using annualized returns
            wfe = (oos_annual_return_pct / is_annual_return_pct) * 100
        
        # Cap WFE at reasonable bounds (-500% to +500%)
        wfe = max(-500.0, min(wfe, 500.0))
        
        return {
            'is_return_pct': is_return,
            'oos_return_pct': oos_return,
            'is_annual_return_pct': is_annual_return_pct,
            'oos_annual_return_pct': oos_annual_return_pct,
            'wfe': wfe,
            'is_period_days': is_days,
            'oos_period_days': oos_days
        }
    
    def _calculate_summary_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary metrics across all walk-forward periods."""
        if not results:
            return {}
        
        # Extract key metrics
        is_returns = [r['wfe_metrics']['is_return_pct'] for r in results]
        oos_returns = [r['wfe_metrics']['oos_return_pct'] for r in results]
        wfe_values = [r['wfe_metrics']['wfe'] for r in results]
        is_sharpe = [r['is_metrics']['sharpe_ratio'] for r in results]
        oos_sharpe = [r['oos_metrics']['sharpe_ratio'] for r in results]
        is_trades = [r['is_metrics']['num_trades'] for r in results]
        oos_trades = [r['oos_metrics']['num_trades'] for r in results]
        
        summary = {
            'total_periods': len(results),
            'avg_is_return_pct': np.mean(is_returns),
            'avg_oos_return_pct': np.mean(oos_returns),
            'median_oos_return_pct': np.median(oos_returns),
            'std_oos_return_pct': np.std(oos_returns),
            'avg_wfe': np.mean(wfe_values),
            'median_wfe': np.median(wfe_values),
            'std_wfe': np.std(wfe_values),
            'avg_is_sharpe': np.mean(is_sharpe),
            'avg_oos_sharpe': np.mean(oos_sharpe),
            'avg_is_trades': np.mean(is_trades),
            'avg_oos_trades': np.mean(oos_trades),
            'positive_oos_periods': sum(1 for r in oos_returns if r > 0),
            'positive_oos_pct': (sum(1 for r in oos_returns if r > 0) / len(oos_returns)) * 100,
            'positive_wfe_periods': sum(1 for w in wfe_values if w > 100),
            'positive_wfe_pct': (sum(1 for w in wfe_values if w > 100) / len(wfe_values)) * 100,
            'is_oos_correlation': np.corrcoef(is_returns, oos_returns)[0, 1] if len(is_returns) > 1 else 0,
            'wfe_consistency': 1.0 / (1.0 + np.std(wfe_values) / 100) if np.std(wfe_values) > 0 else 1.0
        }
        
        return summary
    
    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, float], 
                      output_dir: str, symbol: str) -> None:
        """Save walk-forward results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results to CSV
        self._save_csv_results(results, output_path, symbol)
        
        # Save summary to JSON
        self._save_json_summary(results, summary, output_path, symbol)
        
        # Generate visualizations
        self._generate_walk_forward_charts(results, summary, output_path, symbol)
    
    def _save_csv_results(self, results: List[Dict[str, Any]], output_path: Path, symbol: str) -> None:
        """Save detailed period results to CSV."""
        rows = []
        for r in results:
            row = {
                'period': r['period_num'],
                'is_start': r['is_start'].strftime('%Y-%m-%d'),
                'is_end': r['is_end'].strftime('%Y-%m-%d'),
                'oos_start': r['oos_start'].strftime('%Y-%m-%d'),
                'oos_end': r['oos_end'].strftime('%Y-%m-%d'),
                'is_return_pct': r['wfe_metrics']['is_return_pct'],
                'oos_return_pct': r['wfe_metrics']['oos_return_pct'],
                'wfe': r['wfe_metrics']['wfe'],
                'is_sharpe': r['is_metrics']['sharpe_ratio'],
                'oos_sharpe': r['oos_metrics']['sharpe_ratio'],
                'is_trades': r['is_metrics']['num_trades'],
                'oos_trades': r['oos_metrics']['num_trades'],
                'best_params': json.dumps(r['best_params'])
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = output_path / f"walk_forward_periods_{symbol.replace('/', '_')}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Period results saved to: {csv_path}")
    
    def _save_json_summary(self, results: List[Dict[str, Any]], summary: Dict[str, float], 
                           output_path: Path, symbol: str) -> None:
        """Save summary metrics to JSON."""
        summary_data = {
            'analysis_summary': summary,
            'strategy': self.strategy_class.__name__,
            'symbol': symbol,
            'total_periods': len(results),
            'analysis_parameters': {
                'is_window_months': self.is_window_months,
                'oos_window_months': self.oos_window_months,
                'step_months': self.step_months,
                'window_mode': self.window_mode,
                'optimization_metric': self.optimization_metric,
                'parameter_config': self.parameter_config,
                'risk_manager_type': self.risk_manager_type
            },
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = output_path / f"walk_forward_summary_{symbol.replace('/', '_')}.json"
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        self.logger.info(f"Summary saved to: {json_path}")
    
    def _generate_walk_forward_charts(self, results: List[Dict[str, Any]], summary: Dict[str, float], 
                                      output_path: Path, symbol: str) -> None:
        """Generate walk-forward specific visualizations."""
        try:
            clean_symbol = symbol.replace('/', '_')
            
            # Create 2x2 subplot layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Walk-Forward Analysis - {symbol}', fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            periods = [r['period_num'] for r in results]
            is_returns = [r['wfe_metrics']['is_return_pct'] for r in results]
            oos_returns = [r['wfe_metrics']['oos_return_pct'] for r in results]
            wfe_values = [r['wfe_metrics']['wfe'] for r in results]
            period_dates = [r['oos_start'].strftime('%Y-%m') for r in results]
            
            # 1. IS vs OOS Performance Over Time
            ax1.plot(periods, is_returns, 'bo-', label='In-Sample', alpha=0.8, linewidth=2)
            ax1.plot(periods, oos_returns, 'ro-', label='Out-of-Sample', alpha=0.8, linewidth=2)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title('IS vs OOS Performance Over Time', fontweight='bold')
            ax1.set_xlabel('Period')
            ax1.set_ylabel('Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Walk-Forward Efficiency by Period
            colors = ['green' if w >= 100 else 'orange' if w >= 80 else 'red' for w in wfe_values]
            bars = ax2.bar(periods, wfe_values, color=colors, alpha=0.7)
            ax2.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='100% (Perfect)')
            ax2.set_title('Walk-Forward Efficiency by Period', fontweight='bold')
            ax2.set_xlabel('Period')
            ax2.set_ylabel('WFE (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add WFE values as text on bars
            for bar, wfe in zip(bars, wfe_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{wfe:.0f}%', ha='center', va='bottom', fontsize=9)
            
            # 3. IS vs OOS Scatter Plot
            ax3.scatter(is_returns, oos_returns, alpha=0.7, s=80, c=wfe_values, cmap='RdYlGn', vmin=0, vmax=200)
            
            # Add diagonal line for perfect correlation
            min_val = min(min(is_returns), min(oos_returns))
            max_val = max(max(is_returns), max(oos_returns))
            ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
            
            # Add colorbar
            cbar = plt.colorbar(ax3.collections[0], ax=ax3)
            cbar.set_label('WFE (%)')
            
            correlation = summary.get('is_oos_correlation', 0)
            ax3.set_title(f'IS vs OOS Correlation: {correlation:.3f}', fontweight='bold')
            ax3.set_xlabel('In-Sample Return (%)')
            ax3.set_ylabel('Out-of-Sample Return (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. WFE Summary Statistics
            ax4.axis('off')
            
            # Create summary text
            summary_text = f"""
Walk-Forward Analysis Summary

Total Periods: {summary.get('total_periods', 0)}
Average WFE: {summary.get('avg_wfe', 0):.1f}%
Median WFE: {summary.get('median_wfe', 0):.1f}%
WFE Std Dev: {summary.get('std_wfe', 0):.1f}%

Positive OOS Periods: {summary.get('positive_oos_periods', 0)}/{summary.get('total_periods', 0)} ({summary.get('positive_oos_pct', 0):.1f}%)
Positive WFE Periods: {summary.get('positive_wfe_periods', 0)}/{summary.get('total_periods', 0)} ({summary.get('positive_wfe_pct', 0):.1f}%)

Avg IS Return: {summary.get('avg_is_return_pct', 0):.2f}%
Avg OOS Return: {summary.get('avg_oos_return_pct', 0):.2f}%
IS-OOS Correlation: {summary.get('is_oos_correlation', 0):.3f}

WFE Consistency: {summary.get('wfe_consistency', 0):.3f}
            """.strip()
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            # Add WFE interpretation
            avg_wfe = summary.get('avg_wfe', 0)
            if avg_wfe >= 100:
                interpretation = "🟢 Excellent: Strategy maintains performance OOS"
            elif avg_wfe >= 80:
                interpretation = "🟡 Good: Reasonable OOS robustness"
            elif avg_wfe >= 60:
                interpretation = "🟠 Moderate: Some OOS degradation"
            else:
                interpretation = "🔴 Poor: Likely overfitted to training data"
            
            ax4.text(0.05, 0.05, interpretation, transform=ax4.transAxes,
                    fontsize=14, fontweight='bold', verticalalignment='bottom')
            
            plt.tight_layout()
            chart_path = output_path / f"walk_forward_analysis_{clean_symbol}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Walk-forward chart saved to: {chart_path}")
            
        except Exception as e:
            self.logger.warning(f"Chart generation failed: {e}")
    
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
    
    def _convert_parameter_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameters to correct types based on parameter config."""
        converted_params = {}
        
        for param_name, param_value in params.items():
            if param_name in self.parameter_config:
                param_def = self.parameter_config[param_name]
                param_type = param_def.get('type', 'float')
                
                if param_type == 'int':
                    converted_params[param_name] = int(round(param_value))
                else:
                    converted_params[param_name] = float(param_value)
            else:
                # Default conversion for parameters not in config
                if isinstance(param_value, float) and param_value.is_integer():
                    converted_params[param_name] = int(param_value)
                else:
                    converted_params[param_name] = param_value
        
        return converted_params