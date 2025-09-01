"""
Analysis result container for storing optimization and walk-forward analysis results.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd


class AnalysisResult:
    """
    Container for analysis results from optimization or walk-forward analysis.
    
    This class provides a standardized way to store and access results from
    different types of analysis (optimization, walk-forward, etc.) with
    consistent metrics and metadata.
    """
    
    def __init__(self,
                 analysis_type: str,
                 strategy_name: str,
                 symbol: str,
                 analysis_start_date: datetime,
                 analysis_end_date: datetime,
                 individual_results: List[Any],
                 combined_metrics: Dict[str, float],
                 analysis_parameters: Dict[str, Any],
                 stability_metrics: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize analysis result.
        
        Args:
            analysis_type: Type of analysis ('walk_forward', 'optimization', etc.)
            strategy_name: Name of the strategy analyzed
            symbol: Trading symbol
            analysis_start_date: Start date of analysis
            analysis_end_date: End date of analysis
            individual_results: List of individual backtest results
            combined_metrics: Aggregated metrics across all results
            analysis_parameters: Parameters used for the analysis
            stability_metrics: Metrics measuring performance stability
            metadata: Additional metadata
        """
        self.analysis_type = analysis_type
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.analysis_start_date = analysis_start_date
        self.analysis_end_date = analysis_end_date
        self.individual_results = individual_results
        self.combined_metrics = combined_metrics
        self.analysis_parameters = analysis_parameters
        self.stability_metrics = stability_metrics or {}
        self.metadata = metadata or {}
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        return {
            'analysis_type': self.analysis_type,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'analysis_period': f"{self.analysis_start_date.date()} to {self.analysis_end_date.date()}",
            'total_periods': len(self.individual_results),
            'combined_metrics': self.combined_metrics,
            'stability_metrics': self.stability_metrics
        }
    
    def get_best_periods(self, n: int = 5, metric: str = 'Return [%]') -> List[Any]:
        """
        Get the best performing periods.
        
        Args:
            n: Number of top periods to return
            metric: Metric to sort by (default: 'Return [%]')
            
        Returns:
            List of best performing results
        """
        if not self.individual_results:
            return []
        
        # Sort by the specified metric (assuming it exists in result attributes)
        try:
            sorted_results = sorted(
                self.individual_results, 
                key=lambda r: getattr(r, metric.replace(' [%]', '').replace(' ', '_').lower(), 0),
                reverse=True
            )
            return sorted_results[:n]
        except AttributeError:
            # Fallback to first n results if metric not found
            return self.individual_results[:n]
    
    def get_worst_periods(self, n: int = 5, metric: str = 'Return [%]') -> List[Any]:
        """
        Get the worst performing periods.
        
        Args:
            n: Number of bottom periods to return
            metric: Metric to sort by (default: 'Return [%]')
            
        Returns:
            List of worst performing results
        """
        if not self.individual_results:
            return []
        
        # Sort by the specified metric (ascending for worst)
        try:
            sorted_results = sorted(
                self.individual_results, 
                key=lambda r: getattr(r, metric.replace(' [%]', '').replace(' ', '_').lower(), 0),
                reverse=False
            )
            return sorted_results[:n]
        except AttributeError:
            # Fallback to last n results if metric not found
            return self.individual_results[-n:]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert individual results to a DataFrame for easy analysis.
        
        Returns:
            DataFrame with one row per period and columns for key metrics
        """
        if not self.individual_results:
            return pd.DataFrame()
        
        data = []
        for i, result in enumerate(self.individual_results):
            row = {
                'period': i + 1,
                'start_date': result.metadata.get('test_start') if hasattr(result, 'metadata') else None,
                'end_date': result.metadata.get('test_end') if hasattr(result, 'metadata') else None,
            }
            
            # Add common backtest metrics if available
            common_metrics = [
                'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]',
                'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio',
                'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]',
                'Max. Drawdown Duration', 'Avg. Drawdown Duration',
                '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                'Profit Factor', 'SQN'
            ]
            
            for metric in common_metrics:
                if hasattr(result, metric.replace(' [%]', '').replace(' ', '_').replace('.', '').replace('#', 'num').lower()):
                    row[metric] = getattr(result, metric.replace(' [%]', '').replace(' ', '_').replace('.', '').replace('#', 'num').lower())
                else:
                    # Try alternative attribute names
                    alt_names = [
                        metric.replace(' [%]', '_pct').replace(' ', '_').lower(),
                        metric.replace('[%]', 'pct').replace(' ', '_').lower(),
                        metric.replace(' ', '_').replace('[%]', '').lower()
                    ]
                    for alt_name in alt_names:
                        if hasattr(result, alt_name):
                            row[metric] = getattr(result, alt_name)
                            break
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def __str__(self) -> str:
        """String representation of the analysis result."""
        summary = self.get_summary()
        
        lines = [
            f"=== {self.analysis_type.title()} Analysis Results ===",
            f"Strategy: {summary['strategy_name']}",
            f"Symbol: {summary['symbol']}",
            f"Period: {summary['analysis_period']}",
            f"Total Periods: {summary['total_periods']}",
            "",
            "Combined Metrics:",
        ]
        
        for key, value in summary['combined_metrics'].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        
        if summary['stability_metrics']:
            lines.append("")
            lines.append("Stability Metrics:")
            for key, value in summary['stability_metrics'].items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)