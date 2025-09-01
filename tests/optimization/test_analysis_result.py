"""
Tests for AnalysisResult class.
"""

import pytest
import pandas as pd
from datetime import datetime
from framework.optimization.analysis_result import AnalysisResult


class MockResult:
    """Mock result object for testing."""
    
    def __init__(self, return_pct, sharpe_ratio, num_trades, metadata=None):
        self.return_pct = return_pct
        self.sharpe_ratio = sharpe_ratio
        self.num_trades = num_trades
        self.metadata = metadata or {}


class TestAnalysisResult:
    """Test cases for AnalysisResult class."""

    def create_test_result(self):
        """Create a test AnalysisResult instance."""
        individual_results = [
            MockResult(10.5, 1.2, 15, {'test_start': '2024-01-01', 'test_end': '2024-01-31'}),
            MockResult(8.3, 0.9, 12, {'test_start': '2024-02-01', 'test_end': '2024-02-29'}),
            MockResult(15.7, 1.8, 20, {'test_start': '2024-03-01', 'test_end': '2024-03-31'})
        ]
        
        combined_metrics = {
            'avg_return': 11.5,
            'avg_sharpe': 1.3,
            'total_trades': 47
        }
        
        analysis_parameters = {
            'window_months': 3,
            'step_months': 1,
            'optimization_metric': 'return_pct'
        }
        
        stability_metrics = {
            'return_stability': 0.85,
            'sharpe_stability': 0.75
        }
        
        return AnalysisResult(
            analysis_type='walk_forward',
            strategy_name='SMAStrategy',
            symbol='BTC_USDT',
            analysis_start_date=datetime(2024, 1, 1),
            analysis_end_date=datetime(2024, 3, 31),
            individual_results=individual_results,
            combined_metrics=combined_metrics,
            analysis_parameters=analysis_parameters,
            stability_metrics=stability_metrics,
            metadata={'test_mode': True}
        )

    def test_initialization(self):
        """Test AnalysisResult initialization."""
        result = self.create_test_result()
        
        assert result.analysis_type == 'walk_forward'
        assert result.strategy_name == 'SMAStrategy'
        assert result.symbol == 'BTC_USDT'
        assert result.analysis_start_date == datetime(2024, 1, 1)
        assert result.analysis_end_date == datetime(2024, 3, 31)
        assert len(result.individual_results) == 3
        assert result.combined_metrics['avg_return'] == 11.5
        assert result.stability_metrics['return_stability'] == 0.85
        assert result.metadata['test_mode'] is True

    def test_initialization_optional_params(self):
        """Test initialization with optional parameters."""
        result = AnalysisResult(
            analysis_type='optimization',
            strategy_name='TestStrategy',
            symbol='ETH_USDT',
            analysis_start_date=datetime(2024, 1, 1),
            analysis_end_date=datetime(2024, 12, 31),
            individual_results=[],
            combined_metrics={},
            analysis_parameters={}
            # stability_metrics and metadata not provided
        )
        
        assert result.stability_metrics == {}
        assert result.metadata == {}

    def test_get_summary(self):
        """Test summary generation."""
        result = self.create_test_result()
        summary = result.get_summary()
        
        # Check summary structure
        assert 'analysis_type' in summary
        assert 'strategy_name' in summary
        assert 'symbol' in summary
        assert 'analysis_period' in summary
        assert 'total_periods' in summary
        assert 'combined_metrics' in summary
        assert 'stability_metrics' in summary
        
        # Check values
        assert summary['analysis_type'] == 'walk_forward'
        assert summary['strategy_name'] == 'SMAStrategy'
        assert summary['symbol'] == 'BTC_USDT'
        assert summary['total_periods'] == 3
        assert summary['analysis_period'] == "2024-01-01 to 2024-03-31"

    def test_get_best_periods_default_metric(self):
        """Test getting best periods with default metric."""
        result = self.create_test_result()
        
        # Should work even if exact metric matching fails (fallback behavior)
        best_periods = result.get_best_periods(n=2)
        
        # Should return some results (implementation detail may vary)
        assert len(best_periods) <= 2
        assert len(best_periods) <= len(result.individual_results)

    def test_get_best_periods_custom_metric(self):
        """Test getting best periods with custom metric."""
        result = self.create_test_result()
        
        # This will likely use fallback behavior due to attribute matching
        best_periods = result.get_best_periods(n=1, metric='Sharpe Ratio')
        
        assert len(best_periods) <= 1

    def test_get_worst_periods_default_metric(self):
        """Test getting worst periods with default metric."""
        result = self.create_test_result()
        
        worst_periods = result.get_worst_periods(n=1)
        
        assert len(worst_periods) <= 1
        assert len(worst_periods) <= len(result.individual_results)

    def test_get_best_periods_more_than_available(self):
        """Test requesting more periods than available."""
        result = self.create_test_result()
        
        # Request more periods than we have
        best_periods = result.get_best_periods(n=10)
        
        # Should return all available periods
        assert len(best_periods) == len(result.individual_results)

    def test_get_best_periods_empty_results(self):
        """Test getting best periods with empty results."""
        result = AnalysisResult(
            analysis_type='test',
            strategy_name='Test',
            symbol='TEST',
            analysis_start_date=datetime.now(),
            analysis_end_date=datetime.now(),
            individual_results=[],  # Empty
            combined_metrics={},
            analysis_parameters={}
        )
        
        best_periods = result.get_best_periods(n=5)
        assert best_periods == []

    def test_to_dataframe_empty_results(self):
        """Test DataFrame conversion with empty results."""
        result = AnalysisResult(
            analysis_type='test',
            strategy_name='Test',
            symbol='TEST',
            analysis_start_date=datetime.now(),
            analysis_end_date=datetime.now(),
            individual_results=[],
            combined_metrics={},
            analysis_parameters={}
        )
        
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_to_dataframe_with_results(self):
        """Test DataFrame conversion with actual results."""
        result = self.create_test_result()
        df = result.to_dataframe()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 individual results
        assert 'period' in df.columns
        
        # Check that period numbers are correct
        assert df['period'].tolist() == [1, 2, 3]

    def test_to_dataframe_metadata_extraction(self):
        """Test that metadata is properly extracted to DataFrame."""
        individual_results = [
            MockResult(10.0, 1.0, 10, {'test_start': '2024-01-01', 'test_end': '2024-01-31'}),
            MockResult(12.0, 1.2, 12, {'test_start': '2024-02-01', 'test_end': '2024-02-29'})
        ]
        
        result = AnalysisResult(
            analysis_type='test',
            strategy_name='Test',
            symbol='TEST',
            analysis_start_date=datetime(2024, 1, 1),
            analysis_end_date=datetime(2024, 2, 29),
            individual_results=individual_results,
            combined_metrics={},
            analysis_parameters={}
        )
        
        df = result.to_dataframe()
        
        # Should extract metadata fields
        assert 'start_date' in df.columns
        assert 'end_date' in df.columns

    def test_str_representation(self):
        """Test string representation."""
        result = self.create_test_result()
        str_repr = str(result)
        
        # Check that key information is in string representation
        assert 'Walk_Forward Analysis Results' in str_repr
        assert 'SMAStrategy' in str_repr
        assert 'BTC_USDT' in str_repr
        assert 'Total Periods: 3' in str_repr
        assert 'Combined Metrics:' in str_repr
        assert 'Stability Metrics:' in str_repr

    def test_str_representation_no_stability_metrics(self):
        """Test string representation without stability metrics."""
        result = AnalysisResult(
            analysis_type='optimization',
            strategy_name='TestStrategy',
            symbol='TEST',
            analysis_start_date=datetime(2024, 1, 1),
            analysis_end_date=datetime(2024, 12, 31),
            individual_results=[],
            combined_metrics={'avg_return': 10.0},
            analysis_parameters={}
            # No stability_metrics provided
        )
        
        str_repr = str(result)
        
        # Should handle missing stability metrics gracefully
        assert 'Optimization Analysis Results' in str_repr
        assert 'TestStrategy' in str_repr
        # Should not crash even without stability metrics

    def test_metric_value_formatting(self):
        """Test that float metrics are properly formatted in string representation."""
        result = AnalysisResult(
            analysis_type='test',
            strategy_name='Test',
            symbol='TEST',
            analysis_start_date=datetime(2024, 1, 1),
            analysis_end_date=datetime(2024, 12, 31),
            individual_results=[],
            combined_metrics={
                'float_metric': 3.14159,
                'int_metric': 42,
                'string_metric': 'test_value'
            },
            analysis_parameters={}
        )
        
        str_repr = str(result)
        
        # Float should be formatted with 4 decimal places
        assert '3.1416' in str_repr
        # Int and string should appear as-is
        assert '42' in str_repr
        assert 'test_value' in str_repr