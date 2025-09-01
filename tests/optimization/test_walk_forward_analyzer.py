"""
Tests for Walk-Forward Analyzer.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from framework.optimization.walk_forward_analyzer import WalkForwardAnalyzer
from framework.strategies.sma_strategy import SMAStrategy
from framework.risk.fixed_position_size_manager import FixedPositionSizeManager


class TestWalkForwardAnalyzer:
    """Test cases for WalkForwardAnalyzer."""

    def create_test_data(self, length=500):
        """Create test data for walk-forward analysis."""
        start_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, periods=length, freq='1H')
        
        # Create realistic price data with some trend
        base_price = 100.0
        trend = np.linspace(0, 20, length)
        noise = np.random.normal(0, 1, length)
        prices = base_price + trend + noise
        
        return pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, length)
        }, index=dates)

    def create_param_config(self):
        """Create parameter configuration for testing."""
        return {
            'short_window': {
                'min': 5,
                'max': 10,
                'step': 5,
                'type': 'int'
            },
            'long_window': {
                'min': 15,
                'max': 20,
                'step': 5,
                'type': 'int'
            }
        }

    def test_initialization_valid_params(self):
        """Test initialization with valid parameters."""
        param_config = self.create_param_config()
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            is_window_months=3,
            oos_window_months=1,
            step_months=1,
            window_mode="rolling",
            initial_capital=10000.0,
            commission=0.001,
            margin=0.01,
            risk_manager_type="fixed_position",
            optimization_metric="return_pct",
            maximize=True,
            n_jobs=1
        )
        
        assert analyzer.strategy_class == SMAStrategy
        assert analyzer.parameter_config == param_config
        assert analyzer.is_window_months == 3
        assert analyzer.oos_window_months == 1
        assert analyzer.step_months == 1
        assert analyzer.window_mode == "rolling"
        assert analyzer.optimization_metric == "return_pct"
        assert analyzer.maximize is True

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        param_config = self.create_param_config()
        
        # Empty parameter config
        with pytest.raises(ValueError, match="parameter_config cannot be empty"):
            WalkForwardAnalyzer(SMAStrategy, {})
        
        # Invalid window months
        with pytest.raises(ValueError, match="oos_window_months must be positive"):
            WalkForwardAnalyzer(SMAStrategy, param_config, oos_window_months=0)
        
        with pytest.raises(ValueError, match="step_months must be positive"):
            WalkForwardAnalyzer(SMAStrategy, param_config, step_months=0)
        
        with pytest.raises(ValueError, match="is_window_months must be positive"):
            WalkForwardAnalyzer(SMAStrategy, param_config, is_window_months=0)
        
        # Invalid window mode
        with pytest.raises(ValueError, match="window_mode must be 'rolling' or 'anchored'"):
            WalkForwardAnalyzer(SMAStrategy, param_config, window_mode="invalid")
        
        # Invalid capital/commission/margin
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            WalkForwardAnalyzer(SMAStrategy, param_config, initial_capital=-1000)
        
        with pytest.raises(ValueError, match="commission cannot be negative"):
            WalkForwardAnalyzer(SMAStrategy, param_config, commission=-0.1)
        
        with pytest.raises(ValueError, match="margin must be between 0 and 1"):
            WalkForwardAnalyzer(SMAStrategy, param_config, margin=1.5)

    def test_create_risk_manager_fixed_position(self):
        """Test risk manager creation for fixed position type."""
        param_config = self.create_param_config()
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            risk_manager_type="fixed_position",
            risk_manager_params={'position_size': 0.05}
        )
        
        assert isinstance(analyzer.risk_manager, FixedPositionSizeManager)
        assert analyzer.risk_manager.position_size == 0.05

    def test_create_risk_manager_fixed_risk(self):
        """Test risk manager creation for fixed risk type."""
        from framework.risk.fixed_risk_manager import FixedRiskManager
        
        param_config = self.create_param_config()
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            risk_manager_type="fixed_risk",
            risk_manager_params={'risk_percent': 0.02, 'default_stop_distance': 0.03}
        )
        
        assert isinstance(analyzer.risk_manager, FixedRiskManager)
        assert analyzer.risk_manager.risk_percent == 0.02
        assert analyzer.risk_manager.default_stop_distance == 0.03

    def test_create_risk_manager_invalid_type(self):
        """Test risk manager creation with invalid type."""
        param_config = self.create_param_config()
        
        with pytest.raises(ValueError, match="Unknown risk manager type"):
            WalkForwardAnalyzer(
                strategy_class=SMAStrategy,
                parameter_config=param_config,
                risk_manager_type="invalid_type"
            )

    def test_generate_walk_forward_periods_rolling(self):
        """Test walk-forward period generation in rolling mode."""
        data = self.create_test_data(2000)  # ~3 months of hourly data
        param_config = self.create_param_config()
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            is_window_months=1,
            oos_window_months=1,
            step_months=1,
            window_mode="rolling"
        )
        
        periods = analyzer._generate_walk_forward_periods(data)
        
        # Should have multiple periods
        assert len(periods) > 0
        
        # Check period structure
        for is_start, is_end, oos_start, oos_end in periods:
            assert isinstance(is_start, datetime)
            assert isinstance(is_end, datetime)
            assert isinstance(oos_start, datetime)
            assert isinstance(oos_end, datetime)
            assert is_start < is_end
            assert is_end == oos_start  # OOS should start where IS ends
            assert oos_start < oos_end

    def test_generate_walk_forward_periods_anchored(self):
        """Test walk-forward period generation in anchored mode."""
        data = self.create_test_data(2000)
        param_config = self.create_param_config()
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            is_window_months=2,
            oos_window_months=1,
            step_months=1,
            window_mode="anchored"
        )
        
        periods = analyzer._generate_walk_forward_periods(data)
        
        # Should have multiple periods
        assert len(periods) > 0
        
        # In anchored mode, all IS periods should start from the same date
        first_is_start = periods[0][0]
        for is_start, is_end, oos_start, oos_end in periods:
            assert is_start == first_is_start  # All start from same date

    def test_convert_parameter_types(self):
        """Test parameter type conversion."""
        param_config = {
            'int_param': {'type': 'int'},
            'float_param': {'type': 'float'},
            'unknown_param': {}
        }
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config
        )
        
        params = {
            'int_param': 10.7,  # Should be converted to int
            'float_param': 15.5,  # Should remain float
            'unknown_param': 20.3,  # Should handle gracefully
            'not_in_config': 25.8  # Should handle gracefully
        }
        
        converted = analyzer._convert_parameter_types(params)
        
        assert converted['int_param'] == 11  # Rounded to int
        assert converted['float_param'] == 15.5  # Unchanged
        assert isinstance(converted['unknown_param'], (int, float))
        assert isinstance(converted['not_in_config'], (int, float))

    def test_calculate_wfe_normal_case(self):
        """Test WFE calculation for normal case."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        # Create mock data for calculation
        is_data = pd.DataFrame(index=pd.date_range('2024-01-01', periods=90, freq='D'))
        oos_data = pd.DataFrame(index=pd.date_range('2024-04-01', periods=30, freq='D'))
        
        # Create mock series with returns
        is_row = pd.Series({'return_pct': 10.0, 'sharpe_ratio': 1.0})
        oos_row = pd.Series({'return_pct': 8.0, 'sharpe_ratio': 0.8})
        
        wfe_metrics = analyzer._calculate_wfe(is_row, oos_row, is_data, oos_data)
        
        # Check structure
        assert 'is_return_pct' in wfe_metrics
        assert 'oos_return_pct' in wfe_metrics
        assert 'wfe' in wfe_metrics
        assert 'is_annual_return_pct' in wfe_metrics
        assert 'oos_annual_return_pct' in wfe_metrics
        
        # Check values make sense
        assert wfe_metrics['is_return_pct'] == 10.0
        assert wfe_metrics['oos_return_pct'] == 8.0
        assert isinstance(wfe_metrics['wfe'], float)

    def test_calculate_wfe_edge_cases(self):
        """Test WFE calculation for edge cases."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        is_data = pd.DataFrame(index=pd.date_range('2024-01-01', periods=90, freq='D'))
        oos_data = pd.DataFrame(index=pd.date_range('2024-04-01', periods=30, freq='D'))
        
        # Near-zero IS return
        is_row = pd.Series({'return_pct': 0.05, 'sharpe_ratio': 0.1})
        oos_row = pd.Series({'return_pct': 5.0, 'sharpe_ratio': 0.5})
        
        wfe_metrics = analyzer._calculate_wfe(is_row, oos_row, is_data, oos_data)
        
        # Should handle near-zero IS return
        assert wfe_metrics['wfe'] == 0.0  # Should be set to 0 for near-zero IS return

    def test_calculate_summary_metrics(self):
        """Test summary metrics calculation."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        # Create mock results
        results = [
            {
                'wfe_metrics': {'is_return_pct': 10.0, 'oos_return_pct': 8.0, 'wfe': 80.0},
                'is_metrics': {'sharpe_ratio': 1.0, 'num_trades': 10},
                'oos_metrics': {'sharpe_ratio': 0.8, 'num_trades': 8}
            },
            {
                'wfe_metrics': {'is_return_pct': 15.0, 'oos_return_pct': 12.0, 'wfe': 80.0},
                'is_metrics': {'sharpe_ratio': 1.5, 'num_trades': 12},
                'oos_metrics': {'sharpe_ratio': 1.2, 'num_trades': 10}
            }
        ]
        
        summary = analyzer._calculate_summary_metrics(results)
        
        # Check structure
        assert 'total_periods' in summary
        assert 'avg_is_return_pct' in summary
        assert 'avg_oos_return_pct' in summary
        assert 'avg_wfe' in summary
        assert 'positive_oos_periods' in summary
        assert 'positive_oos_pct' in summary
        
        # Check values
        assert summary['total_periods'] == 2
        assert summary['avg_is_return_pct'] == 12.5  # (10 + 15) / 2
        assert summary['avg_oos_return_pct'] == 10.0  # (8 + 12) / 2
        assert summary['avg_wfe'] == 80.0
        assert summary['positive_oos_periods'] == 2  # Both positive
        assert summary['positive_oos_pct'] == 100.0

    def test_calculate_summary_metrics_empty(self):
        """Test summary metrics calculation with empty results."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        summary = analyzer._calculate_summary_metrics([])
        
        assert summary == {}

    def test_analyze_insufficient_data(self):
        """Test analysis with insufficient data."""
        data = self.create_test_data(50)  # Very small dataset
        param_config = self.create_param_config()
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            is_window_months=1,
            oos_window_months=1
        )
        
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.analyze(data, symbol="TEST")

    def test_analyze_invalid_data_format(self):
        """Test analysis with invalid data format."""
        param_config = self.create_param_config()
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config
        )
        
        # Test with empty data
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.analyze(pd.DataFrame(), symbol="TEST")
        
        # Test with non-datetime index
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [100, 101, 102]
        })
        
        with pytest.raises(ValueError, match="Data must have DatetimeIndex"):
            analyzer.analyze(invalid_data, symbol="TEST")

    @patch('framework.optimization.walk_forward_analyzer.ManualGridSearchOptimizer')
    def test_run_walk_forward_period_success(self, mock_optimizer_class):
        """Test successful walk-forward period execution."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        # Mock optimizer and results
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock IS optimization results
        is_results = pd.DataFrame({
            'short_window': [5],
            'long_window': [15],
            'return_pct': [10.0],
            'sharpe_ratio': [1.0],
            'max_drawdown': [5.0],
            'win_rate': [60.0],
            'num_trades': [10],
            'exposure_time': [50.0],
            'profit_factor': [2.0],
            'avg_trade': [1.0],
            'best_trade': [5.0],
            'worst_trade': [-3.0],
            'calmar_ratio': [2.0],
            'sortino_ratio': [1.2]
        })
        
        # Mock OOS test results
        oos_results = pd.DataFrame({
            'short_window': [5],
            'long_window': [15],
            'return_pct': [8.0],
            'sharpe_ratio': [0.8],
            'max_drawdown': [4.0],
            'win_rate': [55.0],
            'num_trades': [8],
            'exposure_time': [45.0],
            'profit_factor': [1.8],
            'avg_trade': [1.0],
            'best_trade': [4.0],
            'worst_trade': [-2.0],
            'calmar_ratio': [2.0],
            'sortino_ratio': [1.0]
        })
        
        mock_optimizer.optimize.side_effect = [is_results, oos_results]
        
        # Create test data
        data = self.create_test_data(2000)
        
        # Test period execution
        is_start = datetime(2024, 1, 1)
        is_end = datetime(2024, 2, 1)
        oos_start = datetime(2024, 2, 1)
        oos_end = datetime(2024, 3, 1)
        
        result = analyzer._run_walk_forward_period(
            data, is_start, is_end, oos_start, oos_end, "TEST", 1
        )
        
        # Check result structure
        assert result is not None
        assert 'period_num' in result
        assert 'best_params' in result
        assert 'is_metrics' in result
        assert 'oos_metrics' in result
        assert 'wfe_metrics' in result
        
        # Check parameter extraction
        assert result['best_params']['short_window'] == 5
        assert result['best_params']['long_window'] == 15

    def test_run_walk_forward_period_insufficient_data(self):
        """Test walk-forward period with insufficient data."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        data = self.create_test_data(200)
        
        # Very short periods that result in insufficient data
        is_start = datetime(2024, 1, 1)
        is_end = datetime(2024, 1, 2)  # Only 1 day
        oos_start = datetime(2024, 1, 2)
        oos_end = datetime(2024, 1, 3)  # Only 1 day
        
        result = analyzer._run_walk_forward_period(
            data, is_start, is_end, oos_start, oos_end, "TEST", 1
        )
        
        assert result is None  # Should return None for insufficient data

    @patch('framework.optimization.walk_forward_analyzer.ManualGridSearchOptimizer')
    def test_run_walk_forward_period_optimization_failure(self, mock_optimizer_class):
        """Test walk-forward period with optimization failure."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        # Mock optimizer to return empty results
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = pd.DataFrame()  # Empty results
        
        data = self.create_test_data(2000)
        
        is_start = datetime(2024, 1, 1)
        is_end = datetime(2024, 2, 1)
        oos_start = datetime(2024, 2, 1)
        oos_end = datetime(2024, 3, 1)
        
        result = analyzer._run_walk_forward_period(
            data, is_start, is_end, oos_start, oos_end, "TEST", 1
        )
        
        assert result is None  # Should return None for failed optimization

    def test_save_operations_no_crash(self):
        """Test that save operations don't crash."""
        param_config = self.create_param_config()
        analyzer = WalkForwardAnalyzer(SMAStrategy, param_config)
        
        # Mock results
        results = [
            {
                'period_num': 1,
                'is_start': datetime(2024, 1, 1),
                'is_end': datetime(2024, 2, 1),
                'oos_start': datetime(2024, 2, 1),
                'oos_end': datetime(2024, 3, 1),
                'best_params': {'short_window': 5, 'long_window': 15},
                'is_metrics': {'return_pct': 10.0, 'sharpe_ratio': 1.0, 'num_trades': 10},
                'oos_metrics': {'return_pct': 8.0, 'sharpe_ratio': 0.8, 'num_trades': 8},
                'wfe_metrics': {'is_return_pct': 10.0, 'oos_return_pct': 8.0, 'wfe': 80.0}
            }
        ]
        
        summary = {'total_periods': 1, 'avg_wfe': 80.0}
        
        # Mock file operations
        with patch('pandas.DataFrame.to_csv'), \
             patch('builtins.open', create=True), \
             patch('json.dump'), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            analyzer._save_results(results, summary, '/tmp', 'TEST')

    @patch('framework.optimization.walk_forward_analyzer.ManualGridSearchOptimizer')
    def test_analyze_complete_flow(self, mock_optimizer_class):
        """Test complete analysis flow."""
        # Create sufficient test data
        data = self.create_test_data(3000)
        param_config = self.create_param_config()
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock successful optimization results
        mock_is_result = pd.DataFrame({
            'short_window': [5], 'long_window': [15],
            'return_pct': [10.0], 'sharpe_ratio': [1.0], 'max_drawdown': [5.0],
            'win_rate': [60.0], 'num_trades': [10], 'exposure_time': [50.0],
            'profit_factor': [2.0], 'avg_trade': [1.0], 'best_trade': [5.0],
            'worst_trade': [-3.0], 'calmar_ratio': [2.0], 'sortino_ratio': [1.2]
        })
        
        mock_oos_result = pd.DataFrame({
            'short_window': [5], 'long_window': [15],
            'return_pct': [8.0], 'sharpe_ratio': [0.8], 'max_drawdown': [4.0],
            'win_rate': [55.0], 'num_trades': [8], 'exposure_time': [45.0],
            'profit_factor': [1.8], 'avg_trade': [1.0], 'best_trade': [4.0],
            'worst_trade': [-2.0], 'calmar_ratio': [2.0], 'sortino_ratio': [1.0]
        })
        
        mock_optimizer.optimize.side_effect = [mock_is_result, mock_oos_result]
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            is_window_months=2,
            oos_window_months=1,
            step_months=1,
            n_jobs=1
        )
        
        # Mock save operations
        with patch.object(analyzer, '_save_results'):
            result = analyzer.analyze(data, symbol="TEST")
        
        # Check result structure
        assert 'periods' in result
        assert 'summary' in result
        assert 'analysis_info' in result
        
        # Check that we got some periods
        assert len(result['periods']) > 0
        
        # Check analysis info
        assert result['analysis_info']['strategy'] == 'SMAStrategy'
        assert result['analysis_info']['symbol'] == 'TEST'