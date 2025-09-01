"""
Tests for Grid Search Optimizer.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from framework.optimization.grid_search import (
    GridSearchOptimizer, 
    generate_parameter_combinations,
    run_single_backtest
)
from framework.strategies.sma_strategy import SMAStrategy
from framework.risk.fixed_position_size_manager import FixedPositionSizeManager


class TestParameterCombinations:
    """Test parameter combination generation."""

    def test_generate_combinations_values(self):
        """Test generation with direct values."""
        config = {
            'param1': {'values': [1, 2, 3]},
            'param2': {'values': ['a', 'b']}
        }
        combinations = generate_parameter_combinations(config)
        
        assert len(combinations) == 6  # 3 * 2
        assert {'param1': 1, 'param2': 'a'} in combinations
        assert {'param1': 3, 'param2': 'b'} in combinations

    def test_generate_combinations_choices(self):
        """Test generation with choices parameter."""
        config = {
            'method': {'choices': ['method_a', 'method_b', 'method_c']},
            'flag': {'choices': [True, False]}
        }
        combinations = generate_parameter_combinations(config)
        
        assert len(combinations) == 6  # 3 * 2
        assert {'method': 'method_a', 'flag': True} in combinations
        assert {'method': 'method_c', 'flag': False} in combinations

    def test_generate_combinations_int_range(self):
        """Test generation with integer range."""
        config = {
            'window': {
                'min': 10,
                'max': 12,
                'step': 1,
                'type': 'int'
            }
        }
        combinations = generate_parameter_combinations(config)
        
        assert len(combinations) == 3  # 10, 11, 12
        assert {'window': 10} in combinations
        assert {'window': 11} in combinations
        assert {'window': 12} in combinations

    def test_generate_combinations_float_range(self):
        """Test generation with float range."""
        config = {
            'ratio': {
                'min': 1.0,
                'max': 2.0,
                'step': 0.5,
                'type': 'float'
            }
        }
        combinations = generate_parameter_combinations(config)
        
        assert len(combinations) == 3  # 1.0, 1.5, 2.0
        assert {'ratio': 1.0} in combinations
        assert {'ratio': 1.5} in combinations
        assert {'ratio': 2.0} in combinations

    def test_generate_combinations_mixed(self):
        """Test generation with mixed parameter types."""
        config = {
            'window': {'values': [10, 20]},
            'threshold': {'min': 0.1, 'max': 0.2, 'step': 0.1},
            'method': {'choices': ['fast', 'slow']}
        }
        combinations = generate_parameter_combinations(config)
        
        assert len(combinations) == 8  # 2 * 2 * 2
        
        # Verify some specific combinations
        assert {'window': 10, 'threshold': 0.1, 'method': 'fast'} in combinations
        assert {'window': 20, 'threshold': 0.2, 'method': 'slow'} in combinations


class TestSingleBacktest:
    """Test single backtest execution."""

    def create_test_data(self, length=100):
        """Create test data for backtesting."""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
        
        # Create trending data
        base_price = 100.0
        trend = np.linspace(0, 20, length)
        noise = np.random.normal(0, 1, length)
        prices = base_price + trend + noise
        
        return pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': np.random.uniform(1000, 5000, length)
        }, index=dates)

    def test_run_single_backtest_success(self):
        """Test successful single backtest execution."""
        data = self.create_test_data(200)
        params = {'short_window': 10, 'long_window': 20}
        risk_manager = FixedPositionSizeManager(position_size=0.1)
        
        result = run_single_backtest(
            params=params,
            strategy_class=SMAStrategy,
            data=data,
            initial_capital=10000,
            commission=0.001,
            margin=0.01,
            risk_manager=risk_manager
        )
        
        # Check result structure
        assert 'params' in result
        assert 'return_pct' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result
        assert 'num_trades' in result
        
        # Check that params are preserved
        assert result['params'] == params

    @patch('framework.optimization.manual_grid_search.FractionalBacktest')
    def test_run_single_backtest_error_handling(self, mock_backtest):
        """Test error handling in single backtest."""
        # Mock backtest to raise exception
        mock_backtest.return_value.run.side_effect = Exception("Test error")
        
        data = self.create_test_data(100)
        params = {'short_window': 10, 'long_window': 20}
        
        result = run_single_backtest(
            params=params,
            strategy_class=SMAStrategy,
            data=data,
            initial_capital=10000,
            commission=0.001,
            margin=0.01,
            risk_manager=None
        )
        
        # Should return failed result
        assert result['return_pct'] == -100
        assert result['sharpe_ratio'] == -10
        assert 'error' in result
        assert result['params'] == params


class TestGridSearchOptimizer:
    """Test GridSearchOptimizer class."""

    def create_test_data(self, length=200):
        """Create test data for optimization."""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
        
        # Create data with clear trend for SMA strategy testing
        base_price = 100.0
        trend = np.linspace(0, 50, length)
        noise = np.random.normal(0, 2, length)
        prices = base_price + trend + noise
        
        return pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, length)
        }, index=dates)

    def test_initialization(self):
        """Test optimizer initialization."""
        data = self.create_test_data()
        param_config = {
            'short_window': {'values': [10, 20]},
            'long_window': {'values': [30, 40]}
        }
        
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            data=data,
            initial_capital=10000,
            commission=0.001,
            margin=0.01,
            n_jobs=2
        )
        
        assert optimizer.strategy_class == SMAStrategy
        assert optimizer.parameter_config == param_config
        assert optimizer.initial_capital == 10000
        assert optimizer.commission == 0.001
        assert optimizer.margin == 0.01
        assert optimizer.n_jobs == 2
        assert len(optimizer.combinations) == 4  # 2 * 2

    def test_data_preparation(self):
        """Test data preparation (column renaming)."""
        data = self.create_test_data()
        param_config = {'short_window': {'values': [10]}}
        
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            data=data,
            n_jobs=1
        )
        
        # Check that columns are renamed to match backtesting.py expectations
        prepared_data = optimizer.data
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in prepared_data.columns

    def test_n_jobs_auto_detection(self):
        """Test automatic n_jobs detection."""
        data = self.create_test_data()
        param_config = {'short_window': {'values': [10]}}
        
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            data=data,
            n_jobs=-1  # Auto-detect
        )
        
        # Should set to CPU count
        import multiprocessing as mp
        assert optimizer.n_jobs == mp.cpu_count()

    @patch('framework.optimization.manual_grid_search.ProcessPoolExecutor')
    def test_optimize_basic_functionality(self, mock_executor):
        """Test basic optimization functionality."""
        data = self.create_test_data()
        param_config = {
            'short_window': {'values': [10, 20]},
            'long_window': {'values': [30, 40]}
        }
        
        # Mock the executor and futures
        mock_future = Mock()
        mock_future.result.return_value = {
            'params': {'short_window': 10, 'long_window': 30},
            'return_pct': 15.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': 5.0,
            'win_rate': 60.0,
            'num_trades': 10,
            'exposure_time': 50.0,
            'profit_factor': 2.0,
            'avg_trade': 1.5,
            'best_trade': 5.0,
            'worst_trade': -3.0,
            'calmar_ratio': 3.1,
            'sortino_ratio': 1.8
        }
        
        mock_executor_instance = Mock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor_instance.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed to yield our future
        with patch('framework.optimization.manual_grid_search.as_completed') as mock_completed:
            mock_completed.return_value = [mock_future] * 4  # 4 combinations
            
            optimizer = GridSearchOptimizer(
                strategy_class=SMAStrategy,
                parameter_config=param_config,
                data=data,
                n_jobs=1
            )
            
            # Run optimization
            results_df = optimizer.optimize(metric='return_pct', maximize=True)
            
            # Check results structure
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 4
            assert 'short_window' in results_df.columns
            assert 'long_window' in results_df.columns
            assert 'return_pct' in results_df.columns
            assert 'sharpe_ratio' in results_df.columns

    def test_optimize_sorting(self):
        """Test that results are properly sorted by metric."""
        data = self.create_test_data()
        param_config = {'short_window': {'values': [10]}}
        
        # Create mock results with different metric values
        mock_results = [
            {'params': {'short_window': 10}, 'return_pct': 5.0, 'sharpe_ratio': 0.5},
            {'params': {'short_window': 10}, 'return_pct': 15.0, 'sharpe_ratio': 1.5},
            {'params': {'short_window': 10}, 'return_pct': 10.0, 'sharpe_ratio': 1.0}
        ]
        
        with patch.object(ManualGridSearchOptimizer, '_run_parallel_optimization') as mock_run:
            mock_run.return_value = mock_results
            
            optimizer = GridSearchOptimizer(
                strategy_class=SMAStrategy,
                parameter_config=param_config,
                data=data,
                n_jobs=1
            )
            
            # Test maximize=True (descending order)
            results_df = optimizer.optimize(metric='return_pct', maximize=True)
            returns = results_df['return_pct'].tolist()
            assert returns == [15.0, 10.0, 5.0]  # Descending
            
            # Test maximize=False (ascending order)
            results_df = optimizer.optimize(metric='return_pct', maximize=False)
            returns = results_df['return_pct'].tolist()
            assert returns == [5.0, 10.0, 15.0]  # Ascending

    def test_create_visualization_no_crash(self):
        """Test that visualization creation doesn't crash."""
        # Create minimal results DataFrame
        results_df = pd.DataFrame({
            'short_window': [10, 20, 30],
            'long_window': [40, 50, 60],
            'return_pct': [5.0, 10.0, 15.0],
            'win_rate': [55.0, 60.0, 65.0],
            'sharpe_ratio': [0.5, 1.0, 1.5],
            'max_drawdown': [10.0, 8.0, 6.0]
        })
        
        data = self.create_test_data()
        param_config = {'short_window': {'values': [10]}}
        
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            data=data,
            n_jobs=1
        )
        
        # Test that visualization doesn't crash
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            optimizer.create_visualization(results_df, '/tmp/test', 'return_pct')

    def test_save_results_structure(self):
        """Test results saving structure."""
        results_df = pd.DataFrame({
            'short_window': [10, 20],
            'long_window': [40, 50],
            'return_pct': [10.0, 15.0],
            'sharpe_ratio': [1.0, 1.5],
            'win_rate': [60.0, 65.0],
            'max_drawdown': [8.0, 6.0],
            'num_trades': [10, 12],
            'profit_factor': [2.0, 2.5]
        })
        
        data = self.create_test_data()
        param_config = {'short_window': {'values': [10]}}
        
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            data=data,
            n_jobs=1
        )
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            with patch('pandas.DataFrame.to_csv') as mock_csv:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                optimizer.save_results(results_df, '/tmp/test')
                
                # Verify CSV save was called
                mock_csv.assert_called_once()
                
                # Verify JSON save was called with proper structure
                mock_open.assert_called()


class TestGridSearchIntegration:
    """Integration tests for GridSearchOptimizer."""

    def create_test_data(self, length=100):
        """Create test data for integration tests."""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
        
        # Create simple trending data for reliable results
        prices = np.linspace(100, 120, length)
        
        return pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': [1000] * length
        }, index=dates)

    def test_small_optimization_run(self):
        """Test a small optimization run end-to-end."""
        data = self.create_test_data(150)
        
        # Small parameter space for quick test
        param_config = {
            'short_window': {'values': [5, 10]},
            'long_window': {'values': [15, 20]}
        }
        
        risk_manager = FixedPositionSizeManager(position_size=0.1)
        
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            data=data,
            initial_capital=10000,
            commission=0.001,
            margin=0.01,
            risk_manager=risk_manager,
            n_jobs=1  # Single threaded for testing
        )
        
        # Run optimization
        results_df = optimizer.optimize(metric='return_pct', maximize=True)
        
        # Verify results
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 4  # 2 * 2 combinations
        assert 'short_window' in results_df.columns
        assert 'long_window' in results_df.columns
        assert 'return_pct' in results_df.columns
        
        # Check that results are sorted by return_pct (descending)
        returns = results_df['return_pct'].tolist()
        assert returns == sorted(returns, reverse=True)

    def test_invalid_parameter_config(self):
        """Test handling of invalid parameter configurations."""
        data = self.create_test_data()
        
        # Empty parameter config
        with pytest.raises(ValueError):
            GridSearchOptimizer(
                strategy_class=SMAStrategy,
                parameter_config={},
                data=data
            )

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        data = self.create_test_data(10)  # Very small dataset
        param_config = {'short_window': {'values': [5]}}
        
        optimizer = GridSearchOptimizer(
            strategy_class=SMAStrategy,
            parameter_config=param_config,
            data=data,
            n_jobs=1
        )
        
        # Should handle gracefully and return results (even if poor)
        results_df = optimizer.optimize()
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) >= 0