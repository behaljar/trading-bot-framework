#!/usr/bin/env python3
"""Test grid search optimizer implementations."""

import pandas as pd
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from framework.optimization.simple_grid_search import SimpleGridSearchOptimizer
from framework.optimization.parallel_grid_search import ParallelGridSearchOptimizer
from framework.optimization.parameter_space import ParameterSpace
from framework.strategies.sma_strategy import SMAStrategy
from framework.utils.logger import setup_logger


def test_parameter_space():
    """Test parameter space creation and combination generation."""
    space = ParameterSpace()
    space.add_parameter('param1', min_value=1, max_value=3, step=1, param_type='int')
    space.add_parameter('param2', min_value=0.1, max_value=0.2, step=0.1, param_type='float')
    
    combinations = space.get_grid_combinations()
    assert len(combinations) == 6  # 3 * 2 combinations
    assert {'param1': 1, 'param2': 0.1} in combinations
    assert {'param1': 3, 'param2': 0.2} in combinations


def test_simple_grid_search_optimizer():
    """Test simple sequential grid search optimizer."""
    # Setup logging
    logger = setup_logger("INFO")
    
    # Load sample data
    data_file = 'data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv'
    if not Path(data_file).exists():
        pytest.skip(f"Test data file {data_file} not found")
    
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    # Use only small subset for testing
    data = data['2024-01-01':'2024-01-15']
    
    # Create small parameter space
    space = ParameterSpace()
    space.add_parameter('short_window', min_value=10, max_value=20, step=10, param_type='int')
    space.add_parameter('long_window', min_value=30, max_value=40, step=10, param_type='int')
    
    # Create optimizer
    optimizer = SimpleGridSearchOptimizer(
        strategy_class=SMAStrategy,
        parameter_space=space,
        data=data,
        initial_capital=10000,
        commission=0.001,
        margin=0.01,
        n_jobs=1,
        debug=False
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Verify results structure
    assert 'best_params' in results
    assert 'best_stats' in results
    assert 'metric_value' in results
    assert 'total_combinations' in results
    
    # Verify best parameters are within defined ranges
    assert 10 <= results['best_params']['short_window'] <= 20
    assert 30 <= results['best_params']['long_window'] <= 40
    
    # Save results with heatmap
    output_path = 'output/optimizations/test_simple_optimization.json'
    optimizer.save_results(output_path, results, save_heatmap=True)
    
    print(f"Simple optimizer test passed. Best return: {results['metric_value']:.2f}%")


def test_parallel_grid_search_optimizer():
    """Test parallel grid search optimizer."""
    # Setup logging
    logger = setup_logger("INFO")
    
    # Load sample data
    data_file = 'data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv'
    if not Path(data_file).exists():
        pytest.skip(f"Test data file {data_file} not found")
    
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    # Use only small subset for testing
    data = data['2024-01-01':'2024-01-15']
    
    # Create small parameter space
    space = ParameterSpace()
    space.add_parameter('short_window', min_value=10, max_value=20, step=10, param_type='int')
    space.add_parameter('long_window', min_value=30, max_value=40, step=10, param_type='int')
    
    # Create optimizer with parallel execution
    optimizer = ParallelGridSearchOptimizer(
        strategy_class=SMAStrategy,
        parameter_space=space,
        data=data,
        initial_capital=10000,
        commission=0.001,
        margin=0.01,
        n_jobs=2,  # Use 2 cores for testing
        debug=False
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Verify results structure
    assert 'best_params' in results
    assert 'best_stats' in results
    assert 'metric_value' in results
    assert 'total_combinations' in results
    
    # Verify best parameters are within defined ranges
    assert 10 <= results['best_params']['short_window'] <= 20
    assert 30 <= results['best_params']['long_window'] <= 40
    
    print(f"Parallel optimizer test passed. Best return: {results['metric_value']:.2f}%")


if __name__ == '__main__':
    # Run tests directly
    print("Testing Parameter Space...")
    test_parameter_space()
    print("✓ Parameter Space test passed\n")
    
    print("Testing Simple Grid Search Optimizer...")
    test_simple_grid_search_optimizer()
    print("✓ Simple Grid Search test passed\n")
    
    print("Testing Parallel Grid Search Optimizer...")
    test_parallel_grid_search_optimizer()
    print("✓ Parallel Grid Search test passed\n")
    
    print("All tests passed successfully!")