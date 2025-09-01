# Strategy Optimization Guide

This document explains how to use the strategy optimization system to find optimal trading parameters through systematic backtesting.

## Overview

The optimization system performs grid search over strategy parameters, running parallel backtests to find the best parameter combinations based on various performance metrics. It provides comprehensive visualization and detailed results export.

## Quick Start

Basic optimization command:
```bash
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --metric win_rate
```

## Command Line Options

### Required Arguments

- `--strategy`: Strategy to optimize (`sma`, `fvg`, `breakout`)
- `--data-file`: Path to CSV data file with OHLCV data

### Data Options

- `--start`: Start date for optimization period (YYYY-MM-DD)
- `--end`: End date for optimization period (YYYY-MM-DD)

### Optimization Parameters

- `--metric`: Metric to optimize (default: `win_rate`)
  - `return_pct`: Total return percentage
  - `sharpe_ratio`: Risk-adjusted return metric
  - `win_rate`: Percentage of winning trades
  - `max_drawdown`: Maximum peak-to-trough decline
  - `profit_factor`: Gross profit / gross loss ratio
  - `calmar_ratio`: Return / max drawdown ratio
- `--minimize`: Minimize the metric instead of maximizing (useful for drawdown)
- `--param-config`: Custom parameter configuration (JSON string or file path)

### Risk Management

- `--risk-manager`: Risk management type (default: `fixed_risk`)
  - `fixed_position`: Fixed position size (5% of equity)
  - `fixed_risk`: Fixed risk per trade (1% of equity) **[DEFAULT]**

### Backtest Settings

- `--initial-capital`: Starting capital (default: 10000)
- `--commission`: Commission rate (default: 0.001 = 0.1%)
- `--margin`: Margin requirement (default: 0.01 = 100x leverage)

### Performance Options

- `--n-jobs`: Number of parallel workers (default: -1 = all CPU cores)
- `--debug`: Enable debug logging

### Output Options

- `--output`: Custom output path (default: auto-generated in `output/optimizations/`)

## Parameter Configuration

### Default Parameter Ranges

Each strategy has default parameter ranges that will be used if no custom configuration is provided:

**SMA Strategy:**
```json
{
  "short_window": {"min": 5, "max": 50, "step": 5},
  "long_window": {"min": 20, "max": 200, "step": 20},
  "stop_loss_pct": {"min": 0.01, "max": 0.05, "step": 0.01},
  "take_profit_pct": {"min": 0.02, "max": 0.10, "step": 0.02}
}
```

**FVG Strategy:**
```json
{
  "h1_lookback_candles": {"min": 12, "max": 48, "step": 12},
  "risk_reward_ratio": {"min": 1.5, "max": 4.0, "step": 0.5},
  "max_hold_hours": {"min": 2, "max": 8, "step": 2},
  "position_size": {"min": 0.02, "max": 0.10, "step": 0.02}
}
```

**Breakout Strategy:**
```json
{
  "entry_lookback": {"min": 15, "max": 60, "step": 15},
  "exit_lookback": {"min": 5, "max": 30, "step": 5},
  "atr_multiplier": {"min": 1.0, "max": 3.5, "step": 0.5},
  "medium_trend_threshold": {"min": 0.02, "max": 0.10, "step": 0.02},
  "relative_volume_threshold": {"min": 1.2, "max": 3.0, "step": 0.4},
  "cooldown_periods": {"min": 2, "max": 20, "step": 6}
}
```

### Custom Parameter Configuration

You can provide custom parameter ranges in three ways:

#### 1. JSON String
```bash
python scripts/optimize_strategy.py --strategy sma \
  --data-file data.csv \
  --param-config '{"short_window": {"min": 10, "max": 30, "step": 5}, "long_window": {"min": 40, "max": 60, "step": 10}}'
```

#### 2. JSON File
Create a file `params.json`:
```json
{
  "short_window": {"min": 10, "max": 30, "step": 5},
  "long_window": {"min": 40, "max": 60, "step": 10},
  "stop_loss_pct": {"min": 0.02, "max": 0.03, "step": 0.01}
}
```

Then use:
```bash
python scripts/optimize_strategy.py --strategy sma --data-file data.csv --param-config params.json
```

#### 3. Environment Variable
```bash
export PARAMS='{"short_window": {"min": 10, "max": 30, "step": 5}}'
python scripts/optimize_strategy.py --strategy sma --data-file data.csv --param-config PARAMS
```

### Parameter Types

Parameters can be defined with different types:

- **Integer parameters:**
  ```json
  {"type": "int", "min": 5, "max": 50, "step": 5}
  ```

- **Float parameters:**
  ```json
  {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01}
  ```

- **Choice parameters:**
  ```json
  {"choices": [12, 24, 36, 48]}
  ```

- **Direct values list:**
  ```json
  {"values": [10, 20, 30, 50, 100]}
  ```

## Progress Tracking

During optimization, you'll see a real-time progress bar showing:
- Current progress (completed/total combinations)
- Percentage complete
- Elapsed time
- Estimated time remaining (ETA)

Example:
```
Starting optimization of 480 parameter combinations
Using 8 parallel workers
Optimizing for: win_rate (maximize)
--------------------------------------------------------------------------------
|████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 200/480 (41.7%) | Elapsed: 0:02:15 | ETA: 0:03:09
```

## Output Files

The optimization generates several output files:

### 1. CSV Results (`*_all_results.csv`)
Contains all parameter combinations tested with their performance metrics:
- All parameter values
- Return percentage
- Sharpe ratio
- Win rate
- Max drawdown
- Number of trades
- Profit factor
- And more...

### 2. Best Parameters (`*_best_params.json`)
JSON file with the best parameter combination and its metrics:
```json
{
  "best_params": {
    "short_window": 15,
    "long_window": 45,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.05
  },
  "best_metrics": {
    "return_pct": 25.34,
    "sharpe_ratio": 1.82,
    "win_rate": 58.5,
    "max_drawdown": 8.2,
    "profit_factor": 1.95,
    "num_trades": 42
  }
}
```

### 3. Comprehensive Visualization (`*_comprehensive_*.png`)
Multi-panel chart showing:
- Top 30 parameter combinations ranked by metric
- Risk-return scatter plot
- Efficiency metrics (Sharpe vs Win Rate)
- Distribution of key metrics
- Correlation heatmap
- Parameter importance analysis

### 4. Heatmap (`*_heatmap_*.png`)
2D heatmap showing metric values for parameter pairs (when applicable)

### 5. PDF Report (`*_comprehensive_*.pdf`)
High-resolution PDF version of the comprehensive visualization

## Usage Examples

### Example 1: Basic SMA Optimization
```bash
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --metric sharpe_ratio
```

### Example 2: FVG Strategy with Custom Parameters
```bash
python scripts/optimize_strategy.py \
  --strategy fvg \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --param-config '{"h1_lookback_candles": {"choices": [12, 24, 36]}, "risk_reward_ratio": {"min": 2.0, "max": 3.0, "step": 0.5}}' \
  --metric return_pct \
  --n-jobs 8
```

### Example 3: Minimize Drawdown with Risk Manager
```bash
python scripts/optimize_strategy.py \
  --strategy breakout \
  --data-file data/cleaned/ETH_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --metric max_drawdown \
  --minimize \
  --risk-manager fixed_risk \
  --initial-capital 50000
```

### Example 4: Quick Test with Limited Parameters
```bash
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-05-01 \
  --end 2024-05-31 \
  --param-config '{"short_window": {"values": [10, 20]}, "long_window": {"values": [30, 50]}}' \
  --metric win_rate
```

### Example 5: Using Parameter File
Create `optimization_params.json`:
```json
{
  "h1_lookback_candles": {"choices": [12, 24]},
  "risk_reward_ratio": {"min": 1.5, "max": 3.0, "step": 0.5},
  "max_hold_hours": {"values": [2, 4, 6]},
  "position_size": {"min": 0.05, "max": 0.1, "step": 0.05}
}
```

Run optimization:
```bash
python scripts/optimize_strategy.py \
  --strategy fvg \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --param-config optimization_params.json \
  --metric profit_factor
```

## Tips for Effective Optimization

### 1. Start Small
Begin with a limited date range and fewer parameter combinations to test quickly:
```bash
# Test on 1 month with coarse parameter steps
--start 2024-05-01 --end 2024-06-01 \
--param-config '{"short_window": {"min": 10, "max": 30, "step": 10}}'
```

### 2. Use Multiple Metrics
Optimize for different metrics to understand trade-offs:
- `win_rate`: For consistent strategies
- `sharpe_ratio`: For risk-adjusted returns
- `profit_factor`: For profitability assessment
- `max_drawdown` (minimize): For risk management

### 3. Parallel Processing
Use all available CPU cores for faster optimization:
```bash
--n-jobs -1  # Uses all cores
--n-jobs 4   # Uses 4 cores
```

### 4. Data Splitting
Optimize on training data, then validate on test data:
```bash
# Optimization period
--start 2024-01-01 --end 2024-06-01

# Validation period (run backtest with best params)
--start 2024-06-01 --end 2024-08-01
```

### 5. Parameter Ranges
- Start with wide ranges and coarse steps
- Narrow down based on initial results
- Fine-tune with smaller steps around promising values

### 6. Risk Management
Always test with appropriate risk management:
```bash
--risk-manager fixed_risk  # For percentage risk per trade
--risk-manager fixed_position  # For fixed position sizing
```

## Interpreting Results

### Visualization Panels

1. **Top Combinations Bar Chart**: Shows best parameter sets ranked by chosen metric
2. **Risk-Return Profile**: Scatter plot of return vs drawdown (colored by metric)
3. **Efficiency Metrics**: Relationship between win rate and Sharpe ratio
4. **Distribution Histograms**: Frequency distribution of key metrics
5. **Correlation Matrix**: How different metrics relate to each other
6. **Parameter Importance**: Which parameters most affect the target metric

### Key Metrics to Consider

- **Return [%]**: Total percentage return
- **Sharpe Ratio**: Return per unit of risk (> 1.0 is good, > 2.0 is excellent)
- **Win Rate [%]**: Percentage of profitable trades (> 50% is positive)
- **Max Drawdown [%]**: Largest peak-to-trough decline (< 20% is preferred)
- **Profit Factor**: Gross profit / gross loss (> 1.5 is good)
- **Number of Trades**: Total trades executed (ensures statistical significance)

### Warning Signs

- Very high returns with few trades (< 30) may indicate overfitting
- Win rates near 0% or 100% suggest parameter issues
- Sharpe ratio < 0 indicates negative risk-adjusted returns
- Large drawdowns (> 30%) indicate high risk

## Troubleshooting

### No Trades Generated
- Check if parameter ranges are reasonable
- Verify data quality and date range
- Review strategy logic for the given parameters

### All Results Show -100% Return
- Usually indicates strategy errors or invalid parameters
- Check data column names (should be lowercase: open, high, low, close, volume)
- Verify data is properly formatted with datetime index

### Slow Performance
- Reduce parameter combinations by increasing step sizes
- Use fewer CPU cores if memory is limited
- Optimize on shorter time periods

### Memory Issues
- Reduce `--n-jobs` to use fewer parallel workers
- Use smaller data samples
- Close other applications

### "Unrecognized arguments" Error
- Make sure to use `--param-config` (not `--param-space`)
- Ensure JSON is properly formatted with quotes
- On command line, use single quotes around JSON: `'{"key": "value"}'`

## Advanced Usage

### Custom Optimization Metrics
You can extend the system by modifying the `ManualGridSearchOptimizer` class in `framework/optimization/manual_grid_search.py` to add custom metrics.

### Batch Optimization
Create a shell script for multiple optimizations:
```bash
#!/bin/bash
for symbol in BTC_USDT ETH_USDT SOL_USDT; do
  python scripts/optimize_strategy.py \
    --strategy sma \
    --data-file "data/cleaned/${symbol}_binance_15m_2024-01-01_2025-08-20_cleaned.csv" \
    --metric sharpe_ratio \
    --output "output/optimizations/batch_${symbol}"
done
```

### Parameter Sensitivity Analysis
After optimization, analyze the CSV results to understand parameter sensitivity:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('output/optimizations/sma_BTC_USDT_20240901_120000_all_results.csv')

# Plot parameter impact
for param in ['short_window', 'long_window']:
    plt.figure()
    df.groupby(param)['win_rate'].mean().plot(kind='bar')
    plt.title(f'Impact of {param} on Win Rate')
    plt.show()
```

### Running with Best Parameters
After optimization, use the best parameters in a regular backtest:
```bash
# Extract best params from JSON
best_params=$(cat output/optimizations/sma_*_best_params.json | jq -r '.best_params | to_entries | map("\(.key)=\(.value)") | join(",")')

# Run backtest with best params
STRATEGY_PARAMS="{$best_params}" python scripts/run_backtest.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv
```

## Best Practices

1. **Always validate**: Test optimized parameters on out-of-sample data
2. **Consider transaction costs**: Include realistic commission rates
3. **Use appropriate metrics**: Match optimization metric to trading goals
4. **Document results**: Keep notes on optimization runs and findings
5. **Avoid over-optimization**: Too many parameters can lead to curve fitting
6. **Test robustness**: Verify parameters work across different market conditions
7. **Monitor progress**: Use the progress bar to estimate completion time
8. **Save configurations**: Store successful parameter configurations for reuse

## Complete Command Reference

```bash
python scripts/optimize_strategy.py [OPTIONS]

Required:
  --strategy {sma,fvg,breakout}    Strategy to optimize
  --data-file PATH                  Path to CSV data file

Optional:
  --start YYYY-MM-DD               Start date for data
  --end YYYY-MM-DD                 End date for data
  --metric {return_pct,sharpe_ratio,win_rate,max_drawdown,profit_factor,calmar_ratio}
                                   Metric to optimize (default: win_rate)
  --minimize                       Minimize metric instead of maximize
  --param-config JSON/FILE         Custom parameter configuration
  --risk-manager {fixed_position,fixed_risk}
                                   Risk management type
  --initial-capital FLOAT          Initial capital (default: 10000)
  --commission FLOAT               Commission rate (default: 0.001)
  --margin FLOAT                   Margin requirement (default: 0.01)
  --n-jobs INT                     Parallel workers (default: -1 for all cores)
  --output PATH                    Output path for results
  --debug                          Enable debug logging
```

## Conclusion

The optimization system provides a powerful framework for systematic strategy parameter selection. By following this guide and best practices, you can effectively find robust parameter combinations that improve trading strategy performance.