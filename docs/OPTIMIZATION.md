# Strategy Optimization Guide

This guide covers how to optimize trading strategy parameters using the grid search optimizer framework.

## Overview

The optimization framework allows you to systematically test different parameter combinations to find the best performing configuration for your trading strategies. It generates detailed results with performance metrics and heatmap visualizations.

## Quick Start

### Basic Optimization

```bash
# Optimize SMA strategy with default parameters
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --start 2024-01-01 --end 2024-03-01

# Optimize FVG strategy
python scripts/optimize_strategy.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --start 2024-01-01 --end 2024-03-01

# Custom parameter space via JSON
python scripts/optimize_strategy.py --strategy sma --data-file data/file.csv --param-space '{"short_window": {"type": "int", "min": 5, "max": 20, "step": 5}, "long_window": {"type": "int", "min": 30, "max": 100, "step": 10}}'
```

## Strategy-Specific Optimization

### SMA Strategy Optimization

The SMA strategy optimizes moving average windows and optionally risk parameters.

**Default Parameter Space:**
- `short_window`: 5-50 (step: 5) - Short moving average period
- `long_window`: 20-200 (step: 20) - Long moving average period

**Basic Examples:**
```bash
# Standard SMA optimization (uses all CPU cores by default)
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --start 2024-01-01 --end 2024-06-01

# Optimize with specific number of parallel jobs
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --n-jobs 4 --start 2024-01-01 --end 2024-06-01

# Use sequential processing for debugging
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --use-simple --start 2024-01-01 --end 2024-06-01

# Optimize with risk parameters included
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --optimize-risk --start 2024-01-01 --end 2024-06-01

# Optimize for Sharpe ratio instead of returns
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --metric "Sharpe Ratio" --start 2024-01-01 --end 2024-06-01

# Use fixed risk manager
python scripts/optimize_strategy.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --risk-manager fixed_risk --start 2024-01-01 --end 2024-06-01
```

**Risk Parameter Optimization:**
When using `--optimize-risk`, additional parameters are included:
- `stop_loss_pct`: 1-5% (step: 1%)
- `take_profit_pct`: 2-10% (step: 2%)

### FVG Strategy Optimization

The FVG (Fair Value Gap) strategy optimizes multi-timeframe parameters and risk settings.

**Default Parameter Space:**
- `h1_lookback_candles`: 12-48 (step: 12) - H1 context lookback period
- `risk_reward_ratio`: 1.5-4.0 (step: 0.5) - Risk/reward ratio for trades
- `max_hold_hours`: 2-8 (step: 2) - Maximum holding time
- `position_size`: 2-10% (step: 2%) - Position size (if using fixed position manager)

**Basic Examples:**
```bash
# Standard FVG optimization
python scripts/optimize_strategy.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --start 2024-01-01 --end 2024-06-01

# Optimize with fixed risk manager (excludes position_size from optimization)
python scripts/optimize_strategy.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --risk-manager fixed_risk --start 2024-01-01 --end 2024-06-01

# Optimize for win rate
python scripts/optimize_strategy.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --metric "Win Rate [%]" --start 2024-01-01 --end 2024-06-01

# High capital for expensive assets
python scripts/optimize_strategy.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --initial-capital 100000 --start 2024-01-01 --end 2024-06-01
```

## Command Line Options

### Required Parameters
- `--strategy`: Strategy to optimize (`sma`, `fvg`)
- `--data-file`: Path to CSV data file with OHLCV data

### Data Selection
- `--start`: Start date for optimization (YYYY-MM-DD)
- `--end`: End date for optimization (YYYY-MM-DD)

### Optimization Settings
- `--metric`: Metric to optimize (default: `Return [%]`)
  - Available: `Return [%]`, `Sharpe Ratio`, `Win Rate [%]`, `Max. Drawdown [%]`, `Profit Factor`
- `--minimize`: Minimize the metric instead of maximizing (useful for drawdown)
- `--optimize-risk`: Include risk parameters in optimization (SMA only)
- `--param-space`: Custom parameter space as JSON string or environment variable name

### Risk Management
- `--risk-manager`: Risk manager type
  - `fixed_position`: Fixed percentage of equity per trade
  - `fixed_risk`: Fixed percentage risk per trade based on stop loss
- Risk manager parameters are set via environment variables (see examples below)

### Backtest Settings
- `--initial-capital`: Starting capital (default: 10000)
- `--commission`: Commission rate per trade (default: 0.001)
- `--margin`: Margin requirement - 0.01 = 100x leverage (default: 0.01)

### Performance Options
- `--n-jobs`: Number of parallel jobs (default: -1 = all CPU cores)
- `--use-simple`: Force sequential optimizer instead of parallel

### Output
- `--output`: Custom output file path (default: auto-generated timestamp)
- `--debug`: Enable debug logging

## Risk Manager Configuration

### Fixed Position Size Manager
```bash
# Use 5% position size
RISK_PARAMS='{"position_size": 0.05}' python scripts/optimize_strategy.py --strategy sma --data-file data/file.csv --risk-manager fixed_position
```

### Fixed Risk Manager
```bash
# Risk 1% of capital per trade
RISK_PARAMS='{"risk_percent": 0.01, "default_stop_distance": 0.02}' python scripts/optimize_strategy.py --strategy sma --data-file data/file.csv --risk-manager fixed_risk
```

## Output Files

The optimizer generates two files in `output/optimizations/`:

1. **JSON Results** (`strategy_YYYYMMDD_HHMMSS.json`):
   - Best parameters found
   - Performance metrics
   - Complete results for all tested combinations
   - Metadata about the optimization run

2. **Heatmap Visualization** (`strategy_YYYYMMDD_HHMMSS.png`):
   - Generated automatically for 2-parameter optimizations
   - Color-coded performance map
   - Red star marks the best parameter combination

## Custom Parameter Spaces

You can define custom parameter spaces using JSON instead of the default ranges. This gives you full control over which parameters to optimize and their ranges.

### JSON Format

```json
{
  "parameter_name": {
    "type": "int|float|choice",
    "min": minimum_value,
    "max": maximum_value,
    "step": step_size,
    "choices": [list_of_choices]  // Only for choice type
  }
}
```

### Parameter Types

**Integer Parameters:**
```json
{"short_window": {"type": "int", "min": 5, "max": 50, "step": 5}}
```

**Float Parameters:**
```json
{"stop_loss_pct": {"type": "float", "min": 0.01, "max": 0.05, "step": 0.01}}
```

**Choice Parameters:**
```json
{"strategy_mode": {"type": "choice", "choices": ["aggressive", "conservative", "balanced"]}}
```

### Usage Methods

**1. Inline JSON String:**
```bash
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/file.csv \
  --param-space '{"short_window": {"type": "int", "min": 8, "max": 15, "step": 2}, "long_window": {"type": "int", "min": 25, "max": 40, "step": 5}}'
```

**2. Environment Variable:**
```bash
# Set parameter space in environment variable
export PARAM_SPACE='{"short_window": {"type": "int", "min": 10, "max": 30, "step": 5}, "long_window": {"type": "int", "min": 50, "max": 150, "step": 25}}'

# Use environment variable
python scripts/optimize_strategy.py --strategy sma --data-file data/file.csv --param-space PARAM_SPACE
```

### Custom Parameter Examples

**SMA Strategy with Fine-Tuned Windows:**
```bash
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --param-space '{"short_window": {"type": "int", "min": 8, "max": 15, "step": 1}, "long_window": {"type": "int", "min": 20, "max": 40, "step": 2}}'
```

**FVG Strategy with Custom Risk-Reward:**
```bash
python scripts/optimize_strategy.py \
  --strategy fvg \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --param-space '{"h1_lookback_candles": {"type": "choice", "choices": [12, 24, 36, 48]}, "risk_reward_ratio": {"type": "float", "min": 1.5, "max": 5.0, "step": 0.25}}'
```

**SMA with Risk Parameters:**
```bash
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --param-space '{"short_window": {"type": "int", "min": 10, "max": 20, "step": 2}, "long_window": {"type": "int", "min": 30, "max": 60, "step": 10}, "stop_loss_pct": {"type": "float", "min": 0.015, "max": 0.03, "step": 0.005}, "take_profit_pct": {"type": "float", "min": 0.02, "max": 0.06, "step": 0.01}}'
```

## Advanced Examples

### Long-Term SMA Optimization
```bash
# Optimize SMA on full year of data with high capital
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --initial-capital 100000 \
  --commission 0.0005 \
  --risk-manager fixed_risk \
  --output output/optimizations/sma_full_year_optimization.json
```

### FVG Strategy with Conservative Risk
```bash
# Optimize FVG with conservative fixed risk manager
RISK_PARAMS='{"risk_percent": 0.005, "default_stop_distance": 0.015}' \
python scripts/optimize_strategy.py \
  --strategy fvg \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --risk-manager fixed_risk \
  --metric "Sharpe Ratio" \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --debug
```

### Minimize Drawdown Optimization
```bash
# Find parameters that minimize maximum drawdown
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --metric "Max. Drawdown [%]" \
  --minimize \
  --start 2024-01-01 \
  --end 2024-06-01
```

## Performance Considerations

### Parallel vs Sequential Optimization

**Parallel Optimization (Default):**
- Uses all available CPU cores by default (`--n-jobs -1`)
- 2-8x faster depending on your hardware
- Best for large parameter spaces (>20 combinations)
- Uses backtesting.py's built-in multiprocessing

**Sequential Optimization:**
- Single-threaded processing (`--use-simple`)
- Better for debugging and small parameter spaces
- More reliable heatmap generation
- Lower memory usage

### Optimization Time
- **SMA Strategy (Parallel)**: ~100 combinations (2-5 minutes for 6 months of data)
- **SMA Strategy (Sequential)**: ~100 combinations (5-10 minutes for 6 months of data)
- **FVG Strategy (Parallel)**: ~200+ combinations (5-10 minutes for 6 months of data)
- **FVG Strategy (Sequential)**: ~200+ combinations (10-20 minutes for 6 months of data)
- **With Risk Optimization**: 2-4x longer due to additional parameters

### Hardware Recommendations
- **CPU**: 4+ cores recommended for parallel optimization
- **RAM**: 8GB+ for large parameter spaces (>100 combinations)
- **Storage**: SSD recommended for faster data loading

### Data Requirements
- Minimum 3 months of data recommended for reliable results
- Use M15 (15-minute) timeframe for best balance of detail and speed
- Ensure data quality - no gaps or missing values

### Memory Usage
- **Parallel**: Each worker process requires ~100-200MB RAM
- **Sequential**: ~50-100MB RAM total
- Large parameter spaces (>500 combinations) may require 8GB+ RAM
- Results are automatically memory-managed during optimization

## Interpreting Results

### Key Metrics
- **Return [%]**: Total percentage return over the period
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max. Drawdown [%]**: Worst peak-to-trough decline (lower is better)
- **Win Rate [%]**: Percentage of profitable trades
- **# Trades**: Total number of trades executed
- **Profit Factor**: Gross profit / Gross loss ratio

### Heatmap Analysis
- **Green areas**: Better performance
- **Red areas**: Poor performance
- **Red star**: Best parameter combination
- **Patterns**: Look for robust regions (consistent green areas)

### Parameter Selection Guidelines
1. **Avoid overfitting**: Don't pick parameters that work on only narrow data ranges
2. **Check robustness**: Good parameters should show consistent performance across different time periods
3. **Consider trade frequency**: More trades = more statistical significance
4. **Validate out-of-sample**: Test optimized parameters on different time periods

## Troubleshooting

### Common Issues
1. **No trades generated**: Check if strategy parameters are too restrictive
2. **Poor performance**: Strategy may not be suitable for the asset/timeframe
3. **Long optimization time**: Reduce parameter space or data range
4. **Memory errors**: Use smaller parameter ranges or more RAM

### Debug Mode
Add `--debug` flag for detailed logging:
```bash
python scripts/optimize_strategy.py --strategy sma --data-file data/file.csv --debug
```

This will show:
- Parameter combinations being tested
- Strategy initialization details
- Trade execution logs
- Risk management decisions

## Example Workflow

1. **Quick Test**: Run small optimization on 1-2 months of data
2. **Parameter Analysis**: Review heatmap and identify promising regions
3. **Extended Optimization**: Run longer optimization on 6-12 months of data
4. **Out-of-Sample Validation**: Test best parameters on different time periods
5. **Production Deployment**: Use validated parameters in live trading

## Best Practices

1. **Start Small**: Begin with small parameter ranges and short time periods
2. **Use Multiple Metrics**: Optimize for different metrics to find robust parameters
3. **Consider Transaction Costs**: Include realistic commission rates
4. **Validate Results**: Always test optimized parameters on unseen data
5. **Document Results**: Save optimization runs with descriptive names

## Example Scripts

For complete working examples, see `examples/custom_optimization_examples.sh` which demonstrates:

1. **Fine-tuned SMA parameters** - Using step=1 for precise window optimization
2. **Choice-based optimization** - Using discrete parameter choices
3. **Environment variable usage** - Complex parameter spaces via env vars  
4. **Multi-parameter optimization** - Including risk parameters in search space

Run the examples:
```bash
# Run all optimization examples
./examples/custom_optimization_examples.sh

# Or run individual examples
bash examples/custom_optimization_examples.sh
```