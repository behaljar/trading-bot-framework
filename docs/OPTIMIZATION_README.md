# Strategy Optimization Guide

This guide covers how to use the `scripts/optimize.py` tool for parameter optimization and heatmap generation.

## Overview

The optimization script uses the [Backtesting.py](https://kernc.github.io/backtesting.py/) library to find optimal parameters for trading strategies. It generates comprehensive heatmaps showing performance across different parameter combinations.

## Features

- **Multi-parameter optimization** using `Backtest.optimize()` with `return_heatmap=True`
- **Interactive heatmaps** using `backtesting.lib.plot_heatmaps()` 
- **Static PNG heatmaps** for all parameter pairs with mean/max aggregation
- **Summary correlation plots** showing relationships in top-performing results
- **Organized output** with timestamps and comprehensive metadata
- **Multiple data sources** (Yahoo Finance, CCXT exchanges, CSV files)
- **Flexible parameter ranges** and optimization metrics

## Quick Start

### Basic Usage

```bash
# Optimize SMA strategy on AAPL with default parameters
python scripts/optimize.py --strategy sma --symbol AAPL

# Optimize with custom date range
python scripts/optimize.py --strategy sma --symbol AAPL \
  --start 2024-01-01 --end 2024-12-31
```

### Custom Parameter Ranges

```bash
# Specify parameter ranges (format: "min,max" or "val1,val2,val3")
python scripts/optimize.py --strategy sma --symbol AAPL \
  --short-window "10,30" \
  --long-window "40,80" \
  --stop-loss "1,4" \
  --take-profit "3,8" \
  --max-tries 100
```

### Different Data Sources

```bash
# Use Yahoo Finance (default)
python scripts/optimize.py --strategy sma --symbol AAPL --data-source yahoo

# Use cryptocurrency data from CCXT
python scripts/optimize.py --strategy sma --symbol BTCUSDC --data-source ccxt

# Use preprocessed CSV data
python scripts/optimize.py --strategy sma --symbol ETHUSDC --data-source csv_processed
```

### Optimization Metrics

```bash
# Optimize for Sharpe Ratio instead of returns
python scripts/optimize.py --strategy sma --symbol AAPL \
  --metric "Sharpe Ratio"

# Minimize drawdown
python scripts/optimize.py --strategy sma --symbol AAPL \
  --metric "Max. Drawdown [%]" --minimize
```

## Available Strategies

### SMA (Simple Moving Average) Strategy

**Parameters:**
- `--short-window`: Short MA period (default: 5-50)
- `--long-window`: Long MA period (default: 20-100) 
- `--stop-loss`: Stop loss percentage (default: 1-5%)
- `--take-profit`: Take profit percentage (default: 2-10%)

**Example:**
```bash
python scripts/optimize.py --strategy sma --symbol AAPL \
  --short-window "5,25" \
  --long-window "30,100" \
  --stop-loss "0.5,3.0" \
  --take-profit "1.5,6.0"
```

## Parameter Format

Parameters can be specified in two formats:

### Range Format: "min,max"
```bash
--short-window "10,50"  # Tests values from 10 to 50
```

### List Format: "val1,val2,val3"
```bash
--short-window "10,20,30,40,50"  # Tests specific values
```

## Output Files

All outputs are saved to the `output/` directory with timestamp naming:

### File Structure
```
output/
├── optimize_sma_AAPL_20241219_143052_results.json          # Optimization results
├── optimize_sma_AAPL_20241219_143052_heatmap_data.csv      # Raw heatmap data
├── optimize_sma_AAPL_20241219_143052_heatmap_interactive.html  # Interactive heatmap
├── optimize_sma_AAPL_20241219_143052_heatmap_mean.html     # Mean aggregation heatmap
├── optimize_sma_AAPL_20241219_143052_heatmap_*_*_mean.png  # Static heatmaps (mean)
├── optimize_sma_AAPL_20241219_143052_heatmap_*_*_max.png   # Static heatmaps (max)
└── optimize_sma_AAPL_20241219_143052_heatmap_summary.png   # Correlation summary
```

### Results JSON Structure
```json
{
  "best_parameters": {
    "short_window": 25,
    "long_window": 54,
    "stop_loss_pct": 1.4,
    "take_profit_pct": 2.8
  },
  "best_result": {
    "Return [%]": 2.13,
    "Sharpe Ratio": 1.474,
    "Max. Drawdown [%]": -1.52,
    "# Trades": 4
  },
  "optimization_metric": "Return [%]",
  "maximize": true,
  "metadata": {
    "strategy": "sma",
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "data_source": "yahoo",
    "timeframe": "1h",
    "timestamp": "2024-12-19T14:30:52"
  }
}
```

## Heatmap Types

### 1. Interactive Heatmaps (HTML)
- **File**: `*_heatmap_interactive.html`, `*_heatmap_mean.html`
- **Features**: 
  - Interactive Bokeh plots
  - Zoom, pan, hover tooltips
  - Multiple parameter dimensions
  - Generated using `backtesting.lib.plot_heatmaps()`

### 2. Static Heatmaps (PNG)
- **File**: `*_heatmap_{param1}_{param2}_{aggregation}.png`
- **Features**:
  - All parameter pair combinations
  - Mean and max aggregation methods
  - High-resolution PNG format
  - Color-coded performance visualization

### 3. Summary Correlation Plot
- **File**: `*_heatmap_summary.png`
- **Features**:
  - Parameter correlation in top 10% results
  - Identifies parameter relationships
  - Helps understand parameter sensitivity

## Command Line Options

### Required Arguments
- `--strategy`: Strategy name (`sma`, `rsi`)

### Optional Arguments
- `--symbol`: Symbol to optimize (default: from config)
- `--start`: Start date in YYYY-MM-DD format (default: 2024-01-01)
- `--end`: End date in YYYY-MM-DD format (default: 2024-12-31)
- `--data-source`: Data source (`yahoo`, `ccxt`, `csv`)
- `--timeframe`: Timeframe (`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`)
- `--metric`: Optimization metric (default: "Return [%]")
- `--minimize`: Minimize metric instead of maximize
- `--max-tries`: Maximum optimization attempts

### Strategy-Specific Parameters
**SMA Strategy:**
- `--short-window`: Short MA window range
- `--long-window`: Long MA window range  
- `--stop-loss`: Stop loss percentage range
- `--take-profit`: Take profit percentage range

## Advanced Usage

### Grid Search vs Random Search

The optimization uses intelligent sampling. For large parameter spaces, it automatically switches to random sampling:

```bash
# Large parameter space - uses random sampling
python scripts/optimize.py --strategy sma --symbol AAPL \
  --short-window "5,50" \
  --long-window "20,200" \
  --stop-loss "0.5,5.0" \
  --take-profit "1.0,10.0" \
  --max-tries 200
```

### Multiple Timeframes

```bash
# Optimize for different timeframes
python scripts/optimize.py --strategy sma --symbol AAPL --timeframe 1h
python scripts/optimize.py --strategy sma --symbol AAPL --timeframe 4h
python scripts/optimize.py --strategy sma --symbol AAPL --timeframe 1d
```

### Cross-Asset Optimization

```bash
# Stock optimization
python scripts/optimize.py --strategy sma --symbol AAPL --data-source yahoo

# Crypto optimization  
python scripts/optimize.py --strategy sma --symbol BTCUSDC --data-source ccxt

# Custom CSV data (raw)
python scripts/optimize.py --strategy sma --symbol EURUSD --data-source csv

# Processed CSV data with features
python scripts/optimize.py --strategy sma --symbol ETHUSDC --data-source csv_processed --timeframe 15m
```

## Interpreting Results

### Best Parameters
The optimization finds the parameter combination that maximizes (or minimizes) your chosen metric:

```json
"best_parameters": {
  "short_window": 25,
  "long_window": 54,
  "stop_loss_pct": 1.4,
  "take_profit_pct": 2.8
}
```

### Performance Metrics
Key metrics to evaluate:
- **Return [%]**: Total strategy return
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max. Drawdown [%]**: Worst peak-to-trough loss
- **Win Rate [%]**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss ratio

### Heatmap Analysis
1. **Parameter Sensitivity**: How much performance changes with parameter variations
2. **Optimal Regions**: Areas of consistently good performance
3. **Parameter Interactions**: How parameters affect each other
4. **Robustness**: Whether optimal parameters are isolated or part of a stable region

## Best Practices

### 1. Parameter Range Selection
- **Start broad**: Begin with wide parameter ranges
- **Narrow down**: Focus on promising regions from initial optimization
- **Avoid overfitting**: Don't use too narrow ranges

### 2. Data Considerations
- **Sufficient data**: Ensure enough data points for reliable optimization
- **Market regimes**: Consider different market conditions in your date range
- **Out-of-sample testing**: Reserve data for validation

### 3. Optimization Metrics
- **Return [%]**: Good for absolute performance
- **Sharpe Ratio**: Better for risk-adjusted performance
- **Calmar Ratio**: Focus on downside risk
- **Custom metrics**: Use domain-specific measures when appropriate

### 4. Validation
- **Walk-forward analysis**: Test on different time periods
- **Multiple symbols**: Verify parameters work across assets
- **Different timeframes**: Check robustness across timeframes

## Troubleshooting

### Common Issues

**1. "No optimization parameters defined"**
```bash
# Solution: Specify parameter ranges
python scripts/optimize.py --strategy sma --symbol AAPL \
  --short-window "10,30"
```

**2. "No heatmap data available"**
- Increase `--max-tries` for more parameter combinations
- Ensure you're using `return_heatmap=True` (automatic)

**3. "Optimization failed: unhashable type"**
- Check parameter format (use quotes: "10,30")
- Verify parameter values are numeric

**4. "Failed to load data"**
- Check symbol name and data source
- Verify date range has available data
- Ensure CSV files exist for CSV data source

### Performance Tips

**1. Speed up optimization:**
```bash
# Reduce date range for faster testing
python scripts/optimize.py --strategy sma --symbol AAPL \
  --start 2024-06-01 --end 2024-08-01

# Limit optimization attempts
python scripts/optimize.py --strategy sma --symbol AAPL \
  --max-tries 50
```

**2. Parallel processing:**
- The optimization automatically uses parallel processing
- More CPU cores = faster optimization

## Examples

### Complete Optimization Workflow

```bash
# 1. Quick test with small date range
python scripts/optimize.py --strategy sma --symbol AAPL \
  --start 2024-06-01 --end 2024-08-01 \
  --short-window "10,30" \
  --max-tries 20

# 2. Full optimization on promising parameters  
python scripts/optimize.py --strategy sma --symbol AAPL \
  --start 2024-01-01 --end 2024-12-31 \
  --short-window "20,30" \
  --long-window "50,70" \
  --stop-loss "1,3" \
  --take-profit "2,6" \
  --max-tries 100

# 3. Validate on different timeframe
python scripts/optimize.py --strategy sma --symbol AAPL \
  --timeframe 4h \
  --short-window "25" \
  --long-window "60" \
  --stop-loss "2" \
  --take-profit "4"
```

### Multi-Asset Analysis

```bash
# Optimize the same strategy across different assets
for symbol in AAPL MSFT GOOGL; do
  python scripts/optimize.py --strategy sma --symbol $symbol \
    --short-window "15,35" \
    --long-window "45,85" \
    --max-tries 50
done
```

## Integration with Backtesting

After optimization, use the best parameters in your regular backtesting:

```bash
# Use optimized parameters in run_backtest.py
python scripts/run_backtest.py --strategy sma --symbol AAPL \
  --start 2024-01-01 --end 2024-12-31
```

Update your strategy configuration with the optimized parameters or use them in the trading bot's main configuration.

## See Also

- [Quick Start User Guide](README.md) - General framework usage
- [Backtesting Guide](docs/BACKTESTING_README.md) - Single backtest runs
- [Data Preprocessing Guide](scripts/README_preprocessing.md) - Data preparation
- [Backtesting.py Documentation](https://kernc.github.io/backtesting.py/) - Underlying library