# Walk-Forward Analysis

Walk-forward analysis tests trading strategy robustness by optimizing parameters on in-sample (IS) data and testing on out-of-sample (OOS) data across rolling time windows. Uses ManualGridSearchOptimizer for consistent parameter optimization and calculates Walk-Forward Efficiency (WFE) to measure parameter robustness.

## Overview

Walk-forward analysis differs from standard backtesting by:
- Optimizing parameters on in-sample data for each period using ManualGridSearchOptimizer
- Testing optimized parameters on out-of-sample data to validate robustness
- Calculating Walk-Forward Efficiency (WFE) to measure optimization consistency
- Providing insights into strategy performance across different market regimes
- Creating separate directories for organized output

## Quick Start

### Basic Usage

```bash
# Simple walk-forward analysis with SMA strategy (3 months IS, 1 month OOS, rolling)
PARAM_CONFIG='{"short_window": {"values": [8, 10, 12]}, "long_window": {"values": [20, 25, 30]}}' python scripts/run_walk_forward.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --symbol BTC_USDT \
  --is-window-months 3 \
  --oos-window-months 1

# FVG strategy with range parameters
PARAM_CONFIG='{"h1_lookback_candles": {"min": 20, "max": 28, "step": 2}, "risk_reward_ratio": {"choices": [2.5, 3.0, 3.5]}}' python scripts/run_walk_forward.py \
  --strategy fvg \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --symbol BTC_USDT \
  --is-window-months 6 \
  --oos-window-months 1 \
  --window-mode anchored
```

### Advanced Configuration

```bash
# Rolling windows with range parameters
PARAM_CONFIG='{"short_window": {"min": 8, "max": 16, "step": 2, "type": "int"}, "long_window": {"min": 25, "max": 45, "step": 5, "type": "int"}}' \
RISK_PARAMS='{"risk_percent": 0.01}' \
python scripts/run_walk_forward.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --symbol BTC_USDT \
  --is-window-months 6 \
  --oos-window-months 2 \
  --step-months 1 \
  --window-mode rolling \
  --initial-capital 50000 \
  --optimization-metric sharpe_ratio

# Anchored windows with choice parameters
PARAM_CONFIG='{"h1_lookback_candles": {"choices": [20, 24, 28]}, "risk_reward_ratio": {"choices": [2.5, 3.0, 3.5]}}' python scripts/run_walk_forward.py \
  --strategy fvg \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --symbol BTC_USDT \
  --is-window-months 4 \
  --oos-window-months 1 \
  --step-months 1 \
  --window-mode anchored
```

## Parameters

### Parameter Configuration

Parameter configuration is passed via the `PARAM_CONFIG` environment variable using the same format as the optimization script. This defines parameter ranges to optimize during in-sample periods:

**Parameter Configuration Formats:**
- `{"values": [val1, val2, val3]}` - Direct list of values
- `{"choices": [val1, val2, val3]}` - Choice parameters (same as values)
- `{"min": X, "max": Y, "step": Z, "type": "int|float"}` - Range parameters

**SMA Strategy:**
```bash
PARAM_CONFIG='{"short_window": {"min": 8, "max": 16, "step": 2, "type": "int"}, "long_window": {"values": [20, 25, 30]}, "stop_loss_pct": {"choices": [0.015, 0.02, 0.025]}}'
```

**FVG Strategy:**
```bash
PARAM_CONFIG='{"h1_lookback_candles": {"choices": [20, 24, 28]}, "risk_reward_ratio": {"min": 2.0, "max": 3.5, "step": 0.5}, "max_hold_hours": {"values": [3, 4, 5]}}'
```

**Breakout Strategy:**
```bash
PARAM_CONFIG='{"entry_lookback": {"min": 20, "max": 40, "step": 5, "type": "int"}, "atr_multiplier": {"choices": [2.0, 2.5, 3.0]}}'
```

### Optimization Metrics

You can specify which metric to optimize for during the in-sample periods:
```bash
--optimization-metric return_pct        # Default: Total return percentage
--optimization-metric sharpe_ratio      # Risk-adjusted return
--optimization-metric sortino_ratio     # Downside risk-adjusted return
```

### Risk Management Parameters

Risk management parameters are passed via the `RISK_PARAMS` environment variable:

**Fixed Risk Manager:**
```bash
RISK_PARAMS='{"risk_percent": 0.01, "default_stop_distance": 0.02}'
```

**Fixed Position Size Manager:**
```bash
RISK_PARAMS='{"position_size": 0.05}'
```

### Time Window Parameters

- `--is-window-months`: Duration of in-sample period for optimization (default: 3 months)
- `--oos-window-months`: Duration of out-of-sample period for testing (default: 1 month)
- `--step-months`: How far to advance between periods (default: 1 month)
- `--window-mode`: Window mode - 'rolling' or 'anchored' (default: rolling)

**Window Modes:**

**Rolling Mode (Fixed IS Window):**
- IS window length stays constant
- Both IS and OOS windows slide forward by step_months
- Good for testing parameter stability with consistent data amounts

**Anchored Mode (Expanding IS Window):**
- IS window starts from data beginning and expands each step
- OOS window slides forward normally
- Good for testing how more data improves optimization

**Example Configurations:**
- `--is-window-months 3 --oos-window-months 1 --step-months 1 --window-mode rolling`: 3-month IS, 1-month OOS, rolling
- `--is-window-months 6 --oos-window-months 1 --step-months 1 --window-mode anchored`: Expanding IS starting at 6 months
- `--is-window-months 4 --oos-window-months 2 --step-months 2 --window-mode rolling`: 4-month IS, 2-month OOS, larger steps

## Command Line Options

```bash
python scripts/run_walk_forward.py [OPTIONS]

Required:
  --strategy {sma,fvg,breakout}  Strategy to analyze
  --data-file PATH               Path to CSV data file
  
Optional:
  --symbol TEXT                  Symbol identifier (default: UNKNOWN)
  --start DATE                   Start date (YYYY-MM-DD)
  --end DATE                     End date (YYYY-MM-DD)
  --is-window-months INT         Months of in-sample data for optimization (default: 3)
  --oos-window-months INT        Months of out-of-sample data for testing (default: 1)
  --step-months INT              Months to step forward between periods (default: 1)
  --window-mode {rolling,anchored}  Window mode (default: rolling)
  --optimization-metric TEXT     Metric to optimize ('return_pct', 'sharpe_ratio', 'sortino_ratio')
  --minimize                     Minimize metric instead of maximize
  --initial-capital FLOAT        Starting capital (default: 10000)
  --commission FLOAT             Commission rate (default: 0.001)
  --margin FLOAT                 Margin requirement (default: 0.01 = 100x leverage)
  --risk-manager {fixed_position,fixed_risk}  Risk manager type (default: fixed_risk)
  --n-jobs INT                   Number of parallel jobs (default: auto-detect)
  --debug                        Enable debug logging
```

## Output and Results

### Console Output

The script provides a comprehensive summary including:

```
================================================================================
WALK-FORWARD ANALYSIS RESULTS - FVGStrategy
================================================================================
Symbol: BTC_USDT
Total Periods: 5
IS Window: 6 months, OOS Window: 1 months
Window Mode: rolling, Step: 1 months
Optimization Metric: return_pct

WALK-FORWARD EFFICIENCY (WFE):
----------------------------------------
Average WFE:           45.2%
Median WFE:            52.1%
WFE Std Dev:           28.3%
Positive WFE Periods:  2/5 (40.0%)
WFE Consistency:       0.672

PERFORMANCE METRICS:
----------------------------------------
Avg IS Return:         12.43%
Avg OOS Return:        5.62%
Median OOS Return:     4.83%
OOS Return Std Dev:    8.45%
Positive OOS Periods:  3/5 (60.0%)
IS-OOS Correlation:    0.423
Avg IS Sharpe:         1.205
Avg OOS Sharpe:        0.687
Avg IS Trades:         45.2
Avg OOS Trades:        12.8

🟠 Moderate: Strategy shows some degradation out-of-sample

TOP 3 OOS PERFORMING PERIODS:
----------------------------------------
  1. 12.0% OOS (WFE: 79%) - 2024-03-01 to 2024-03-31
  2. 8.5% OOS (WFE: 65%) - 2024-06-01 to 2024-06-30
  3. 3.2% OOS (WFE: 52%) - 2024-09-01 to 2024-09-30
```

### File Outputs

Results are saved to separate timestamped directories in `output/walk_forward/`:

**Directory Structure:**
```
output/walk_forward/sma_BTC_USDT_20250901_143022/
├── walk_forward_periods_BTC_USDT.csv         # Period-by-period results
├── walk_forward_summary_BTC_USDT.json        # Summary metrics and analysis
└── walk_forward_analysis_BTC_USDT.png        # WFE visualization charts
```

**File Contents:**
1. **Period CSV**: Period-by-period data with IS/OOS metrics, WFE, and optimal parameters
2. **Summary JSON**: Complete analysis metrics, WFE statistics, and configuration parameters
3. **Analysis Chart**: 2x2 visualization with IS vs OOS performance, WFE by period, correlation analysis, and summary statistics

## Interpreting Results

### Walk-Forward Efficiency (WFE)

- **Average WFE**: Mean efficiency across all periods (higher = better parameter consistency)
- **WFE Standard Dev**: Variability of efficiency (lower = more stable optimization)
- **Positive WFE Periods**: Percentage of periods where OOS performance exceeded expectations
- **WFE Formula**: (Annualized OOS Return) / (Annualized IS Return) × 100%

### Combined Metrics

- **IS/OOS Average Return**: Comparison of in-sample vs out-of-sample performance
- **IS/OOS Average Sharpe**: Risk-adjusted return comparison
- **Return Std Dev**: Variability of OOS returns (lower = more consistent)
- **Positive Periods**: Percentage of OOS periods with positive returns
- **Average/Worst Max DD**: Drawdown statistics for OOS periods

### Stability Metrics

- **Return Volatility**: Standard deviation of OOS period returns (lower = more stable)
- **Return Consistency**: Mean OOS return / standard deviation (higher = more consistent)
- **Temporal Stability**: Measures OOS performance consistency over time (0-1, higher = more stable)
- **Trend Consistency**: Measures directional consistency of OOS performance changes

### Interpretation Guidelines

**Good Walk-Forward Results:**
- High average WFE (>80%) indicating good parameter generalization
- High percentage of positive OOS periods (>60%)
- Low WFE standard deviation (<30%) showing stable optimization
- Consistent IS vs OOS performance ratios
- High temporal stability (>0.7)

**Warning Signs:**
- Low average WFE (<50%) indicating poor parameter generalization
- High WFE volatility (>50%) showing unstable optimization
- Large performance gap between IS and OOS results
- Very few positive OOS periods (<30%)
- Low temporal stability (<0.3)

## Best Practices

### 1. Parameter Space Definition

Define appropriate parameter ranges for optimization:
```bash
# Define parameter space around expected optimal values
PARAMETER_SPACE='{"short_window": [8, 10, 12, 14], "long_window": [20, 25, 30, 35]}' python scripts/run_walk_forward.py --strategy sma --data-file data.csv

# For FVG strategy, test different lookback periods and risk ratios
PARAMETER_SPACE='{"h1_lookback_candles": [16, 20, 24, 28], "risk_reward_ratio": [2.0, 2.5, 3.0, 3.5]}' python scripts/run_walk_forward.py --strategy fvg --data-file data.csv
```

### 2. Time Window Configuration

**For High-Frequency Strategies (15m, 1h):**
- IS window: 2-4 months, OOS window: 2-4 weeks
- Step size: 1-2 weeks (use rolling mode)

**For Lower-Frequency Strategies (4h, 1d):**
- IS window: 6-12 months, OOS window: 1-3 months  
- Step size: 1-2 months (rolling or anchored mode)

### 3. Data Requirements

- Minimum 1 year of data for meaningful analysis
- At least 3-4 complete test windows
- High-quality, clean data without gaps

### 4. Risk Management

Always use appropriate risk management:
```bash
# Conservative risk management
RISK_PARAMS='{"risk_percent": 0.005}' # 0.5% risk per trade

# Position-based management for testing
RISK_PARAMS='{"position_size": 0.05}' # 5% of equity per trade
```

## Common Use Cases

### 1. Strategy Validation

Validate parameter optimization robustness across multiple periods:
```bash
# Test SMA parameter space for consistent optimization (rolling windows)
PARAMETER_SPACE='{"short_window": [6, 8, 10, 12], "long_window": [18, 21, 24, 27]}' python scripts/run_walk_forward.py \
  --strategy sma --data-file data.csv --is-window-months 4 --oos-window-months 1 --step-months 1 --window-mode rolling
```

### 2. Market Regime Analysis

Identify which market conditions favor your strategy and how parameter optimization adapts:
```bash
# Shorter windows to identify specific market periods (rolling mode)
PARAMETER_SPACE='{"h1_lookback_candles": [20, 24, 28], "risk_reward_ratio": [2.5, 3.0, 3.5]}' python scripts/run_walk_forward.py \
  --strategy fvg --data-file data.csv --is-window-months 2 --oos-window-months 1 --step-months 1 --window-mode rolling

# Anchored mode to see how expanding data affects optimization
PARAMETER_SPACE='{"h1_lookback_candles": [20, 24, 28], "risk_reward_ratio": [2.5, 3.0, 3.5]}' python scripts/run_walk_forward.py \
  --strategy fvg --data-file data.csv --is-window-months 3 --oos-window-months 1 --step-months 1 --window-mode anchored
```

### 3. Parameter Robustness Testing

Test how the optimization process handles different parameter ranges:
```bash
# Test narrow parameter ranges around expected optimums
PARAMETER_SPACE='{"short_window": [9, 10, 11], "long_window": [19, 20, 21]}'

# Test wider parameter ranges to assess optimization stability
PARAMETER_SPACE='{"short_window": [5, 8, 10, 12, 15], "long_window": [15, 20, 25, 30, 35]}'
```

## Integration with Other Tools

### Comprehensive Analysis Workflow

```bash
# 1. Define parameter optimization space based on strategy knowledge
PARAMETER_SPACE='{"short_window": [8, 10, 12], "long_window": [20, 25, 30]}'

# 2. Run walk-forward analysis with optimization
python scripts/run_walk_forward.py --strategy sma --data-file data.csv

# 3. Analyze WFE results to assess parameter consistency
# 4. Review charts showing IS vs OOS performance
# 5. Check period details for parameter adaptation patterns
```

### Before Live Trading

Walk-forward analysis should be the final validation step before deploying a strategy:

1. **Strategy Development** → Design and implement strategy
2. **Walk-Forward Analysis** → Optimize parameters on IS data and validate on OOS data across time periods
3. **Parameter Robustness Assessment** → Analyze WFE metrics and parameter consistency
4. **Paper Trading** → Test in simulated live environment
5. **Live Trading** → Deploy with confidence

## Troubleshooting

### Common Issues

**"No successful walk-forward periods"**
- Check that data has sufficient length for the specified time windows
- Reduce test window months or increase step months
- Verify parameter space contains valid parameter combinations
- Ensure parameter space is not too restrictive

**"Insufficient data"**
- Ensure at least 1 year of data for meaningful analysis
- Check data quality and remove any gaps
- Consider using longer timeframe data (4h instead of 15m)

**Poor WFE or OOS Performance**
- May indicate overfitting during IS optimization
- Consider expanding parameter space for more robust optimization
- Review if optimization metric aligns with trading objectives
- Check for market regime changes affecting parameter effectiveness

### Performance Optimization

- Use `--n-jobs 1` for sequential processing (more stable)
- For large datasets, consider reducing test frequency or window size
- Monitor memory usage with very long datasets

## Output Files

Results are automatically saved to `output/walk_forward/` with timestamped filenames:

- **`*_summary.json`**: Complete analysis results with WFE metrics, IS/OOS comparison, and stability measures
- **`*_periods.csv`**: Detailed period-by-period data including optimal parameters, IS/OOS performance, and WFE scores
- **`*_charts.png`**: Comprehensive visualizations showing performance comparison, WFE distribution, and correlation analysis

These files provide complete insight into parameter optimization robustness and can be used for strategy refinement and deployment decisions.