# Walk-Forward Analysis Documentation

## Overview

Walk-forward analysis is a robust method for testing trading strategies that helps prevent overfitting and provides a more realistic assessment of strategy performance. The `walk_forward.py` script implements this methodology by:

1. Dividing historical data into multiple consecutive windows
2. Optimizing parameters on a training period
3. Testing those parameters on an out-of-sample test period
4. Calculating walk-forward efficiency to assess robustness

## Usage

### Basic Command

```bash
python scripts/walk_forward.py --strategy sma --symbol SPY --start 2022-01-01 --end 2024-12-31 --train-period 180 --test-period 60
```

### Parameters

#### Required Parameters:
- `--strategy`: Strategy name (sma, rsi)
- `--train-period`: Training period in days (default: 180)
- `--test-period`: Test period in days (default: 60)

#### Optional Parameters:
- `--symbol`: Trading symbol (defaults to first symbol in config)
- `--start`: Start date in YYYY-MM-DD format (default: 2022-01-01)
- `--end`: End date in YYYY-MM-DD format (default: 2024-12-31)
- `--data-source`: Data source override (yahoo, ccxt, csv, csv_processed)
- `--timeframe`: Timeframe override (1m, 5m, 15m, 30m, 1h, 1d)
- `--metric`: Optimization metric (default: 'Return [%]')
- `--minimize`: Minimize metric instead of maximize
- `--max-tries`: Maximum optimization attempts

#### Strategy-Specific Parameters:

For SMA strategy:
- `--short-window`: Short window range (e.g., "5,20" for range or "5,10,15" for specific values)
- `--long-window`: Long window range (e.g., "20,50" or "20,30,40,50")
- `--stop-loss`: Stop loss percentage range (e.g., "1,5")
- `--take-profit`: Take profit percentage range (e.g., "2,10")

### Examples

#### Example 1: Basic SMA Walk-Forward
```bash
python scripts/walk_forward.py --strategy sma --symbol AAPL --train-period 90 --test-period 30 --short-window 10,30 --long-window 20,60
```

#### Example 2: Crypto with CCXT
```bash
python scripts/walk_forward.py --strategy sma --symbol BTC/USDT --data-source ccxt --timeframe 1h --train-period 30 --test-period 10
```

#### Example 3: Using Processed CSV Data
```bash
python scripts/walk_forward.py --strategy rsi --symbol SPY --data-source csv_processed --start 2020-01-01 --end 2023-12-31
```

## Walk-Forward Process

### 1. Window Generation
The script divides the data into consecutive train-test windows:
- Training period: Used for parameter optimization
- Test period: Used for out-of-sample validation
- Windows overlap by advancing the start date by the test period length

Example with 180-day train, 60-day test:
```
Window 1: Train [Jan 1 - Jun 30], Test [Jul 1 - Aug 30]
Window 2: Train [Jul 1 - Dec 28], Test [Dec 29 - Feb 27]
...
```

### 2. Optimization Phase
For each window:
- Extract training data
- Run parameter optimization using the specified metric
- Find best parameter combination

### 3. Testing Phase
- Apply optimized parameters to test period
- Record performance metrics
- No further optimization allowed

### 4. Efficiency Calculation
Walk-forward efficiency = (Average Test Performance / Average Train Performance) × 100

## Output Files

The script generates several output files in the `output/` directory:

### 1. Results JSON
`walk_forward_{strategy}_{symbol}_{timestamp}_results.json`

Contains:
- Detailed results for each window
- Best parameters per window
- Train and test performance metrics
- Walk-forward efficiency statistics

### 2. Performance Plot
`walk_forward_performance_{timestamp}.png`

Shows:
- Training vs test performance across windows
- Visual comparison of in-sample vs out-of-sample results

### 3. Analysis Plot
`walk_forward_analysis_{timestamp}.png`

Contains four subplots:
- Walk-forward efficiency bar
- Performance distribution histogram
- Train vs test correlation scatter plot
- Summary statistics

## Interpreting Results

### Walk-Forward Efficiency

The efficiency percentage indicates how well the strategy maintains its performance out-of-sample:

- **80%+ (Excellent)**: Strategy shows strong out-of-sample robustness
- **60-80% (Good)**: Strategy shows reasonable out-of-sample performance
- **40-60% (Moderate)**: Strategy shows some overfitting concerns
- **Below 40% (Poor)**: Strategy appears to be overfitted

### Key Metrics

1. **Efficiency**: Overall walk-forward efficiency percentage
2. **Average Train Metric**: Mean performance during training periods
3. **Average Test Metric**: Mean performance during test periods
4. **Test Std Dev**: Consistency of test results
5. **Correlation**: Relationship between train and test performance
6. **Win Rate**: Percentage of profitable test windows

### Red Flags

Watch out for:
- Very low efficiency (<30%)
- Negative correlation between train and test
- High train performance but near-zero test performance
- Large standard deviation in test results
- Low percentage of profitable test windows

## Best Practices

### 1. Period Selection
- Training period should be long enough to capture various market conditions
- Test period should be meaningful but not too long
- Common ratios: 3:1 or 6:1 (train:test)

### 2. Parameter Ranges
- Use reasonable parameter ranges based on strategy logic
- Avoid overly wide ranges that lead to curve fitting
- Consider market regime when setting ranges

### 3. Sample Size
- Ensure enough windows for statistical significance (minimum 5-10)
- Longer data history provides more windows
- Balance between window count and period length

### 4. Multiple Runs
- Test different train/test period combinations
- Vary parameter ranges to check sensitivity
- Compare results across different symbols

## Advanced Usage

### Custom Metrics
Optimize for different metrics:
```bash
python scripts/walk_forward.py --strategy sma --metric "Sharpe Ratio" --train-period 180 --test-period 60
```

### Minimize Instead of Maximize
For metrics like drawdown:
```bash
python scripts/walk_forward.py --strategy sma --metric "Max. Drawdown [%]" --minimize --train-period 180 --test-period 60
```

### Controlling Optimization Attempts
Limit optimization iterations:
```bash
python scripts/walk_forward.py --strategy sma --max-tries 100 --train-period 180 --test-period 60
```

## Troubleshooting

### Common Issues

1. **No valid windows generated**
   - Check data availability for the date range
   - Ensure train + test periods don't exceed data length
   - Verify data source configuration

2. **Few or No Trades in Test Periods**
   This is a common issue with crossover-based strategies like SMA:
   - **Problem**: SMA crossover strategies only generate signals at the exact crossover points
   - **Impact**: Short test periods (e.g., 60 days) may have no crossovers
   - **Solutions**:
     - Use longer test periods (90-180 days) to increase probability of crossovers
     - Use position-based strategies that maintain positions throughout the period
     - Consider more active strategies (e.g., RSI mean reversion)
     - Use shorter moving average periods for more frequent signals

3. **Very low or negative efficiency**
   - Consider longer training periods
   - Review parameter ranges for overfitting
   - Test on different time periods
   - May indicate the strategy is curve-fitted to training data

### Performance Tips

1. Use processed CSV data for faster execution
2. Reasonable parameter ranges reduce optimization time
3. Smaller test periods create more windows
4. Consider parallel execution for multiple symbols

## Integration with Other Tools

### After Walk-Forward Analysis

1. Use best parameters in production:
   - Take parameters from best-performing windows
   - Consider averaging parameters across windows
   - Monitor live performance against walk-forward results

2. Combine with regular optimization:
   - Use walk-forward to validate optimization results
   - Compare in-sample optimization vs walk-forward efficiency

3. Risk management:
   - Use walk-forward statistics to set position sizing
   - Adjust risk based on efficiency scores
   - Monitor strategy degradation over time

## Example Workflow

1. **Initial Optimization**
   ```bash
   python scripts/optimize.py --strategy sma --symbol SPY
   ```

2. **Walk-Forward Validation**
   ```bash
   python scripts/walk_forward.py --strategy sma --symbol SPY --train-period 180 --test-period 60
   ```

3. **Production Deployment**
   - If efficiency > 60%, consider for live trading
   - Use average of best parameters from windows
   - Set up monitoring based on test performance statistics

4. **Ongoing Monitoring**
   - Re-run walk-forward analysis periodically
   - Compare live results to walk-forward expectations
   - Adjust or retire strategy if efficiency degrades

## Notes on Strategy Implementation

The walk-forward script includes a position-based implementation of the SMA strategy that:
- Maintains long positions when short MA > long MA
- Maintains short positions when short MA < long MA
- Switches positions when the relationship changes

This ensures trades occur in test periods, unlike the original crossover-only strategy which may not generate any signals in short test windows.

For production use, consider:
1. Implementing your strategies with position-based logic
2. Using longer test periods for crossover strategies
3. Testing multiple strategy types to find those that perform well in walk-forward analysis