# Monte Carlo Simulation for Trading Strategy Analysis

## Overview

The Monte Carlo simulation tool (`scripts/run_monte_carlo.py`) performs statistical analysis on trading strategy results by randomizing the order of trades to assess strategy robustness and potential drawdown scenarios.

## What is Monte Carlo Simulation in Trading?

Monte Carlo simulation in trading involves:
1. **Taking your actual trade results** (wins/losses from backtesting)
2. **Randomly reshuffling the order** of these trades thousands of times
3. **Calculating statistics** for each random sequence (returns, drawdowns, etc.)
4. **Analyzing the distribution** of possible outcomes

This helps answer critical questions:
- "What if the losing trades had happened first?"
- "How bad could my drawdown have been with different trade timing?"
- "How robust is my strategy's performance?"
- "What's the probability of significant losses?"

## Key Metrics Analyzed

### Return Statistics
- **Mean Return**: Average final return across all simulations
- **Return Distribution**: 5th to 95th percentiles of possible returns
- **Probability of Loss**: Chance of ending with negative returns

### Drawdown Analysis
- **Maximum Drawdown**: Worst peak-to-trough decline in account value
- **Average Drawdown**: Mean of all negative drawdown periods
- **Drawdown Duration**: How long drawdown periods typically last
- **Percentile Analysis**: Best case (5th percentile) to worst case (95th percentile)

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns (return per unit of volatility)
- **Sortino Ratio**: Risk-adjusted returns using only downside volatility
- **Win Rate**: Percentage of profitable trades
- **Risk Assessment**: Probabilities of exceeding specific drawdown thresholds

## Usage Examples

### Basic Usage (Auto-detect Most Recent Backtest)
```bash
# Run Monte Carlo on most recent backtest with default 1000 simulations
python scripts/run_monte_carlo.py

# Run with more simulations for higher precision
python scripts/run_monte_carlo.py --simulations 5000
```

### Specific Files
```bash
# Run on specific trades CSV file
python scripts/run_monte_carlo.py --trades-file output/backtests/sma_BTC_USDT_20240101_trades.csv

# Run on specific backtest directory
python scripts/run_monte_carlo.py --backtest-dir output/backtests/sma_BTC_USDT_20240101

# Custom output directory
python scripts/run_monte_carlo.py --output-dir my_monte_carlo_results
```

### Advanced Options
```bash
# Different initial capital (affects absolute dollar calculations)
python scripts/run_monte_carlo.py --initial-capital 50000 --simulations 2000

# Enable debug logging
python scripts/run_monte_carlo.py --debug --simulations 100
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--trades-file` | Path to trades CSV file | Auto-detect |
| `--backtest-dir` | Path to backtest directory | Auto-detect |
| `--simulations` | Number of Monte Carlo simulations | 1000 |
| `--initial-capital` | Initial capital amount | 10000 |
| `--output-dir` | Custom output directory | Auto-generated |
| `--debug` | Enable debug logging | False |

## Output Files

The simulation creates three output files in `output/monte_carlo/{strategy}_{timestamp}/`:

### 1. Comprehensive Chart (`*_monte_carlo.png`)
A detailed visualization showing:
- **Sample Equity Curves**: 100 randomly selected simulation paths
- **Maximum Drawdown Distribution**: Histogram of worst-case drawdowns
- **Final Return Distribution**: Range of possible strategy outcomes
- **Sharpe Ratio Distribution**: Risk-adjusted performance spread
- **Risk-Return Scatter**: Relationship between risk and reward
- **Drawdown Duration**: How long bad periods might last
- **Statistics Summary Table**: Key metrics and probabilities

### 2. Results Summary (`*_monte_carlo_results.json`)
Statistical summary including:
```json
{
  "num_simulations": 1000,
  "num_trades": 68,
  "statistics": {
    "final_returns": {
      "mean": 15.2,
      "std": 8.3,
      "percentiles": {
        "p5": 2.1,
        "p25": 9.4,
        "p50": 14.8,
        "p75": 20.1,
        "p95": 30.5
      }
    },
    "max_drawdowns": {
      "mean": -12.4,
      "percentiles": {
        "p5": -22.1,
        "p95": -5.2
      }
    }
  }
}
```

### 3. Detailed Metrics (`*_monte_carlo_metrics.csv`)
Raw simulation data with columns:
- `max_drawdowns`: Maximum drawdown for each simulation
- `final_returns`: Final return for each simulation  
- `sharpe_ratios`: Sharpe ratio for each simulation
- `sortino_ratios`: Sortino ratio for each simulation
- `win_rates`: Win rate for each simulation
- `max_dd_durations`: Duration of maximum drawdown period

## Integration with Backtest Workflow

### Standard Workflow
```bash
# 1. Run backtest
python scripts/run_backtest.py --strategy sma --symbol BTC_USDT --data-file data/cleaned/BTC_USDT.csv

# 2. Run Monte Carlo analysis (auto-detects most recent backtest)
python scripts/run_monte_carlo.py --simulations 2000

# 3. Review results in output/monte_carlo/ directory
```

### Batch Analysis
```bash
# Analyze multiple strategies
for strategy in sma fvg breakout; do
    python scripts/run_backtest.py --strategy $strategy --symbol BTC_USDT --data-file data.csv
    python scripts/run_monte_carlo.py --simulations 1000
done
```

## Interpreting Results

### Return Analysis
- **Mean Return**: Expected performance based on your trades
- **5th Percentile**: "Bad luck" scenario - only 5% chance of worse performance
- **95th Percentile**: "Good luck" scenario - only 5% chance of better performance
- **Standard Deviation**: Volatility of outcomes

### Drawdown Analysis
- **Mean Max Drawdown**: Typical worst-case loss you should expect
- **95th Percentile Drawdown**: Near worst-case scenario (1 in 20 chance)
- **5th Percentile Drawdown**: Best-case scenario for drawdowns

### Risk Assessment Examples
```
Prob(DD > 20%): 15.0%  # 15% chance of >20% drawdown
Prob(DD > 30%): 3.2%   # 3.2% chance of >30% drawdown  
Prob(Loss): 8.5%       # 8.5% chance of losing money overall
```

## Strategy Robustness Evaluation

### Robust Strategy Indicators
- **Low return standard deviation**: Consistent performance regardless of trade order
- **Reasonable worst-case drawdowns**: 95th percentile drawdown not excessive
- **High probability of profit**: Low chance of overall loss
- **Stable Sharpe ratios**: Risk-adjusted performance doesn't vary wildly

### Warning Signs
- **Very high return standard deviation**: Performance highly dependent on trade order
- **Extreme worst-case scenarios**: 95th percentile shows catastrophic drawdowns
- **High probability of loss**: >20% chance of losing money
- **Wide Sharpe ratio distribution**: Inconsistent risk-adjusted performance

## Best Practices

### Simulation Count
- **100 simulations**: Quick test, rough estimates
- **1,000 simulations**: Standard analysis, good precision
- **5,000+ simulations**: High precision for critical decisions

### Minimum Trade Count
- **<20 trades**: Results may be unreliable due to small sample size
- **50+ trades**: Good statistical foundation
- **100+ trades**: Excellent statistical reliability

### Analysis Frequency
- **After strategy development**: Validate robustness before live trading
- **Before capital allocation**: Understand worst-case scenarios
- **Periodic review**: Confirm ongoing performance consistency

## Technical Notes

### Assumptions
- **Independence**: Assumes trade results are independent (no market regime effects)
- **Stationarity**: Assumes future market conditions similar to backtest period
- **No Sequence Effects**: Ignores potential benefits/drawbacks of trade timing

### Limitations
- **Market Regime Changes**: Cannot predict performance in different market conditions
- **Liquidity Constraints**: Doesn't account for real-world execution limitations
- **Correlation Effects**: May underestimate risk during market stress periods

### Statistical Validity
- Results become more reliable with larger numbers of trades and simulations
- Confidence intervals provide ranges rather than exact predictions
- Should be combined with other analysis methods for complete strategy evaluation

## Example Interpretation

Consider a strategy with Monte Carlo results showing:
- Mean return: 15.2%, Std dev: 8.3%
- 5th percentile return: 2.1%
- Mean max drawdown: -12.4%
- 95th percentile max drawdown: -22.1%
- Probability of loss: 5.2%

**Interpretation:**
- The strategy typically returns around 15% but could vary significantly (±8.3%)
- Even in bad scenarios (5th percentile), still expect some profit (2.1%)
- Typical worst drawdown around 12%, but could be as bad as 22% in rare cases
- Very low chance (5.2%) of losing money overall
- This appears to be a robust strategy with acceptable risk characteristics

This analysis helps set realistic expectations and appropriate position sizing for live trading.