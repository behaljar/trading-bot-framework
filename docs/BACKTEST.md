# Backtesting Guide

This guide covers the comprehensive backtesting capabilities of the trading framework. The framework uses the industry-standard `backtesting.py` library with custom strategy wrappers to provide professional-grade backtesting with advanced risk management features.

## Overview

The backtesting system evaluates trading strategies on historical data to assess performance, risk metrics, and profitability before deploying strategies in live trading. The framework provides:

- **Professional Backtesting Engine**: Built on `backtesting.py` library
- **Strategy Integration**: Seamless integration with framework strategies
- **Risk Management**: Built-in stop loss, take profit, and position sizing
- **Comprehensive Metrics**: Industry-standard performance indicators
- **Interactive Visualizations**: HTML plots and detailed reports
- **High-Precision Trading**: Fractional share support for high-priced assets

## Quick Start

### Basic Backtesting

```bash
# Simple backtest with SMA strategy
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT.csv --symbol BTC_USDT

# Backtest with custom parameters
STRATEGY_PARAMS='{"short_window": 10, "long_window": 20, "position_size": 0.1}' \
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT.csv --symbol BTC_USDT

# Backtest with stop loss and take profit
STRATEGY_PARAMS='{"short_window": 10, "long_window": 20, "position_size": 0.1, "stop_loss_pct": 0.02, "take_profit_pct": 0.04}' \
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT.csv --symbol BTC_USDT
```

### Advanced Configuration

```bash
# Full configuration with custom capital and commission
uv run python scripts/run_backtest.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT.csv \
  --symbol BTC_USDT \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --initial-capital 100000 \
  --commission 0.001 \
  --debug
```

## Backtest Script Reference

### Command Line Options

```bash
python scripts/run_backtest.py [OPTIONS]
```

**Required Arguments:**
- `--strategy STRATEGY`: Strategy to use (currently supports: `sma`)
- `--data-file PATH`: Path to CSV data file
- `--symbol SYMBOL`: Symbol identifier for results

**Optional Arguments:**
- `--start YYYY-MM-DD`: Start date for backtesting
- `--end YYYY-MM-DD`: End date for backtesting
- `--initial-capital AMOUNT`: Initial capital (default: 10000)
- `--commission RATE`: Commission rate (default: 0.001 = 0.1%)
- `--use-standard`: Use standard Backtest instead of FractionalBacktest
- `--debug`: Enable debug logging

### Strategy Parameters

Strategy parameters are passed via the `STRATEGY_PARAMS` environment variable as JSON:

```bash
# SMA Strategy Parameters
STRATEGY_PARAMS='{
  "short_window": 10,
  "long_window": 20,
  "position_size": 0.1,
  "stop_loss_pct": 0.02,
  "take_profit_pct": 0.04
}'
```

## Supported Strategies

### SMA (Simple Moving Average) Strategy

**Description**: Classic moving average crossover strategy that generates buy signals when the short-term MA crosses above the long-term MA.

**Parameters:**
- `short_window` (int): Period for short moving average (default: 20)
- `long_window` (int): Period for long moving average (default: 50)
- `position_size` (float): Fraction of portfolio per trade (0.0-1.0, default: 0.01)
- `stop_loss_pct` (float, optional): Stop loss percentage (e.g., 0.02 for 2%)
- `take_profit_pct` (float, optional): Take profit percentage (e.g., 0.04 for 4%)

**Example Configuration:**
```json
{
  "short_window": 10,
  "long_window": 20,
  "position_size": 0.1,
  "stop_loss_pct": 0.02,
  "take_profit_pct": 0.04
}
```

**Strategy Logic:**
1. Calculate short and long moving averages
2. Generate buy signal when short MA crosses above long MA
3. Generate sell signal when short MA crosses below long MA
4. Apply stop loss and take profit if configured

## Risk Management Features

### Stop Loss and Take Profit

The framework supports strategy-level stop loss and take profit configuration:

**Stop Loss**: Automatically exits positions when losses exceed the specified percentage.
**Take Profit**: Automatically exits positions when profits reach the specified percentage.

**Configuration Example:**
```bash
STRATEGY_PARAMS='{"stop_loss_pct": 0.02, "take_profit_pct": 0.04}' \
uv run python scripts/run_backtest.py --strategy sma --data-file data.csv --symbol BTC_USDT
```

### Position Sizing

Control risk through position sizing:

- `position_size`: Fraction of portfolio allocated per trade
- Recommended range: 0.01 (1%) to 0.1 (10%) for most strategies
- Higher position sizes increase both potential returns and risk

### Commission and Slippage

Realistic trading costs are incorporated:

- **Commission**: Configurable percentage fee per trade (default: 0.1%)
- **Slippage**: Market impact is modeled by the backtesting engine
- **Bid-Ask Spread**: Accounted for in execution prices

## Backtesting Engine

### FractionalBacktest (Default)

The framework uses `FractionalBacktest` by default for several advantages:

**Benefits:**
- **High-Precision Trading**: Supports fractional shares for high-priced assets
- **Accurate Position Sizing**: No "insufficient margin" issues with Bitcoin/expensive stocks
- **Realistic Execution**: Better modeling of actual trading conditions
- **Scalable Capital**: Works with any initial capital amount

**When to Use Standard Backtest:**
Use `--use-standard` flag only for:
- Low-priced assets where fractional shares aren't needed
- Compatibility testing with integer share requirements
- Academic research requiring whole-share constraints

### Data Requirements

**Input Format**: CSV files with the following columns:
- `timestamp`: DateTime index
- `open`, `high`, `low`, `close`: Price data (lowercase column names)
- `volume`: Trading volume

**Data Quality**: Ensure data is preprocessed and validated:
- No gaps in timestamps
- Valid OHLC relationships
- Non-negative volumes
- Chronological ordering

## Performance Metrics

The backtesting system generates comprehensive performance statistics:

### Core Metrics

1. **Return Metrics**:
   - Total Return: Overall percentage gain/loss
   - Annualized Return: Yearly return rate
   - CAGR: Compound Annual Growth Rate
   - Buy & Hold Return: Passive investment comparison

2. **Risk Metrics**:
   - Volatility: Annual price volatility
   - Maximum Drawdown: Largest peak-to-trough decline
   - Sharpe Ratio: Risk-adjusted return measure
   - Sortino Ratio: Downside-focused risk metric

3. **Trade Metrics**:
   - Total Trades: Number of completed trade pairs
   - Win Rate: Percentage of profitable trades
   - Profit Factor: Ratio of gross profit to gross loss
   - Average Trade Duration: Mean time in positions

### Advanced Metrics

4. **Statistical Measures**:
   - Alpha: Excess return vs benchmark
   - Beta: Correlation with market movements
   - Kelly Criterion: Optimal position sizing
   - SQN (System Quality Number): Strategy quality metric

5. **Drawdown Analysis**:
   - Maximum Drawdown Duration: Longest losing period
   - Average Drawdown: Mean drawdown depth
   - Recovery Time: Time to recover from drawdowns

## Output and Results

### Result Files

Backtests generate multiple output files in `output/backtests/`:

1. **JSON Results**: `{strategy}_{timestamp}_results.json`
   - Complete performance metrics
   - Trade-by-trade details
   - Configuration parameters

2. **Interactive Plot**: `{strategy}_{timestamp}_plot.html`
   - Equity curve visualization
   - Buy/sell markers
   - Interactive candlestick charts
   - Performance overlay

### Example Results Output

```
============================================================
BACKTEST RESULTS - SMA Strategy
============================================================
Start                     2024-01-01 00:00:00
End                       2024-12-31 00:00:00
Duration                      365 days 00:00:00
Exposure Time [%]                    45.2
Equity Final [$]                  12,350.00
Equity Peak [$]                   13,200.00
Return [%]                           23.5
Buy & Hold Return [%]                15.2
Max. Drawdown [%]                    -8.4
Sharpe Ratio                         1.65
Win Rate [%]                        58.3
Total Trades                           42
Profit Factor                        1.87
```

### Interpreting Results

**Good Strategy Indicators:**
- Positive total return
- Sharpe ratio > 1.0
- Win rate > 50%
- Profit factor > 1.5
- Maximum drawdown < 20%
- Outperforming buy & hold

**Warning Signs:**
- Excessive drawdowns (>25%)
- Low win rate with few big winners (risky)
- Sharpe ratio < 0.5
- Too few trades for statistical significance
- High turnover with low returns

## Strategy Development

### Creating Custom Strategies

Extend the `BaseStrategy` class to create new strategies:

```python
from framework.strategies.base_strategy import BaseStrategy
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1=10, param2=20, position_size=0.01,
                 stop_loss_pct=None, take_profit_pct=None):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['position_size'] = self.position_size
        signals['stop_loss'] = None
        signals['take_profit'] = None
        
        # Your strategy logic here
        # ...
        
        # Add stop loss and take profit if configured
        if self.stop_loss_pct is not None:
            buy_mask = signals['signal'] == 1
            signals.loc[buy_mask, 'stop_loss'] = data.loc[buy_mask, 'close'] * (1 - self.stop_loss_pct)
        
        if self.take_profit_pct is not None:
            buy_mask = signals['signal'] == 1
            signals.loc[buy_mask, 'take_profit'] = data.loc[buy_mask, 'close'] * (1 + self.take_profit_pct)
        
        return signals
    
    def get_description(self) -> str:
        return f"Custom Strategy with params: {self.param1}, {self.param2}"
```

### Strategy Integration

Add new strategies to the backtest script:

```python
# In run_backtest.py, update the strategy mapping:
framework_strategies = {
    'sma': SMAStrategy,
    'custom': MyCustomStrategy,  # Add your strategy here
}
```

## Best Practices

### Data Preparation

1. **Use Clean Data**: Always preprocess data before backtesting
2. **Sufficient History**: Ensure enough data for strategy warmup
3. **Realistic Timeframes**: Match data frequency to strategy needs
4. **Market Hours**: Consider trading sessions and gaps

### Backtest Design

1. **Walk-Forward Analysis**: Test strategies on rolling windows
2. **Out-of-Sample Testing**: Reserve data for final validation
3. **Parameter Sensitivity**: Test robustness across parameter ranges
4. **Multiple Markets**: Validate across different assets/timeframes

### Risk Management

1. **Position Sizing**: Start with small positions (1-5%)
2. **Stop Losses**: Always define maximum acceptable loss
3. **Diversification**: Test strategies across multiple assets
4. **Correlation**: Avoid highly correlated strategies

### Performance Evaluation

1. **Statistical Significance**: Ensure sufficient trades for confidence
2. **Risk-Adjusted Returns**: Focus on Sharpe ratio, not just returns
3. **Drawdown Analysis**: Understand worst-case scenarios
4. **Benchmark Comparison**: Compare against buy-and-hold
5. **Transaction Costs**: Include realistic fees and slippage

## Common Pitfalls

### Overfitting

**Problem**: Strategy works perfectly on historical data but fails in live trading.
**Solutions**:
- Use out-of-sample testing
- Avoid excessive parameter optimization
- Test on multiple timeframes and markets
- Keep strategies simple and logical

### Look-Ahead Bias

**Problem**: Using future information in historical decisions.
**Solutions**:
- Ensure indicators only use past data
- Be careful with preprocessing steps
- Validate signal generation logic
- Use proper data alignment

### Survivorship Bias

**Problem**: Testing only on successful assets that "survived."
**Solutions**:
- Include delisted/failed assets in universe
- Test across broad asset classes
- Be aware of historical context

### Data Snooping

**Problem**: Finding patterns that don't generalize due to extensive testing.
**Solutions**:
- Limit hypothesis testing
- Use proper statistical corrections
- Maintain discipline in testing protocols
- Document all tested approaches

## Optimization and Parameter Tuning

### Parameter Optimization

Use systematic approaches to find optimal parameters:

```bash
# Test multiple parameter combinations
for short in 5 10 15 20; do
  for long in 25 30 35 40 50; do
    STRATEGY_PARAMS='{"short_window":'$short',"long_window":'$long',"position_size":0.1}' \
    uv run python scripts/run_backtest.py --strategy sma --data-file data.csv --symbol BTC_USDT
  done
done
```

### Walk-Forward Analysis

Implement rolling window validation:

```bash
# Test strategy across different time periods
for year in 2020 2021 2022 2023; do
  start_date="${year}-01-01"
  end_date="${year}-12-31"
  uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data.csv \
    --symbol BTC_USDT \
    --start $start_date \
    --end $end_date
done
```

## Integration with Paper Trading

Backtested strategies can be deployed in paper trading:

```bash
# 1. Validate strategy in backtest
uv run python scripts/run_backtest.py --strategy sma --data-file data.csv --symbol BTC_USDT

# 2. Deploy to paper trading (if implemented)
python scripts/run_paper_trading.py --strategy sma --symbol BTC_USDT --capital 10000
```

## Troubleshooting

### Common Issues

**No Trades Generated:**
- Check strategy parameters (MA periods too long?)
- Verify data has sufficient history
- Ensure signal generation logic is correct

**Poor Performance:**
- Review strategy logic and assumptions
- Check for overfitting on specific periods
- Validate data quality and preprocessing
- Consider transaction costs impact

**Memory Issues:**
- Use smaller date ranges for testing
- Process data in chunks for large datasets
- Close unnecessary processes during backtesting

**Inconsistent Results:**
- Verify data preprocessing is deterministic
- Check for floating-point precision issues
- Ensure consistent parameter passing

### Performance Optimization

1. **Data Loading**: Cache preprocessed data files
2. **Strategy Logic**: Vectorize operations where possible
3. **Parallel Processing**: Run multiple backtests simultaneously
4. **Memory Management**: Use appropriate data types and cleanup

The backtesting framework provides a robust foundation for strategy development and validation, enabling you to build confidence in your trading approaches before risking real capital.