# Backtesting Guide

This guide covers how to run backtests using the trading framework with the backtesting.py library.

## Quick Start

```bash
# Basic backtest with SMA strategy
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT
```

## Basic Examples

### Simple Backtest
```bash
# Run backtest with default settings (uses FixedRiskManager with 1% risk by default)
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT
```

### Custom Strategy Parameters
```bash
# Modify strategy parameters (MA periods, stop loss, take profit)
STRATEGY_PARAMS='{"short_window": 10, "long_window": 20, "stop_loss_pct": 0.02, "take_profit_pct": 0.04}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT
```

### Date Range Filtering
```bash
# Test specific time period
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --start 2024-01-01 \
    --end 2024-03-31
```

## Risk Management

The framework supports two risk management approaches:

### 1. Fixed Risk Manager (Default & Recommended)

Risk a fixed percentage of your account on each trade. The position size is automatically calculated based on stop loss distance.

```bash
# Risk 1% of account on each trade (DEFAULT)
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_risk

# Custom risk parameters - 1% risk with 2% default stop
RISK_PARAMS='{"risk_percent": 0.01, "default_stop_distance": 0.02}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_risk

# Conservative approach - Risk only 0.5% per trade
RISK_PARAMS='{"risk_percent": 0.005, "default_stop_distance": 0.02}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_risk

# Aggressive approach - Risk 2% per trade (use with caution)
RISK_PARAMS='{"risk_percent": 0.02, "default_stop_distance": 0.02}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_risk
```

#### How Fixed Risk Works:
- If you risk 1% with a 2% stop loss → position size = 50% of equity
- If you risk 1% with a 0.5% stop loss → position size = 200% of equity (uses leverage)
- If you risk 1% with a 5% stop loss → position size = 20% of equity
- Maximum loss per trade is always limited to your risk percentage

### 2. Fixed Position Size Manager

Use a fixed percentage of equity for each trade, regardless of stop loss distance.

```bash
# Default - Use 10% of equity per trade
RISK_PARAMS='{"position_size": 0.1}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_position

# Conservative - Use 5% of equity per trade
RISK_PARAMS='{"position_size": 0.05}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_position

# For high leverage - Use only 0.1% of equity
RISK_PARAMS='{"position_size": 0.001}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_position
```

## Advanced Options

### Initial Capital
```bash
# Increase initial capital for expensive assets like BTC
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --initial-capital 100000
```

### Commission Settings
```bash
# Set custom commission rate (0.05% maker fee)
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --commission 0.0005
```

### Debug Mode
```bash
# Enable detailed logging for troubleshooting
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --debug
```

### Backtest Engine Type
```bash
# Use standard backtest instead of fractional (not recommended for crypto)
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --use-standard
```

## Leverage and Margin Trading

The backtest engine uses **100x leverage by default** (margin=0.01). This allows trading positions larger than your capital, which is common in crypto derivatives trading.

### Important Considerations:

1. **With FixedRiskManager (Default)**: The leverage is handled automatically. The manager calculates safe position sizes based on your risk percentage and stop loss distance. You don't need to worry about over-leveraging.

2. **With FixedPositionSizeManager**: You must manually adjust position sizes for high leverage:
   - No leverage (1x): `position_size: 0.1` (10% of equity)
   - 10x leverage: `position_size: 0.01` (1% of equity)
   - 100x leverage: `position_size: 0.001` (0.1% of equity)

### Why Use Leverage?
- Allows trading high-priced assets (like BTC) with smaller capital
- Enables proper position sizing with tight stop losses
- Required for short selling in spot markets
- Common in crypto derivatives (futures, perpetuals)

### Risk Warning
High leverage amplifies both gains and losses. With 100x leverage:
- A 1% market move in your favor = 100% gain
- A 1% market move against you = 100% loss (liquidation)
Always use proper risk management!

## Output Files

Each backtest generates three output files in `output/backtests/`:

### 1. Results JSON (`{strategy}_{symbol}_{timestamp}_results.json`)
Contains complete backtest statistics:
- Total Return, Sharpe Ratio, Maximum Drawdown
- Win Rate, Number of Trades, Average Trade Duration
- All performance metrics from backtesting.py

### 2. Trades CSV (`{strategy}_{symbol}_{timestamp}_trades.csv`)
Detailed record of every trade:
- EntryTime, ExitTime (formatted as YYYY-MM-DD HH:MM:SS)
- EntryPrice, ExitPrice (4 decimal places)
- Size (position size in units or fraction)
- PnL (profit/loss in currency, 2 decimal places)
- ReturnPct (return percentage)
- DurationHours (how long position was held)

### 3. Interactive Plot (`{strategy}_{symbol}_{timestamp}_plot.html`)
Visual analysis tool with:
- Price chart with buy/sell signals
- Equity curve showing portfolio value over time
- Drawdown periods
- Volume indicators

## Complete Example

Here's a full example combining multiple features:

```bash
# Run a conservative backtest with fixed risk management
STRATEGY_PARAMS='{"short_window": 20, "long_window": 50, "stop_loss_pct": 0.015, "take_profit_pct": 0.03}' \
RISK_PARAMS='{"risk_percent": 0.01, "default_stop_distance": 0.015}' \
uv run python scripts/run_backtest.py \
    --strategy sma \
    --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
    --symbol BTC_USDT \
    --risk-manager fixed_risk \
    --initial-capital 50000 \
    --commission 0.001 \
    --start 2024-01-01 \
    --end 2024-06-30 \
    --debug
```

This example:
- Uses SMA strategy with 20/50 period moving averages
- Risks 1% per trade with 1.5% stop loss and 3% take profit
- Starts with $50,000 capital
- Uses 0.06% commission rate (realistic for crypto exchanges)
- Tests first half of 2024
- Enables debug logging for detailed analysis

## Understanding Results

### Key Metrics to Watch

**Return Metrics:**
- `Return [%]`: Total percentage gain/loss
- `Annualized Return [%]`: Yearly return if strategy continues performing similarly
- `Buy & Hold Return [%]`: Comparison with just holding the asset

**Risk Metrics:**
- `Max. Drawdown [%]`: Largest peak-to-trough decline (smaller is better)
- `Sharpe Ratio`: Risk-adjusted return (higher is better, >1.0 is good, >2.0 is excellent)
- `Sortino Ratio`: Similar to Sharpe but focuses on downside risk

**Trade Metrics:**
- `Win Rate [%]`: Percentage of profitable trades (>50% is good but not required)
- `# Trades`: Total number of completed trades
- `Avg. Trade [%]`: Average return per trade
- `Profit Factor`: Gross profit / Gross loss (>1.5 is good)

### What Makes a Good Strategy?

✅ **Good Signs:**
- Positive return with reasonable drawdown (<20%)
- Sharpe Ratio > 1.0
- Consistent profits across different time periods
- Win rate combined with good risk/reward ratio
- Profit Factor > 1.5

⚠️ **Warning Signs:**
- Maximum drawdown > 30%
- Sharpe Ratio < 0.5
- Very few trades (< 20) - not statistically significant
- All profits from one or two lucky trades
- Performance degrades in recent periods

## Tips for Better Results

### 1. Risk Management is Key
- Always use either FixedRiskManager or appropriate position sizing
- Never risk more than 2% per trade (1% is safer)
- Consider correlation between trades

### 2. Test Robustness
- Run backtests on different time periods
- Try various parameter combinations
- Test on multiple assets/symbols
- Use walk-forward analysis

### 3. Be Realistic
- Include realistic commission rates (0.05-0.1% for crypto)
- Account for slippage in volatile markets
- Consider funding rates for perpetuals
- Remember that past performance doesn't guarantee future results

### 4. Avoid Common Pitfalls
- **Overfitting**: Don't optimize parameters too much on historical data
- **Survivorship Bias**: Test on various assets, not just successful ones
- **Look-Ahead Bias**: Ensure strategies only use past data
- **Small Sample Size**: Need enough trades for statistical significance

### 5. Progressive Development
1. Start with simple strategies (like SMA)
2. Test with conservative risk settings
3. Gradually increase complexity
4. Always validate on out-of-sample data
5. Paper trade before going live

## Available Strategies

### Currently Implemented:
- `sma` - Simple Moving Average Crossover
  - Generates long/short signals based on MA crossovers
  - Configurable periods, stop loss, and take profit
  - Suitable for trending markets

### Adding New Strategies:
New strategies can be added by:
1. Creating a new class inheriting from `BaseStrategy` in `framework/strategies/`
2. Implementing `generate_signals()` method
3. Adding the strategy to the mapping in `scripts/run_backtest.py`

## Troubleshooting

### No Trades Generated
- Check if MA periods are too long for your data
- Verify data has sufficient history (at least long_window periods)
- Try shorter MA periods or different parameters

### Poor Performance
- This is normal - most strategies need optimization
- Try different parameter combinations
- Test on different time periods
- Consider market conditions (trending vs ranging)

### "Insufficient Margin" Warnings
- Reduce position size in risk manager
- Increase initial capital
- Make sure you're using FractionalBacktest (default)

### Memory Issues with Large Datasets
- Use date filtering (--start and --end)
- Process data in chunks
- Close other applications during backtesting

## Next Steps

1. **Optimize Parameters**: Use grid search or optimization algorithms
2. **Develop New Strategies**: Implement RSI, MACD, Bollinger Bands, etc.
3. **Combine Strategies**: Create portfolio of uncorrelated strategies
4. **Forward Testing**: Paper trade successful strategies
5. **Live Trading**: Deploy thoroughly tested strategies with small capital

Remember: Successful backtesting doesn't guarantee future profits. Always start with small positions and monitor performance carefully.