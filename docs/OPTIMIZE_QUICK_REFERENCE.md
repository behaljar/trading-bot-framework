# Optimization Quick Reference

## Basic Commands

```bash
# Simple optimization
python scripts/optimize.py --strategy sma --symbol AAPL

# Custom parameters
python scripts/optimize.py --strategy sma --symbol AAPL \
  --short-window "10,30" --long-window "40,80" \
  --max-tries 100

# Different data source
python scripts/optimize.py --strategy sma --symbol BTCUSDC \
  --data-source ccxt --timeframe 1h

# Use processed CSV data
python scripts/optimize.py --strategy sma --symbol ETHUSDC \
  --data-source csv_processed --timeframe 15m

# Optimize for Sharpe Ratio
python scripts/optimize.py --strategy sma --symbol AAPL \
  --metric "Sharpe Ratio"
```

## Parameter Formats

| Format | Example | Description |
|--------|---------|-------------|
| Range | `"10,50"` | Tests values from 10 to 50 |
| List | `"10,20,30,40,50"` | Tests specific values only |

## Output Files

| File | Description |
|------|-------------|
| `*_results.json` | Best parameters and metrics |
| `*_heatmap_data.csv` | Raw optimization data |
| `*_heatmap_interactive.html` | Interactive Bokeh heatmap |
| `*_heatmap_*_*.png` | Static parameter pair heatmaps |
| `*_heatmap_summary.png` | Correlation analysis |

## SMA Strategy Parameters

| Parameter | Flag | Default Range | Description |
|-----------|------|---------------|-------------|
| Short Window | `--short-window` | 5-50 | Fast moving average period |
| Long Window | `--long-window` | 20-100 | Slow moving average period |
| Stop Loss | `--stop-loss` | 1-5% | Stop loss percentage |
| Take Profit | `--take-profit` | 2-10% | Take profit percentage |

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--start` | Start date | `--start 2024-01-01` |
| `--end` | End date | `--end 2024-12-31` |
| `--data-source` | Data source | `--data-source yahoo` |
| `--timeframe` | Timeframe | `--timeframe 1h` |
| `--metric` | Optimization metric | `--metric "Sharpe Ratio"` |
| `--minimize` | Minimize instead of maximize | `--minimize` |
| `--max-tries` | Max optimization attempts | `--max-tries 200` |

## Optimization Metrics

| Metric | Good For |
|--------|----------|
| `"Return [%]"` | Absolute performance |
| `"Sharpe Ratio"` | Risk-adjusted returns |
| `"Calmar Ratio"` | Drawdown-adjusted returns |
| `"Max. Drawdown [%]"` | Risk management (use `--minimize`) |
| `"Win Rate [%]"` | Trade consistency |

## Data Sources

| Source | Symbols | Example |
|--------|---------|---------|
| `yahoo` | Stocks, ETFs, Indices | `AAPL`, `SPY`, `^GSPC` |
| `ccxt` | Cryptocurrencies | `BTCUSDC`, `ETHUSDT` |
| `csv` | Custom raw data | Any symbol with CSV file |
| `csv_processed` | Preprocessed data with features | `ETHUSDC` (from `data/processed/`) |

## Timeframes

| Format | Description |
|--------|-------------|
| `1m`, `5m`, `15m`, `30m` | Minutes |
| `1h`, `2h`, `4h`, `6h`, `12h` | Hours |
| `1d` | Daily |
| `1w` | Weekly |

## Workflow Example

```bash
# 1. Quick test
python scripts/optimize.py --strategy sma --symbol AAPL \
  --start 2024-06-01 --end 2024-08-01 --max-tries 20

# 2. Full optimization  
python scripts/optimize.py --strategy sma --symbol AAPL \
  --short-window "20,30" --long-window "50,70" \
  --max-tries 100

# 3. Use best parameters in backtest
python scripts/run_backtest.py --strategy sma --symbol AAPL
```