# Trading Framework

A comprehensive Python framework for algorithmic trading strategy development, backtesting, and analysis.

## Features

- **Strategy Development**: Modular architecture with abstract base classes for consistent strategy implementation
- **Backtesting Engine**: Comprehensive backtesting with risk management integration and performance metrics
- **Data Management**: Support for multiple data sources (Yahoo Finance, CCXT exchanges)
- **Risk Management**: Built-in position sizing, stop-loss, and portfolio risk controls
- **Performance Analysis**: Detailed metrics including Sharpe ratio, maximum drawdown, and win rates

## Quick Start

### Download Data
```bash
# Download stock data
python scripts/download_data.py --source yahoo --symbol AAPL --start 2023-01-01

# Download crypto data
python scripts/download_data.py --source ccxt --symbol BTC/USDT --exchange binance
```

### Run Backtest
```bash
python scripts/run_backtest.py --strategy sma --data-file data/raw/AAPL_yahoo_1d_2023-01-01_2023-12-31.csv --symbol AAPL
```

## Architecture

- `framework/strategies/` - Trading strategy implementations
- `framework/backtesting/` - Backtesting engine and results
- `framework/data/` - Data sources and preprocessing
- `framework/risk/` - Risk management system
- `scripts/` - Utility scripts for data download and backtesting
- `tests/` - Comprehensive test suite

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=framework
```

See `CLAUDE.md` for detailed development guidance.