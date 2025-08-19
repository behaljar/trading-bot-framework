# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Python trading framework for algorithmic trading strategy development, backtesting, and analysis. The framework provides a modular architecture for building, testing, and evaluating trading strategies with built-in risk management, data handling, and visualization capabilities.

## Architecture

### Core Components

**Framework Structure:**
- `framework/strategies/` - Trading strategy implementations
  - `base_strategy.py` - Abstract base class all strategies must inherit from
  - `sma_strategy.py` - Simple Moving Average crossover strategy implementation
  - `detectors/` - Technical analysis pattern detectors (pivot points, etc.)
  - `utils/` - Strategy utilities (candlestick patterns, volume profile analysis)

- `framework/backtesting/` - Backtesting engine and results
  - `backtest_engine.py` - Core backtesting engine with risk management integration
  - `backtest_result.py` - Results container with performance metrics
  - `trade.py` - Trade execution and tracking

- `framework/data/` - Data management and preprocessing
  - `downloaders/` - Data source integrations (CCXT, Yahoo Finance)
  - `preprocessors/` - Data cleaning and preparation utilities

- `framework/risk/` - Risk management system
  - `position_calculator.py` - Position sizing and risk calculations

- `framework/utils/` - Common utilities
  - `logger.py` - Structured JSON logging with trading-specific features

**Scripts:**
- `scripts/run_backtest.py` - Universal backtest runner for any strategy
- `scripts/download_data.py` - Data fetching utilities
- `scripts/preprocess_data.py` - Data preprocessing pipeline

**Data Organization:**
- `data/csv/` - Raw CSV data files
- `data/processed/` - Preprocessed data ready for backtesting
- `output/backtests/` - Backtest results and reports
- `logs/` - Application logs with JSON structured format

## Common Development Commands

### Running Backtests
```bash
# Basic backtest with SMA strategy
python scripts/run_backtest.py --strategy sma --data-file data/csv/BTCUSDT.csv --symbol BTC_USDT

# Backtest with date range and debug logging
python scripts/run_backtest.py --strategy sma --data-file data/csv/BTCUSDT.csv --symbol BTC_USDT --start 2023-01-01 --end 2023-12-31 --debug

# Backtest with custom strategy parameters via environment variables
STRATEGY_PARAMS='{"short_window": 5, "long_window": 20, "position_size": 0.5}' python scripts/run_backtest.py --strategy sma --data-file data/csv/BTCUSDT.csv --symbol BTC_USDT
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/strategies/test_pivot_detector.py

# Run tests with coverage
python -m pytest tests/ --cov=framework

# Run tests with visualization output
python tests/strategies/test_pivot_detector.py
```

### Strategy Development

**Strategy Implementation Pattern:**
1. Inherit from `BaseStrategy` in `framework/strategies/base_strategy.py`
2. Implement required methods: `generate_signals()` and `get_description()`
3. Use `validate_data()` to ensure input data integrity
4. Return DataFrame with `signal` column (1=buy, -1=sell, 0=hold) and `position_size` column (0.0-1.0)

**Key Strategy Requirements:**
- Input data must have columns: `['open', 'high', 'low', 'close', 'volume']`
- Index must be DatetimeIndex
- Signal generation must handle insufficient data periods gracefully
- Position sizing should respect risk management constraints

### Risk Management Integration

The framework supports risk management through the `risk_manager` parameter in strategies:
- Strategies can be enhanced with position sizing, stop-loss, and risk metrics
- Risk managers integrate with the backtesting engine for dynamic position management
- Risk decisions can override strategy signals based on portfolio constraints

### Data Management

**Supported Data Sources:**
- CCXT for cryptocurrency exchange data
- Yahoo Finance for traditional financial instruments
- Direct CSV file loading with standard OHLCV format

**Data Format Requirements:**
- CSV files with DatetimeIndex
- Standard OHLCV columns: open, high, low, close, volume
- Data must be sorted chronologically
- No missing values in price data

### Logging and Debugging

The framework uses structured JSON logging:
- Logs are saved to `logs/trading_bot_YYYYMMDD.log`
- Debug level can be controlled via `--debug` flag or logger setup
- Trading-specific log fields include portfolio values, trade execution details, and risk management decisions

### Performance Analysis

Backtest results include comprehensive metrics:
- Total return and percentage return
- Maximum drawdown analysis  
- Sharpe ratio calculation
- Win rate based on paired buy/sell trades
- Portfolio value tracking over time
- Detailed trade history with timestamps and values

## Testing Strategy

The test suite includes:
- Unit tests for individual strategy components
- Integration tests with real market data
- Visualization tests that generate candlestick charts with technical indicators
- Performance metric validation
- Data integrity checks

Test outputs including charts are saved to `output/tests/` directory.