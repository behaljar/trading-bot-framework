# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development
- `make dev` - Start bot in development mode (paper trading)
- `make test` - Run tests with coverage (`python -m pytest tests/ -v --cov=trading_bot`)
- `make install` - Install dependencies (`pip install -r requirements.txt`)
- `make clean` - Clean up Python cache files and test artifacts

### Trading Operations
- `make backtest` - Run backtest with SMA strategy on AAPL
- `make paper` - Start paper trading (single run)
- `make live` - Start live trading (scheduled mode)

### Development Tools
- `make dev-dashboard` - Start Streamlit dashboard on port 8501
- `make dev-notebook` - Start Jupyter Lab on port 8888
- `make dev-all` - Start bot + dashboard together

### Monitoring
- `make monitor` - Start performance monitoring
- `make logs` - Show recent logs (tails current day's log file)

### Testing Individual Components
```bash
# Test specific strategy
python -m pytest tests/test_strategies.py::TestSMAStrategy

# Run backtest with custom parameters
python scripts/run_backtest.py --strategy sma --symbol AAPL --start 2022-01-01 --end 2023-12-31
```

### Data Preprocessing
```bash
# Preprocess single ticker
python scripts/preprocess_data.py --ticker AAPL

# Preprocess all CSV files
python scripts/preprocess_data.py --all

# Generate sample data for testing
python scripts/generate_sample_data.py
```

## Architecture Overview

### Configuration System
- Environment-based configuration via `.env` file and environment variables
- `config/settings.py` - Main configuration class with environment variable parsing
- Supports multiple data sources: `yahoo`, `ccxt`, `csv`
- Strategy parameters configurable via `STRATEGY_*` environment variables

### Data Layer Architecture
- **Base**: `data/base_data_source.py` - Abstract interface
- **Yahoo Finance**: `data/yahoo_finance.py` - Stock market data
- **CCXT**: `data/ccxt_source.py` - Cryptocurrency exchange data with sandbox support
- **CSV**: `data/csv_source.py` - Custom data with pre-calculated indicators support

### Strategy Framework
- **Base Class**: `strategies/base_strategy.py` - Abstract strategy interface with Signal enum
- **Implementation**: Strategies must implement `generate_signals()` and `get_strategy_name()`
- **Features**: Supports custom indicators from CSV, position sizing, signal history
- **Built-in**: SMA crossover (`trend_following.py`), RSI mean reversion

### Key Components
- **Risk Management**: `risk/risk_manager.py` - Position sizing and risk controls
- **Execution**: `execution/paper_trader.py` - Paper trading simulation
- **Monitoring**: `monitoring/alert_system.py` - Alert and notification system
- **Logging**: `utils/logger.py` - Centralized logging with date-based log files
- **Data Preprocessing**: `scripts/preprocess_data.py` - Feature engineering pipeline with 50+ technical indicators

### Data Flow
1. Configuration loaded from environment variables
2. Data source initialized based on `DATA_SOURCE` setting
3. Strategy selected based on `STRATEGY_NAME` (defaults to SMA if unknown)
4. Main loop processes each symbol: data → signals → risk management → execution
5. Performance summary logged at end

### CSV Data Source Features
- Supports pre-calculated indicators beyond OHLCV
- Automatic column recognition for dates, prices, volume
- Feature columns preserved and accessible in strategies via `get_feature_columns()`
- File naming convention: `{SYMBOL}.csv` in `CSV_DATA_DIRECTORY`

### Data Preprocessing Pipeline
- **Raw Data**: `data/csv/` - Original OHLCV CSV files
- **Processed Data**: `data/processed/` - Feature-rich datasets with 50+ indicators
- **Student Section**: Dedicated area in preprocessing script for custom feature engineering
- **Features**: Basic price features, moving averages, technical indicators, volume indicators, custom features, ML labels
- **Usage**: `python scripts/preprocess_data.py --ticker SYMBOL` or `--all` for batch processing

### Environment Configuration Patterns
- Uses inline comment stripping for .env values
- Default symbols vary by data source (stocks vs crypto)
- Sandbox mode support for CCXT exchanges
- Strategy parameters: `STRATEGY_SHORT_WINDOW`, `STRATEGY_LONG_WINDOW`, etc.

## Testing Framework
- Uses pytest with coverage reporting
- Strategy-specific test classes in `tests/test_strategies.py`
- Coverage target appears to be focused on `trading_bot` module
- Supports testing individual strategies and components

## Deployment Modes
- **Development**: Paper trading with debug logging
- **Production**: Live trading mode (requires API credentials)
- **Backtest**: Historical analysis via `scripts/run_backtest.py`