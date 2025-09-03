# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Python trading framework for algorithmic trading strategy development, backtesting, and analysis. The framework provides a modular architecture for building, testing, and evaluating trading strategies with built-in risk management, data handling, and visualization capabilities.

## Architecture

### Important Development Notes

**Pandas Resampling:**
- Use modern pandas resampling frequency strings: `'15min'`, `'4h'`, `'1D'` instead of deprecated `'15T'`, `'4H'`, `'1D'`
- Avoid deprecated frequency aliases to prevent FutureWarnings
- Standard frequencies: `'1min'`, `'5min'`, `'15min'`, `'30min'`, `'1h'`, `'4h'`, `'1D'`, `'1W'`

### Core Components

**Framework Structure:**
- `framework/strategies/` - Trading strategy implementations
  - `base_strategy.py` - Abstract base class all strategies must inherit from
  - `sma_strategy.py` - Simple Moving Average crossover strategy implementation
  - `fvg_strategy.py` - Fair Value Gap multi-timeframe strategy implementation
  - `breakout_strategy.py` - High/Low Breakout trend-following strategy with ATR-based stops
  - `detectors/` - Technical analysis pattern detectors (pivot points, FVG detection, etc.)
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

**Basic Examples:**
```bash
# Basic backtest with SMA strategy
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

# Basic backtest with FVG strategy (requires M15 data)
uv run python scripts/run_backtest.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

# Basic backtest with Breakout strategy
uv run python scripts/run_backtest.py --strategy breakout --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

# Backtest with custom strategy parameters (SMA)
STRATEGY_PARAMS='{"short_window": 10, "long_window": 20, "stop_loss_pct": 0.02, "take_profit_pct": 0.04}' uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

# Backtest with custom FVG strategy parameters
STRATEGY_PARAMS='{"h1_lookback_candles": 24, "risk_reward_ratio": 3.0, "max_hold_hours": 4, "position_size": 0.05}' uv run python scripts/run_backtest.py --strategy fvg --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

# Backtest with custom Breakout strategy parameters
STRATEGY_PARAMS='{"entry_lookback": 30, "exit_lookback": 15, "atr_multiplier": 3.0, "use_trend_filter": true, "use_volume_filter": false}' uv run python scripts/run_backtest.py --strategy breakout --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT

# Backtest with date range
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --start 2024-01-01 --end 2024-03-31
```

**Risk Management Examples:**
```bash
# Fixed Risk Manager - Risk 1% of account on each trade (RECOMMENDED)
RISK_PARAMS='{"risk_percent": 0.01, "default_stop_distance": 0.02}' uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --risk-manager fixed_risk

# Fixed Risk Manager - Conservative 0.5% risk per trade
RISK_PARAMS='{"risk_percent": 0.005, "default_stop_distance": 0.02}' uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --risk-manager fixed_risk

# Fixed Position Size Manager - Use fixed 10% of equity (default)
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --risk-manager fixed_position

# Fixed Position Size Manager - Small position for high leverage (0.1% of equity)
RISK_PARAMS='{"position_size": 0.001}' uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --risk-manager fixed_position
```

**Advanced Options:**
```bash
# High initial capital for expensive assets
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --initial-capital 100000

# Custom commission rate
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --commission 0.0005

# Debug logging for troubleshooting
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --debug

# Use standard backtest instead of fractional (not recommended for crypto)
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv --symbol BTC_USDT --use-standard
```

**Note on Leverage:**
The backtest uses 100x leverage by default (margin=0.01). When using high leverage, adjust position sizes accordingly:
- With FixedRiskManager: The manager automatically calculates safe position sizes based on stop loss distance
- With FixedPositionSizeManager: Use smaller position sizes (e.g., 0.001 = 0.1% instead of 0.1 = 10%)
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

## Development Guidelines

### Documentation
- All new documentation files must be created in the `docs/` folder only
- Do not create duplicate documentation in the root directory
- Keep documentation organized and up-to-date

### Script Development
- Do not create new scripts for every feature
- Extend existing scripts or framework modules instead
- Keep the framework simple and uncluttered while maintaining full functionality
- Minimize the number of scripts to maintain clean project structure

### Code Organization
- Prefer extending existing modules over creating new ones
- Keep the codebase streamlined and professional
- All heavy logic should be in framework modules, scripts should be simple wrappers

### Framework Rules
- All new documentation files must be created in the `docs/` folder only
- Do not create new scripts for every feature - extend existing ones
- Keep the framework simple and uncluttered while maintaining full functionality
- Minimize script count to maintain clean project structure

### Documentation Rules
- All documentation files (except README.md and CLAUDE.md) must be created in the `docs/` folder only
- Use UPPERCASE naming convention for docs files (e.g., STRATEGY_NAME.md, not strategy_name.md)  
- Follow the documentation style and format used by existing docs in the project
- Do not create multiple documentation files for the same feature/strategy
- Only create documentation when explicitly requested by the user