# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development
- `make dev` - Start bot in development mode (sandbox trading)
- `make test` - Run tests with coverage (`python -m pytest tests/ -v --cov=trading_bot`)
- `make install` - Install dependencies (`pip install -r requirements.txt`)
- `make clean` - Clean up Python cache files and test artifacts

### Trading Operations
- `make backtest` - Run backtest with SMA strategy on AAPL
- `make paper` - Start sandbox trading with CCXT (single run)
- `make live` - Start live trading (real money - be careful!)

### Order Testing
- `make test-orders` - Test order execution with simple strategy
- `make test-positions` - Test position synchronization
- Uses testnet/sandbox mode for safe order testing

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
- **Live Trading**: CCXT and IBKR data sources supported (`DATA_SOURCE=ccxt` or `DATA_SOURCE=ibkr`)
- Strategy parameters configurable via `STRATEGY_*` environment variables
- Multiple config templates: `.env.sandbox`, `.env.live`, `.env.test`, `.env.ibkr.live`

### Data Layer Architecture
- **Base**: `data/base_data_source.py` - Abstract interface
- **Yahoo Finance**: `data/yahoo_finance.py` - Stock market data
- **CCXT**: `data/ccxt_source.py` - Cryptocurrency exchange data with sandbox support
- **CSV**: `data/csv_source.py` - Custom data with pre-calculated indicators support
- **IBKR**: `data/ibkr_source.py` - Interactive Brokers data with ib_async integration

### Strategy Framework
- **Base Class**: `strategies/base_strategy.py` - Abstract strategy interface with Signal enum
- **Implementation**: Strategies must implement `generate_signals()` and `get_strategy_name()`
- **Features**: Supports custom indicators from CSV, position sizing, signal history
- **Built-in**: SMA crossover (`trend_following.py`), Z-Score mean reversion (`mean_reversion.py`)

### Key Components
- **Live Execution**: 
  - `execution/ccxt/` - Live trading with CCXT exchanges
    - `ccxt_trader.py` - Main trading engine with error handling and state management
    - `state_persistence.py` - Abstract state storage interface (supports PostgreSQL migration)
    - `file_state_store.py` - File-based state storage implementation
    - `position_sync.py` - Position synchronization with exchange
    - `data_manager.py` - Efficient historical data caching
  - `execution/ibkr/` - Live trading with Interactive Brokers
    - `ibkr_trader.py` - Async IBKR trading engine with comprehensive error handling
    - `ibkr_sync_trader.py` - Synchronous wrapper for main.py integration
    - `ibkr_state_store.py` - IBKR-specific state persistence
    - `ibkr_position_sync.py` - Position synchronization with IBKR account
- **Risk Management**: `risk/risk_manager.py` - Position sizing and risk controls
- **Monitoring**: `monitoring/alert_system.py` - Alert and notification system
- **Logging**: `utils/logger.py` - Centralized logging with date-based log files
- **Data Preprocessing**: `scripts/preprocess_data.py` - Feature engineering pipeline with 50+ technical indicators

### Data Flow
1. Configuration loaded from environment variables
2. Trader initialized based on `DATA_SOURCE` (CCXT or IBKR) with connection and state recovery
3. Strategy selected based on `STRATEGY_NAME` (defaults to SMA if unknown)
4. Main loop for each symbol:
   - Fetch/cache historical data for signal generation
   - Generate trading signals using completed bars
   - Get real-time price for execution
   - Determine action based on position and signal
   - Execute orders with comprehensive error handling
   - Update position tracking and save state
5. Continuous operation with automatic recovery on restart

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

## Trading Modes
- **Sandbox**: Paper trading with real market data (`USE_SANDBOX=true`)
- **Live**: Real money trading (`USE_SANDBOX=false`)
- **Test**: Order execution testing with alternating buy/sell strategy
- **Backtest**: Historical analysis via `scripts/run_backtest.py` (CSV/Yahoo data)

## Configuration Templates
- `.env.example` - Complete configuration reference
- `.env.sandbox` - Safe testing configuration
- `.env.live` - Live trading template (real money)
- `.env.test` - Order testing configuration
- `.env.ibkr.paper` - IBKR paper trading configuration
- `.env.ibkr.live` - IBKR live trading configuration

## IBKR Integration
- **Connection Management**: Professional-grade connection with auto-reconnect via ib_async
- **Data Sources**: Real-time and historical market data from Interactive Brokers
- **Paper/Live Trading**: Seamless switching between paper and live accounts
- **Market Data Types**: Live, delayed, and frozen market data support
- **Account Management**: Multi-account support with proper authentication
- **Error Handling**: Comprehensive IBKR error code mapping and recovery
- **Rate Limiting**: Built-in API rate limiting to respect IBKR limits

### IBKR Commands
```bash
# Test IBKR connection and functionality
make ibkr-test

# Start IBKR paper trading (port 7497)
make ibkr-paper

# Start IBKR live trading (port 7496) - REAL MONEY
make ibkr-live

# Paper trading with IBKR data
make paper-ibkr
```

### IBKR Configuration
- **Paper Trading**: Port 7497 (TWS) or 4002 (Gateway)
- **Live Trading**: Port 7496 (TWS) or 4001 (Gateway)
- **Market Data**: Requires API acknowledgement for off-platform data
- **Authentication**: Manual TWS/Gateway login required (no headless mode)
- **Client IDs**: Unique per connection (0-32)

### IBKR Prerequisites
1. **TWS or IB Gateway**: Must be running and logged in
2. **API Configuration**: Enable API connections in TWS settings
3. **Market Data**: Complete API acknowledgement form
4. **Account Access**: Proper account permissions for trading
5. **Memory**: Increase TWS memory allocation to 4GB+ for stability

## Live Trading Features
- **State Persistence**: Crash-resistant with automatic recovery
- **Position Synchronization**: Handles pre-existing positions
- **Order Management**: Comprehensive error handling and retry logic
- **Safety Features**: Daily loss limits, emergency stops, instance locking
- **Real-time Data**: Smart caching with new bar detection

## Commit Message Guidelines
- NEVER include "Claude", "Claude Code", or AI-related references in commit messages
- Keep commit messages professional and focused on the actual changes made
- Use conventional commit format (feat:, fix:, docs:, etc.)
- NEVER use the auto-generated commit format with "🤖 Generated with [Claude Code]" footer