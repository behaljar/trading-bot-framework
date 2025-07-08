# 🤖 Professional Trading Bot Framework

An automated trading bot framework supporting multiple asset classes including cryptocurrencies, stocks, ETFs, and more. Features comprehensive paper trading, risk management, and data source integration. Under active development with production deployment planned for the future.

## ⚡ Quick Start

### 1. Setup
```bash
# Clone and install
git clone <repository-url>
cd robotdreams-trading-fw
make install
```

### 2. Configure for Testing (Recommended First)
```bash
# Copy sandbox configuration
cp .env.sandbox .env

# Edit .env with your testnet API keys
# Get testnet keys from: https://testnet.binance.vision/
nano .env
```

### 3. Test Order Execution
```bash
# Test with safe alternating buy/sell strategy
make test-orders
```

### 4. Start Paper Trading
```bash
# Local paper trading with virtual portfolio
make paper

# Paper trading with different data sources
make paper-yahoo    # Stock market data (Yahoo Finance)
make paper-ccxt     # Cryptocurrency data
make paper-ibkr     # IBKR stock data (requires TWS/Gateway)

# Or start sandbox trading with exchange testnet
make sandbox
```

### 5. Go Live (After Testing)
```bash
# Copy live configuration
cp .env.live .env

# Edit with production API keys
nano .env

# Start live trading (REAL MONEY!)
make live
```

## 🎯 Features

### 🔒 Safety First
- **Paper Trading**: Local simulation with virtual portfolio and realistic execution
- **Multiple Data Sources**: Yahoo Finance (stocks), CCXT (crypto), IBKR (professional)
- **Sandbox Mode**: Test with real market data on exchange testnets
- **State Persistence**: Crash-resistant with automatic recovery
- **Risk Management**: Position sizing, stop losses, exposure limits
- **Comprehensive Testing**: Extensive test suite for all components

### 📊 Trading Capabilities
- **Multi-Asset Support**: Cryptocurrencies, stocks, ETFs, and more
- **Multiple Strategies**: SMA, RSI, Breakout, and custom strategies
- **Professional Data Sources**: IBKR, CCXT, Yahoo Finance integration
- **Smart Data Management**: Efficient historical data caching and preprocessing
- **Flexible Execution**: Paper trading, sandbox, and live trading modes

### 🛠️ Technical Features
- **Multiple Data Sources**: CCXT (100+ crypto exchanges), IBKR (stocks), Yahoo Finance
- **Professional IBKR Integration**: Native API with TWS/Gateway support
- **Error Recovery**: Automatic retry logic for API failures
- **Comprehensive Logging**: Detailed execution logs with rotation
- **Flexible Configuration**: Environment-based settings

## 📈 Supported Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **SMA** | Simple Moving Average crossover | Trending markets |
| **Mean Reversion** | Z-Score based mean reversion with RSI filter | Range-bound markets |
| **Breakout** | Price breakout with volume confirmation | Volatile markets |
| **Test** | Alternating buy/sell (testing only) | Order execution testing |

## 🔧 Configuration

### Quick Setup
```bash
# For testing
cp .env.sandbox .env

# For live trading  
cp .env.live .env

# View current status
make status
```

### Key Settings
```bash
# Trading mode
USE_SANDBOX=true          # true=safe testing, false=real money

# Exchange settings
EXCHANGE_NAME=binance     # binance, kraken, coinbase, etc.
SYMBOLS=BTC/USDT,ETH/USDT # Comma-separated trading pairs
TIMEFRAME=1h              # 1m, 5m, 15m, 1h, 4h, 1d

# Strategy
STRATEGY_NAME=sma         # sma, mean_reversion, breakout, test

# Risk management
MAX_POSITION_SIZE=0.05    # 5% max position size
STOP_LOSS_PCT=0.02        # 2% stop loss
INITIAL_CAPITAL=1000      # Your trading capital
```

See [Configuration Guide](docs/CONFIGURATION.md) for complete options.

## 📊 Paper Trading

### Local Paper Trading
The built-in paper trader provides realistic simulation without using real money or exchange APIs:

```bash
# Start with default settings (Yahoo Finance data)
make paper

# Paper trading with different data sources
make paper-yahoo    # Stock market data (Yahoo Finance)
make paper-ccxt     # Cryptocurrency data (exchanges)
make paper-ibkr     # IBKR stock data (professional grade)

# Custom paper trading
python scripts/run_paper_trading.py \
  --source yahoo \
  --symbols AAPL MSFT GOOGL \
  --initial-capital 50000 \
  --commission 0.001 \
  --spread 10 \
  --slippage 5

# IBKR paper trading (requires TWS/Gateway running)
python scripts/run_paper_trading.py \
  --source ibkr \
  --symbols SPY AAPL MSFT \
  --timeframe 1h \
  --strategy sma
```

### Paper Trading Features
- **Virtual Portfolio**: Track positions, balance, and P&L
- **Realistic Execution**: Simulates spreads, slippage, and commissions
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, etc.
- **State Persistence**: Resume trading sessions after interruption
- **Multiple Data Sources**: Yahoo Finance, CCXT, or CSV files

### Performance Report
The paper trader generates comprehensive reports including:
- Total and annualized returns
- Risk metrics (volatility, Sharpe ratio, max drawdown)
- Trading statistics (win rate, profit factor, expectancy)
- Trade history and position analysis

## 🚀 Usage

### Development Commands
```bash
make help           # Show all commands
make install        # Install dependencies
make test           # Run unit tests
make clean          # Clean Python cache
```

### Trading Commands
```bash
make test-orders    # Test order execution (safe)
make paper          # Local paper trading with virtual portfolio
make paper-yahoo    # Paper trading with Yahoo Finance data
make paper-ccxt     # Paper trading with crypto exchange data
make paper-ibkr     # Paper trading with IBKR data
make sandbox        # Exchange testnet trading
make live           # Live trading (real money!)
make status         # Show current configuration
make logs           # Show recent logs
```

### IBKR Commands
```bash
make ibkr-test      # Test IBKR connection and functionality
make ibkr-paper     # IBKR paper trading mode
make ibkr-live      # IBKR live trading (REAL MONEY!)
```

### Safety Checks
The Makefile includes automatic safety checks:
- ✅ Verifies `.env` file exists
- ✅ Checks `USE_SANDBOX` setting
- ✅ Warns about real money usage
- ✅ 5-second countdown for live trading

## 📊 Monitoring

### Real-time Status
```bash
# Show current status
make status

# Follow logs
make logs

# Check positions (saved state)
cat data/state/positions.json
```

### Performance Tracking
The bot automatically tracks:
- 📈 Daily P&L and returns
- 💰 Current positions
- 📋 Order history
- ⚠️ Error rates
- 💾 State persistence
- 📊 Risk metrics (Sharpe ratio, max drawdown, volatility)
- 🎯 Trading statistics (win rate, profit factor, expectancy)

## 🏗️ Architecture

```
trading-bot/
├── execution/
│   ├── ccxt/               # Live trading engine
│   │   ├── ccxt_trader.py  # Main trading logic
│   │   ├── state_persistence.py # State management interface
│   │   ├── file_state_store.py # File-based storage
│   │   ├── position_sync.py # Position synchronization
│   │   └── data_manager.py # Data caching
│   └── paper/              # Paper trading engine
│       ├── paper_trader.py # Paper trading logic
│       ├── virtual_portfolio.py # Virtual portfolio management
│       ├── order_simulator.py # Order execution simulation
│       └── performance_tracker.py # Performance metrics
├── strategies/             # Trading strategies
│   ├── trend_following.py  # SMA strategy
│   ├── mean_reversion.py   # RSI strategy
│   ├── breakout_strategy.py # Breakout strategy
│   └── test_strategy.py    # Testing strategy
├── config/                 # Configuration
│   └── settings.py         # Settings management
├── data/                   # Data sources
│   ├── ccxt_source.py      # Cryptocurrency exchange data
│   ├── yahoo_finance.py    # Stock market data
│   ├── ibkr_source.py      # IBKR professional data source
│   ├── ibkr_connection.py  # IBKR connection management
│   └── ibkr_sync_wrapper.py # IBKR synchronous wrapper
├── config/                 # Configuration
│   ├── settings.py         # Main configuration
│   └── ibkr_config.py      # IBKR-specific configuration
└── utils/                  # Utilities
```

## 🔐 Security

### API Key Safety
- 🔑 Store API keys in `.env` file (not in code)
- 🧪 Use testnet keys for testing
- 🔒 Limit API permissions to trading only
- 🚫 Never commit `.env` to version control

### Trading Safety
- 📊 Start with paper trading to validate strategies
- 🧪 Then test with sandbox mode using real market data
- 💰 Start with small capital amounts in live trading
- 📊 Monitor trades closely initially
- 🛑 Set appropriate stop losses
- 📱 Enable exchange notifications

## 🐛 Troubleshooting

### Testing & Diagnostics
```bash
# Test strategy with paper trading
make paper

# Test position synchronization
make test-positions

# Test order execution
make test-orders

# Check bot status
make status

# Follow logs
make logs
```

### Common Issues

**Connection Errors**
```bash
# Check API keys and permissions
make status

# Verify exchange name
EXCHANGE_NAME=binance  # correct
EXCHANGE_NAME=Binance  # incorrect (case sensitive)
```

**Order Failures**
```bash
# Check minimum order sizes
# BTC/USDT minimum: ~0.0001 BTC (~$5)
# ETH/USDT minimum: ~0.001 ETH (~$2)

# Verify account balance
# Ensure sufficient funds for position + fees
```

**Position Sync Issues**
```bash
# Test position detection
make test-positions

# Clear saved state and restart
rm -rf data/state/
make sandbox
```

### Getting Help
1. 📋 Check logs: `make logs`
2. 📊 Verify status: `make status`
3. 🧪 Test positions: `make test-positions`
4. 🧪 Test orders: `make test-orders`
5. 📖 Read [Configuration Guide](docs/CONFIGURATION.md)

## 🚧 Development

### Adding New Strategies
1. Create new file in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signals()` method
4. Add to strategy selection in `main.py`

### Database Migration
The state persistence uses an abstract interface. To switch to PostgreSQL:
1. Implement `PostgreSQLStateStore` class
2. Replace `FileStateStore()` in `ccxt_trader.py`
3. No other code changes needed

### Testing
```bash
# Run unit tests
make test

# Test specific strategy
python -m pytest tests/test_strategies.py::TestSMAStrategy

# Test paper trading components
python -m pytest tests/test_paper_trader.py

# Test order execution
make test-orders
```

## ⚖️ Disclaimer

**This software is for educational and research purposes. Trading financial instruments involves substantial risk of loss. Use at your own risk.**

- 🚧 **Under Development**: This framework is actively being developed and is not production-ready
- 💰 You are responsible for all trading decisions and outcomes
- 📉 Past performance does not guarantee future results
- 🔍 Always thoroughly test strategies before any live trading
- 💡 Start with paper trading and small amounts when going live
- 📊 Suitable for learning algorithmic trading concepts and strategy development

## 📄 License

MIT License - see LICENSE file for details.

---

🤖 **Happy Learning & Trading!** Remember: Education first, test thoroughly, trade smart, manage risk.

> **Status**: 🚧 Active development - Production deployment planned for the future