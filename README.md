# 🤖 Cryptocurrency Trading Bot

A production-ready automated trading bot for cryptocurrency exchanges with comprehensive risk management, state persistence, and error handling.

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
# Start sandbox trading (safe)
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
- **Sandbox Mode**: Test with real market data, no real money
- **State Persistence**: Crash-resistant with automatic recovery
- **Position Sync**: Handles pre-existing positions on restart
- **Daily Loss Limits**: Automatic emergency stops
- **Instance Locking**: Prevents multiple bots running simultaneously

### 📊 Trading Capabilities
- **Multiple Strategies**: SMA, RSI, Breakout, and custom strategies
- **Real-time Execution**: Live order placement with comprehensive error handling
- **Smart Data Management**: Efficient historical data caching
- **Risk Management**: Position sizing, stop losses, exposure limits

### 🛠️ Technical Features
- **CCXT Integration**: Supports 100+ cryptocurrency exchanges
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
make sandbox        # Paper trading with testnet
make live           # Live trading (real money!)
make status         # Show current configuration
make logs           # Show recent logs
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
- 📈 Daily P&L
- 💰 Current positions
- 📋 Order history
- ⚠️ Error rates
- 💾 State persistence

## 🏗️ Architecture

```
trading-bot/
├── execution/ccxt/          # Live trading engine
│   ├── ccxt_trader.py      # Main trading logic
│   ├── state_persistence.py # State management interface
│   ├── file_state_store.py # File-based storage
│   ├── position_sync.py    # Position synchronization
│   └── data_manager.py     # Data caching
├── strategies/             # Trading strategies
│   ├── trend_following.py  # SMA strategy
│   ├── mean_reversion.py   # RSI strategy
│   ├── breakout_strategy.py # Breakout strategy
│   └── test_strategy.py    # Testing strategy
├── config/                 # Configuration
│   └── settings.py         # Settings management
├── data/                   # Data sources
└── utils/                  # Utilities
```

## 🔐 Security

### API Key Safety
- 🔑 Store API keys in `.env` file (not in code)
- 🧪 Use testnet keys for testing
- 🔒 Limit API permissions to trading only
- 🚫 Never commit `.env` to version control

### Trading Safety
- 🧪 Always test with sandbox mode first
- 💰 Start with small capital amounts
- 📊 Monitor trades closely initially
- 🛑 Set appropriate stop losses
- 📱 Enable exchange notifications

## 🐛 Troubleshooting

### Testing & Diagnostics
```bash
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

# Test order execution
make test-orders
```

## ⚖️ Disclaimer

**This software is for educational purposes. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.**

- 💰 You are responsible for all trading decisions
- 📉 Past performance does not guarantee future results
- 🔍 Always review and test thoroughly before live trading
- 💡 Start with small amounts and paper trading

## 📄 License

MIT License - see LICENSE file for details.

---

🤖 **Happy Trading!** Remember: Test first, trade smart, manage risk.