# 🚀 Getting Started Guide

A step-by-step guide to get your trading bot running safely.

## 📋 Prerequisites

- Python 3.8+ installed
- Basic understanding of cryptocurrency trading
- Exchange account with API access (Binance recommended)

## 🔧 Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd robotdreams-trading-fw
make install
```

### 2. Get Testnet API Keys
**⚠️ Start with testnet - never use real money for testing!**

For Binance Spot:
1. Go to https://testnet.binance.vision/
2. Create account and generate API keys

For Binance Futures:
1. Go to https://testnet.binancefuture.com/
2. Login with your Binance account
3. Generate API keys for futures testnet
4. Save your API key and secret

## 🧪 Safe Testing (Recommended First Step)

### 1. Setup Test Configuration

**For Spot Trading:**
```bash
# Copy spot sandbox configuration
cp .env.sandbox .env
```

**For Futures Trading:**
```bash
# Copy futures configuration
cp .env.futures .env
```

Then edit with your testnet credentials:
```bash
nano .env
```

Update these lines in `.env`:
```bash
EXCHANGE_API_KEY=your_testnet_api_key_here
EXCHANGE_API_SECRET=your_testnet_secret_here
```

### 2. Test Order Execution
```bash
# Test basic order handling (safe)
make test-orders
```

This will:
- ✅ Connect to exchange testnet
- ✅ Test buy/sell orders with small amounts
- ✅ Verify position tracking
- ✅ Test error handling

### 3. Test Strategy Trading
```bash
# Start sandbox trading
make sandbox
```

This runs continuous trading with:
- 🧪 Real market data
- 💰 Fake money (testnet)
- 📊 Real strategy execution
- 📝 Full logging

## 📊 Monitor Your Test Trading

### Check Status
```bash
# Show current configuration and state
make status
```

### View Logs
```bash
# Follow live logs
make logs
```

### Check Positions
```bash
# View saved positions
cat data/state/positions.json

# Check orders
cat data/state/orders.json
```

## 🎯 Test Different Strategies

Edit `.env` and change strategy:
```bash
# Simple moving average (trend following)
STRATEGY_NAME=sma
STRATEGY_SHORT_WINDOW=10
STRATEGY_LONG_WINDOW=20

# RSI mean reversion
STRATEGY_NAME=rsi

# Breakout strategy
STRATEGY_NAME=breakout
```

Restart with:
```bash
make sandbox
```

## 💰 Going Live (After Successful Testing)

### ⚠️ Important Safety Checklist
- [ ] ✅ Tested extensively with sandbox
- [ ] ✅ Understand the strategy behavior
- [ ] ✅ Verified all settings
- [ ] ✅ Started with small capital
- [ ] ✅ Set appropriate stop losses
- [ ] ✅ Have monitoring plan

### 1. Get Production API Keys
1. Log into your exchange (Binance, etc.)
2. Go to API Management
3. Create new API key
4. **Enable spot trading only** (don't enable withdrawals)
5. Note the key and secret

### 2. Setup Live Configuration
```bash
# Copy live template
cp .env.live .env

# Edit with production credentials
nano .env
```

Update key settings:
```bash
USE_SANDBOX=false                    # REAL MONEY MODE!
EXCHANGE_API_KEY=your_production_key
EXCHANGE_API_SECRET=your_production_secret
INITIAL_CAPITAL=100                  # Start small!
MAX_POSITION_SIZE=0.02              # 2% max position
```

### 3. Final Safety Check
```bash
# Verify configuration
make status
```

Should show:
- 💰 Mode: LIVE (real money)
- 🔗 Exchange: your exchange
- 📈 Strategy: your chosen strategy

### 4. Start Live Trading
```bash
# THIS USES REAL MONEY!
make live
```

The system will:
- 🚨 Show warnings
- ⏰ Wait 5 seconds (time to abort)
- 🚀 Start live trading

## 📱 Monitoring Live Trading

### Real-time Monitoring
```bash
# Check status frequently
make status

# Watch logs
make logs

# Check positions
cat data/state/positions.json
```

### Exchange Monitoring
- 📱 Enable exchange app notifications
- 📊 Monitor positions on exchange directly
- 💰 Check balance changes

### Emergency Stop
If something goes wrong:
1. **Ctrl+C** to stop the bot
2. Manually close positions on exchange if needed
3. Review logs to understand what happened

## 🔧 Common Configuration Tweaks

### Position Sizing
```bash
# Conservative (1% positions)
MAX_POSITION_SIZE=0.01

# Moderate (5% positions) 
MAX_POSITION_SIZE=0.05

# Aggressive (10% positions)
MAX_POSITION_SIZE=0.10
```

### Stop Losses
```bash
# Tight stop loss (2%)
STOP_LOSS_PCT=0.02

# Moderate stop loss (5%)
STOP_LOSS_PCT=0.05

# Wide stop loss (10%)
STOP_LOSS_PCT=0.10
```

### Trading Frequency
```bash
# High frequency (1 minute)
TIMEFRAME=1m

# Medium frequency (1 hour)
TIMEFRAME=1h

# Low frequency (4 hours)
TIMEFRAME=4h
```

## ❌ Common Mistakes to Avoid

1. **Starting with live trading** - Always test with sandbox first
2. **Large position sizes** - Start with 1-2% maximum
3. **No stop losses** - Always set stop losses
4. **Ignoring logs** - Monitor logs for errors
5. **Wrong API permissions** - Only enable spot trading
6. **Leaving bot unmonitored** - Check regularly, especially first days

## 🆘 Getting Help

### If Something Goes Wrong
1. 🛑 Stop the bot: `Ctrl+C`
2. 📋 Check logs: `make logs`
3. 📊 Check status: `make status`
4. 💰 Check exchange balance directly

### Debug Steps
```bash
# Test connection
make test-orders

# Check configuration
make status

# Clear state and restart
rm -rf data/state/
make sandbox
```

### Support
- 📖 Read the full [README.md](README.md)
- 📋 Check [Configuration Guide](docs/CONFIGURATION.md)
- 🐛 Review error logs
- 🧪 Test with sandbox mode

---

🎯 **Remember**: Start small, test thoroughly, monitor closely!