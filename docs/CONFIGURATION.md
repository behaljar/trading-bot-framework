# Trading Bot Configuration Guide

## Quick Start

1. **Copy configuration template:**
   ```bash
   cp .env.example .env
   ```

2. **For testing (recommended first):**
   ```bash
   cp .env.sandbox .env
   # Edit with your testnet API keys
   ```

3. **For live trading (after testing):**
   ```bash
   cp .env.live .env
   # Edit with your production API keys
   ```

## Configuration Files

| File | Purpose | Safety |
|------|---------|--------|
| `.env.example` | Template with all options | N/A |
| `.env.sandbox` | Spot testnet/paper trading | ✅ Safe |
| `.env.futures` | Futures testnet configuration | ✅ Safe |
| `.env.live` | Live trading template | ⚠️ Real money |
| `.env.test` | Order testing config | ✅ Safe |

## Key Settings

### Trading Mode
```bash
USE_SANDBOX=true   # Paper trading (safe)
USE_SANDBOX=false  # Live trading (real money)
```

### Data Source
```bash
DATA_SOURCE=ccxt   # REQUIRED for live trading
EXCHANGE_NAME=binance  # binance, kraken, coinbase, etc.
```

### Symbols & Timeframes
```bash
# Spot trading
SYMBOLS=BTC/USDT,ETH/USDT          # Spot pairs

# Futures trading  
SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT # Futures pairs (USDT margined)
TIMEFRAME=1h                        # 1m, 5m, 15m, 1h, 4h, 1d
TRADING_TYPE=future                 # spot or future
```

### Strategies
```bash
STRATEGY_NAME=sma      # Available: sma, mean_reversion, breakout, test
```

#### Strategy Parameters

**SMA Strategy:**
```bash
STRATEGY_SHORT_WINDOW=20
STRATEGY_LONG_WINDOW=50
```

**Breakout Strategy:**
```bash
STRATEGY_ENTRY_LOOKBACK=20
STRATEGY_EXIT_LOOKBACK=10
STRATEGY_ATR_PERIOD=14
STRATEGY_ATR_MULTIPLIER=2.0
```

**Test Strategy:**
- No parameters needed
- Alternates buy/sell every bar
- Use only for testing order execution

### Risk Management
```bash
INITIAL_CAPITAL=1000        # Your trading capital
MAX_POSITION_SIZE=0.05      # 5% max position size
STOP_LOSS_PCT=0.02         # 2% global stop loss
ALLOW_SHORT=false          # Enable/disable short selling
```

### API Keys

**For Testing (Testnet):**
- Binance Spot: https://testnet.binance.vision/
- Binance Futures: https://testnet.binancefuture.com/
- Get testnet API keys (no real money)

**For Live Trading:**
- Get production API keys from your exchange
- Ensure trading permissions are enabled
- Keep keys secure and never share

```bash
EXCHANGE_API_KEY=your_key_here
EXCHANGE_API_SECRET=your_secret_here
```

## Safety Guidelines

### Before Live Trading:
1. ✅ Test with sandbox/testnet first
2. ✅ Start with small capital
3. ✅ Verify API key permissions
4. ✅ Set conservative position sizes
5. ✅ Enable stop losses
6. ✅ Monitor initial trades closely

### Configuration Checklist:
- [ ] `USE_SANDBOX=false` for live trading
- [ ] Production API keys configured
- [ ] `STRATEGY_NAME` is NOT "test"
- [ ] `MAX_POSITION_SIZE` is reasonable (≤10%)
- [ ] `STOP_LOSS_PCT` is set (2-5%)
- [ ] `INITIAL_CAPITAL` matches account balance

## Testing Order Execution

Use the test strategy to verify order handling:

```bash
# Copy test configuration
cp .env.test .env

# Edit with your testnet API keys
# Run test script
python test_orders.py
```

This will:
- ✅ Test exchange connection
- ✅ Verify order placement
- ✅ Test position tracking
- ✅ Validate error handling

## Common Issues

### API Connection Errors
- Check API key format
- Verify exchange name spelling
- Ensure API permissions include trading
- Check if sandbox mode matches key type

### Position Size Errors
- Check minimum order sizes for your symbols
- Verify account balance
- Adjust `MAX_POSITION_SIZE` if needed

### Strategy Errors
- Ensure sufficient historical data
- Check strategy parameter ranges
- Verify symbol format (BTC/USDT, not BTCUSDT)

## Environment Variables Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATA_SOURCE` | string | ccxt | Data source (ccxt only for live) |
| `EXCHANGE_NAME` | string | binance | Exchange name |
| `USE_SANDBOX` | boolean | true | Sandbox mode |
| `SYMBOLS` | string | BTC/USDT | Trading symbols |
| `TIMEFRAME` | string | 1h | Candlestick timeframe |
| `STRATEGY_NAME` | string | sma | Strategy to use |
| `INITIAL_CAPITAL` | float | 1000 | Starting capital |
| `MAX_POSITION_SIZE` | float | 0.05 | Max position as % of capital |
| `STOP_LOSS_PCT` | float | 0.02 | Stop loss percentage |
| `COMMISSION` | float | 0.001 | Commission rate |
| `SLIPPAGE` | float | 0.0005 | Slippage factor |
| `ALLOW_SHORT` | boolean | false | Allow short selling |
| `LOG_LEVEL` | string | INFO | Logging level |