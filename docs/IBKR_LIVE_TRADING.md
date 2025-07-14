# IBKR Live Trading with main.py

This guide explains how to use the IBKR trader with the main.py live trading system.

## Prerequisites

1. **Interactive Brokers Account**: You need an active IBKR account
2. **TWS or IB Gateway**: Must be running and logged in
3. **API Configuration**: Enable API connections in TWS/Gateway settings
4. **Market Data**: Complete API acknowledgement form for market data
5. **Dependencies**: Install required packages: `pip install ib_async`

## Configuration

### 1. Environment Variables

Copy and customize the IBKR configuration:

```bash
cp .env.ibkr.live.example .env
```

Key settings:

- `DATA_SOURCE=ibkr` - Use IBKR as data source
- `IBKR_ACCOUNT_TYPE=live` - Set to 'live' for real money trading
- `IBKR_PORT=7496` - Live TWS port (7496) or Gateway port (4001)
- `IBKR_ACCOUNT_ID=your_account_id` - Your IBKR account ID
- `SYMBOLS=AAPL,MSFT,GOOGL` - Symbols to trade

### 2. Paper Trading vs Live Trading

**Paper Trading:**
```bash
IBKR_ACCOUNT_TYPE=paper
IBKR_PORT=7497  # Paper TWS port
```

**Live Trading:**
```bash
IBKR_ACCOUNT_TYPE=live
IBKR_PORT=7496  # Live TWS port
```

## Running Live Trading

### 1. Start TWS/Gateway

- Launch TWS or IB Gateway
- Log in with your credentials
- Ensure API connections are enabled

### 2. Start the Trading Bot

```bash
python main.py
```

The bot will:
- Connect to IBKR via TWS/Gateway
- Initialize the selected strategy
- Start the main trading loop
- Display trading mode (PAPER/LIVE)

### 3. Monitor Trading

The bot logs:
- Connection status
- Trading signals
- Order execution
- Performance metrics
- Error messages

## Trading Process

### 1. Data Flow

1. **Historical Data**: Fetched from IBKR for signal generation
2. **Real-time Prices**: Current market prices for order execution
3. **Signal Generation**: Strategy processes data and generates signals
4. **Order Execution**: Market orders placed through IBKR API
5. **Position Tracking**: Positions synced with IBKR account

### 2. Order Management

- **Market Orders**: Primary order type for immediate execution
- **Position Sizing**: Based on account balance and risk parameters
- **Order Tracking**: All orders tracked with unique IDs
- **State Persistence**: Order and position state saved to disk

### 3. Risk Management

- **Daily P&L Monitoring**: Tracks daily profit/loss
- **Position Limits**: Enforces maximum position sizes
- **Emergency Stop**: Manual stop mechanism for emergencies
- **Account Balance**: Real-time account balance monitoring

## Safety Features

### 1. Pre-Flight Checks

Before each trading cycle:
- Verify IBKR connection
- Check emergency stop status
- Validate daily loss limits
- Ensure sufficient data

### 2. Error Handling

- **Connection Issues**: Auto-reconnect with backoff
- **API Errors**: Comprehensive error mapping
- **Data Issues**: Fallback to cached data
- **Order Failures**: Retry mechanism with logging

### 3. State Recovery

- **Crash Recovery**: Automatic state recovery on restart
- **Position Sync**: Sync with actual IBKR positions
- **Order Tracking**: Resume tracking existing orders

## Configuration Examples

### Conservative Day Trading
```bash
STRATEGY_NAME=sma
STRATEGY_SHORT_WINDOW=10
STRATEGY_LONG_WINDOW=20
MAX_POSITION_SIZE=0.05
TIMEFRAME=5m
```

### Swing Trading
```bash
STRATEGY_NAME=sma
STRATEGY_SHORT_WINDOW=20
STRATEGY_LONG_WINDOW=50
MAX_POSITION_SIZE=0.1
TIMEFRAME=1h
```

### Breakout Strategy
```bash
STRATEGY_NAME=breakout
STRATEGY_ENTRY_LOOKBACK=20
STRATEGY_EXIT_LOOKBACK=10
STRATEGY_ATR_PERIOD=14
MAX_POSITION_SIZE=0.08
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify TWS/Gateway is running
   - Check port configuration
   - Ensure API is enabled

2. **No Market Data**
   - Complete API acknowledgement form
   - Check market data subscriptions
   - Verify market hours

3. **Order Rejected**
   - Check account permissions
   - Verify symbol format
   - Ensure sufficient buying power

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG
IBKR_LOG_LEVEL=DEBUG
```

## Performance Monitoring

### Real-time Metrics
- Daily P&L
- Open positions
- Order status
- Account balance

### Log Analysis
- Check `logs/` directory for detailed logs
- Monitor for errors and warnings
- Track performance over time

## Important Notes

⚠️ **WARNING**: This is live trading with real money. Always:
- Test thoroughly with paper trading first
- Start with small position sizes
- Monitor actively during trading hours
- Have emergency stop procedures ready
- Understand the risks involved

📊 **Market Data**: Live market data requires subscriptions and acknowledgement forms with IBKR.

🕐 **Trading Hours**: The bot respects market hours and will not trade outside regular hours.

💾 **State Management**: All trading state is persisted to disk for crash recovery.

## Support

For issues with:
- IBKR API: Contact Interactive Brokers support
- Trading Bot: Check logs and error messages
- Strategy Issues: Review strategy documentation