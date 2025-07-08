# 📈 IBKR Integration Guide

Interactive Brokers (IBKR) integration for multi-asset trading with TWS/IB Gateway connectivity, real-time market data, and comprehensive paper trading support. Part of an educational trading framework under active development.

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Install TWS or IB Gateway
# Download from: https://www.interactivebrokers.com/en/trading/tws.php

# Install dependencies
make install
```

### 2. TWS/Gateway Setup
1. **Install TWS or IB Gateway**
   - TWS: Full trading platform with GUI
   - IB Gateway: Headless API server (recommended for automated trading)

2. **Configure API Settings**
   - Enable API connections in TWS/Gateway
   - Set port: 7497 (paper) or 7496 (live)
   - Add trusted IP: 127.0.0.1
   - Set client ID range (default: 0-32)

3. **Market Data Permissions**
   - Paper trading: Free delayed data (15 min delay)
   - Real-time data: Requires market data subscriptions (most US stocks free)

### 3. Configuration
```bash
# Copy IBKR paper trading configuration
cp .env.ibkr.paper .env

# Or manually configure in .env:
DATA_SOURCE=ibkr
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                    # 7497=paper, 7496=live
IBKR_CLIENT_ID=1
IBKR_ACCOUNT_TYPE=paper
IBKR_MARKET_DATA_TYPE=3           # 3=delayed, 1=live
```

### 4. Test Connection
```bash
# Test IBKR connection and functionality
make ibkr-test

# Expected output:
# ✅ Connected to IBKR with accounts: ['DU9502108']
# ✅ Test data request successful: SPY @ $620.41
```

### 5. Start Trading
```bash
# Paper trading with IBKR data
make paper-ibkr

# Or directly:
python scripts/run_paper_trading.py \
  --source ibkr \
  --symbols SPY AAPL MSFT \
  --timeframe 1h \
  --strategy sma
```

## 🏗️ Architecture

### Core Components

#### 1. **IBKRConfig** (`config/ibkr_config.py`)
```python
@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # 7496=live, 7497=paper
    client_id: int = 1
    account_type: IBKRAccountType = IBKRAccountType.PAPER
    market_data_type: IBKRMarketDataType = IBKRMarketDataType.DELAYED
```

#### 2. **IBKRConnectionManager** (`data/ibkr_connection.py`)
- Professional connection management with auto-reconnect
- Event-driven error handling and monitoring
- Rate limiting and API compliance
- Contract qualification and creation

#### 3. **IBKRDataSource** (`data/ibkr_source.py`)
- Asynchronous market data retrieval
- Historical data with proper timezone handling (UTC)
- Real-time streaming market data
- Comprehensive caching system

#### 4. **IBKRSyncWrapper** (`data/ibkr_sync_wrapper.py`)
- Synchronous wrapper for paper trading integration
- Handles async/sync impedance mismatch
- Thread-based async execution management

## 📊 Data Sources & Formats

### Historical Data
```python
# Get historical data
df = await ibkr_source.get_historical_data(
    symbol='SPY',
    start_date='2025-01-01',
    end_date='2025-01-07',
    timeframe='1h'
)

# Returns DataFrame with columns:
# Date (timezone-aware index), Open, High, Low, Close, Volume
```

### Real-time Data
```python
# Get current price
price_data = await ibkr_source.get_current_price('SPY')
# Returns: {'symbol': 'SPY', 'price': 620.41, 'bid': 620.40, 'ask': 620.42, ...}
```

### Supported Timeframes
- `1m`, `5m`, `15m`, `30m` - Intraday
- `1h`, `2h`, `4h` - Hourly
- `1d`, `1w`, `1M` - Daily and longer

### Supported Symbols
- **US Stocks**: AAPL, MSFT, GOOGL, TSLA, etc.
- **ETFs**: SPY, QQQ, IWM, DIA, etc.
- **Indices**: Auto-detection based on symbol
- **Forex**: EUR/USD format (limited in paper trading)

## 🔧 Configuration Reference

### Environment Variables
```bash
# Connection Settings
IBKR_HOST=127.0.0.1               # TWS/Gateway host
IBKR_PORT=7497                    # Paper: 7497, Live: 7496
IBKR_CLIENT_ID=1                  # Unique client ID (0-32)

# Account Settings
IBKR_ACCOUNT_TYPE=paper           # paper or live
IBKR_ACCOUNT_ID=DU9502108         # Your account ID (optional)

# Market Data Settings
IBKR_MARKET_DATA_TYPE=3           # 1=live, 2=frozen, 3=delayed, 4=delayed_frozen

# Connection Management
IBKR_CONNECT_TIMEOUT=10           # Connection timeout (seconds)
IBKR_READ_TIMEOUT=30              # Read timeout (seconds)
IBKR_AUTO_RECONNECT=true          # Enable auto-reconnection
IBKR_MAX_RECONNECT_ATTEMPTS=5     # Max reconnection attempts
IBKR_RECONNECT_DELAY=5            # Delay between reconnections (seconds)

# Rate Limiting
IBKR_MAX_REQUESTS_PER_SECOND=50   # Max API requests per second
IBKR_HISTORICAL_DATA_TIMEOUT=60   # Historical data timeout (seconds)

# Trading Bot Settings
DATA_SOURCE=ibkr                  # Use IBKR as data source
STRATEGY_NAME=sma                 # Trading strategy
INITIAL_CAPITAL=10000.0           # Starting capital
```

### Market Data Types
| Type | Value | Description | Cost |
|------|-------|-------------|------|
| Live | 1 | Real-time market data | Subscription required |
| Frozen | 2 | Last available quote | Free |
| Delayed | 3 | 15-minute delayed data | Free |
| Delayed Frozen | 4 | Delayed last quote | Free |

## 📋 Commands Reference

### Testing Commands
```bash
# Test IBKR connection
make ibkr-test

# Test paper trading integration
python tests/test_ibkr_paper_integration.py
```

### Paper Trading Commands
```bash
# IBKR paper trading with default settings
make paper-ibkr

# Custom paper trading
python scripts/run_paper_trading.py \
  --source ibkr \
  --symbols SPY AAPL MSFT \
  --timeframe 1h \
  --strategy sma \
  --initial-capital 50000
```

### Live Trading Commands
```bash
# IBKR paper trading mode (safe)
make ibkr-paper

# IBKR live trading (REAL MONEY!)
make ibkr-live
```

## 💰 Market Data Costs

### Free Data
- **US Stocks/ETFs**: Free real-time from Cboe One and IEX
- **Delayed Data**: 15-minute delayed quotes (free)
- **Snapshot Quotes**: Up to 100 free per month

### Paid Subscriptions
- **Real-time Consolidated**: $1-10/month per exchange
- **Level II Data**: $10-30/month per exchange
- **International Markets**: Varies by exchange

### Paper Trading Considerations
- Delayed data is sufficient for strategy development
- Real-time subscriptions work with both live and paper accounts
- Market data subscriptions require $500+ account balance

## 🛠️ Development & Testing

### Testing Framework
```bash
# Run all IBKR tests
python -m pytest tests/test_ibkr* -v

# Test specific functionality
python tests/test_ibkr_connection.py        # Connection testing
python tests/test_ibkr_paper_integration.py # Paper trading integration
```

### Development Tips
1. **Use Paper Account**: Always start with paper trading
2. **Test with Delayed Data**: Develop strategies with free delayed data
3. **Handle Rate Limits**: Built-in rate limiting prevents API violations
4. **Monitor Connections**: Auto-reconnection handles temporary disconnections
5. **Contract Qualification**: Automatic contract resolution for symbols

### Adding New Features
1. **Execution Engine**: Implement order placement and management
2. **Position Sync**: Add position synchronization with IBKR accounts
3. **Risk Management**: Integrate position sizing and risk controls
4. **Portfolio Analytics**: Add portfolio tracking and reporting

## 🔐 Security & Best Practices

### API Security
- **Local Connections**: TWS/Gateway runs locally (127.0.0.1)
- **Client ID Management**: Use unique client IDs per application
- **IP Restrictions**: Configure trusted IPs in TWS settings
- **Account Isolation**: Separate paper and live account configurations

### Trading Safety
- **Paper First**: Always test strategies in paper trading
- **Start Small**: Begin with small position sizes in live trading
- **Monitor Closely**: Watch initial live trades carefully
- **Risk Limits**: Set appropriate stop losses and position limits

### Error Handling
- **Connection Recovery**: Automatic reconnection on network issues
- **API Rate Limits**: Built-in rate limiting prevents violations
- **Data Validation**: Comprehensive error checking and logging
- **Graceful Degradation**: Handles missing market data gracefully

## 🐛 Troubleshooting

### Common Issues

#### Connection Problems
```
Error: Failed to connect to IBKR
```
**Solutions:**
- Ensure TWS/IB Gateway is running
- Check port settings (7497 for paper, 7496 for live)
- Verify API connections are enabled in TWS
- Confirm client ID is not in use

#### Market Data Issues
```
Warning 10167: Requested market data is not subscribed
```
**Solutions:**
- This is normal for delayed data (working as intended)
- For real-time data: Subscribe in Client Portal
- Enable data sharing for paper accounts
- Wait 24 hours for subscription changes to propagate

#### Contract Resolution Errors
```
Error: Could not qualify contract for SYMBOL
```
**Solutions:**
- Verify symbol format (e.g., 'SPY' not 'SPY.US')
- Check if symbol is available on IBKR
- Ensure market is open for contract qualification

#### Account Access Issues
```
Error: Account DU123456 not accessible
```
**Solutions:**
- Update IBKR_ACCOUNT_ID in configuration
- Verify account is properly funded
- Check account permissions in Client Portal

### Getting Help
1. **Check Logs**: Monitor detailed connection and error logs
2. **Test Connection**: Use `make ibkr-test` for diagnostics
3. **Verify Configuration**: Review `.env` settings
4. **IBKR Documentation**: Consult official IBKR API documentation
5. **Community Forums**: IBKR API community and support

## 📚 Additional Resources

### IBKR Documentation
- [IBKR API Documentation](https://www.interactivebrokers.com/en/trading/ib-api.php)
- [TWS API Guide](https://www.interactivebrokers.com/campus/ibkr-api-page/)
- [Market Data Subscriptions](https://www.interactivebrokers.com/en/pricing/market-data-pricing.php)

### Configuration Templates
- `.env.ibkr.paper` - Paper trading configuration
- `.env.ibkr.live` - Live trading configuration template

### Code Examples
- `tests/test_ibkr_connection.py` - Connection testing examples
- `tests/test_ibkr_paper_integration.py` - Integration examples
- `scripts/run_paper_trading.py` - Paper trading implementation

---

🏦 **Educational IBKR Integration** - Comprehensive framework for learning algorithmic trading with institutional-grade connectivity and robust error handling.

> **Status**: 🚧 Under active development - Production deployment planned for the future