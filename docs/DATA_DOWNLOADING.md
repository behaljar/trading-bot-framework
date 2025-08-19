# Data Downloading Guide

This guide covers how to download financial data for use with the trading framework. The framework supports multiple data sources with automatic chunking for large date ranges.

## Supported Data Sources

### 1. CCXT (Cryptocurrency Exchanges)
- **Purpose**: Download cryptocurrency data from various exchanges
- **Supported Exchanges**: Binance, Coinbase, Kraken, and many others
- **Data Types**: OHLCV (Open, High, Low, Close, Volume) candlestick data
- **Rate Limiting**: Built-in rate limiting and chunking for large date ranges
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M (exchange-dependent)

### 2. Yahoo Finance 
- **Purpose**: Download stock market and traditional financial instrument data
- **Coverage**: Stocks, ETFs, indices, forex, commodities
- **Data Types**: OHLCV historical data and current prices
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- **Limitations**: No order book data available

## Quick Start

### Download Stock Data (Yahoo Finance)
```bash
# Download 1 year of Apple stock data (default date range)
python scripts/download_data.py --source yahoo --symbol AAPL

# Download with custom date range
python scripts/download_data.py --source yahoo --symbol MSFT --start 2020-01-01 --end 2023-12-31

# Download hourly data
python scripts/download_data.py --source yahoo --symbol GOOGL --timeframe 1h --start 2023-01-01
```

### Download Cryptocurrency Data (CCXT)
```bash
# Download 1 year of Bitcoin data from Binance
python scripts/download_data.py --source ccxt --symbol BTC/USDT --exchange binance

# Download from 2017 to present (automatically chunked)
python scripts/download_data.py --source ccxt --symbol BTC/USDT --start 2017-01-01 --exchange binance

# Download from sandbox/testnet
python scripts/download_data.py --source ccxt --symbol ETH/USDT --sandbox --exchange binance

# Download 15-minute data
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 15m --start 2024-01-01
```

## Data Download Script Reference

### Command Line Options

```bash
python scripts/download_data.py [OPTIONS]
```

**Required Arguments:**
- `--source {yahoo,ccxt}`: Data source to use
- `--symbol SYMBOL`: Symbol to download (e.g., AAPL, BTC/USDT)

**Optional Arguments:**
- `--start YYYY-MM-DD`: Start date (default: 1 year ago)
- `--end YYYY-MM-DD`: End date (default: today)
- `--timeframe TIMEFRAME`: Data timeframe (default: 1d)
- `--exchange EXCHANGE`: Exchange name for CCXT (default: binance)
- `--sandbox`: Use sandbox/testnet instead of live exchange (CCXT only)
- `--output-dir DIR`: Output directory (default: data/csv)
- `--filename FILENAME`: Custom filename (default: SYMBOL.csv)

### Automatic Chunking

The download script automatically handles large date ranges by breaking them into smaller chunks:

- **Yahoo Finance**: Chunks requests for date ranges > 3 years into 12-month segments
- **CCXT**: Always chunks requests into 3-month segments due to exchange API limits
- **Rate Limiting**: Built-in delays between requests to respect API limits
- **Deduplication**: Automatically removes duplicate entries at chunk boundaries

### Data Output Format

All downloaded data is saved as CSV files with standardized columns:

```csv
timestamp,Open,High,Low,Close,Volume
2023-01-01 00:00:00,16625.08,16625.08,16531.0,16688.74,1200.5
2023-01-02 00:00:00,16688.74,16950.0,16688.74,16830.0,950.2
...
```

**Column Specifications:**
- `timestamp`: DateTime index in ISO format
- `Open/High/Low/Close`: Price data as float values
- `Volume`: Trading volume as float

## Data Source Implementation Details

### CCXT Source (`framework/data/downloaders/ccxt_downloader.py`)

**Features:**
- Dynamic exchange initialization with error handling
- Symbol normalization (converts BTCUSDT to BTC/USDT format)
- Automatic market loading and validation
- Rate limiting compliance
- Support for both sandbox and live trading environments
- Order book data retrieval (when supported by exchange)

**Configuration:**
```python
source = CCXTSource(
    exchange_name="binance",  # Exchange name
    api_key="",               # Optional for public data
    api_secret="",           # Optional for public data  
    sandbox=True             # Use testnet/sandbox
)
```

**Symbol Format:**
- Input: `BTCUSDT`, `BTC/USDT`, `ETH/BTC`
- Auto-normalization to exchange format
- Validation against available markets

### Yahoo Finance Source (`framework/data/downloaders/yahoo_downloader.py`)

**Features:**
- Direct integration with yfinance library
- Automatic timeframe mapping
- Column standardization to match framework expectations
- Current price retrieval
- Error handling for unavailable symbols/dates

**Supported Symbols:**
- Stocks: `AAPL`, `MSFT`, `GOOGL`
- ETFs: `SPY`, `QQQ`, `VTI`
- Indices: `^GSPC`, `^IXIC`, `^RUT`
- Forex: `EURUSD=X`, `GBPUSD=X`
- Commodities: `GC=F` (Gold), `CL=F` (Oil)

## Usage Examples by Asset Class

### Stocks and ETFs
```bash
# Major tech stocks
python scripts/download_data.py --source yahoo --symbol AAPL --start 2020-01-01
python scripts/download_data.py --source yahoo --symbol MSFT --start 2020-01-01
python scripts/download_data.py --source yahoo --symbol GOOGL --start 2020-01-01

# Popular ETFs
python scripts/download_data.py --source yahoo --symbol SPY --start 2020-01-01
python scripts/download_data.py --source yahoo --symbol QQQ --start 2020-01-01
```

### Cryptocurrencies
```bash
# Major crypto pairs
python scripts/download_data.py --source ccxt --symbol BTC/USDT --exchange binance
python scripts/download_data.py --source ccxt --symbol ETH/USDT --exchange binance
python scripts/download_data.py --source ccxt --symbol ADA/USDT --exchange binance

# Alternative exchanges
python scripts/download_data.py --source ccxt --symbol BTC/USD --exchange coinbase
python scripts/download_data.py --source ccxt --symbol ETH/EUR --exchange kraken
```

### High-Frequency Data
```bash
# 1-minute data for day trading strategies
python scripts/download_data.py --source ccxt --symbol BTC/USDT --timeframe 1m --start 2024-01-01 --end 2024-01-07

# 15-minute data for swing trading
python scripts/download_data.py --source yahoo --symbol AAPL --timeframe 15m --start 2024-01-01 --end 2024-02-01
```

## Error Handling and Troubleshooting

### Common Issues

**1. Module Import Errors**
```
ModuleNotFoundError: No module named 'framework.data.yahoo_finance'
```
- Solution: Check that module paths match actual file structure
- Ensure you're running from the project root directory

**2. API Rate Limiting**
```
CCXT error downloading data: Exchange rate limit exceeded
```
- Solution: Script automatically handles rate limiting with delays
- For persistent issues, increase chunk size or add longer delays

**3. Symbol Format Errors**
```
Symbol BTC-USDT is not available
```
- Solution: Use proper symbol format (BTC/USDT for CCXT, BTCUSD for Yahoo)
- Check available symbols using the exchange's documentation

**4. Date Range Issues**
```
No data retrieved for SYMBOL
```
- Solution: Verify symbol existed during the requested date range
- Check if market was trading during specified dates
- Ensure date format is YYYY-MM-DD

### Best Practices

1. **Start with small date ranges** when testing new symbols
2. **Use sandbox mode** for CCXT when developing/testing
3. **Check output CSV files** for data quality before using in strategies
4. **Monitor rate limits** especially when downloading multiple symbols
5. **Store data locally** to avoid repeated API calls during development

## Integration with Trading Framework

Downloaded CSV files can be directly used with the backtesting system:

```bash
# Download data
python scripts/download_data.py --source ccxt --symbol BTC/USDT --start 2023-01-01

# Run backtest using downloaded data
python scripts/run_backtest.py --strategy sma --data-file data/csv/BTC_USDT.csv --symbol BTC_USDT
```

The framework expects data in the standardized format produced by the download scripts, making the integration seamless.