# Backtesting Guide for Trading Strategies

This guide explains how to backtest trading strategies using the trading bot framework, including support for custom indicators and features beyond basic OHLCV data.

## Quick Start

### Basic Backtesting Command

```bash
# Using the Makefile
make backtest

# Or directly with Python
python scripts/run_backtest.py --strategy sma --symbol AAPL --start 2022-01-01 --end 2023-12-31
```

### Available Parameters

```bash
python scripts/run_backtest.py \
    --strategy sma \                    # Strategy name
    --symbol AAPL \                     # Trading symbol  
    --start 2022-01-01 \               # Start date
    --end 2023-12-31 \                 # End date
    --data-source csv                   # Data source (optional override)
```

## Data Sources for Backtesting

The framework supports three data sources for backtesting:

### 1. Yahoo Finance (Default for Stocks)
```bash
# Environment configuration
DATA_SOURCE=yahoo
SYMBOLS=AAPL,MSFT,GOOGL

# Backtest command
python scripts/run_backtest.py --strategy sma --symbol AAPL --data-source yahoo
```

### 2. CCXT (Cryptocurrency Exchanges)
```bash
# Environment configuration  
DATA_SOURCE=ccxt
EXCHANGE_NAME=binance
SYMBOLS=BTC/USDT,ETH/USDT
USE_SANDBOX=true

# Backtest command
python scripts/run_backtest.py --strategy sma --symbol BTC/USDT --data-source ccxt
```

### 3. CSV Files (Recommended for Custom Data)
```bash
# Environment configuration
DATA_SOURCE=csv
CSV_DATA_DIRECTORY=data/csv
SYMBOLS=AAPL,CUSTOM_SYMBOL

# Backtest command
python scripts/run_backtest.py --strategy sma --symbol AAPL --data-source csv
```

## Using Custom Indicators and Features

### CSV Format with Additional Indicators

Beyond basic OHLCV data, you can include pre-calculated indicators in your CSV files:

```csv
Date,Open,High,Low,Close,Volume,SMA_20,SMA_50,RSI_14,MACD,Signal_Line,BB_Upper,BB_Lower
2023-01-01,100.0,105.0,99.0,104.0,1000000,102.5,98.5,65.2,1.2,0.8,110.0,95.0
2023-01-02,104.0,108.0,103.0,107.0,1200000,103.1,99.2,68.5,1.5,1.0,111.0,96.0
2023-01-03,107.0,110.0,106.0,109.0,800000,104.2,99.8,72.1,1.8,1.2,112.0,97.0
```

### Strategy Implementation with Custom Features

Create strategies that utilize additional features:

```python
from strategies.base_strategy import BaseStrategy
import pandas as pd

class CustomIndicatorStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Access standard OHLCV data
        close_prices = data['Close']
        
        # Access custom indicators from CSV
        if 'RSI_14' in data.columns:
            rsi = data['RSI_14']
        else:
            # Calculate RSI if not provided
            rsi = self.calculate_rsi(close_prices, 14)
            
        if 'MACD' in data.columns and 'Signal_Line' in data.columns:
            macd = data['MACD']
            signal_line = data['Signal_Line']
        else:
            # Calculate MACD if not provided
            macd, signal_line = self.calculate_macd(close_prices)
            
        # Generate signals using multiple indicators
        signals = pd.Series(0, index=data.index)
        
        # Buy when RSI < 30 and MACD crosses above signal line
        buy_condition = (rsi < 30) & (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        
        # Sell when RSI > 70 and MACD crosses below signal line  
        sell_condition = (rsi > 70) & (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def get_strategy_name(self) -> str:
        return "Custom Indicator Strategy"
```

### Accessing Feature Columns in Strategies

Use the base strategy helper methods:

```python
class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Get list of additional feature columns
        feature_columns = self.get_feature_columns(data)
        print(f"Available features: {feature_columns}")
        
        # Check if specific indicators are available
        has_custom_indicator = 'My_Custom_Indicator' in data.columns
        
        if has_custom_indicator:
            # Use pre-calculated indicator
            indicator_values = data['My_Custom_Indicator']
        else:
            # Calculate indicator on the fly
            indicator_values = self.calculate_custom_indicator(data)
            
        # Generate signals based on indicators
        return self.generate_signals_from_indicators(data, indicator_values)
```

## Directory Structure for CSV Backtesting

```
data/csv/
├── README.md                    # CSV format documentation
├── AAPL.csv                    # Apple stock data with indicators
├── MSFT.csv                    # Microsoft stock data  
├── BTC-USD.csv                 # Bitcoin data
├── custom_features/
│   ├── AAPL_with_ml_features.csv   # Extended feature set
│   └── portfolio_data.csv          # Multi-asset data
```

## Common CSV Column Names

The CSV data source automatically recognizes these column variations:

### Standard OHLCV Columns
- **Date columns**: `Date`, `DateTime`, `Timestamp`, `Time`
- **Price columns**: `Open`, `High`, `Low`, `Close`, `Adj Close`
- **Volume**: `Volume`

### Common Technical Indicators
- **Moving Averages**: `SMA_20`, `EMA_50`, `MA_200`
- **Oscillators**: `RSI_14`, `MACD`, `Signal_Line`, `Stochastic_K`
- **Bands**: `BB_Upper`, `BB_Middle`, `BB_Lower`
- **Volatility**: `ATR_14`, `VIX`
- **Custom**: Any column name not in OHLCV will be preserved as a feature

## Advanced Backtesting Examples

### 1. Multi-Timeframe Strategy

```bash
# Backtest on different timeframes
python scripts/run_backtest.py --strategy sma --symbol AAPL --start 2020-01-01 --end 2023-12-31
python scripts/run_backtest.py --strategy rsi --symbol AAPL --start 2020-01-01 --end 2023-12-31
```

### 2. Portfolio Backtesting

```bash
# Test multiple symbols (requires loop or custom script)
for symbol in AAPL MSFT GOOGL; do
    python scripts/run_backtest.py --strategy sma --symbol $symbol --start 2022-01-01 --end 2023-12-31
done
```

### 3. Parameter Optimization

Create a parameter sweep script:

```python
# scripts/optimize_strategy.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_backtest import run_backtest

# Parameter ranges
short_windows = [10, 15, 20, 25]
long_windows = [40, 50, 60, 70]

best_return = -float('inf')
best_params = None

for short in short_windows:
    for long in long_windows:
        if short >= long:
            continue
            
        # Update strategy parameters
        os.environ['STRATEGY_SHORT_WINDOW'] = str(short)
        os.environ['STRATEGY_LONG_WINDOW'] = str(long)
        
        # Run backtest
        results = run_backtest('sma', 'AAPL', '2022-01-01', '2023-12-31')
        
        if results and results['Return [%]'] > best_return:
            best_return = results['Return [%]']
            best_params = (short, long)
            
print(f"Best parameters: Short={best_params[0]}, Long={best_params[1]}")
print(f"Best return: {best_return:.2f}%")
```

## Backtest Output and Analysis

### Standard Metrics Reported
- **Total Return [%]**: Overall percentage return
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown [%]**: Largest peak-to-trough decline
- **Number of Trades**: Total trades executed
- **Win Rate [%]**: Percentage of profitable trades
- **Data Points**: Number of price bars analyzed
- **Date Range**: Actual date range of data used

### Visualization
The backtest automatically generates plots showing:
- Price chart with buy/sell signals
- Portfolio value over time
- Drawdown periods
- Trade markers

### Accessing Raw Results

```python
# In your custom backtest script
results = run_backtest('sma', 'AAPL', '2022-01-01', '2023-12-31')

# Access detailed metrics
print(f"Annual Return: {results['Return [%]']}")
print(f"Volatility: {results['Volatility [%]']}")
print(f"Max Drawdown Duration: {results['Max. Drawdown Duration']}")

# Access trade-by-trade data
trades = results._trades
print(trades.head())
```

## Tips for Effective Backtesting

### 1. Data Quality
- Ensure CSV files have consistent date formats
- Handle missing data points appropriately
- Verify indicator calculations are correct
- Check for look-ahead bias in indicators

### 2. Strategy Design
- Use `add_indicators()` method for calculated indicators
- Access pre-calculated features from CSV columns
- Implement proper position sizing
- Consider transaction costs and slippage

### 3. Validation
- Test on out-of-sample data
- Use walk-forward analysis
- Compare against buy-and-hold benchmark
- Validate results with different time periods

### 4. Performance Optimization
- Cache calculated indicators using the CSV approach
- Use vectorized operations in pandas
- Limit indicator recalculation
- Consider using smaller datasets for parameter optimization

## Troubleshooting Common Issues

### CSV File Issues
```bash
# Error: "Missing required columns"
# Solution: Ensure CSV has Open, High, Low, Close columns

# Error: "Symbol not found"  
# Solution: Check filename matches symbol (AAPL.csv for symbol AAPL)

# Error: "No date column found"
# Solution: Include Date, DateTime, or Timestamp column
```

### Strategy Issues
```bash
# Error: "Unknown strategy"
# Solution: Use supported strategy names (sma, rsi, trend_following, mean_reversion)

# Error: "Strategy parameters missing"
# Solution: Check .env file has STRATEGY_SHORT_WINDOW and STRATEGY_LONG_WINDOW
```

### Data Source Issues
```bash
# Error: "Unknown data source"
# Solution: Use yahoo, ccxt, or csv as DATA_SOURCE

# Error: "Failed to load data"
# Solution: Check symbol exists and date range is valid
```

This comprehensive backtesting system allows you to test strategies with rich feature sets, custom indicators, and multiple data sources for robust strategy development and validation.