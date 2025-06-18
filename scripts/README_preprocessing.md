# Data Preprocessing Guide

This guide explains how to use the data preprocessing script to prepare trading data for backtesting and strategy development.

## Overview

The `preprocess_data.py` script transforms raw OHLCV data into feature-rich datasets suitable for:
- Technical analysis strategies
- Machine learning models
- Advanced backtesting with custom indicators

## Quick Start

### Single Ticker
```bash
python scripts/preprocess_data.py --ticker AAPL
```

### All Tickers
```bash
python scripts/preprocess_data.py --all
```

### Date Range
```bash
python scripts/preprocess_data.py --ticker AAPL --start 2023-01-01 --end 2023-12-31
```

### Different Timeframes
```bash
# Weekly data
python scripts/preprocess_data.py --ticker AAPL --timeframe 1W

# Hourly data  
python scripts/preprocess_data.py --ticker AAPL --timeframe 1H

# Monthly data
python scripts/preprocess_data.py --ticker AAPL --timeframe 1M
```

### Combine Options
```bash
# Process all tickers with weekly timeframe
python scripts/preprocess_data.py --all --timeframe 1W

# Date range with custom timeframe
python scripts/preprocess_data.py --ticker AAPL --start 2023-01-01 --end 2023-12-31 --timeframe 4H
```

## Input Data Format

The script accepts CSV files with flexible column naming:

### Supported Column Names
- **Date**: `Date`, `DateTime`, `Timestamp`, `date`, `datetime`, `timestamp`
- **OHLC**: `Open`/`open`, `High`/`high`, `Low`/`low`, `Close`/`close`
- **Volume**: `Volume`/`volume` (optional, defaults to 0)
- **Ignored**: `ticker`, `rown` (additional columns are ignored)

### Example CSV Formats
```csv
# Standard format
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,99.0,104.0,1000000

# Lowercase format
ticker,rown,date,open,high,low,close,volume
AAPL,1,2023-01-01,100.0,105.0,99.0,104.0,1000000
```

## Timeframe Resampling

The script can resample data to different timeframes:

### Supported Timeframes
- **Intraday**: `1H`, `4H`, `6H`, `12H`
- **Daily**: `1D` (default, no resampling)
- **Weekly**: `1W`
- **Monthly**: `1M`
- **Custom**: Any pandas-compatible frequency string

### Resampling Rules
- **Open**: First value in period
- **High**: Maximum value in period  
- **Low**: Minimum value in period
- **Close**: Last value in period
- **Volume**: Sum of all volumes in period

## Features Generated

The script provides a framework for implementing features across several categories:

### 1. Basic Features
- Price changes (absolute and percentage)
- Daily range (high-low)
- Gap analysis
- Close position within daily range

### 2. Moving Averages
- Simple Moving Averages (SMA): 5, 10, 20, 50, 100, 200 periods
- Exponential Moving Averages (EMA): 12, 26, 50 periods
- Volume Weighted Moving Average (VWMA): 20 periods

### 3. Technical Indicators
- **RSI** (Relative Strength Index): 14 periods
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**: Upper, Lower, Width, Position
- **ATR** (Average True Range): 14 periods
- **Stochastic Oscillator**: %K and %D

### 4. Volume Indicators
- **OBV** (On Balance Volume)
- Volume moving average and ratio
- **MFI** (Money Flow Index): 14 periods

### 5. Custom Features (Student Section)
- Candlestick patterns (bullish, bearish, doji)
- Support and resistance levels
- Volatility measures
- **Your custom features go here!**

### 6. Labels for ML
- Future returns (5-period default)
- Binary classification labels
- Multi-class labels (Buy/Hold/Sell)

## Adding Custom Features

The script includes a dedicated section for students to add their own features:

```python
def add_custom_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    ========================================================================
    STUDENT FEATURE ENGINEERING SECTION
    ========================================================================
    
    Add your custom features here!
    """
    
    # Example: Your custom indicator
    df['My_Custom_Indicator'] = your_calculation_here
    
    return df
```

### Ideas for Custom Features

1. **Pattern Recognition**
   ```python
   # Example: Detect hammer candlestick pattern
   body = abs(df['Close'] - df['Open'])
   lower_shadow = df['Open'].where(df['Close'] >= df['Open'], df['Close']) - df['Low']
   df['Hammer'] = ((lower_shadow > 2 * body) & (df['High'] - df['Close'] < body * 0.3)).astype(int)
   ```

2. **Market Microstructure**
   ```python
   # Example: Spread estimation
   df['Spread_Estimate'] = 2 * np.sqrt(df['Daily_Range'] * df['Volume'])
   ```

3. **Trend Strength**
   ```python
   # Example: ADX (Average Directional Index)
   plus_dm = df['High'].diff()
   minus_dm = -df['Low'].diff()
   # ... continue ADX calculation
   ```

4. **Seasonality**
   ```python
   # Example: Day of week effects
   df['DayOfWeek'] = df.index.dayofweek
   df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
   df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
   ```

## Output Format

Processed files are saved to `data/processed/` with the naming convention:
- `{TICKER}_processed.csv`

Each row contains:
- Original OHLCV data
- All calculated features
- Labels for supervised learning

## Using Processed Data

### With Backtesting
```python
from data.csv_source import CSVDataSource

# Load processed data
data_source = CSVDataSource(data_directory="data/processed")
data = data_source.get_historical_data("AAPL_processed", start, end)
```

### With Custom Strategies
```python
class MLStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Use any of the 50+ features
        if 'RSI_14' in data.columns and 'MACD' in data.columns:
            buy_signal = (data['RSI_14'] < 30) & (data['MACD'] > 0)
            # ... generate signals based on features
```

## Data Quality Notes

1. **Warm-up Period**: Many indicators need historical data to calculate. The script drops rows with NaN values, typically losing:
   - First 199 rows for 200-period SMA
   - First 59 rows for 60-period volatility
   - Adjust your dataset size accordingly

2. **Forward-Looking Bias**: Labels use future data and should only be used for training, not backtesting

3. **Missing Data**: The script forward-fills missing values before dropping remaining NaNs

## Performance Tips

1. **Batch Processing**: Use `--all` flag to process multiple tickers efficiently
2. **Date Filtering**: Use `--start` and `--end` to limit data processing
3. **Custom Directories**: Specify input/output directories for different datasets

## Troubleshooting

### "Missing required columns"
Ensure your CSV has: Date, Open, High, Low, Close columns (Volume is optional)

### "No data after cleaning"
Your dataset is too small. Most features need 200+ rows of data.

### "Memory issues"
Process tickers individually instead of using `--all` for large datasets.

## Example Workflow

1. **Generate sample data** (for testing):
   ```bash
   python scripts/generate_sample_data.py
   ```

2. **Preprocess all tickers**:
   ```bash
   python scripts/preprocess_data.py --all
   ```

3. **Verify output**:
   ```bash
   ls data/processed/
   head data/processed/AAPL_processed.csv
   ```

4. **Use in strategy**:
   - Update your strategy to use the preprocessed features
   - Run backtests with the enriched data

## Next Steps

1. Add your custom features in the designated section
2. Experiment with different feature combinations
3. Use the labels for ML model training
4. Backtest strategies using the enriched features