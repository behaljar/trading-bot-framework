# CSV Data Directory

Place your CSV files with historical price data in this directory for backtesting.

## File Format

CSV files should have the following structure:

### Required Columns
- **Date/DateTime/Timestamp**: Date column (various formats supported)
- **Open**: Opening price
- **High**: Highest price  
- **Low**: Lowest price
- **Close**: Closing price

### Optional Columns
- **Volume**: Trading volume (will default to 0 if missing)
- **Adj Close**: Adjusted closing price

### Example CSV Format

```csv
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,99.0,104.0,1000000
2023-01-02,104.0,108.0,103.0,107.0,1200000
2023-01-03,107.0,110.0,106.0,109.0,800000
```

## File Naming Convention

- Name your CSV files with the symbol name (e.g., `AAPL.csv`, `BTCUSDT.csv`)
- The symbol in your configuration should match the filename (without .csv extension)

## Supported Date Formats

The CSV data source automatically detects and converts various date formats:
- `2023-01-01`
- `2023-01-01 09:30:00`
- `01/01/2023`
- Unix timestamps

## Usage

Set your environment variables:

```bash
DATA_SOURCE=csv
CSV_DATA_DIRECTORY=data/csv
SYMBOLS=AAPL,MSFT,GOOGL  # Should match your CSV filenames
```

The CSVDataSource will automatically scan this directory for `.csv` files and make them available for backtesting.