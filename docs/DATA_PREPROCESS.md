# Data Preprocessing Guide

This guide covers the data preprocessing pipeline in the trading framework. The preprocessing system automatically cleans and validates OHLCV data files to ensure data quality for backtesting and analysis.

## Overview

The preprocessing system uses a pipeline of specialized processors to clean trading data. It automatically detects and fixes common data quality issues in financial data, making the data suitable for strategy development and backtesting.

**Key Features:**
- Automatic timestamp column detection
- OHLC relationship validation
- NaN and infinite value handling
- Data quality validation
- Time gap detection
- Batch processing capabilities

## Quick Start

### Process a Single File

```bash
# Basic preprocessing
python scripts/preprocess_data.py --input data/raw/BTCUSDT.csv --output data/cleaned/BTCUSDT.csv

# Auto-generate output path
python scripts/preprocess_data.py --input data/raw/BTCUSDT.csv
# Output: data/cleaned/BTCUSDT_cleaned.csv
```

### Process Directory of Files

```bash
# Process all CSV files in a directory
python scripts/preprocess_data.py --input data/raw/ --output data/cleaned/

# Process with custom suffix
python scripts/preprocess_data.py --input data/raw/ --output data/cleaned/ --suffix "_processed"

# Process without adding suffix
python scripts/preprocess_data.py --input data/raw/ --output data/cleaned/ --no-suffix
```

## Command Line Options

### Basic Usage

```bash
python scripts/preprocess_data.py [OPTIONS]
```

### Required Arguments

- `--input, -i PATH`: Path to input CSV file or directory containing CSV files

### Optional Arguments

- `--output, -o PATH`: Path to output CSV file or directory. If not specified, creates "cleaned" subdirectory
- `--suffix, -s STRING`: Suffix to add to output files when processing directory (default: "_cleaned")
- `--timestamp, -t COLUMN`: Name of timestamp column (auto-detected if not specified)
- `--no-suffix`: Do not add suffix to output files (keeps original names)
- `--debug`: Enable debug logging

## Preprocessing Pipeline

The preprocessing system applies the following processors automatically:

### 1. Data Quality Validator
- **Purpose**: Validates overall data structure and identifies quality issues
- **Checks**: Column presence, data types, basic statistics
- **Location**: `framework/data/preprocessors/data_quality_validator_preprocessor.py`

### 2. NaN Value Preprocessor  
- **Purpose**: Handles missing values (NaN) in the dataset
- **Actions**: Identifies and processes NaN values based on context
- **Location**: `framework/data/preprocessors/nan_value_preprocessor.py`

### 3. Infinite Value Preprocessor
- **Purpose**: Handles infinite values that can cause calculation errors
- **Actions**: Identifies and corrects infinite values in price/volume data
- **Location**: `framework/data/preprocessors/infinite_value_preprocessor.py`

### 4. OHLC Validator Preprocessor
- **Purpose**: Validates Open-High-Low-Close relationships
- **Validation Rules**:
  - High ≥ max(Open, Close)
  - Low ≤ min(Open, Close)
  - All prices > 0
- **Location**: `framework/data/preprocessors/ohlc_validator_preprocessor.py`

### 5. Time Gap Detector
- **Purpose**: Identifies gaps and irregularities in timestamp sequences
- **Detection**: Missing periods, irregular intervals, timestamp jumps
- **Location**: `framework/data/preprocessors/time_gap_detector_preprocessor.py`

## Data Requirements

### Input Data Format

The preprocessor expects CSV files with OHLCV data:

**Required Columns** (case-insensitive):
- `timestamp`, `date`, `datetime` (or similar): Date/time information
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price  
- `close`: Closing price
- `volume`: Trading volume

**Example Input:**
```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,16625.08,16625.08,16531.0,16688.74,1200.5
2023-01-02 00:00:00,16688.74,16950.0,16688.74,16830.0,950.2
```

### Timestamp Column Detection

The preprocessor automatically detects timestamp columns using these patterns:
- `timestamp`, `Timestamp`, `TIMESTAMP`
- `date`, `Date`, `DATE`
- `datetime`, `DateTime`, `DATETIME`
- `time`, `Time`, `TIME`
- `index`, `Index`, `INDEX`

If auto-detection fails, specify the column manually:
```bash
python scripts/preprocess_data.py --input data.csv --timestamp "Date"
```

## Output Format

Processed files maintain the same structure with:
- DateTime index properly set
- Clean OHLCV data
- Invalid values corrected or removed
- Consistent column naming

**Example Output:**
```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,16625.08,16625.08,16531.0,16688.74,1200.5
2023-01-02 00:00:00,16688.74,16950.0,16688.74,16830.0,950.2
```

## Automatic Path Management

### Single File Processing

When processing a single file without specifying output:

```bash
# Input: data/raw/BTCUSDT.csv
python scripts/preprocess_data.py --input data/raw/BTCUSDT.csv

# Output: data/cleaned/BTCUSDT_cleaned.csv (auto-generated)
```

### Directory Processing

When processing directories:

```bash
# Input directory: data/raw/
python scripts/preprocess_data.py --input data/raw/

# Output directory: data/cleaned/ (auto-generated)
# Files: file1_cleaned.csv, file2_cleaned.csv, etc.
```

## Usage Examples

### Common Workflows

```bash
# Download and preprocess Bitcoin data
python scripts/download_data.py --source ccxt --symbol BTC/USDT --exchange binance
python scripts/preprocess_data.py --input data/csv/BTC_USDT.csv

# Batch process multiple downloaded files
python scripts/preprocess_data.py --input data/csv/ --output data/cleaned/

# Process with specific timestamp column
python scripts/preprocess_data.py --input data/csv/custom_data.csv --timestamp "Date"

# Process with debug output
python scripts/preprocess_data.py --input data/csv/BTCUSDT.csv --debug
```

### Integration with Backtesting

```bash
# Complete workflow: Download → Preprocess → Backtest
python scripts/download_data.py --source ccxt --symbol BTC/USDT --start 2023-01-01
python scripts/preprocess_data.py --input data/csv/BTC_USDT.csv
uv run python scripts/run_backtest.py --strategy sma --data-file data/cleaned/BTC_USDT_cleaned.csv --symbol BTC_USDT
```

## Preprocessing Manager

The preprocessing pipeline is managed by the `preprocessor_manager.py` module:

```python
from framework.data.preprocessors.preprocessor_manager import create_default_manager

# Create default preprocessing pipeline
manager = create_default_manager()

# Run preprocessing on DataFrame
cleaned_df = manager.run(df)
```

### Custom Preprocessing

For programmatic use, you can create custom preprocessing pipelines:

```python
from framework.data.preprocessors.base_preprocessor import BasePreprocessor

class CustomPreprocessor(BasePreprocessor):
    def preprocess(self, df):
        # Your custom logic here
        return df
```

## Logging and Debug Output

### Standard Logging

The preprocessor provides detailed logging during execution:

```
INFO - Loading file: data/raw/BTCUSDT.csv
INFO - Loaded 8760 rows, 6 columns
INFO - Auto-detected timestamp column: 'timestamp'
INFO - Set 'timestamp' as datetime index
INFO - Applying preprocessing pipeline...
INFO - Processing complete:
INFO -   Input: 8760 rows, 6 columns
INFO -   Output: 8645 rows, 6 columns
INFO -   Rows removed: 115
INFO -   Saved to: data/cleaned/BTCUSDT_cleaned.csv
```

### Debug Mode

Enable detailed debug information:

```bash
python scripts/preprocess_data.py --input data.csv --debug
```

Debug mode provides:
- Detailed processing steps for each preprocessor
- Data quality metrics and validation results
- Row-by-row processing information
- Performance timing information

## Error Handling

### Common Issues and Solutions

**File Not Found:**
```
Error: Input path does not exist: data/raw/file.csv
```
Solution: Verify the file path is correct

**Invalid CSV Format:**
```
Error: Input file must be a CSV file: data/raw/file.txt
```
Solution: Ensure input files have .csv extension

**Timestamp Column Issues:**
```
Warning: Could not identify or create datetime index
```
Solution: Specify timestamp column manually with `--timestamp`

**No CSV Files Found:**
```
Warning: No CSV files found in data/raw/
```
Solution: Verify directory contains CSV files

### Processing Failures

If processing fails for individual files:
- Check the debug output for specific error details
- Verify data format and column names
- Ensure sufficient data quality for preprocessing

The script will continue processing other files and provide a summary of successful/failed files.

## Best Practices

### Data Quality

1. **Always preprocess before backtesting**: Ensures clean, validated data
2. **Keep raw data**: Preserve original files before preprocessing
3. **Use debug mode**: For troubleshooting data quality issues
4. **Batch processing**: Process multiple files efficiently

### File Organization

```
data/
├── raw/           # Original downloaded files
├── csv/           # Direct downloads (legacy)
├── cleaned/       # Preprocessed files ready for backtesting
└── processed/     # Custom processed files
```

### Workflow Integration

1. **Download**: Use `download_data.py` to get raw data
2. **Preprocess**: Use `preprocess_data.py` to clean data  
3. **Backtest**: Use `run_backtest.py` with cleaned data
4. **Analysis**: Use clean data for strategy development

This preprocessing pipeline ensures your trading data is reliable and ready for analysis and backtesting.