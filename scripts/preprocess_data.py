#!/usr/bin/env python3
"""
Script to preprocess and clean trading data files.
Applies a comprehensive data cleaning pipeline to CSV files containing OHLCV data.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from framework.data.preprocessors.preprocessor_manager import create_default_manager
from framework.utils.logger import setup_logger, get_logger


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = "DEBUG" if debug else "INFO"
    # Setup the main logger
    setup_logger(log_level=level, use_json=False)
    
    # Also configure the root logger to ensure preprocessor logs are shown
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(levelname)s - %(name)s - %(message)s',
        force=True
    )
    
    # Return a logger for this script
    return get_logger("TradingBot.PreprocessData")


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """
    Automatically detect the timestamp column in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Name of the timestamp column or None if not found
    """
    # Common timestamp column names
    timestamp_candidates = [
        'timestamp', 'Timestamp', 'TIMESTAMP',
        'date', 'Date', 'DATE',
        'datetime', 'DateTime', 'DATETIME',
        'time', 'Time', 'TIME',
        'index', 'Index', 'INDEX'
    ]
    
    for col in timestamp_candidates:
        if col in df.columns:
            return col
    
    # Check if index is already a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return None  # Already using datetime index
    
    # Try to find columns that look like dates
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Try to parse first few values as dates
                sample = df[col].head(5)
                pd.to_datetime(sample)
                return col
            except:
                continue
    
    return None


def load_csv_file(file_path: Path, timestamp_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV file and prepare it for preprocessing.
    
    Args:
        file_path: Path to the CSV file
        timestamp_column: Name of timestamp column (auto-detected if None)
        
    Returns:
        DataFrame with datetime index
    """
    logging.info(f"Loading file: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Handle timestamp column
    if timestamp_column:
        if timestamp_column not in df.columns:
            raise ValueError(f"Specified timestamp column '{timestamp_column}' not found in file")
        timestamp_col = timestamp_column
    else:
        timestamp_col = detect_timestamp_column(df)
        if timestamp_col:
            logging.info(f"Auto-detected timestamp column: '{timestamp_col}'")
    
    # Set datetime index if timestamp column found
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.set_index(timestamp_col, inplace=True)
        logging.info(f"Set '{timestamp_col}' as datetime index")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse the index as datetime
        try:
            df.index = pd.to_datetime(df.index)
            logging.info("Converted index to datetime")
        except:
            logging.warning("Could not identify or create datetime index")
    
    return df


def process_file(
    input_path: Path,
    output_path: Path,
    timestamp_column: Optional[str] = None
) -> bool:
    """
    Process a single CSV file with the data cleaning pipeline.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        timestamp_column: Name of timestamp column (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the CSV file
        df = load_csv_file(input_path, timestamp_column)
        
        # Get initial stats
        initial_rows = len(df)
        initial_cols = len(df.columns)
        
        # Apply preprocessing pipeline
        logging.info("Applying preprocessing pipeline...")
        manager = create_default_manager()
        df_clean = manager.run(df)
        
        # Get final stats
        final_rows = len(df_clean)
        final_cols = len(df_clean.columns)
        
        # Save cleaned data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path)
        
        # Log summary
        logging.info(f"Processing complete:")
        logging.info(f"  Input: {initial_rows} rows, {initial_cols} columns")
        logging.info(f"  Output: {final_rows} rows, {final_cols} columns")
        logging.info(f"  Rows removed: {initial_rows - final_rows}")
        logging.info(f"  Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    suffix: str = "_cleaned",
    timestamp_column: Optional[str] = None
) -> tuple[int, int]:
    """
    Process all CSV files in a directory.
    
    Args:
        input_dir: Input directory containing CSV files
        output_dir: Output directory for cleaned files
        suffix: Suffix to add to output filenames
        timestamp_column: Name of timestamp column (optional)
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        logging.warning(f"No CSV files found in {input_dir}")
        return 0, 0
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        # Determine output filename
        if suffix:
            output_name = f"{csv_file.stem}{suffix}.csv"
        else:
            output_name = csv_file.name
        
        output_path = output_dir / output_name
        
        logging.info(f"\nProcessing {csv_file.name}...")
        if process_file(csv_file, output_path, timestamp_column):
            successful += 1
        else:
            failed += 1
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Clean and preprocess trading data files using a comprehensive pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python scripts/preprocess_data.py --input data/raw/BTCUSDT.csv --output data/cleaned/BTCUSDT.csv
  
  # Process all files in a directory
  python scripts/preprocess_data.py --input data/raw/ --output data/cleaned/
  
  # Process with auto-generated output names
  python scripts/preprocess_data.py --input data/raw/BTCUSDT.csv
  
  # Specify timestamp column
  python scripts/preprocess_data.py --input data/raw/ --output data/cleaned/ --timestamp date
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file or directory containing CSV files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output CSV file or directory. If not specified, creates "cleaned" subdirectory'
    )
    
    parser.add_argument(
        '--suffix', '-s',
        type=str,
        default='_cleaned',
        help='Suffix to add to output files when processing directory (default: _cleaned)'
    )
    
    parser.add_argument(
        '--timestamp', '-t',
        type=str,
        help='Name of timestamp column (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--no-suffix',
        action='store_true',
        help='Do not add suffix to output files (keeps original names)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Parse input path
    input_path = Path(args.input)
    
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Handle suffix option
    suffix = "" if args.no_suffix else args.suffix
    
    # Process based on input type
    if input_path.is_file():
        # Single file processing
        if not input_path.suffix.lower() == '.csv':
            logging.error(f"Input file must be a CSV file: {input_path}")
            sys.exit(1)
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Create output in "cleaned" directory at the same level as parent directory
            # e.g., if input is data/raw/file.csv, output goes to data/cleaned/file.csv
            if input_path.parent.name == 'raw':
                # Replace 'raw' with 'cleaned' in the path
                output_dir = input_path.parent.parent / "cleaned"
            else:
                # If not in 'raw' folder, create 'cleaned' at same level
                output_dir = input_path.parent.parent / "cleaned"
            
            output_dir.mkdir(exist_ok=True, parents=True)
            if suffix:
                output_path = output_dir / f"{input_path.stem}{suffix}.csv"
            else:
                output_path = output_dir / input_path.name
        
        # Process the file
        logging.info(f"Processing single file: {input_path}")
        success = process_file(input_path, output_path, args.timestamp)
        
        if success:
            logging.info("✓ File processing completed successfully")
            sys.exit(0)
        else:
            logging.error("✗ File processing failed")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Directory processing
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            # Create "cleaned" directory at the same level as input directory
            # e.g., if input is data/raw/, output goes to data/cleaned/
            if input_path.name == 'raw':
                # Replace 'raw' with 'cleaned'
                output_dir = input_path.parent / "cleaned"
            else:
                # If not 'raw' folder, create 'cleaned' at same level
                output_dir = input_path.parent / f"{input_path.name}_cleaned"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Processing directory: {input_path}")
        logging.info(f"Output directory: {output_dir}")
        
        # Process all CSV files
        successful, failed = process_directory(
            input_path,
            output_dir,
            suffix,
            args.timestamp
        )
        
        # Summary
        total = successful + failed
        logging.info("\n" + "="*50)
        logging.info(f"Processing complete: {successful}/{total} files successful")
        
        if failed > 0:
            logging.warning(f"{failed} files failed to process")
            sys.exit(1)
        else:
            logging.info("✓ All files processed successfully")
            sys.exit(0)
    
    else:
        logging.error(f"Invalid input path: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()