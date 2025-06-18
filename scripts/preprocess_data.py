#!/usr/bin/env python3
"""
Data Preprocessing Script for Trading Bot Framework

This script preprocesses raw CSV data and adds technical indicators and custom features.
Students will implement features manually in the designated sections.

Usage:
    python scripts/preprocess_data.py --ticker AAPL
    python scripts/preprocess_data.py --ticker AAPL --start 2022-01-01 --end 2023-12-31
    python scripts/preprocess_data.py --all  # Process all CSV files in data/csv/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

# Setup logger
logger = setup_logger("INFO")


class DataPreprocessor:
    """Data preprocessing class with technical indicators and feature engineering"""
    
    def __init__(self, input_dir: str = "data/csv", output_dir: str = "data/processed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self, ticker: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, timeframe: Optional[str] = None) -> pd.DataFrame:
        """Load CSV data for a given ticker"""
        filepath = os.path.join(self.input_dir, f"{ticker}.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Normalize column names to handle both uppercase and lowercase
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['date', 'datetime', 'timestamp']:
                column_mapping[col] = 'Date'
            elif col_lower == 'open':
                column_mapping[col] = 'Open'
            elif col_lower == 'high':
                column_mapping[col] = 'High'
            elif col_lower == 'low':
                column_mapping[col] = 'Low'
            elif col_lower == 'close':
                column_mapping[col] = 'Close'
            elif col_lower == 'volume':
                column_mapping[col] = 'Volume'
            elif col_lower in ['ticker', 'rown']:  # Additional columns to ignore
                continue
        
        # Rename columns to standard format
        df.rename(columns=column_mapping, inplace=True)
        
        # Find and parse date column
        date_columns = ['Date', 'DateTime', 'Timestamp']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
                
        if date_col is None:
            raise ValueError(f"No date column found in {filepath}. Available columns: {list(df.columns)}")
        
        # Parse dates and set as index
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        
        # Filter by date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
        
        # Add Volume if missing
        if 'Volume' not in df.columns:
            df['Volume'] = 0
            
        # Resample to different timeframe if specified
        if timeframe and timeframe != '1D':
            df = self.resample_data(df, timeframe)
            logger.info(f"Resampled data to {timeframe} timeframe")
            
        logger.info(f"Loaded {len(df)} rows of data for {ticker}")
        return df
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframes
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Target timeframe (e.g., '1H', '4H', '1D', '1W', '1M')
            
        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling data to {timeframe}")
        
        # Define aggregation rules for OHLCV data
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Remove any columns not in OHLCV for resampling
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in ohlcv_cols if col in df.columns]
        
        # Create aggregation dict only for available columns
        resample_agg = {col: agg_dict[col] for col in available_cols}
        
        # Resample the data
        resampled = df[available_cols].resample(timeframe).agg(resample_agg)
        
        # Remove rows with NaN (incomplete periods)
        resampled = resampled.dropna()
        
        return resampled
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with basic features added
        """
        logger.info("Adding basic features...")

        return df

    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare final dataset"""
        logger.info("Cleaning data...")
        
        # Remove any infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Forward fill NaN values (common for indicators that need warm-up period)
        df.fillna(method='ffill', inplace=True)
        
        # Drop rows with any remaining NaN values
        initial_len = len(df)
        df.dropna(inplace=True)
        
        if len(df) < initial_len:
            logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        return df
    
    def save_data(self, df: pd.DataFrame, ticker: str, timeframe: Optional[str] = None) -> str:
        """Save preprocessed data"""
        if timeframe and timeframe != '1D':
            output_path = os.path.join(self.output_dir, f"{ticker}_{timeframe}_processed.csv")
        else:
            output_path = os.path.join(self.output_dir, f"{ticker}_processed.csv")
        df.to_csv(output_path)
        logger.info(f"Saved preprocessed data to {output_path}")
        return output_path
    
    def preprocess_ticker(self, ticker: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, timeframe: Optional[str] = None) -> str:
        """Main preprocessing pipeline for a single ticker"""
        logger.info(f"Starting preprocessing for {ticker}")
        
        try:
            # Load data
            df = self.load_data(ticker, start_date, end_date, timeframe)
            
            # Add features in sequence
            df = self.add_basic_features(df)
            
            # Clean data
            df = self.clean_data(df)
            
            # Save processed data
            output_path = self.save_data(df, ticker, timeframe)
            
            # Print summary
            logger.info(f"Preprocessing complete for {ticker}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Features: {len(df.columns)} columns")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error preprocessing {ticker}: {str(e)}")
            raise
    
    def preprocess_all(self, timeframe: Optional[str] = None) -> Dict[str, str]:
        """Preprocess all CSV files in the input directory"""
        csv_files = [f for f in os.listdir(self.input_dir) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.input_dir}")
            return {}
        
        results = {}
        for csv_file in csv_files:
            ticker = csv_file.replace('.csv', '')
            try:
                output_path = self.preprocess_ticker(ticker, timeframe=timeframe)
                results[ticker] = output_path
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {str(e)}")
                results[ticker] = None
                
        return results


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Preprocess trading data with technical indicators and custom features'
    )
    parser.add_argument('--ticker', type=str, help='Ticker symbol to process')
    parser.add_argument('--all', action='store_true', help='Process all CSV files')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--input-dir', type=str, default='data/csv',
                       help='Input directory for CSV files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for processed files')
    parser.add_argument('--timeframe', type=str, 
                       help='Resample to timeframe (e.g., 1H, 4H, 1D, 1W, 1M)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ticker and not args.all:
        parser.error('Either --ticker or --all must be specified')
    
    if args.ticker and args.all:
        parser.error('Cannot specify both --ticker and --all')
    
    # Create preprocessor
    preprocessor = DataPreprocessor(args.input_dir, args.output_dir)
    
    # Process data
    if args.ticker:
        output_path = preprocessor.preprocess_ticker(args.ticker, args.start, args.end, args.timeframe)
        print(f"\nProcessed data saved to: {output_path}")
    else:
        results = preprocessor.preprocess_all(args.timeframe)
        print("\nProcessing results:")
        for ticker, path in results.items():
            status = "SUCCESS" if path else "FAILED"
            print(f"  {ticker}: {status}")
            if path:
                print(f"    -> {path}")


if __name__ == "__main__":
    main()