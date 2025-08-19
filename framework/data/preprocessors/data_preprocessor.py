"""
Data Preprocessing Module for Trading Bot Framework

This module handles preprocessing of raw CSV data and adds technical indicators.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from framework.utils.logger import setup_logger

logger = setup_logger("INFO")


class DataPreprocessor:
    """Main data preprocessing class for adding technical indicators and features"""
    
    def __init__(self, input_dir: str = "data/csv", output_dir: str = "data/processed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure directories exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, ticker: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, timeframe: Optional[str] = None) -> pd.DataFrame:
        """Load raw CSV data"""
        if timeframe and timeframe != '1D':
            file_path = os.path.join(self.input_dir, f"{ticker}_{timeframe}.csv")
        else:
            file_path = os.path.join(self.input_dir, f"{ticker}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Try to identify date column
        date_columns = ['date', 'Date', 'timestamp', 'Timestamp', 'datetime', 'DateTime']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        elif 'date' not in df.columns and df.index.name != 'date':
            # Assume first column is date if no date column found
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
        
        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = []
        
        # Map common column names
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
        }
        
        # Apply mapping
        df.rename(columns=column_mapping, inplace=True)
        
        for col in required_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} rows for {ticker}")
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price and volume features"""
        df = df.copy()
        
        # Price features
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['open_close_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Technical indicators
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df['close'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df['atr_14'] = self.calculate_atr(df, 14)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close})
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        df = df.copy()
        
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
            logger.error(f"Error preprocessing {ticker}: {e}")
            raise
    
    def preprocess_all(self, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[str, str]:
        """Preprocess all CSV files in input directory"""
        results = {}
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(self.input_dir) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.input_dir}")
            return results
        
        logger.info(f"Found {len(csv_files)} CSV files to preprocess")
        
        for csv_file in csv_files:
            # Extract ticker from filename
            ticker = csv_file.replace('.csv', '')
            if timeframe and timeframe != '1D':
                ticker = ticker.replace(f'_{timeframe}', '')
            
            try:
                output_path = self.preprocess_ticker(ticker, start_date, end_date, timeframe)
                results[ticker] = output_path
            except Exception as e:
                logger.error(f"Failed to preprocess {ticker}: {e}")
                results[ticker] = None
        
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Preprocessing complete: {successful}/{len(csv_files)} files successful")
        
        return results