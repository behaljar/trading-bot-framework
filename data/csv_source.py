"""
CSV data source implementation for backtesting
"""
import pandas as pd
import os
from typing import Dict, Any, List
from datetime import datetime
from .base_data_source import DataSource


class CSVDataSource(DataSource):
    """CSV data source for backtesting with local files"""

    def __init__(self, data_directory: str = "data/csv"):
        """
        Initialize CSV data source
        
        Args:
            data_directory: Directory containing CSV files
        """
        self.data_directory = data_directory
        self.loaded_data = {}  # Cache for loaded data
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created data directory: {self.data_directory}")
        
        # Scan for available CSV files
        self.available_files = self._scan_csv_files()
        print(f"CSV data source initialized with {len(self.available_files)} files")

    def _scan_csv_files(self) -> Dict[str, str]:
        """Scan directory for CSV files and map symbols to file paths"""
        files = {}
        
        if not os.path.exists(self.data_directory):
            return files
            
        for filename in os.listdir(self.data_directory):
            if filename.endswith('.csv'):
                # Extract symbol from filename (remove .csv extension)
                symbol = filename[:-4].upper()
                files[symbol] = os.path.join(self.data_directory, filename)
                
        return files

    def get_available_symbols(self) -> List[str]:
        """Returns list of available symbols from CSV files"""
        return list(self.available_files.keys())

    def _load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load and standardize CSV file format"""
        try:
            # Try to read the CSV file
            df = pd.read_csv(file_path)
            
            # Common column name mappings
            column_mappings = {
                'date': 'Date',
                'datetime': 'Date', 
                'timestamp': 'Date',
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adj close': 'Adj Close',
                'adj_close': 'Adj Close'
            }
            
            # Normalize column names
            df.columns = [col.strip() for col in df.columns]
            df_lower = df.columns.str.lower()
            
            # Apply mappings
            new_columns = {}
            for old_col, lower_col in zip(df.columns, df_lower):
                if lower_col in column_mappings:
                    new_columns[old_col] = column_mappings[lower_col]
                    
            df = df.rename(columns=new_columns)
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Handle date column
            date_columns = ['Date', 'Datetime', 'Timestamp']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
                    
            if date_col is None:
                # Try to use index if it looks like a date
                if df.index.name and any(word in df.index.name.lower() for word in ['date', 'time']):
                    df = df.reset_index()
                    date_col = df.columns[0]
                else:
                    raise ValueError("No date column found. Expected columns: Date, Datetime, or Timestamp")
            
            # Convert date column to datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.index.name = 'timestamp'
            
            # Ensure numeric columns are numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add Volume column if missing
            if 'Volume' not in df.columns:
                df['Volume'] = 0
                
            # Sort by date
            df = df.sort_index()
            
            # Remove any rows with NaN in OHLC data
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            # Store information about additional feature columns
            feature_columns = []
            for col in df.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    feature_columns.append(col)
            
            if feature_columns:
                print(f"Found additional feature columns: {', '.join(feature_columns)}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading CSV file {file_path}: {e}")

    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Get historical data from CSV file
        
        Args:
            symbol: Trading symbol (should match CSV filename without extension)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Timeframe (ignored for CSV, returns raw data)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check if symbol is available
            if symbol not in self.available_files:
                # Try to find similar symbols
                available = self.get_available_symbols()
                similar = [s for s in available if symbol.lower() in s.lower() or s.lower() in symbol.lower()]
                
                if similar:
                    raise ValueError(f"Symbol {symbol} not found. Did you mean: {', '.join(similar)}?")
                else:
                    raise ValueError(f"Symbol {symbol} not found. Available symbols: {', '.join(available[:10])}")
            
            # Load data if not cached
            if symbol not in self.loaded_data:
                file_path = self.available_files[symbol]
                self.loaded_data[symbol] = self._load_csv_file(file_path)
                
            data = self.loaded_data[symbol].copy()
            
            # Filter by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
                
            return data
            
        except Exception as e:
            print(f"Error getting historical data from CSV for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get the most recent price from CSV data"""
        try:
            if symbol not in self.available_files:
                return 0.0
                
            # Load data if not cached
            if symbol not in self.loaded_data:
                file_path = self.available_files[symbol]
                self.loaded_data[symbol] = self._load_csv_file(file_path)
                
            data = self.loaded_data[symbol]
            if data.empty:
                return 0.0
                
            # Return the most recent close price
            return float(data['Close'].iloc[-1])
            
        except Exception as e:
            print(f"Error getting current price from CSV for {symbol}: {e}")
            return 0.0

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """CSV data doesn't contain order book information"""
        return {"bids": [], "asks": [], "timestamp": None}

    def refresh_available_files(self):
        """Refresh the list of available CSV files"""
        self.available_files = self._scan_csv_files()
        # Clear cache to force reload
        self.loaded_data = {}
        
    def add_csv_file(self, symbol: str, file_path: str):
        """Add a specific CSV file for a symbol"""
        if os.path.exists(file_path):
            self.available_files[symbol.upper()] = file_path
            # Remove from cache to force reload
            if symbol.upper() in self.loaded_data:
                del self.loaded_data[symbol.upper()]
        else:
            raise FileNotFoundError(f"CSV file not found: {file_path}")