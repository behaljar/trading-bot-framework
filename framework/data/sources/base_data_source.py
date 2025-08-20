"""
Base data source interface
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume and timestamp index
        """
        pass
        
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price as float
        """
        pass
        
    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get order book data
        
        Args:
            symbol: Trading symbol
            limit: Number of bid/ask levels to return
            
        Returns:
            Dictionary with bids, asks, timestamp, and symbol
        """
        pass