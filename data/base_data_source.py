"""
Abstract base class for data sources
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class DataSource(ABC):
    """Abstract class for data sources"""

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Timeframe (e.g., '1d', '1h', '5m')
            
        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price as float
        """
        pass

    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get order book data for a symbol
        
        Args:
            symbol: Trading symbol
            limit: Number of order book levels
            
        Returns:
            Dictionary with bids, asks, and timestamp
        """
        pass