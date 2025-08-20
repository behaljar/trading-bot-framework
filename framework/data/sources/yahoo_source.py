"""
Yahoo Finance data source implementation
"""
import pandas as pd
import yfinance as yf
from typing import Dict, Any
from .base_data_source import DataSource


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source for stocks"""

    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
        """Downloads historical data from Yahoo Finance"""
        try:
            # Yahoo Finance has different timeframe formats
            yf_intervals = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1wk", "1M": "1mo"
            }
            interval = yf_intervals.get(timeframe, "1d")

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            # Standardize column names to lowercase
            if not data.empty:
                # Rename columns to lowercase
                data = data.rename(columns={
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                data.index.name = 'timestamp'

            return data
        except Exception as e:
            print(f"Error downloading data from Yahoo Finance for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Gets current price"""
        try:
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d")
            # Use lowercase column name
            if 'Close' in current_data.columns:
                return float(current_data['Close'].iloc[-1])
            elif 'close' in current_data.columns:
                return float(current_data['close'].iloc[-1])
            else:
                return 0.0
        except Exception as e:
            print(f"Error getting current price from Yahoo Finance for {symbol}: {e}")
            return 0.0

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Yahoo Finance does not support order book data"""
        return {"bids": [], "asks": [], "timestamp": None}