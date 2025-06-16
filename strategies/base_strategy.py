"""
Abstract base class for trading strategies
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Tuple
from enum import Enum

class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class BaseStrategy(ABC):
    """Abstract base class for all strategies"""

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.positions = {}  # Current positions for each symbol
        self.signals_history = []  # Signal history

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on data

        Args:
            data: DataFrame with OHLCV data and optionally additional feature columns

        Returns:
            DataFrame with columns:
            - signal: int (1=BUY, -1=SELL, 0=HOLD)
            - stop_loss: float (optional, stop loss price)
            - take_profit: float (optional, take profit price)
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Returns strategy name"""
        pass

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators to the dataset
        Can be overridden in subclasses to add custom indicators
        
        Note: If your CSV already contains pre-calculated indicators,
        they will be preserved and available in the generate_signals method
        """
        return data
    
    def get_feature_columns(self, data: pd.DataFrame) -> list:
        """
        Returns list of additional feature columns beyond OHLCV
        """
        standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return [col for col in data.columns if col not in standard_columns]

    def get_position_size(self, symbol: str, signal: Signal, capital: float) -> float:
        """
        Determines position size
        Can be overridden for more sophisticated position sizing
        """
        if signal == Signal.HOLD:
            return 0.0
        return capital * 0.1  # Default 10% of capital