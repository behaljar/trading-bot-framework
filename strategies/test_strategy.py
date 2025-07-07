"""
Simple test strategy for order handling verification
"""
import pandas as pd
import logging
from .base_strategy import BaseStrategy, Signal


class TestStrategy(BaseStrategy):
    """
    Simple test strategy that alternates between buy and sell
    - If no position: BUY
    - If in position: SELL
    
    This is purely for testing order execution and should NOT be used for real trading
    """
    
    def __init__(self, params: dict = None):
        """
        Initialize test strategy
        
        Args:
            params: Strategy parameters (not used for this simple strategy)
        """
        super().__init__(params or {})
        self.min_bars_required = 2  # Very minimal requirement
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate alternating signals for testing
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Series with signals (-1, 0, 1)
        """
        if len(data) < self.min_bars_required:
            self.logger.warning(f"Insufficient data: {len(data)} bars, need {self.min_bars_required}")
            return pd.Series([Signal.HOLD.value] * len(data), index=data.index)
            
        # Create signals series
        signals = pd.Series([Signal.HOLD.value] * len(data), index=data.index)
        
        # For testing: always generate BUY signal on every new bar
        # The trader will handle position checking and decide whether to buy or sell
        signals.iloc[-1] = Signal.BUY.value
        
        return signals
        
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "TestStrategy"
        
    def get_position_size(self, symbol: str, signal: float, capital: float) -> float:
        """
        Calculate position size for testing
        
        Args:
            symbol: Trading symbol
            signal: Signal value
            capital: Available capital
            
        Returns:
            Position size (small for testing)
        """
        # Use very small position size for testing (1% of capital)
        if signal != Signal.HOLD.value:
            return capital * 0.01  # 1% of capital
        return 0.0
        
    def should_enter_position(self, data: pd.DataFrame) -> bool:
        """Check if we should enter a position"""
        # Always try to enter for testing
        return True
        
    def should_exit_position(self, data: pd.DataFrame, entry_price: float, current_price: float) -> bool:
        """Check if we should exit a position"""
        # For this test strategy, we'll exit based on the alternating logic
        # This will be handled by the main trading logic
        return False