"""
Fixed position size risk manager implementation.

This module provides a simple risk manager that always uses
a fixed position size regardless of other factors.
"""

from typing import Optional
from .base_risk_manager import BaseRiskManager


class FixedPositionSizeManager(BaseRiskManager):
    """
    Risk manager that uses a fixed position size for all trades.
    
    This is the simplest risk management strategy where every trade
    uses the same fraction of available equity.
    """
    
    def __init__(self, position_size: float = 0.1, **params):
        """
        Initialize the fixed position size manager.
        
        Args:
            position_size: Fixed position size as fraction of equity (default: 0.1 = 10%)
            **params: Additional parameters passed to base class
        """
        super().__init__(position_size=position_size, **params)
        self.position_size = position_size
        
        if not 0 < self.position_size <= 1.0:
            raise ValueError(f"Position size must be between 0 and 1, got {self.position_size}")
    
    def calculate_position_size(
        self,
        signal: int,
        current_price: float,
        equity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate the position size for a given trade.
        
        Always returns the fixed position size regardless of other parameters.
        
        Args:
            signal: Trading signal (1 for long, -1 for short, 0 for no position)
            current_price: Current asset price
            equity: Current account equity
            stop_loss: Stop loss price (ignored)
            take_profit: Take profit price (ignored)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Fixed position size as a fraction of equity
        """
        if signal == 0:
            return 0.0
        
        return self.position_size
    
    def get_description(self) -> str:
        """
        Get a description of this risk manager.
        
        Returns:
            String description of the risk manager
        """
        return f"Fixed Position Size Manager (size={self.position_size:.1%})"