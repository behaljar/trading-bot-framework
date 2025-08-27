"""
Fixed risk percentage risk manager implementation.

This module provides a risk manager that calculates position sizes
based on a fixed percentage of account equity to risk per trade.
"""

from typing import Optional
from .base_risk_manager import BaseRiskManager


class FixedRiskManager(BaseRiskManager):
    """
    Risk manager that calculates position size based on fixed risk percentage.
    
    This manager ensures that each trade risks only a fixed percentage of
    the account equity by adjusting position size based on the stop loss distance.
    Position sizes can exceed 100% of equity if the stop loss is tight enough
    (useful for leveraged trading).
    """
    
    def __init__(self, risk_percent: float = 0.01, 
                 default_stop_distance: float = 0.02, **params):
        """
        Initialize the fixed risk manager.
        
        Args:
            risk_percent: Percentage of equity to risk per trade (default: 0.01 = 1%)
            default_stop_distance: Default stop loss distance as percentage if no stop loss provided (default: 0.02 = 2%)
            **params: Additional parameters passed to base class
        """
        super().__init__(risk_percent=risk_percent,
                        default_stop_distance=default_stop_distance, **params)
        self.risk_percent = risk_percent
        self.default_stop_distance = default_stop_distance
        
        if not 0 < self.risk_percent <= 0.1:
            raise ValueError(f"Risk percent must be between 0 and 0.1 (10%), got {self.risk_percent}")
        if not 0 < self.default_stop_distance <= 0.5:
            raise ValueError(f"Default stop distance must be between 0 and 0.5 (50%), got {self.default_stop_distance}")
    
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
        Calculate the position size based on fixed risk percentage.
        
        Position size is calculated to ensure that if the stop loss is hit,
        the loss will equal the specified risk percentage of equity.
        No maximum position size limit - can exceed 100% of equity if
        stop loss is tight enough (for leveraged trading).
        
        Args:
            signal: Trading signal (1 for long, -1 for short, 0 for no position)
            current_price: Current asset price
            equity: Current account equity
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            **kwargs: Additional parameters
            
        Returns:
            Position size as a fraction of equity or
            absolute number of units if calculation results in units >= 1.0
        """
        if signal == 0:
            return 0.0
        
        # Calculate risk amount in currency
        risk_amount = equity * self.risk_percent
        
        # Determine stop loss distance
        if stop_loss and stop_loss > 0:
            if signal == 1:  # Long position
                risk_per_share = current_price - stop_loss
            else:  # Short position (signal == -1)
                risk_per_share = stop_loss - current_price
            
            if risk_per_share <= 0:
                # Invalid stop loss (would result in immediate loss)
                return 0.0
            
            # Calculate exact number of units to risk the specified percentage
            units = risk_amount / risk_per_share
            
            # If units >= 1, return absolute unit count for backtesting.py
            if units >= 1.0:
                return round(units)
            
            # Otherwise, convert to fraction of equity
            position_value = units * current_price
            return position_value / equity
            
        else:
            # No stop loss provided, use default stop distance
            # Position fraction = risk_percent / default_stop_distance
            return self.risk_percent / self.default_stop_distance
    
    def get_description(self) -> str:
        """
        Get a description of this risk manager.
        
        Returns:
            String description of the risk manager
        """
        return (f"Fixed Risk Manager (risk={self.risk_percent:.1%}, "
                f"default_stop={self.default_stop_distance:.1%})")