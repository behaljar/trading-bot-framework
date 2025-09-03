"""
Strategy Position Size Risk Manager

This risk manager respects the position sizes calculated by the strategy,
allowing strategies to implement their own sophisticated position sizing logic
based on factors like signal strength, confluence, volatility, etc.
"""

from typing import Optional
from .base_risk_manager import BaseRiskManager


class StrategyPositionSizeManager(BaseRiskManager):
    """
    Risk manager that uses position sizes calculated by the strategy.
    
    This manager allows strategies to control their own position sizing
    while providing optional safety constraints and limits.
    """
    
    def __init__(self, 
                 max_position_size: float = 1.0,
                 min_position_size: float = 0.001,
                 apply_safety_limits: bool = True,
                 **params):
        """
        Initialize the strategy position size manager.
        
        Args:
            max_position_size: Maximum allowed position size as fraction of equity (default: 1.0 = 100%)
            min_position_size: Minimum position size threshold (default: 0.001 = 0.1%)
            apply_safety_limits: Whether to apply min/max limits (default: True)
            **params: Additional parameters passed to base class
        """
        super().__init__(
            max_position_size=max_position_size,
            min_position_size=min_position_size,
            apply_safety_limits=apply_safety_limits,
            **params
        )
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.apply_safety_limits = apply_safety_limits
        
        if not 0 < self.max_position_size <= 2.0:
            raise ValueError(f"Max position size must be between 0 and 2.0 (200%), got {self.max_position_size}")
        if not 0 < self.min_position_size < self.max_position_size:
            raise ValueError(f"Min position size must be between 0 and max_position_size, got {self.min_position_size}")
    
    def calculate_position_size(
        self,
        signal: int,
        current_price: float,
        equity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_position_size: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Use the position size calculated by the strategy with optional safety limits.
        
        Args:
            signal: Trading signal (1 for long, -1 for short, 0 for no position)
            current_price: Current asset price
            equity: Current account equity
            stop_loss: Stop loss price (optional, used for validation)
            take_profit: Take profit price (optional, used for validation)
            strategy_position_size: Position size calculated by strategy
            **kwargs: Additional parameters
            
        Returns:
            Position size as calculated by strategy (with optional limits applied)
        """
        if signal == 0:
            return 0.0
        
        # Use strategy's position size if provided
        if strategy_position_size is not None and strategy_position_size > 0:
            position_size = strategy_position_size
            
            # Apply safety limits if enabled
            if self.apply_safety_limits:
                if position_size < self.min_position_size:
                    position_size = 0.0  # Below minimum threshold, reject trade
                elif position_size > self.max_position_size:
                    position_size = self.max_position_size  # Cap at maximum
            
            return position_size
        
        # Fallback: if strategy doesn't provide position size, return 0 (no trade)
        return 0.0
    
    def get_description(self) -> str:
        """
        Get a description of this risk manager.
        
        Returns:
            String description of the risk manager
        """
        if self.apply_safety_limits:
            return (f"Strategy Position Size Manager "
                   f"(min={self.min_position_size:.1%}, max={self.max_position_size:.1%})")
        else:
            return "Strategy Position Size Manager (no limits)"