"""
Base risk manager class for position sizing and risk management.

This module provides the abstract base class for all risk managers
in the trading framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseRiskManager(ABC):
    """
    Abstract base class for all risk management strategies.
    
    Risk managers determine position sizes based on various criteria
    such as fixed size, fixed risk percentage, Kelly criterion, etc.
    """
    
    def __init__(self, **params):
        """
        Initialize the risk manager with optional parameters.
        
        Args:
            **params: Risk manager specific parameters
        """
        self.params = params
    
    @abstractmethod
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
        
        Args:
            signal: Trading signal (1 for long, -1 for short, 0 for no position)
            current_price: Current asset price
            equity: Current account equity
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Position size as a fraction of equity (0.0 to 1.0) or
            absolute number of units if >= 1.0
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of this risk manager.
        
        Returns:
            String description of the risk manager
        """
        pass