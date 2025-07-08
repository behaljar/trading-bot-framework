"""
Position size calculator with fixed risk and balance percentage modes.
"""
import logging
from typing import Optional


class PositionCalculator:
    """
    Calculate position sizes based on risk management rules.
    
    Supports two modes:
    1. Balance percentage: Size = (Balance * Percentage) / Price
    2. Fixed risk with stop loss: Size = (Balance * Risk%) / (Entry - StopLoss)
    """
    
    def __init__(self, 
                 balance_percentage: float = 0.1,  # 10% of balance per trade
                 risk_percentage: float = 0.02,    # 2% risk per trade
                 max_position_pct: float = 0.25):  # Max 25% in one position
        """
        Initialize position calculator.
        
        Args:
            balance_percentage: Default position size as % of balance (0.1 = 10%)
            risk_percentage: Risk per trade when using stop loss (0.02 = 2%)
            max_position_pct: Maximum position size as % of balance (0.25 = 25%)
        """
        self.balance_percentage = balance_percentage
        self.risk_percentage = risk_percentage
        self.max_position_pct = max_position_pct
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self,
                              balance: float,
                              entry_price: float,
                              stop_loss: Optional[float] = None,
                              position_type: str = 'long') -> float:
        """
        Calculate position size based on available balance and risk parameters.
        
        Args:
            balance: Available balance
            entry_price: Entry price for the position
            stop_loss: Stop loss price (optional)
            position_type: 'long' or 'short'
            
        Returns:
            Position size (number of units to buy/sell)
        """
        if balance <= 0 or entry_price <= 0:
            return 0.0
            
        # Method 1: Fixed risk based on stop loss
        if stop_loss and stop_loss > 0:
            # Calculate risk per unit
            if position_type == 'long':
                # For long positions, stop loss should be below entry
                if stop_loss >= entry_price:
                    self.logger.warning(
                        f"Invalid stop loss for long position: "
                        f"SL ${stop_loss:.2f} >= Entry ${entry_price:.2f}"
                    )
                    return self._calculate_balance_based_size(balance, entry_price)
                    
                risk_per_unit = entry_price - stop_loss
            else:  # short
                # For short positions, stop loss should be above entry
                if stop_loss <= entry_price:
                    self.logger.warning(
                        f"Invalid stop loss for short position: "
                        f"SL ${stop_loss:.2f} <= Entry ${entry_price:.2f}"
                    )
                    return self._calculate_balance_based_size(balance, entry_price)
                    
                risk_per_unit = stop_loss - entry_price
                
            # Calculate position size based on risk
            risk_amount = balance * self.risk_percentage
            position_size = risk_amount / risk_per_unit
            
            # Log the calculation
            self.logger.info(
                f"Risk-based sizing: Balance=${balance:.2f}, Risk={self.risk_percentage:.1%}, "
                f"Entry=${entry_price:.2f}, SL=${stop_loss:.2f}, "
                f"Risk/unit=${risk_per_unit:.2f}, Size={position_size:.4f}"
            )
            
            # For fixed risk with stop loss, don't apply max position cap
            # The risk is already controlled by the stop loss distance
            return position_size
            
        else:
            # Method 2: Simple balance percentage
            position_size = self._calculate_balance_based_size(balance, entry_price)
            
            # Apply maximum position size limit only for balance-based sizing
            max_position_value = balance * self.max_position_pct
            max_position_size = max_position_value / entry_price
            
            if position_size > max_position_size:
                self.logger.warning(
                    f"Position size {position_size:.4f} exceeds max {max_position_size:.4f}, "
                    f"capping at {self.max_position_pct:.1%} of balance"
                )
                position_size = max_position_size
                
            return position_size
        
    def _calculate_balance_based_size(self, balance: float, entry_price: float) -> float:
        """Calculate position size as percentage of balance."""
        position_value = balance * self.balance_percentage
        position_size = position_value / entry_price
        
        self.logger.info(
            f"Balance-based sizing: Balance=${balance:.2f}, "
            f"Percentage={self.balance_percentage:.1%}, "
            f"Entry=${entry_price:.2f}, Size={position_size:.4f}"
        )
        
        return position_size
        
    def calculate_shares_for_amount(self, amount: float, price: float) -> float:
        """Calculate number of shares for a given dollar amount."""
        if price <= 0:
            return 0.0
        return amount / price
        
    def calculate_risk_amount(self, balance: float) -> float:
        """Calculate dollar amount to risk based on balance."""
        return balance * self.risk_percentage