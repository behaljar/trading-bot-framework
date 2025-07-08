"""
Order execution simulation with realistic market conditions.
"""
from typing import Optional

from config.settings import TradingConfig
import logging
from .virtual_portfolio import VirtualOrder


class OrderSimulator:
    """Simulates order execution with realistic market conditions."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.commission_rate = getattr(config, 'paper_commission_rate', 0.001)  # 0.1% default
        self.spread_bps = getattr(config, 'paper_spread_bps', 10)  # 10 bps default
        self.slippage_bps = getattr(config, 'paper_slippage_bps', 5)  # 5 bps default
        self.logger = logging.getLogger(__name__)
        
    def simulate_market_order(self, order: VirtualOrder, current_price: float, 
                            bid: Optional[float] = None, ask: Optional[float] = None) -> VirtualOrder:
        """Simulate market order execution with spread and slippage."""
        # If bid/ask not provided, estimate from spread
        if bid is None or ask is None:
            spread = current_price * (self.spread_bps / 10000)
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
        # Determine execution price based on side
        if order.side == 'buy':
            base_price = ask
            slippage_multiplier = 1 + (self.slippage_bps / 10000)
        else:
            base_price = bid
            slippage_multiplier = 1 - (self.slippage_bps / 10000)
            
        # Apply slippage
        filled_price = base_price * slippage_multiplier
        
        # Calculate commission
        commission = abs(order.size * filled_price * self.commission_rate)
        
        # Update order
        order.filled_price = filled_price
        order.commission = commission
        order.slippage = abs(filled_price - current_price)
        order.status = 'filled'
        
        self.logger.info(
            f"Simulated {order.side} order: size={order.size}, "
            f"mid_price={current_price:.4f}, filled_at={filled_price:.4f}, "
            f"slippage={order.slippage:.4f}, commission={commission:.4f}"
        )
        
        return order
        
    def simulate_limit_order(self, order: VirtualOrder, current_price: float,
                           bid: Optional[float] = None, ask: Optional[float] = None) -> VirtualOrder:
        """Simulate limit order execution."""
        # If bid/ask not provided, estimate from spread
        if bid is None or ask is None:
            spread = current_price * (self.spread_bps / 10000)
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
        # Check if limit order would fill
        if order.side == 'buy' and order.price >= ask:
            # Buy limit fills at ask or better
            filled_price = min(order.price, ask)
            order.filled_price = filled_price
            order.commission = abs(order.size * filled_price * self.commission_rate)
            order.status = 'filled'
        elif order.side == 'sell' and order.price <= bid:
            # Sell limit fills at bid or better
            filled_price = max(order.price, bid)
            order.filled_price = filled_price
            order.commission = abs(order.size * filled_price * self.commission_rate)
            order.status = 'filled'
        else:
            # Order doesn't fill
            self.logger.debug(
                f"Limit order not filled: {order.side} @ {order.price}, "
                f"bid={bid:.4f}, ask={ask:.4f}"
            )
            
        return order