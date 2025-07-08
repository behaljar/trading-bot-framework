"""
Virtual portfolio management for paper trading.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import logging


@dataclass
class VirtualPosition:
    """Virtual position tracking."""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    side: str  # 'long' or 'short'
    unrealized_pnl: float
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    entry_time: datetime = None
    stop_loss: float = None
    take_profit: float = None
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()


@dataclass
class VirtualOrder:
    """Virtual order representation."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: str  # 'market' or 'limit'
    price: Optional[float] = None
    filled_price: Optional[float] = None
    status: str = 'pending'  # pending, filled, cancelled
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class VirtualPortfolio:
    """Manages virtual balances and positions."""
    
    def __init__(self, initial_balance: float, base_currency: str = 'USD'):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.base_currency = base_currency
        self.positions: Dict[str, VirtualPosition] = {}
        self.order_history: List[VirtualOrder] = []
        self.trade_history: List[Dict] = []
        self.logger = logging.getLogger(__name__)
        
    def update_position_price(self, symbol: str, current_price: float):
        """Update position with current price and calculate unrealized PnL."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            
            if pos.side == 'long':
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            else:  # short
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
                
    def execute_order(self, order: VirtualOrder) -> bool:
        """Execute a filled order and update portfolio."""
        if order.status != 'filled':
            return False
            
        symbol = order.symbol
        size = order.size
        price = order.filled_price
        
        # Calculate total cost including commission
        total_cost = abs(size * price) + order.commission
        
        # Check if we have enough balance
        if total_cost > self.balance:
            self.logger.error(f"Insufficient balance: need {total_cost}, have {self.balance}")
            order.status = 'rejected'
            return False
            
        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            
            # Check if closing or reducing position
            if (pos.side == 'long' and order.side == 'sell') or \
               (pos.side == 'short' and order.side == 'buy'):
                # Calculate realized PnL
                if pos.side == 'long':
                    realized = (price - pos.entry_price) * min(size, pos.size)
                else:
                    realized = (pos.entry_price - price) * min(size, pos.size)
                    
                pos.realized_pnl += realized
                
                # Update or close position
                if size >= pos.size:
                    # Close position
                    self.balance += pos.size * price - order.commission
                    self.trade_history.append({
                        'symbol': symbol,
                        'side': 'close',
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'exit_price': price,
                        'pnl': realized,
                        'commission': order.commission,
                        'timestamp': order.timestamp
                    })
                    del self.positions[symbol]
                else:
                    # Reduce position
                    pos.size -= size
                    self.balance += size * price - order.commission
            else:
                # Adding to position
                # Calculate new average price
                total_value = pos.size * pos.entry_price + size * price
                pos.size += size
                pos.entry_price = total_value / pos.size
                pos.commission_paid += order.commission
                self.balance -= total_cost
        else:
            # New position
            side = 'long' if order.side == 'buy' else 'short'
            self.positions[symbol] = VirtualPosition(
                symbol=symbol,
                size=size,
                entry_price=price,
                current_price=price,
                side=side,
                unrealized_pnl=0.0,
                commission_paid=order.commission
            )
            self.balance -= total_cost
            
        self.order_history.append(order)
        return True
        
    def get_total_value(self) -> float:
        """Calculate total portfolio value including positions."""
        total = self.balance
        for pos in self.positions.values():
            total += pos.size * pos.current_price
        return total
        
    def get_performance_summary(self) -> Dict:
        """Calculate performance metrics."""
        total_value = self.get_total_value()
        total_pnl = total_value - self.initial_balance
        total_return = (total_value / self.initial_balance - 1) * 100
        
        # Calculate realized and unrealized PnL
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Add closed positions PnL
        realized_pnl += sum(trade['pnl'] for trade in self.trade_history)
        
        # Calculate commissions
        total_commission = sum(order.commission for order in self.order_history)
        
        return {
            'total_value': total_value,
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_commission': total_commission,
            'num_trades': len(self.order_history),
            'open_positions': len(self.positions)
        }