"""
Paper trading implementation
"""
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    
@dataclass
class Trade:
    symbol: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    position_type: str
    pnl: float
    pnl_pct: float

class PaperTrader:
    """Paper trading simulation"""
    
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.initial_capital
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.commission = config.commission
        self.slippage = config.slippage
        self.logger = logging.getLogger(__name__)
        
    def place_order(self, symbol: str, quantity: float, current_price: Optional[float] = None) -> bool:
        """
        Place a paper trading order
        
        Args:
            symbol: Trading symbol
            quantity: Positive for buy, negative for sell
            current_price: Current market price (optional)
            
        Returns:
            bool: True if order was successful
        """
        try:
            if current_price is None:
                # In real implementation, you'd fetch current price
                self.logger.warning(f"No current price provided for {symbol}")
                return False
                
            # Apply slippage
            if quantity > 0:  # Buy order
                execution_price = current_price * (1 + self.slippage)
            else:  # Sell order
                execution_price = current_price * (1 - self.slippage)
                
            # Check if we have existing position
            if symbol in self.positions:
                return self._modify_position(symbol, quantity, execution_price)
            else:
                return self._open_position(symbol, quantity, execution_price)
                
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            return False
            
    def _open_position(self, symbol: str, quantity: float, price: float) -> bool:
        """Open a new position"""
        position_value = abs(quantity * price)
        commission_cost = position_value * self.commission
        total_cost = position_value + commission_cost
        
        if total_cost > self.current_capital:
            self.logger.warning(f"Insufficient capital for {symbol}. Need: {total_cost}, Have: {self.current_capital}")
            return False
            
        self.current_capital -= total_cost
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            position_type='long' if quantity > 0 else 'short'
        )
        
        self.positions[symbol] = position
        self.logger.info(f"Opened {position.position_type} position: {symbol} x {quantity} @ {price}")
        return True
        
    def _modify_position(self, symbol: str, quantity: float, price: float) -> bool:
        """Modify existing position (close or add to position)"""
        position = self.positions[symbol]
        
        # Check if this closes the position
        if (position.quantity > 0 and quantity < 0 and abs(quantity) >= position.quantity) or \
           (position.quantity < 0 and quantity > 0 and quantity >= abs(position.quantity)):
            return self._close_position(symbol, price)
        else:
            # Partial close or add to position - simplified implementation
            self.logger.info(f"Modifying position for {symbol} (not fully implemented)")
            return True
            
    def _close_position(self, symbol: str, exit_price: float) -> bool:
        """Close an existing position"""
        if symbol not in self.positions:
            self.logger.warning(f"No position to close for {symbol}")
            return False
            
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.position_type == 'long':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # short position
            pnl = (position.entry_price - exit_price) * abs(position.quantity)
            
        position_value = abs(position.quantity * exit_price)
        commission_cost = position_value * self.commission
        net_pnl = pnl - commission_cost
        
        # Update capital
        self.current_capital += position_value - commission_cost
        if position.position_type == 'long':
            self.current_capital += net_pnl
        
        # Calculate percentage return
        entry_value = abs(position.quantity * position.entry_price)
        pnl_pct = (net_pnl / entry_value) * 100 if entry_value > 0 else 0
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            position_type=position.position_type,
            pnl=net_pnl,
            pnl_pct=pnl_pct
        )
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        self.logger.info(f"Closed {position.position_type} position: {symbol} @ {exit_price}, P&L: {net_pnl:.2f} ({pnl_pct:.2f}%)")
        return True
        
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.current_capital
        
        if current_prices:
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    if position.position_type == 'long':
                        unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)
                    portfolio_value += unrealized_pnl
                    
        return portfolio_value
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.trades:
            return {
                "Total Trades": 0,
                "Total P&L": 0.0,
                "Total Return %": 0.0,
                "Win Rate %": 0.0,
                "Average Trade P&L": 0.0,
                "Current Capital": self.current_capital,
                "Open Positions": len(self.positions)
            }
            
        total_pnl = sum(trade.pnl for trade in self.trades)
        winning_trades = [trade for trade in self.trades if trade.pnl > 0]
        win_rate = (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0
        avg_trade_pnl = total_pnl / len(self.trades) if self.trades else 0
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        return {
            "Total Trades": len(self.trades),
            "Total P&L": total_pnl,
            "Total Return %": total_return_pct,
            "Win Rate %": win_rate,
            "Average Trade P&L": avg_trade_pnl,
            "Current Capital": self.current_capital,
            "Open Positions": len(self.positions)
        }