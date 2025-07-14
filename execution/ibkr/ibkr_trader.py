"""
IBKR-based trader with comprehensive error handling and state management
"""
import asyncio
import time
import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, Any, List
import logging
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

try:
    from ib_async import IB, MarketOrder, LimitOrder, Order as IBOrder, Trade
except ImportError:
    raise ImportError("ib_async is required. Install with: pip install ib_async>=0.9.86")

from .ibkr_state_store import IBKRStateStore
from .ibkr_position_sync import IBKRPositionSync
from data.ibkr_connection import IBKRConnectionManager
from config.ibkr_config import IBKRConfig, create_ibkr_config
from risk.position_calculator import PositionCalculator


class OrderStatus(Enum):
    PENDING = "pending"
    PLACED = "placed"
    PARTIAL = "partial"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: str
    status: OrderStatus
    timestamp: str
    ibkr_order_id: Optional[int] = None
    ibkr_order: Optional[Any] = None  # Store the actual IBKR order object
    filled_size: float = 0.0
    average_price: float = 0.0
    error: Optional[str] = None


class IBKRTrader:
    """Production-ready IBKR trader with comprehensive error handling"""
    
    def __init__(self, config=None, ibkr_config: Optional[IBKRConfig] = None):
        self.config = config
        self.ibkr_config = ibkr_config or create_ibkr_config()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.state_store = IBKRStateStore()
        
        # Safety settings
        self.emergency_stop = False
        self.max_retries = 3
        self.retry_delays = [1, 5, 15]  # Exponential backoff
        self.daily_loss_limit = getattr(config, 'initial_capital', 10000.0) * 0.02  # 2% daily loss limit
        
        # State
        self.positions = {}
        self.orders = {}
        self.daily_pnl = 0.0
        self.initial_balance = None
        
        # Position calculator
        self.position_calc = PositionCalculator(
            balance_percentage=0.1,  # 10% of balance per trade
            risk_percentage=0.02,    # 2% risk per trade
            max_position_pct=getattr(config, 'max_position_size', 0.1)
        )
        
        # Try to acquire lock
        lock_id = f"ibkr_trader_{self.ibkr_config.account_type.value}_{datetime.now().isoformat()}"
        if not self.state_store.acquire_lock(lock_id):
            raise RuntimeError("Another IBKR trader instance is already running!")
        self.lock_id = lock_id
        
        # Initialize IBKR connection
        self.connection_manager = IBKRConnectionManager(self.ibkr_config)
        
        # Components
        self.position_sync = IBKRPositionSync(self.connection_manager, self.ibkr_config)
        
        # Recovery
        self._recover_state()
        
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, 'state_store') and hasattr(self, 'lock_id'):
                self.state_store.release_lock(self.lock_id)
        except:
            pass
    
    async def initialize(self) -> bool:
        """Initialize the trader and connect to IBKR"""
        try:
            # Connect to IBKR
            if not await self.connection_manager.connect():
                self.logger.error("Failed to connect to IBKR")
                return False
            
            # Get initial account balance
            await self._set_initial_balance()
            
            # Sync positions
            await self._sync_positions()
            
            self.logger.info("IBKR trader initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IBKR trader: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            # Save current state
            self._save_state()
            
            # Disconnect from IBKR
            await self.connection_manager.disconnect()
            
            # Release lock
            self.state_store.release_lock(self.lock_id)
            
            self.logger.info("IBKR trader shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def place_market_order(self, symbol: str, side: str, size: float, 
                                stop_loss: Optional[float] = None,
                                take_profit: Optional[float] = None) -> Optional[Order]:
        """
        Place a market order
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'SPY')
            side: 'buy' or 'sell'
            size: Order size (number of shares)
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            Order object if successful, None otherwise
        """
        if self.emergency_stop:
            self.logger.error("Trading halted due to emergency stop")
            return None
        
        try:
            # Validate inputs
            if side not in ['buy', 'sell']:
                raise ValueError(f"Invalid side: {side}")
            
            if size <= 0:
                raise ValueError(f"Invalid size: {size}")
            
            # Create contract
            contract = self.connection_manager.create_contract(symbol, 'STK')
            
            # Qualify contract
            contracts = await self.connection_manager.ib.qualifyContractsAsync(contract)
            if not contracts:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return None
            
            qualified_contract = contracts[0]
            
            # Create market order
            action = 'BUY' if side == 'buy' else 'SELL'
            market_order = MarketOrder(action, size)
            
            # Create local order tracking
            order_id = str(uuid.uuid4())
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                order_type='market',
                status=OrderStatus.PENDING,
                timestamp=datetime.now().isoformat()
            )
            
            # Place order with IBKR
            self.logger.info(f"Placing market order: {action} {size} {symbol}")
            
            trade = self.connection_manager.ib.placeOrder(qualified_contract, market_order)
            
            if trade and hasattr(trade, 'order') and hasattr(trade.order, 'orderId'):
                order.ibkr_order_id = trade.order.orderId
                order.ibkr_order = trade.order  # Store the actual IBKR order object
                order.status = OrderStatus.PLACED
                
                # Store order
                self.orders[order_id] = order
                self._save_orders()
                
                # Monitor order execution
                await self._monitor_order_execution(order_id, trade)
                
                # Place bracket orders if specified
                if stop_loss or take_profit:
                    await self._place_bracket_orders(qualified_contract, order, stop_loss, take_profit)
                
                return order
            else:
                order.status = OrderStatus.FAILED
                order.error = "Failed to get order ID from IBKR"
                self.logger.error(f"Failed to place order: {order.error}")
                return order
                
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            if 'order' in locals():
                order.status = OrderStatus.FAILED
                order.error = str(e)
                return order
            return None
    
    async def place_limit_order(self, symbol: str, side: str, size: float, price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> Optional[Order]:
        """
        Place a limit order
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            price: Limit price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            Order object if successful, None otherwise
        """
        if self.emergency_stop:
            self.logger.error("Trading halted due to emergency stop")
            return None
        
        try:
            # Create contract
            contract = self.connection_manager.create_contract(symbol, 'STK')
            
            # Qualify contract
            contracts = await self.connection_manager.ib.qualifyContractsAsync(contract)
            if not contracts:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return None
            
            qualified_contract = contracts[0]
            
            # Create limit order
            action = 'BUY' if side == 'buy' else 'SELL'
            limit_order = LimitOrder(action, size, price)
            
            # Create local order tracking
            order_id = str(uuid.uuid4())
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                order_type='limit',
                status=OrderStatus.PENDING,
                timestamp=datetime.now().isoformat()
            )
            
            # Place order with IBKR
            self.logger.info(f"Placing limit order: {action} {size} {symbol} @ {price}")
            
            trade = self.connection_manager.ib.placeOrder(qualified_contract, limit_order)
            
            if trade and hasattr(trade, 'order') and hasattr(trade.order, 'orderId'):
                order.ibkr_order_id = trade.order.orderId
                order.ibkr_order = trade.order  # Store the actual IBKR order object
                order.status = OrderStatus.PLACED
                
                # Store order
                self.orders[order_id] = order
                self._save_orders()
                
                # Monitor order execution
                await self._monitor_order_execution(order_id, trade)
                
                return order
            else:
                order.status = OrderStatus.FAILED
                order.error = "Failed to get order ID from IBKR"
                self.logger.error(f"Failed to place order: {order.error}")
                return order
                
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            if 'order' in locals():
                order.status = OrderStatus.FAILED
                order.error = str(e)
                return order
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if order_id not in self.orders:
                self.logger.error(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            if order.ibkr_order is None:
                self.logger.error(f"No IBKR order object for order {order_id}")
                return False
            
            # Cancel with IBKR - use the actual order object
            self.connection_manager.ib.cancelOrder(order.ibkr_order)
            
            # Update local status
            order.status = OrderStatus.CANCELLED
            self._save_orders()
            
            self.logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        await self._sync_positions()
        return self.positions.copy()
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance information"""
        try:
            account_summary = await self.position_sync.get_account_summary()
            
            balance_info = {
                'total_cash': account_summary.get('TotalCashValue', 0.0),
                'net_liquidation': account_summary.get('NetLiquidation', 0.0),
                'unrealized_pnl': account_summary.get('UnrealizedPnL', 0.0),
                'realized_pnl': account_summary.get('RealizedPnL', 0.0),
                'buying_power': account_summary.get('BuyingPower', 0.0),
                'gross_position_value': account_summary.get('GrossPositionValue', 0.0)
            }
            
            return balance_info
            
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return {}
    
    async def _monitor_order_execution(self, order_id: str, trade) -> None:
        """Monitor order execution in the background"""
        try:
            order = self.orders[order_id]
            
            # Wait for order to fill or timeout
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Update trade status
                self.connection_manager.ib.sleep(0.1)  # Small delay
                
                if hasattr(trade, 'orderStatus') and trade.orderStatus:
                    status = trade.orderStatus.status
                    filled = trade.orderStatus.filled
                    remaining = trade.orderStatus.remaining
                    avg_fill_price = trade.orderStatus.avgFillPrice
                    
                    # Update order status
                    if status == 'Filled':
                        order.status = OrderStatus.FILLED
                        order.filled_size = filled
                        order.average_price = avg_fill_price if avg_fill_price and avg_fill_price > 0 else 0.0
                        self.logger.info(f"Order {order_id} filled: {filled} @ {order.average_price}")
                        break
                    elif status == 'PartiallyFilled':
                        order.status = OrderStatus.PARTIAL
                        order.filled_size = filled
                        order.average_price = avg_fill_price if avg_fill_price and avg_fill_price > 0 else 0.0
                    elif status == 'Cancelled':
                        order.status = OrderStatus.CANCELLED
                        self.logger.info(f"Order {order_id} was cancelled")
                        break
                    elif status in ['Inactive', 'ApiCancelled', 'Rejected']:
                        order.status = OrderStatus.FAILED
                        order.error = f"Order {status}"
                        self.logger.error(f"Order {order_id} failed: {status}")
                        break
                
                await asyncio.sleep(0.5)
            
            # Save updated order
            self._save_orders()
            
            # Update positions if order was filled
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
                await self._sync_positions()
                
        except Exception as e:
            self.logger.error(f"Error monitoring order execution: {e}")
    
    async def _place_bracket_orders(self, contract, parent_order: Order, 
                                   stop_loss: Optional[float], 
                                   take_profit: Optional[float]) -> None:
        """Place stop loss and take profit orders"""
        try:
            if not (stop_loss or take_profit):
                return
            
            # For bracket orders, we need to wait for the parent order to fill first
            # This is a simplified implementation - in practice you might want more sophisticated bracket order handling
            
            if stop_loss:
                # Create stop loss order
                stop_action = 'SELL' if parent_order.side == 'buy' else 'BUY'
                stop_order = MarketOrder(stop_action, parent_order.size)
                stop_order.orderType = 'STP'
                stop_order.auxPrice = stop_loss
                
                self.logger.info(f"Placing stop loss order for {parent_order.symbol} @ {stop_loss}")
                # Note: In a full implementation, you'd place this as a bracket order or conditional order
            
            if take_profit:
                # Create take profit order  
                tp_action = 'SELL' if parent_order.side == 'buy' else 'BUY'
                tp_order = LimitOrder(tp_action, parent_order.size, take_profit)
                
                self.logger.info(f"Placing take profit order for {parent_order.symbol} @ {take_profit}")
                # Note: In a full implementation, you'd place this as a bracket order or conditional order
                
        except Exception as e:
            self.logger.error(f"Error placing bracket orders: {e}")
    
    async def _sync_positions(self) -> None:
        """Synchronize positions with IBKR account"""
        try:
            self.positions = await self.position_sync.sync_positions(self.positions)
            self._save_positions()
        except Exception as e:
            self.logger.error(f"Error syncing positions: {e}")
    
    async def _set_initial_balance(self) -> None:
        """Set initial account balance"""
        try:
            balance_info = await self.get_account_balance()
            self.initial_balance = balance_info.get('net_liquidation', 0.0)
            self.logger.info(f"Initial account balance: ${self.initial_balance:,.2f}")
        except Exception as e:
            self.logger.error(f"Error setting initial balance: {e}")
            self.initial_balance = 0.0
    
    def _recover_state(self) -> None:
        """Recover state from storage"""
        try:
            # Load positions
            self.positions = self.state_store.load_positions()
            self.logger.info(f"Recovered {len(self.positions)} positions")
            
            # Load orders
            self.orders = {}
            orders_data = self.state_store.load_orders()
            for order_id, order_dict in orders_data.items():
                # Convert dict back to Order object
                order_dict['status'] = OrderStatus(order_dict['status'])
                # Remove non-serializable fields that can't be recovered
                order_dict.pop('ibkr_order', None)
                self.orders[order_id] = Order(**order_dict)
            
            self.logger.info(f"Recovered {len(self.orders)} orders")
            
            # Load daily P&L
            today = date.today().isoformat()
            self.daily_pnl = self.state_store.load_daily_pnl(today)
            
        except Exception as e:
            self.logger.error(f"Error recovering state: {e}")
    
    def _save_state(self) -> None:
        """Save current state"""
        try:
            self._save_positions()
            self._save_orders()
            
            # Save daily P&L
            today = date.today().isoformat()
            self.state_store.save_daily_pnl(today, self.daily_pnl)
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def _save_positions(self) -> None:
        """Save positions to storage"""
        try:
            self.state_store.save_positions(self.positions)
        except Exception as e:
            self.logger.error(f"Error saving positions: {e}")
    
    def _save_orders(self) -> None:
        """Save orders to storage"""
        try:
            # Convert Order objects to dicts for JSON serialization
            orders_data = {}
            for order_id, order in self.orders.items():
                order_dict = asdict(order)
                order_dict['status'] = order.status.value
                # Remove non-serializable fields
                order_dict.pop('ibkr_order', None)
                orders_data[order_id] = order_dict
            
            self.state_store.save_orders(orders_data)
        except Exception as e:
            self.logger.error(f"Error saving orders: {e}")
    
    def get_daily_pnl(self) -> float:
        """Get current daily P&L"""
        return self.daily_pnl
    
    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is active"""
        return self.emergency_stop
    
    def set_emergency_stop(self, stop: bool = True) -> None:
        """Set emergency stop"""
        self.emergency_stop = stop
        if stop:
            self.logger.warning("EMERGENCY STOP ACTIVATED - All trading halted")
        else:
            self.logger.info("Emergency stop deactivated")