"""
IBKR position synchronization
"""
import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

try:
    from ib_async import PortfolioItem
except ImportError:
    raise ImportError("ib_async is required. Install with: pip install ib_async>=0.9.86")

from data.ibkr_connection import IBKRConnectionManager
from config.ibkr_config import IBKRConfig


class IBKRPositionSync:
    """Synchronizes positions between trading bot and IBKR account"""
    
    def __init__(self, connection_manager: IBKRConnectionManager, config: IBKRConfig):
        self.connection_manager = connection_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def get_account_positions(self) -> Dict[str, Dict]:
        """Get all positions from IBKR account"""
        try:
            if not self.connection_manager.is_connected():
                await self.connection_manager.connect()
            
            positions = {}
            
            # Get portfolio positions
            portfolio_items = self.connection_manager.ib.portfolio()
            
            for item in portfolio_items:
                if hasattr(item, 'contract') and hasattr(item, 'position'):
                    symbol = self._normalize_symbol(item.contract.symbol)
                    
                    if item.position != 0:  # Only include non-zero positions
                        positions[symbol] = {
                            'symbol': symbol,
                            'size': float(item.position),
                            'market_price': float(item.marketPrice) if hasattr(item, 'marketPrice') and item.marketPrice else 0.0,
                            'market_value': float(item.marketValue) if hasattr(item, 'marketValue') and item.marketValue else 0.0,
                            'average_cost': float(item.averageCost) if hasattr(item, 'averageCost') and item.averageCost else 0.0,
                            'unrealized_pnl': float(item.unrealizedPNL) if hasattr(item, 'unrealizedPNL') and item.unrealizedPNL else 0.0,
                            'realized_pnl': float(item.realizedPNL) if hasattr(item, 'realizedPNL') and item.realizedPNL else 0.0,
                            'contract': {
                                'symbol': item.contract.symbol,
                                'secType': item.contract.secType,
                                'exchange': item.contract.exchange,
                                'currency': item.contract.currency,
                                'conId': item.contract.conId if hasattr(item.contract, 'conId') else None
                            },
                            'timestamp': datetime.now().isoformat()
                        }
            
            self.logger.info(f"Retrieved {len(positions)} positions from IBKR account")
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get account positions: {e}")
            return {}
    
    async def get_account_summary(self) -> Dict[str, float]:
        """Get account summary information"""
        try:
            if not self.connection_manager.is_connected():
                await self.connection_manager.connect()
            
            # Request account summary
            account_summary = self.connection_manager.ib.accountSummary()
            
            summary = {}
            for item in account_summary:
                if hasattr(item, 'tag') and hasattr(item, 'value'):
                    try:
                        # Convert common numeric fields
                        if item.tag in ['TotalCashValue', 'NetLiquidation', 'UnrealizedPnL', 
                                       'RealizedPnL', 'BuyingPower', 'GrossPositionValue']:
                            summary[item.tag] = float(item.value)
                        else:
                            summary[item.tag] = item.value
                    except (ValueError, TypeError):
                        summary[item.tag] = item.value
            
            self.logger.info(f"Retrieved account summary with {len(summary)} fields")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get account summary: {e}")
            return {}
    
    async def sync_positions(self, local_positions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Synchronize local positions with IBKR account positions
        
        Args:
            local_positions: Current local position tracking
            
        Returns:
            Updated positions combining local and IBKR data
        """
        try:
            # Get current IBKR positions
            ibkr_positions = await self.get_account_positions()
            
            # Merge positions
            synced_positions = {}
            
            # Start with IBKR positions as the source of truth
            for symbol, ibkr_pos in ibkr_positions.items():
                synced_positions[symbol] = ibkr_pos.copy()
                
                # If we have local tracking for this position, merge additional data
                if symbol in local_positions:
                    local_pos = local_positions[symbol]
                    synced_positions[symbol].update({
                        'entry_price': local_pos.get('entry_price'),
                        'entry_time': local_pos.get('entry_time'),
                        'stop_loss': local_pos.get('stop_loss'),
                        'take_profit': local_pos.get('take_profit'),
                        'strategy': local_pos.get('strategy'),
                        'local_tracking': True
                    })
                else:
                    # Position exists in IBKR but not in local tracking
                    synced_positions[symbol]['local_tracking'] = False
                    self.logger.warning(f"Found untracked position in IBKR: {symbol} size={ibkr_pos['size']}")
            
            # Check for local positions that don't exist in IBKR
            for symbol, local_pos in local_positions.items():
                if symbol not in synced_positions:
                    self.logger.warning(f"Local position {symbol} not found in IBKR account")
                    # Keep local position but mark as inconsistent
                    synced_positions[symbol] = local_pos.copy()
                    synced_positions[symbol]['ibkr_sync_error'] = True
                    synced_positions[symbol]['size'] = 0  # Override size to 0 since not in IBKR
            
            self.logger.info(f"Position sync completed: {len(synced_positions)} total positions")
            return synced_positions
            
        except Exception as e:
            self.logger.error(f"Failed to sync positions: {e}")
            return local_positions  # Return original positions on error
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for consistency"""
        # Remove any exchange suffixes and clean up
        symbol = symbol.split('.')[0].split(':')[0].upper()
        return symbol
    
    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders from IBKR"""
        try:
            if not self.connection_manager.is_connected():
                await self.connection_manager.connect()
            
            open_orders = []
            orders = self.connection_manager.ib.orders()
            
            for order in orders:
                if hasattr(order, 'order') and hasattr(order, 'contract'):
                    order_data = {
                        'orderId': order.order.orderId if hasattr(order.order, 'orderId') else None,
                        'symbol': self._normalize_symbol(order.contract.symbol),
                        'action': order.order.action if hasattr(order.order, 'action') else None,
                        'totalQuantity': float(order.order.totalQuantity) if hasattr(order.order, 'totalQuantity') else 0.0,
                        'orderType': order.order.orderType if hasattr(order.order, 'orderType') else None,
                        'lmtPrice': float(order.order.lmtPrice) if hasattr(order.order, 'lmtPrice') and order.order.lmtPrice else None,
                        'status': order.orderStatus.status if hasattr(order, 'orderStatus') and hasattr(order.orderStatus, 'status') else 'Unknown',
                        'filled': float(order.orderStatus.filled) if hasattr(order, 'orderStatus') and hasattr(order.orderStatus, 'filled') else 0.0,
                        'remaining': float(order.orderStatus.remaining) if hasattr(order, 'orderStatus') and hasattr(order.orderStatus, 'remaining') else 0.0,
                        'timestamp': datetime.now().isoformat()
                    }
                    open_orders.append(order_data)
            
            self.logger.info(f"Retrieved {len(open_orders)} open orders from IBKR")
            return open_orders
            
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """
        Cancel all open orders, optionally filtered by symbol
        
        Args:
            symbol: If provided, only cancel orders for this symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            open_orders = await self.get_open_orders()
            
            cancelled_count = 0
            for order_data in open_orders:
                if symbol is None or order_data['symbol'] == symbol:
                    order_id = order_data['orderId']
                    if order_id:
                        # Create a dummy order to cancel
                        self.connection_manager.ib.cancelOrder(order_id)
                        cancelled_count += 1
                        self.logger.info(f"Cancelled order {order_id} for {order_data['symbol']}")
            
            self.logger.info(f"Cancelled {cancelled_count} orders")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel orders: {e}")
            return False