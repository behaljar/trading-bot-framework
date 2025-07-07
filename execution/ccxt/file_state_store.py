"""
File-based implementation of state persistence
"""
import json
import os
from datetime import datetime, date
from typing import Dict, Any, Optional
from filelock import FileLock
import logging

from .state_persistence import StateStore


class FileStateStore(StateStore):
    """File-based state storage implementation"""
    
    def __init__(self, state_dir: str = "data/state"):
        self.state_dir = state_dir
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(state_dir, exist_ok=True)
        
        # File paths
        self.positions_file = os.path.join(state_dir, "positions.json")
        self.orders_file = os.path.join(state_dir, "orders.json")
        self.checkpoint_file = os.path.join(state_dir, "checkpoint.json")
        self.lock_file = os.path.join(state_dir, "trader.lock")
        
        # Lock for preventing multiple instances
        self.lock = None
        
    def _atomic_write(self, filepath: str, data: Any) -> bool:
        """Write data atomically to prevent corruption"""
        try:
            temp_file = filepath + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(temp_file, filepath)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write {filepath}: {e}")
            return False
            
    def _safe_read(self, filepath: str, default: Any = None) -> Any:
        """Read data safely with default fallback"""
        if not os.path.exists(filepath):
            return default
            
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read {filepath}: {e}")
            return default
            
    def save_positions(self, positions: Dict[str, Any]) -> bool:
        """Save positions to file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'positions': positions
        }
        return self._atomic_write(self.positions_file, data)
        
    def load_positions(self) -> Dict[str, Any]:
        """Load positions from file"""
        data = self._safe_read(self.positions_file, {'positions': {}})
        return data.get('positions', {})
        
    def save_orders(self, orders: Dict[str, Any]) -> bool:
        """Save orders to file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'orders': orders
        }
        return self._atomic_write(self.orders_file, data)
        
    def load_orders(self) -> Dict[str, Any]:
        """Load orders from file"""
        data = self._safe_read(self.orders_file, {'orders': {}})
        
        # Clean up old orders (older than 24 hours)
        current_time = datetime.now()
        cleaned_orders = {}
        
        for order_id, order in data.get('orders', {}).items():
            if 'timestamp' in order:
                order_time = datetime.fromisoformat(order['timestamp'])
                if (current_time - order_time).total_seconds() < 86400:  # 24 hours
                    cleaned_orders[order_id] = order
                    
        return cleaned_orders
        
    def save_daily_pnl(self, date_str: str, pnl: float) -> bool:
        """Save daily P&L"""
        pnl_file = os.path.join(self.state_dir, f"daily_pnl_{date_str}.json")
        data = {
            'date': date_str,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat()
        }
        return self._atomic_write(pnl_file, data)
        
    def load_daily_pnl(self, date_str: str) -> float:
        """Load daily P&L for given date"""
        pnl_file = os.path.join(self.state_dir, f"daily_pnl_{date_str}.json")
        data = self._safe_read(pnl_file, {'pnl': 0.0})
        return data.get('pnl', 0.0)
        
    def acquire_lock(self, lock_id: str) -> bool:
        """Acquire lock to prevent multiple instances"""
        try:
            self.lock = FileLock(self.lock_file, timeout=1)
            self.lock.acquire()
            
            # Write lock info
            lock_data = {
                'lock_id': lock_id,
                'pid': os.getpid(),
                'timestamp': datetime.now().isoformat()
            }
            self._atomic_write(self.lock_file + '.info', lock_data)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to acquire lock: {e}")
            return False
            
    def release_lock(self, lock_id: str) -> bool:
        """Release the lock"""
        try:
            if self.lock:
                self.lock.release()
                
            # Remove lock info
            lock_info_file = self.lock_file + '.info'
            if os.path.exists(lock_info_file):
                os.remove(lock_info_file)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to release lock: {e}")
            return False
            
    def save_checkpoint(self, data: Dict[str, Any]) -> bool:
        """Save full checkpoint of state"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        return self._atomic_write(self.checkpoint_file, checkpoint)
        
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint"""
        checkpoint = self._safe_read(self.checkpoint_file)
        if checkpoint and 'data' in checkpoint:
            return checkpoint['data']
        return None