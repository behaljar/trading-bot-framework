"""
IBKR-specific state persistence using file storage
"""
import json
import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from execution.ccxt.state_persistence import StateStore


class IBKRStateStore(StateStore):
    """File-based state storage for IBKR execution"""
    
    def __init__(self, state_dir: str = "data/state/ibkr"):
        self.state_dir = state_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure state directory exists
        os.makedirs(state_dir, exist_ok=True)
        
        # File paths
        self.positions_file = os.path.join(state_dir, "positions.json")
        self.orders_file = os.path.join(state_dir, "orders.json")
        self.pnl_file = os.path.join(state_dir, "daily_pnl.json")
        self.lock_file = os.path.join(state_dir, "trader.lock")
        self.checkpoint_file = os.path.join(state_dir, "checkpoint.json")
        
    def save_positions(self, positions: Dict[str, Any]) -> bool:
        """Save positions to file"""
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(positions, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save positions: {e}")
            return False
            
    def load_positions(self) -> Dict[str, Any]:
        """Load positions from file"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")
            return {}
            
    def save_orders(self, orders: Dict[str, Any]) -> bool:
        """Save orders to file"""
        try:
            with open(self.orders_file, 'w') as f:
                json.dump(orders, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save orders: {e}")
            return False
            
    def load_orders(self) -> Dict[str, Any]:
        """Load orders from file"""
        try:
            if os.path.exists(self.orders_file):
                with open(self.orders_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load orders: {e}")
            return {}
            
    def save_daily_pnl(self, date: str, pnl: float) -> bool:
        """Save daily P&L"""
        try:
            pnl_data = {}
            if os.path.exists(self.pnl_file):
                with open(self.pnl_file, 'r') as f:
                    pnl_data = json.load(f)
            
            pnl_data[date] = pnl
            
            with open(self.pnl_file, 'w') as f:
                json.dump(pnl_data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save daily P&L: {e}")
            return False
            
    def load_daily_pnl(self, date: str) -> float:
        """Load daily P&L for given date"""
        try:
            if os.path.exists(self.pnl_file):
                with open(self.pnl_file, 'r') as f:
                    pnl_data = json.load(f)
                return float(pnl_data.get(date, 0.0))
            return 0.0
        except Exception as e:
            self.logger.error(f"Failed to load daily P&L: {e}")
            return 0.0
            
    def acquire_lock(self, lock_id: str) -> bool:
        """Acquire a lock to prevent multiple instances"""
        try:
            if os.path.exists(self.lock_file):
                # Check if lock is stale (older than 1 hour)
                lock_age = time.time() - os.path.getmtime(self.lock_file)
                if lock_age > 3600:  # 1 hour
                    self.logger.warning("Removing stale lock file")
                    os.remove(self.lock_file)
                else:
                    self.logger.error("Another instance is already running")
                    return False
            
            # Create lock file
            lock_data = {
                'lock_id': lock_id,
                'timestamp': time.time(),
                'pid': os.getpid()
            }
            
            with open(self.lock_file, 'w') as f:
                json.dump(lock_data, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to acquire lock: {e}")
            return False
            
    def release_lock(self, lock_id: str) -> bool:
        """Release the lock"""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
            return True
        except Exception as e:
            self.logger.error(f"Failed to release lock: {e}")
            return False
            
    def save_checkpoint(self, data: Dict[str, Any]) -> bool:
        """Save full checkpoint of state"""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False
            
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                return checkpoint_data.get('data')
            return None
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None