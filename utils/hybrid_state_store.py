"""
Hybrid state store that uses PostgreSQL as primary with file-based fallback
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .postgres_state_store import PostgreSQLStateStore, DatabaseConfig
from execution.ccxt.file_state_store import FileStateStore


class HybridStateStore:
    """
    Hybrid state store that uses PostgreSQL as primary storage 
    with file-based storage as fallback
    """
    
    def __init__(self, instance_id: str = None, exchange: str = 'CCXT'):
        self.instance_id = instance_id or f"trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        
        # Initialize stores
        self.postgres_store = None
        self.file_store = FileStateStore()
        
        # Try to initialize PostgreSQL store
        self._initialize_postgres()
    
    def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL store with error handling"""
        try:
            self.logger.info(
                "Initializing PostgreSQL state store...",
                extra={
                    'instance_id': self.instance_id,
                    'exchange': self.exchange
                }
            )
            
            self.postgres_store = PostgreSQLStateStore(
                instance_id=self.instance_id,
                exchange=self.exchange
            )
            
            # Test connection
            conn = self.postgres_store._get_connection()
            self.logger.info(
                "PostgreSQL state store initialized successfully",
                extra={
                    'instance_id': self.instance_id,
                    'exchange': self.exchange,
                    'connection_status': 'connected'
                }
            )
            
        except Exception as e:
            self.logger.warning(
                "Failed to initialize PostgreSQL state store",
                extra={
                    'instance_id': self.instance_id,
                    'exchange': self.exchange,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            )
            self.logger.info(
                "Falling back to file-based state store",
                extra={
                    'instance_id': self.instance_id,
                    'exchange': self.exchange
                }
            )
            self.postgres_store = None
    
    def save_positions(self, positions: Dict[str, Any]) -> None:
        """Save positions to primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                self.postgres_store.save_positions(positions)
                self.logger.debug("Positions saved to PostgreSQL")
            else:
                self.file_store.save_positions(positions)
                self.logger.debug("Positions saved to file (fallback)")
                
        except Exception as e:
            self.logger.error(f"Failed to save positions to PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for positions")
                try:
                    self.file_store.save_positions(positions)
                    self.logger.debug("Positions saved to file (fallback)")
                except Exception as file_e:
                    self.logger.error(f"Failed to save positions to file: {file_e}")
    
    def load_positions(self) -> Dict[str, Any]:
        """Load positions from primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                positions = self.postgres_store.load_positions()
                self.logger.debug("Positions loaded from PostgreSQL")
                return positions
            else:
                positions = self.file_store.load_positions()
                self.logger.debug("Positions loaded from file (fallback)")
                return positions
                
        except Exception as e:
            self.logger.error(f"Failed to load positions from PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for positions")
                try:
                    positions = self.file_store.load_positions()
                    self.logger.debug("Positions loaded from file (fallback)")
                    return positions
                except Exception as file_e:
                    self.logger.error(f"Failed to load positions from file: {file_e}")
                    return {}
            return {}
    
    def save_orders(self, orders: Dict[str, Any]) -> None:
        """Save orders to primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                self.postgres_store.save_orders(orders)
                self.logger.debug("Orders saved to PostgreSQL")
            else:
                self.file_store.save_orders(orders)
                self.logger.debug("Orders saved to file (fallback)")
                
        except Exception as e:
            self.logger.error(f"Failed to save orders to PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for orders")
                try:
                    self.file_store.save_orders(orders)
                    self.logger.debug("Orders saved to file (fallback)")
                except Exception as file_e:
                    self.logger.error(f"Failed to save orders to file: {file_e}")
    
    def load_orders(self) -> Dict[str, Any]:
        """Load orders from primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                orders = self.postgres_store.load_orders()
                self.logger.debug("Orders loaded from PostgreSQL")
                return orders
            else:
                orders = self.file_store.load_orders()
                self.logger.debug("Orders loaded from file (fallback)")
                return orders
                
        except Exception as e:
            self.logger.error(f"Failed to load orders from PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for orders")
                try:
                    orders = self.file_store.load_orders()
                    self.logger.debug("Orders loaded from file (fallback)")
                    return orders
                except Exception as file_e:
                    self.logger.error(f"Failed to load orders from file: {file_e}")
                    return {}
            return {}
    
    def save_daily_pnl(self, date: str, pnl: float) -> None:
        """Save daily P&L to primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                self.postgres_store.save_daily_pnl(date, pnl)
                self.logger.debug("Daily P&L saved to PostgreSQL")
            else:
                self.file_store.save_daily_pnl(date, pnl)
                self.logger.debug("Daily P&L saved to file (fallback)")
                
        except Exception as e:
            self.logger.error(f"Failed to save daily P&L to PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for daily P&L")
                try:
                    self.file_store.save_daily_pnl(date, pnl)
                    self.logger.debug("Daily P&L saved to file (fallback)")
                except Exception as file_e:
                    self.logger.error(f"Failed to save daily P&L to file: {file_e}")
    
    def load_daily_pnl(self, date: str) -> float:
        """Load daily P&L from primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                pnl = self.postgres_store.load_daily_pnl(date)
                self.logger.debug("Daily P&L loaded from PostgreSQL")
                return pnl
            else:
                pnl = self.file_store.load_daily_pnl(date)
                self.logger.debug("Daily P&L loaded from file (fallback)")
                return pnl
                
        except Exception as e:
            self.logger.error(f"Failed to load daily P&L from PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for daily P&L")
                try:
                    pnl = self.file_store.load_daily_pnl(date)
                    self.logger.debug("Daily P&L loaded from file (fallback)")
                    return pnl
                except Exception as file_e:
                    self.logger.error(f"Failed to load daily P&L from file: {file_e}")
                    return 0.0
            return 0.0
    
    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save checkpoint to primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                self.postgres_store.save_checkpoint(checkpoint)
                self.logger.debug("Checkpoint saved to PostgreSQL")
            else:
                self.file_store.save_checkpoint(checkpoint)
                self.logger.debug("Checkpoint saved to file (fallback)")
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for checkpoint")
                try:
                    self.file_store.save_checkpoint(checkpoint)
                    self.logger.debug("Checkpoint saved to file (fallback)")
                except Exception as file_e:
                    self.logger.error(f"Failed to save checkpoint to file: {file_e}")
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from primary store (PostgreSQL) with file fallback"""
        try:
            if self.postgres_store:
                checkpoint = self.postgres_store.load_checkpoint()
                self.logger.debug("Checkpoint loaded from PostgreSQL")
                return checkpoint
            else:
                checkpoint = self.file_store.load_checkpoint()
                self.logger.debug("Checkpoint loaded from file (fallback)")
                return checkpoint
                
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for checkpoint")
                try:
                    checkpoint = self.file_store.load_checkpoint()
                    self.logger.debug("Checkpoint loaded from file (fallback)")
                    return checkpoint
                except Exception as file_e:
                    self.logger.error(f"Failed to load checkpoint from file: {file_e}")
                    return {}
            return {}
    
    def acquire_lock(self, lock_id: str) -> bool:
        """Acquire a trading lock"""
        try:
            if self.postgres_store:
                return self.postgres_store.acquire_lock(lock_id)
            else:
                return self.file_store.acquire_lock(lock_id)
                
        except Exception as e:
            self.logger.error(f"Failed to acquire lock from PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for lock")
                try:
                    return self.file_store.acquire_lock(lock_id)
                except Exception as file_e:
                    self.logger.error(f"Failed to acquire lock from file: {file_e}")
                    return False
            return False
    
    def release_lock(self, lock_id: str) -> bool:
        """Release a trading lock"""
        try:
            if self.postgres_store:
                return self.postgres_store.release_lock(lock_id)
            else:
                return self.file_store.release_lock(lock_id)
                
        except Exception as e:
            self.logger.error(f"Failed to release lock from PostgreSQL: {e}")
            if self.postgres_store:
                self.logger.info("Falling back to file storage for lock")
                try:
                    return self.file_store.release_lock(lock_id)
                except Exception as file_e:
                    self.logger.error(f"Failed to release lock from file: {file_e}")
                    return False
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about current storage backend"""
        return {
            'primary_store': 'PostgreSQL' if self.postgres_store else 'File',
            'fallback_available': True,
            'postgres_available': self.postgres_store is not None,
            'instance_id': self.instance_id,
            'exchange': self.exchange
        }
    
    def close(self) -> None:
        """Close connections"""
        if self.postgres_store:
            self.postgres_store.close()
        # File store doesn't need explicit closing