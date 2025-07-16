import os
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List
import asyncpg
from contextlib import asynccontextmanager
import json

from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseLogger:
    """Handles logging of trades and other data to PostgreSQL database."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.db_config = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'port': int(os.environ.get('DB_PORT', 5432)),
            'database': os.environ.get('DB_NAME', 'trading_bot'),
            'user': os.environ.get('DB_USER', 'trading_user'),
            'password': os.environ.get('DB_PASSWORD', 'trading_password'),
        }
        
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
            
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
            
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        async with self.pool.acquire() as connection:
            yield connection
            
    async def log_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        status: str = 'pending',
        order_type: str = 'market',
        strategy_name: Optional[str] = None,
        signal_reason: Optional[str] = None,
        exchange: Optional[str] = None,
        account_id: Optional[str] = None,
        commission: Decimal = Decimal('0'),
        commission_asset: Optional[str] = None,
        executed_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a trade to the database."""
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO trading.trades (
                        trade_id, symbol, side, quantity, price, status,
                        order_type, strategy_name, signal_reason, exchange,
                        account_id, commission, commission_asset, executed_at, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        commission = EXCLUDED.commission,
                        commission_asset = EXCLUDED.commission_asset,
                        executed_at = EXCLUDED.executed_at,
                        updated_at = NOW(),
                        metadata = EXCLUDED.metadata
                    RETURNING id
                """
                
                result = await conn.fetchrow(
                    query,
                    trade_id, symbol, side, quantity, price, status,
                    order_type, strategy_name, signal_reason, exchange,
                    account_id, commission, commission_asset, executed_at,
                    json.dumps(metadata) if metadata else '{}'
                )
                
                logger.info(f"Trade logged: {trade_id} - {symbol} {side} {quantity} @ {price}")
                return result['id']
                
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            raise
            
    async def update_position(
        self,
        symbol: str,
        quantity: Decimal,
        average_price: Decimal,
        exchange: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        current_price: Optional[Decimal] = None,
        unrealized_pnl: Optional[Decimal] = None,
        realized_pnl: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update or create a position record."""
        try:
            async with self.get_connection() as conn:
                # Close position if quantity is 0
                if quantity == 0:
                    query = """
                        UPDATE trading.positions
                        SET quantity = 0,
                            is_active = FALSE,
                            closed_at = NOW(),
                            updated_at = NOW(),
                            realized_pnl = COALESCE($1, realized_pnl),
                            metadata = COALESCE($2::jsonb, metadata)
                        WHERE symbol = $3
                          AND COALESCE(exchange, '') = COALESCE($4, '')
                          AND COALESCE(account_id, '') = COALESCE($5, '')
                          AND COALESCE(strategy_name, '') = COALESCE($6, '')
                          AND is_active = TRUE
                    """
                    await conn.execute(
                        query,
                        realized_pnl, json.dumps(metadata) if metadata else None,
                        symbol, exchange, account_id, strategy_name
                    )
                else:
                    # Update or insert position
                    query = """
                        INSERT INTO trading.positions (
                            symbol, quantity, average_price, exchange, account_id,
                            strategy_name, current_price, unrealized_pnl, realized_pnl,
                            metadata, is_active
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, TRUE)
                        ON CONFLICT (symbol, exchange, account_id, strategy_name)
                        DO UPDATE SET
                            quantity = EXCLUDED.quantity,
                            average_price = EXCLUDED.average_price,
                            current_price = EXCLUDED.current_price,
                            unrealized_pnl = EXCLUDED.unrealized_pnl,
                            realized_pnl = COALESCE(EXCLUDED.realized_pnl, positions.realized_pnl),
                            updated_at = NOW(),
                            metadata = COALESCE(EXCLUDED.metadata, positions.metadata),
                            is_active = TRUE,
                            closed_at = NULL
                    """
                    await conn.execute(
                        query,
                        symbol, quantity, average_price, exchange, account_id,
                        strategy_name, current_price, unrealized_pnl, realized_pnl,
                        json.dumps(metadata) if metadata else '{}'
                    )
                    
                logger.info(f"Position updated: {symbol} - {quantity} @ {average_price}")
                
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
            raise
            
    async def log_signal(
        self,
        symbol: str,
        strategy_name: str,
        signal_type: str,
        price: Decimal,
        signal_strength: Optional[Decimal] = None,
        indicators: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a strategy signal."""
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO trading.strategy_signals (
                        symbol, strategy_name, signal_type, signal_strength,
                        price, indicators, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """
                
                await conn.execute(
                    query,
                    symbol, strategy_name, signal_type, signal_strength,
                    price,
                    json.dumps(indicators) if indicators else '{}',
                    json.dumps(metadata) if metadata else '{}'
                )
                
                logger.debug(f"Signal logged: {symbol} - {strategy_name} - {signal_type}")
                
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")
            # Don't raise - signal logging is not critical
            
    async def update_balance(
        self,
        account_id: str,
        asset: str,
        free_balance: Decimal,
        locked_balance: Decimal,
        exchange: Optional[str] = None
    ):
        """Update account balance."""
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO trading.account_balances (
                        account_id, exchange, asset, free_balance, locked_balance
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (account_id, exchange, asset)
                    DO UPDATE SET
                        free_balance = EXCLUDED.free_balance,
                        locked_balance = EXCLUDED.locked_balance,
                        updated_at = NOW()
                """
                
                await conn.execute(
                    query,
                    account_id, exchange, asset, free_balance, locked_balance
                )
                
        except Exception as e:
            logger.error(f"Failed to update balance: {e}")
            # Don't raise - balance logging is not critical
            
    async def log_performance_metrics(
        self,
        strategy_name: str,
        symbol: Optional[str] = None,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        total_pnl: Decimal = Decimal('0'),
        win_rate: Optional[Decimal] = None,
        sharpe_ratio: Optional[Decimal] = None,
        max_drawdown: Optional[Decimal] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics."""
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO trading.performance_metrics (
                        strategy_name, symbol, total_trades, winning_trades,
                        losing_trades, total_pnl, win_rate, sharpe_ratio,
                        max_drawdown, period_start, period_end, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """
                
                await conn.execute(
                    query,
                    strategy_name, symbol, total_trades, winning_trades,
                    losing_trades, total_pnl, win_rate, sharpe_ratio,
                    max_drawdown, period_start, period_end,
                    json.dumps(metadata) if metadata else '{}'
                )
                
                logger.info(f"Performance metrics logged for {strategy_name}")
                
        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")
            # Don't raise - metrics logging is not critical
            
    async def get_recent_trades(
        self,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent trades from database."""
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT * FROM trading.trades
                    WHERE ($1::text IS NULL OR symbol = $1)
                      AND ($2::text IS NULL OR strategy_name = $2)
                    ORDER BY created_at DESC
                    LIMIT $3
                """
                
                rows = await conn.fetch(query, symbol, strategy_name, limit)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []


# Singleton instance
_db_logger: Optional[DatabaseLogger] = None


async def get_db_logger() -> DatabaseLogger:
    """Get or create the database logger instance."""
    global _db_logger
    if _db_logger is None:
        _db_logger = DatabaseLogger()
        await _db_logger.initialize()
    return _db_logger


async def close_db_logger():
    """Close the database logger."""
    global _db_logger
    if _db_logger:
        await _db_logger.close()
        _db_logger = None