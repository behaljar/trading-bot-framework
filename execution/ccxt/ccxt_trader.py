"""
CCXT-based trader with comprehensive error handling and state management
"""
import time
import ccxt
import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, Any, List
import logging
from enum import Enum
from dataclasses import dataclass, asdict

from .state_persistence import StateStore
from .file_state_store import FileStateStore
from .position_sync import PositionSynchronizer
from .data_manager import DataManager


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
    exchange_order_id: Optional[str] = None
    filled_size: float = 0.0
    average_price: float = 0.0
    error: Optional[str] = None


class CCXTTrader:
    """Production-ready CCXT trader with comprehensive error handling"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # State management - easily switchable to PostgreSQL later
        self.state_store: StateStore = FileStateStore()
        
        # Safety settings (need these before _initialize_exchange)
        self.emergency_stop = False
        self.max_retries = 3
        self.retry_delays = [1, 5, 15]  # Exponential backoff
        self.daily_loss_limit = config.initial_capital * 0.02  # 2% daily loss limit
        
        # State
        self.positions = {}
        self.orders = {}
        self.daily_pnl = 0.0
        self.initial_balance = None
        
        # Try to acquire lock
        lock_id = f"trader_{config.exchange_name}_{datetime.now().isoformat()}"
        if not self.state_store.acquire_lock(lock_id):
            raise RuntimeError("Another instance is already running!")
        self.lock_id = lock_id
        
        # Initialize exchange
        self._initialize_exchange()
        
        # Components
        self.data_manager = DataManager(
            max_bars=500,
            cache_duration=60  # Will be updated after initialization
        )
        self.position_sync = PositionSynchronizer(self.exchange, self.logger)
        
        # Update cache duration after data_manager is created
        cache_duration = self._get_cache_duration()
        self.data_manager.cache_duration = cache_duration
        
        # Get initial balance now that position_sync is created
        self._set_initial_balance()
        
        # Recovery
        self._recover_state()
        
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'state_store') and hasattr(self, 'lock_id'):
            self.state_store.release_lock(self.lock_id)
            
    def _initialize_exchange(self):
        """Initialize exchange with comprehensive error handling"""
        for attempt in range(self.max_retries):
            try:
                exchange_class = getattr(ccxt, self.config.exchange_name)
                
                # Exchange configuration
                exchange_config = {
                    'enableRateLimit': True,
                    'rateLimit': 1000,  # ms between requests
                    'timeout': 30000,
                    'options': {
                        'defaultType': getattr(self.config, 'trading_type', 'spot'),  # spot or future
                        'adjustForTimeDifference': True,
                    }
                }
                
                # Add credentials if provided
                if self.config.api_key and self.config.api_key != "":
                    exchange_config['apiKey'] = self.config.api_key
                    exchange_config['secret'] = self.config.api_secret
                    
                # Enable sandbox mode if configured
                if self.config.use_sandbox:
                    exchange_config['sandbox'] = True
                    
                self.exchange = exchange_class(exchange_config)
                
                # Test connection and load markets
                self.markets = self.exchange.load_markets()
                self.logger.info(
                    f"Exchange initialized: {self.config.exchange_name} "
                    f"{'SANDBOX' if self.config.use_sandbox else 'LIVE'} mode, "
                    f"{len(self.markets)} markets loaded"
                )
                
                # Exchange initialized successfully
                # Initial balance will be set after position_sync is created
                    
                return
                
            except Exception as e:
                self.logger.error(f"Exchange init attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                else:
                    raise
                    
    def _set_initial_balance(self):
        """Set initial balance after position_sync is available"""
        try:
            if self.config.api_key:
                account_info = self.position_sync.get_account_info()
                self.initial_balance = account_info.get('total_balance_usdt', self.config.initial_capital)
                self.logger.info(f"Account balance: {self.initial_balance} USDT")
            else:
                self.initial_balance = self.config.initial_capital
                self.logger.info(f"No API key provided, using config capital: {self.initial_balance}")
        except Exception as e:
            self.logger.warning(f"Could not get initial balance: {e}")
            self.initial_balance = self.config.initial_capital
                    
    def _recover_state(self):
        """Recover from saved state and sync with exchange"""
        try:
            # Load saved state
            saved_positions = self.state_store.load_positions()
            saved_orders = self.state_store.load_orders()
            
            # Load today's P&L
            today = date.today().isoformat()
            self.daily_pnl = self.state_store.load_daily_pnl(today)
            
            self.logger.info(f"Loaded state: {len(saved_positions)} positions, {len(saved_orders)} orders")
            
            # Sync positions with exchange
            self.logger.info("Synchronizing positions with exchange...")
            self.positions = self.position_sync.sync_positions(
                self.config.symbols,
                saved_positions
            )
            
            # Log discovered positions
            active_positions = {k: v for k, v in self.positions.items() if abs(v.get('size', 0)) > 0.0001}
            if active_positions:
                self.logger.warning(f"Found {len(active_positions)} existing positions:")
                for symbol, pos in active_positions.items():
                    side = pos.get('side', 'unknown')
                    size = pos.get('size', 0)
                    entry_price = pos.get('entry_price', 'unknown')
                    self.logger.warning(f"  {symbol}: {side} {abs(size)} @ {entry_price}")
            else:
                self.logger.info("No existing positions found")
            
            # Restore orders (only keeping recent ones)
            self.orders = saved_orders
            
            self.logger.info(f"State recovery complete: {len(active_positions)} active positions")
            
        except Exception as e:
            self.logger.error(f"State recovery failed: {e}")
            # Continue with empty state
            self.positions = {}
            self.orders = {}
            
    def run_trading_cycle(self, symbol: str, strategy):
        """Enhanced trading cycle with error recovery"""
        try:
            # Safety checks
            if self.emergency_stop:
                self.logger.error("Emergency stop active - no trading")
                return
                
            if abs(self.daily_pnl) > self.daily_loss_limit:
                self.logger.error(f"Daily loss limit reached: {self.daily_pnl}")
                self.emergency_stop = True
                return
                
            # Get data with retry
            df = self._get_data_with_retry(symbol)
            if df is None or len(df) < strategy.min_bars_required:
                self.logger.warning(f"Insufficient data for {symbol}")
                return
                
            # Generate signals
            signals = strategy.generate_signals(df)
            latest_signal = signals.iloc[-1]
            
            # Get current market state
            ticker = self._fetch_ticker_with_retry(symbol)
            if ticker is None:
                return
                
            current_price = ticker['last']
            
            # Get or sync position
            current_position = self._get_or_sync_position(symbol)
            
            # Determine action
            action = self._determine_action(
                current_position, latest_signal, current_price, strategy
            )
            
            # Execute if needed
            if action['type'] != 'hold':
                self._execute_action_safe(symbol, action, current_price)
                
            # Save state after each cycle
            self._save_current_state()
            
        except ccxt.RateLimitExceeded:
            self.logger.error(f"Rate limit exceeded for {symbol}")
            time.sleep(60)
        except ccxt.NetworkError as e:
            self.logger.error(f"Network error for {symbol}: {e}")
            time.sleep(5)
        except Exception as e:
            self.logger.error(f"Unexpected error for {symbol}: {e}", exc_info=True)
            
    def _get_data_with_retry(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self.data_manager.get_data_for_signals(
                    symbol, self.exchange, self.config.timeframe
                )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Data fetch attempt {attempt + 1} failed: {e}")
                    time.sleep(self.retry_delays[attempt])
                else:
                    self.logger.error(f"Failed to get data after {self.max_retries} attempts")
                    return None
                    
    def _fetch_ticker_with_retry(self, symbol: str) -> Optional[dict]:
        """Fetch ticker with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self.exchange.fetch_ticker(symbol)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                else:
                    self.logger.error(f"Failed to fetch ticker after {self.max_retries} attempts")
                    return None
                    
    def _get_or_sync_position(self, symbol: str) -> dict:
        """Get position, syncing with exchange if needed"""
        if symbol not in self.positions:
            # Try to sync this specific position
            try:
                balance = self.exchange.fetch_balance()
                base = symbol.split('/')[0]
                
                if base in balance and balance[base]['total'] > 0.0001:
                    self.positions[symbol] = {
                        'size': balance[base]['total'],
                        'entry_price': None,
                        'side': 'long',
                        'synchronized': True
                    }
                    self.logger.info(f"Found untracked position: {symbol} size={balance[base]['total']}")
                else:
                    self.positions[symbol] = {'size': 0, 'entry_price': None}
                    
            except Exception as e:
                self.logger.error(f"Failed to sync position for {symbol}: {e}")
                return {'size': 0}
                
        return self.positions.get(symbol, {'size': 0})
        
    def _determine_action(self, position: dict, signal: float, current_price: float, strategy) -> dict:
        """Determine what action to take based on position and signal"""
        position_size = position.get('size', 0)
        strategy_name = strategy.get_strategy_name()
        
        # Special handling for test strategy - alternating buy/sell
        if strategy_name == "TestStrategy":
            if position_size == 0 or position_size < 0.0001:
                # No position - BUY
                if signal > 0:
                    size = self._calculate_position_size(current_price, strategy)
                    self.logger.info(f"TEST: Opening position, size={size}")
                    return {'type': 'open_long', 'size': size}
            else:
                # Have position - SELL (close it)
                self.logger.info(f"TEST: Closing position, size={position_size}")
                return {'type': 'close_long', 'size': -position_size}
                
            return {'type': 'hold', 'size': 0}
        
        # Normal strategy logic
        # No position
        if position_size == 0 or position_size < 0.0001:
            if signal > 0:
                size = self._calculate_position_size(current_price, strategy)
                return {'type': 'open_long', 'size': size}
            elif signal < 0 and self.config.allow_short:  # Only if shorting is allowed
                size = self._calculate_position_size(current_price, strategy)
                return {'type': 'open_short', 'size': -size}
                
        # Long position
        elif position_size > 0:
            if signal <= 0:  # Exit signal
                return {'type': 'close_long', 'size': -position_size}
            elif self._should_stop_loss(position, current_price):
                return {'type': 'stop_loss', 'size': -position_size}
                
        # Short position
        elif position_size < 0:
            if signal >= 0:  # Exit signal
                return {'type': 'close_short', 'size': -position_size}
            elif self._should_stop_loss(position, current_price):
                return {'type': 'stop_loss', 'size': -position_size}
                
        return {'type': 'hold', 'size': 0}
        
    def _calculate_position_size(self, current_price: float, strategy) -> float:
        """Calculate position size based on risk management rules"""
        # Get account balance
        try:
            account_info = self.position_sync.get_account_info()
            available_balance = account_info.get('balances', {}).get('USDT', {}).get('free', 0)
            if available_balance == 0:
                # Fallback to total USDT balance
                available_balance = account_info.get('total_balance_usdt', self.config.initial_capital * 0.1)
        except Exception as e:
            self.logger.warning(f"Could not get balance, using fallback: {e}")
            available_balance = self.config.initial_capital * 0.1  # Conservative fallback
            
        # Apply max position size limit
        max_position_value = min(
            available_balance * 0.95,  # Use 95% of available balance max
            self.config.initial_capital * self.config.max_position_size
        )
        
        # Calculate size in base currency
        position_size = max_position_value / current_price
        
        # Apply strategy-specific sizing if available
        if hasattr(strategy, 'get_position_size'):
            strategy_size = strategy.get_position_size(None, 1, available_balance)
            position_size = min(position_size, strategy_size / current_price)
            
        return position_size
        
    def _should_stop_loss(self, position: dict, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if not position.get('entry_price'):
            return False
            
        entry_price = position['entry_price']
        position_size = position['size']
        
        if position_size > 0:  # Long position
            loss_pct = (entry_price - current_price) / entry_price
        else:  # Short position
            loss_pct = (current_price - entry_price) / entry_price
            
        return loss_pct >= self.config.stop_loss_pct
        
    def _execute_action_safe(self, symbol: str, action: dict, current_price: float):
        """Execute action with comprehensive error handling"""
        order_id = f"{symbol}_{action['type']}_{int(time.time())}"
        
        try:
            # Pre-execution validation
            if not self._validate_order(symbol, action['size'], current_price):
                return
                
            # Create order tracking
            order = Order(
                id=order_id,
                symbol=symbol,
                side='buy' if action['size'] > 0 else 'sell',
                size=abs(action['size']),
                order_type='market',
                status=OrderStatus.PENDING,
                timestamp=datetime.now().isoformat()
            )
            self.orders[order_id] = asdict(order)
            
            # Execute order
            exchange_order = self._place_order_with_retry(symbol, action['size'])
            
            if exchange_order:
                # Update order tracking
                self.orders[order_id]['exchange_order_id'] = exchange_order['id']
                self.orders[order_id]['status'] = OrderStatus.PLACED.value
                
                # Wait for fill
                filled_order = self._wait_for_fill(exchange_order['id'], symbol)
                
                if filled_order:
                    # Update position
                    self._update_position_from_order(symbol, filled_order, action['type'])
                    self.orders[order_id]['status'] = OrderStatus.FILLED.value
                    self.orders[order_id]['filled_size'] = filled_order.get('filled', filled_order.get('amount'))
                    self.orders[order_id]['average_price'] = filled_order.get('average', filled_order.get('price'))
                    
                    self.logger.info(
                        f"Order filled: {action['type']} {symbol} "
                        f"size={filled_order.get('filled')} @ {filled_order.get('average')}"
                    )
                else:
                    self.orders[order_id]['status'] = OrderStatus.FAILED.value
                    
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            self.orders[order_id]['status'] = OrderStatus.FAILED.value
            self.orders[order_id]['error'] = str(e)
            
    def _validate_order(self, symbol: str, size: float, current_price: float) -> bool:
        """Validate order before execution"""
        try:
            # Check market info
            market = self.markets.get(symbol)
            if not market:
                self.logger.error(f"Market {symbol} not found")
                return False
                
            # Check minimum order size
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if abs(size) < min_amount:
                self.logger.error(f"Order size {abs(size)} below minimum {min_amount}")
                return False
                
            # Check precision
            amount_precision = market.get('precision', {}).get('amount', 8)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False
            
    def _place_order_with_retry(self, symbol: str, size: float) -> Optional[dict]:
        """Place order with retry and error handling"""
        for attempt in range(self.max_retries):
            try:
                if size > 0:
                    order = self.exchange.create_market_buy_order(symbol, size)
                else:
                    order = self.exchange.create_market_sell_order(symbol, abs(size))
                    
                self.logger.info(f"Order placed: {order['id']} for {symbol}")
                return order
                
            except ccxt.InsufficientFunds as e:
                self.logger.error(f"Insufficient funds: {e}")
                return None
                
            except ccxt.InvalidOrder as e:
                self.logger.error(f"Invalid order: {e}")
                return None
                
            except ccxt.ExchangeNotAvailable as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Exchange unavailable, retrying...")
                    time.sleep(self.retry_delays[attempt])
                else:
                    return None
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                    time.sleep(self.retry_delays[attempt])
                else:
                    return None
                    
    def _wait_for_fill(self, order_id: str, symbol: str, timeout: int = 30) -> Optional[dict]:
        """Wait for order to fill with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order = self.exchange.fetch_order(order_id, symbol)
                
                if order['status'] == 'closed':
                    return order
                elif order['status'] == 'canceled':
                    self.logger.warning(f"Order {order_id} was cancelled")
                    return None
                    
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error checking order status: {e}")
                time.sleep(2)
                
        self.logger.error(f"Order {order_id} timeout after {timeout}s")
        return None
        
    def _update_position_from_order(self, symbol: str, order: dict, action_type: str):
        """Update position based on filled order"""
        filled_size = order.get('filled', order.get('amount', 0))
        average_price = order.get('average', order.get('price', 0))
        
        if symbol not in self.positions:
            self.positions[symbol] = {'size': 0, 'entry_price': None}
            
        current_position = self.positions[symbol]
        
        if action_type in ['open_long', 'open_short']:
            # New position
            self.positions[symbol] = {
                'size': filled_size if 'long' in action_type else -filled_size,
                'entry_price': average_price,
                'side': 'long' if 'long' in action_type else 'short',
                'entry_time': datetime.now().isoformat()
            }
        else:
            # Closing position - calculate P&L
            if current_position.get('entry_price'):
                if current_position['size'] > 0:  # Was long
                    pnl = (average_price - current_position['entry_price']) * filled_size
                else:  # Was short
                    pnl = (current_position['entry_price'] - average_price) * filled_size
                    
                # Consider fees
                fee_amount = filled_size * average_price * self.config.commission
                net_pnl = pnl - fee_amount
                
                self.daily_pnl += net_pnl
                self.logger.info(f"Position closed: P&L = {net_pnl:.2f} USDT")
                
            # Clear position
            self.positions[symbol] = {'size': 0, 'entry_price': None}
            
    def _save_current_state(self):
        """Save current state to disk"""
        try:
            self.state_store.save_positions(self.positions)
            self.state_store.save_orders(self.orders)
            self.state_store.save_daily_pnl(date.today().isoformat(), self.daily_pnl)
            
            # Save checkpoint
            checkpoint = {
                'positions': self.positions,
                'orders': self.orders,
                'daily_pnl': self.daily_pnl,
                'emergency_stop': self.emergency_stop,
                'timestamp': datetime.now().isoformat()
            }
            self.state_store.save_checkpoint(checkpoint)
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            
    def _get_cache_duration(self) -> int:
        """Get appropriate cache duration based on timeframe"""
        # Convert timeframe to seconds and use half of it for cache
        timeframe_seconds = self.data_manager.get_timeframe_seconds(self.config.timeframe)
        return max(60, timeframe_seconds // 2)  # At least 60 seconds
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            account_info = self.position_sync.get_account_info()
            current_balance = account_info.get('total_balance_usdt', self.initial_balance)
            
            return {
                "initial_balance": self.initial_balance,
                "current_balance": current_balance,
                "daily_pnl": self.daily_pnl,
                "total_pnl": current_balance - self.initial_balance if self.initial_balance else 0,
                "open_positions": len([p for p in self.positions.values() if p.get('size', 0) != 0]),
                "emergency_stop": self.emergency_stop,
                "mode": "SANDBOX" if self.config.use_sandbox else "LIVE"
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}