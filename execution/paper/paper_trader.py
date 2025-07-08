"""
Main paper trading engine with real-time data streaming.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

import pandas as pd

from data.base_data_source import DataSource
from strategies.base_strategy import BaseStrategy, Signal
import logging
from config.settings import TradingConfig
from .virtual_portfolio import VirtualPortfolio, VirtualPosition, VirtualOrder
from .order_simulator import OrderSimulator
from .performance_tracker import PerformanceTracker
from risk.position_calculator import PositionCalculator


class PaperTrader:
    """Main paper trading engine."""
    
    def __init__(self, config: TradingConfig, data_source: DataSource, 
                 strategy: BaseStrategy, position_size_pct: float = 0.1,
                 risk_pct: float = 0.02):
        self.config = config
        self.data_source = data_source
        self.strategy = strategy
        self.logger = logging.getLogger("TradingBot")
        
        # Initialize position calculator
        self.position_calc = PositionCalculator(
            balance_percentage=position_size_pct,
            risk_percentage=risk_pct,
            max_position_pct=config.max_position_size
        )
        
        # Initialize components
        self.portfolio = VirtualPortfolio(config.initial_capital)
        self.order_simulator = OrderSimulator(config)
        self.performance_tracker = PerformanceTracker()
        
        # State management
        self.state_dir = Path(config.state_directory) / 'paper'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.order_counter = 0
        
        # Performance tracking
        self.start_time = None
        self.performance_history = []
        
        # Data caching for efficiency
        self.cached_data = {}  # symbol -> DataFrame
        self.last_bar_time = {}  # symbol -> datetime
        
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"PAPER_{int(time.time())}_{self.order_counter}"
        
    def _wait_for_next_candle(self, interval: str):
        """Wait until the next candle close time + buffer for data availability."""
        now = datetime.now()
        
        # Parse interval and add 1 second buffer for data availability
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            # Calculate next minute boundary aligned to interval
            current_minute = now.minute
            next_minute = ((current_minute // minutes) + 1) * minutes
            
            if next_minute >= 60:
                next_minute = 0
                next_time = now.replace(hour=now.hour + 1, minute=next_minute, second=1, microsecond=0)
            else:
                next_time = now.replace(minute=next_minute, second=1, microsecond=0)
                
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            # Calculate next hour boundary aligned to interval
            current_hour = now.hour
            next_hour = ((current_hour // hours) + 1) * hours
            
            if next_hour >= 24:
                next_hour = 0
                next_time = now.replace(day=now.day + 1, hour=next_hour, minute=0, second=1, microsecond=0)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=1, microsecond=0)
                
        elif interval == '1d':
            # Next day at midnight + 1 second
            next_time = now.replace(hour=0, minute=0, second=1, microsecond=0) + timedelta(days=1)
        else:
            # Default: wait 60 seconds
            next_time = now + timedelta(seconds=60)
        
        # Calculate sleep time
        sleep_seconds = (next_time - now).total_seconds()
        
        if sleep_seconds > 0:
            self.logger.info(f"Waiting {sleep_seconds:.1f}s until next {interval} candle closes at {next_time.strftime('%H:%M:%S')}")
            time.sleep(sleep_seconds)
        
    def _save_state(self):
        """Save current state to disk."""
        state = {
            'portfolio': {
                'balance': self.portfolio.balance,
                'positions': {
                    symbol: asdict(pos) for symbol, pos in self.portfolio.positions.items()
                },
                'initial_balance': self.portfolio.initial_balance
            },
            'order_history': [asdict(order) for order in self.portfolio.order_history[-100:]],
            'trade_history': self.portfolio.trade_history[-100:],
            'performance_history': self.performance_history[-1000:],
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = self.state_dir / 'paper_trader_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
    def _load_state(self) -> bool:
        """Load state from disk if exists."""
        state_file = self.state_dir / 'paper_trader_state.json'
        if not state_file.exists():
            return False
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            # Restore portfolio
            self.portfolio.balance = state['portfolio']['balance']
            self.portfolio.initial_balance = state['portfolio']['initial_balance']
            
            # Restore positions
            for symbol, pos_data in state['portfolio']['positions'].items():
                # Convert string datetime back to datetime object
                if 'entry_time' in pos_data and isinstance(pos_data['entry_time'], str):
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                self.portfolio.positions[symbol] = VirtualPosition(**pos_data)
                
            # Restore history
            self.portfolio.trade_history = state.get('trade_history', [])
            self.performance_history = state.get('performance_history', [])
            
            self.logger.info("State loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
            
    def execute_signal(self, symbol: str, signal, current_data: pd.Series) -> Optional[VirtualOrder]:
        """Execute trading signal."""
        # Convert signal value to Signal enum if needed
        if isinstance(signal, (int, float)):
            if signal == 1:
                signal = Signal.BUY
            elif signal == -1:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
        
        self.logger.info(f"Processing signal: {signal} (value: {signal.value if hasattr(signal, 'value') else signal})")
        
        current_price = float(current_data['Close'])
        
        # Update position prices
        self.portfolio.update_position_price(symbol, current_price)
        
        # Check current position
        has_position = symbol in self.portfolio.positions
        position = self.portfolio.positions.get(symbol)
        
        self.logger.info(f"Current position status: has_position={has_position}, balance=${self.portfolio.balance:.2f}")
        
        # Determine action
        if signal == Signal.BUY and not has_position:
            self.logger.info(f"BUY signal with no position - opening long position")
            # Get stop loss and take profit from strategy data if available
            stop_loss = current_data.get('stop_loss', None)
            take_profit = current_data.get('take_profit', None)
            
            # Calculate position size using position calculator
            portfolio_value = self.portfolio.get_total_value()
            size = self.position_calc.calculate_position_size(
                balance=portfolio_value,
                entry_price=current_price,
                stop_loss=stop_loss,
                position_type='long'
            )
            
            if size > 0:
                # Create and simulate order
                order = VirtualOrder(
                    order_id=self._generate_order_id(),
                    symbol=symbol,
                    side='buy',
                    size=size,
                    order_type='market'
                )
                
                # Get bid/ask if available
                try:
                    ticker = self.data_source.get_current_price(symbol)
                    if isinstance(ticker, dict):
                        bid = ticker.get('bid', current_price)
                        ask = ticker.get('ask', current_price)
                    else:
                        bid = ask = current_price
                except:
                    bid = ask = current_price
                
                # Simulate execution
                order = self.order_simulator.simulate_market_order(
                    order, current_price, bid, ask
                )
                
                # Execute in portfolio
                if self.portfolio.execute_order(order):
                    # Store stop loss and take profit in position
                    if symbol in self.portfolio.positions:
                        position = self.portfolio.positions[symbol]
                        position.stop_loss = stop_loss
                        position.take_profit = take_profit
                    
                    self.logger.info(
                        f"BUY executed: {symbol} | Size: {size:.4f} | "
                        f"Price: ${order.filled_price:,.2f} | "
                        f"Commission: ${order.commission:.2f} | "
                        f"SL: ${stop_loss:.2f if stop_loss else 'None'} | "
                        f"TP: ${take_profit:.2f if take_profit else 'None'}"
                    )
                    return order
                    
        elif signal == Signal.SELL:
            if has_position and position.side == 'long':
                self.logger.info(f"SELL signal with long position - closing position")
                # Close long position
                order = VirtualOrder(
                    order_id=self._generate_order_id(),
                    symbol=symbol,
                    side='sell',
                    size=position.size,
                    order_type='market'
                )
            elif not has_position:
                # For test strategy: SELL signal with no position = open new long position  
                self.logger.info(f"SELL signal with no position - opening long position (test strategy)")
                # Get stop loss and take profit from strategy data if available
                stop_loss = current_data.get('stop_loss', None)
                take_profit = current_data.get('take_profit', None)
                
                # Calculate position size using position calculator
                portfolio_value = self.portfolio.get_total_value()
                size = self.position_calc.calculate_position_size(
                    balance=portfolio_value,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    position_type='long'
                )
                
                if size > 0:
                    # Create and simulate order
                    order = VirtualOrder(
                        order_id=self._generate_order_id(),
                        symbol=symbol,
                        side='buy',
                        size=size,
                        order_type='market'
                    )
                else:
                    self.logger.warning(f"Position size calculation returned 0 for {symbol}")
                    return None
            else:
                self.logger.info(f"SELL signal but no appropriate position to close")
                return None
            
            # Get bid/ask if available
            try:
                ticker = self.data_source.get_current_price(symbol)
                if isinstance(ticker, dict):
                    bid = ticker.get('bid', current_price)
                    ask = ticker.get('ask', current_price)
                else:
                    bid = ask = current_price
            except:
                bid = ask = current_price
            
            # Simulate execution
            order = self.order_simulator.simulate_market_order(
                order, current_price, bid, ask
            )
            
            # Execute in portfolio
            if self.portfolio.execute_order(order):
                # Get updated position info after order execution
                updated_position = self.portfolio.positions.get(symbol)
                if updated_position and order.side == 'sell':
                    # Closing position
                    self.logger.info(
                        f"SELL executed: {symbol} | Size: {order.size:.4f} | "
                        f"Price: ${order.filled_price:,.2f} | "
                        f"Commission: ${order.commission:.2f} | "
                        f"P&L: ${updated_position.unrealized_pnl:.2f}"
                    )
                elif order.side == 'buy':
                    # Opening position (test strategy: SELL signal = BUY order)
                    self.logger.info(
                        f"BUY executed (test strategy): {symbol} | Size: {order.size:.4f} | "
                        f"Price: ${order.filled_price:,.2f} | "
                        f"Commission: ${order.commission:.2f}"
                    )
                else:
                    # Fallback logging
                    self.logger.info(
                        f"Order executed: {symbol} | Side: {order.side} | Size: {order.size:.4f} | "
                        f"Price: ${order.filled_price:,.2f} | Commission: ${order.commission:.2f}"
                    )
                return order
                
        return None
        
    def _fetch_and_cache_data(self, symbol: str, interval: str, max_bars: int = 1000) -> pd.DataFrame:
        """Efficiently fetch data using caching and sliding window."""
        current_time = datetime.now()
        
        # Check if we have cached data for this symbol
        if symbol in self.cached_data and symbol in self.last_bar_time:
            cached_df = self.cached_data[symbol]
            last_time = self.last_bar_time[symbol]
            
            # Try to fetch only new data since last bar
            try:
                # Add buffer to ensure we get the latest data
                start_date = (last_time + timedelta(minutes=1)).strftime('%Y-%m-%d')
                end_date = (current_time + timedelta(days=1)).strftime('%Y-%m-%d')
                
                self.logger.info(f"Fetching new data for {symbol} from {start_date}")
                
                new_df = self.data_source.get_historical_data(
                    symbol, 
                    start_date,
                    end_date,
                    timeframe=interval
                )
                
                if new_df is not None and not new_df.empty and len(new_df) > 0:
                    # Remove any overlapping data to avoid duplicates
                    new_df = new_df[new_df.index > last_time]
                    
                    if not new_df.empty:
                        self.logger.info(f"Got {len(new_df)} new bars for {symbol}")
                        
                        # Concatenate with cached data
                        combined_df = pd.concat([cached_df, new_df])
                        
                        # Apply sliding window - keep only last max_bars
                        if len(combined_df) > max_bars:
                            combined_df = combined_df.tail(max_bars)
                            self.logger.info(f"Applied sliding window, keeping last {max_bars} bars")
                        
                        # Update cache
                        self.cached_data[symbol] = combined_df
                        self.last_bar_time[symbol] = combined_df.index[-1]
                        
                        return combined_df
                    else:
                        self.logger.info(f"No new data for {symbol}, using cached data")
                        return cached_df
                else:
                    self.logger.info(f"No new data received for {symbol}, using cached data")
                    return cached_df
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch incremental data for {symbol}: {e}, falling back to full fetch")
        
        # Initial fetch or fallback - get full dataset
        self.logger.info(f"Performing initial data fetch for {symbol}")
        
        # Calculate required bars and date range
        required_bars = getattr(self.strategy, 'min_bars_required', 50)
        
        # Use larger of required bars or max_bars to ensure we have enough history
        fetch_bars = max(required_bars, max_bars)
        
        # Calculate days needed based on timeframe
        if interval == '1m':
            days_needed = max(1, (fetch_bars // 1440) + 2)
        elif interval == '5m':
            days_needed = max(1, (fetch_bars // 288) + 2)
        elif interval == '15m':
            days_needed = max(1, (fetch_bars // 96) + 2)
        elif interval == '1h':
            days_needed = max(1, (fetch_bars // 24) + 2)
        elif interval == '4h':
            days_needed = max(1, (fetch_bars // 6) + 2)
        elif interval == '1d':
            days_needed = max(1, fetch_bars + 5)
        else:
            days_needed = self.config.data_lookback_days
            
        end_date = current_time + timedelta(days=1)
        start_date = end_date - timedelta(days=days_needed)
        
        df = self.data_source.get_historical_data(
            symbol, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            timeframe=interval
        )
        
        if df is not None and not df.empty:
            # Apply sliding window immediately
            if len(df) > max_bars:
                df = df.tail(max_bars)
                self.logger.info(f"Applied sliding window on initial fetch, keeping last {max_bars} bars")
            
            # Cache the data
            self.cached_data[symbol] = df
            self.last_bar_time[symbol] = df.index[-1]
            
            self.logger.info(f"Cached {len(df)} bars for {symbol}")
            
        return df

    def run_paper_trading(self, symbols: List[str], interval: str = '1h'):
        """Run paper trading with real-time data updates."""
        self.logger.info(f"Starting paper trading for {symbols} with {interval} bars")
        self.running = True
        self.start_time = datetime.now()
        
        # Load previous state if exists
        self._load_state()
        
        try:
            while self.running:
                # Wait for next candle close time
                self._wait_for_next_candle(interval)
                
                for symbol in symbols:
                    try:
                        self.logger.info(f"Processing symbol: {symbol}")
                        
                        # Get data using efficient caching
                        df = self._fetch_and_cache_data(symbol, interval, max_bars=1000)
                        
                        if df is None or df.empty:
                            self.logger.warning(f"No data received for {symbol}")
                            continue
                            
                        self.logger.info(f"Using {len(df)} bars for {symbol} (cached: {symbol in self.cached_data})")
                            
                        # Generate signals
                        self.logger.info(f"Generating signals for {symbol}")
                        signals_df = self.strategy.generate_signals(df)
                        self.logger.info(f"Generated {len(signals_df)} signals, unique values: {signals_df.unique()}")
                        
                        # Merge signals with original data
                        signals_df.name = 'signal'  # Ensure the Series has a name
                        df = pd.concat([df, signals_df], axis=1)
                        current_bar = df.iloc[-1]
                        
                        # Get current price
                        current_price = float(current_bar['Close'])
                        self.portfolio.update_position_price(symbol, current_price)
                        
                        # Debug output
                        bar_time = df.index[-1]
                        signal_value = current_bar.get('signal', Signal.HOLD.value)
                        self.logger.info(
                            f"{symbol} - Last bar: {bar_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                            f"Close: ${current_price:,.2f} | "
                            f"Signal: {signal_value} | "
                            f"Cached bars: {len(self.cached_data.get(symbol, []))}"
                        )
                        
                        # Execute signal if any
                        if 'signal' in current_bar and current_bar['signal'] != Signal.HOLD.value:
                            self.logger.info(f"Executing signal: {current_bar['signal']} for {symbol}")
                            result = self.execute_signal(symbol, current_bar['signal'], current_bar)
                            if result:
                                self.logger.info(f"Order executed successfully: {result.order_id}")
                            else:
                                self.logger.warning(f"Signal execution failed for {symbol}")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        
                # Update performance tracker
                current_time = datetime.now()
                portfolio_value = self.portfolio.get_total_value()
                self.performance_tracker.update_equity(current_time, portfolio_value)
                
                # Calculate comprehensive metrics
                metrics = self.performance_tracker.calculate_metrics(
                    self.portfolio, self.start_time
                )
                metrics['timestamp'] = current_time.isoformat()
                self.performance_history.append(metrics)
                
                # Log performance
                self.logger.info(
                    f"[PERFORMANCE] Value: ${metrics['total_value']:,.2f} | "
                    f"P&L: ${metrics['total_pnl']:+,.2f} ({metrics['total_return_pct']:+.2f}%) | "
                    f"Trades: {metrics['num_trades']} | "
                    f"Open Pos: {metrics['open_positions']} | "
                    f"Win Rate: {metrics.get('win_rate', 0):.1%}"
                )
                
                # Save state
                self._save_state()
                
        except KeyboardInterrupt:
            self.logger.info("Paper trading stopped by user")
        except Exception as e:
            self.logger.error(f"Paper trading error: {e}")
        finally:
            self.running = False
            self._save_state()
            self._print_final_report()
            
    def _print_final_report(self):
        """Print final performance report."""
        # Calculate final metrics
        metrics = self.performance_tracker.calculate_metrics(
            self.portfolio, self.start_time
        )
        
        # Generate and print comprehensive report
        report = self.performance_tracker.generate_report(metrics)
        print(report)
        
        # Save detailed report to file
        report_file = self.state_dir / f"paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        self.logger.info(f"Detailed report saved to {report_file}")