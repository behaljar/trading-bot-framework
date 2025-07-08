"""
Order book data source for high-frequency trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
from .base_data_source import DataSource


class OrderBookDataSource(DataSource):
    """
    Order book data source that provides:
    - Real-time bid/ask data
    - Order book depth
    - Market microstructure indicators
    - Tick-by-tick data
    """
    
    def __init__(self, exchange_source=None, symbols: List[str] = None):
        """
        Initialize order book data source.
        
        Args:
            exchange_source: CCXT exchange instance or similar
            symbols: List of symbols to track
        """
        self.exchange = exchange_source
        self.symbols = symbols or []
        self.logger = logging.getLogger(__name__)
        self.orderbook_cache = {}
        self.last_update = {}
        
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1m") -> pd.DataFrame:
        """
        Get historical order book data aggregated into OHLCV format.
        
        Note: This simulates historical data. In production, you'd need
        stored order book snapshots.
        """
        try:
            # For now, we'll simulate historical data based on current order book
            # In production, you'd load stored order book snapshots
            current_ob = self.get_current_orderbook(symbol)
            
            if not current_ob:
                return pd.DataFrame()
            
            # Generate synthetic historical data
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            
            # Create time range
            freq_map = {
                '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
                '1h': '1H', '4h': '4H', '1d': '1D'
            }
            
            time_range = pd.date_range(
                start=start_dt, 
                end=end_dt, 
                freq=freq_map.get(timeframe, '1T')
            )
            
            # Simulate price movement around current mid price
            mid_price = (current_ob['best_bid'] + current_ob['best_ask']) / 2
            
            # Generate synthetic OHLCV data
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0, 0.001, len(time_range))  # 0.1% volatility
            
            prices = [mid_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV bars
            data = []
            for i in range(len(time_range)):
                base_price = prices[i]
                high = base_price * (1 + abs(np.random.normal(0, 0.0005)))
                low = base_price * (1 - abs(np.random.normal(0, 0.0005)))
                close = base_price * (1 + np.random.normal(0, 0.0002))
                
                data.append({
                    'Open': base_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': np.random.uniform(1000, 10000),
                    'bid': close * 0.9995,  # Simulated bid
                    'ask': close * 1.0005,  # Simulated ask
                    'spread': close * 0.001,  # Simulated spread
                    'bid_volume': np.random.uniform(100, 1000),
                    'ask_volume': np.random.uniform(100, 1000)
                })
            
            df = pd.DataFrame(data, index=time_range)
            df.index.name = 'timestamp'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical order book data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_orderbook(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """
        Get current order book for symbol.
        
        Args:
            symbol: Trading symbol
            depth: Order book depth (number of levels)
            
        Returns:
            Dict with order book data or None if not available
        """
        try:
            if self.exchange:
                # Real order book from exchange
                ob = self.exchange.fetch_order_book(symbol, depth)
                
                # Process order book
                processed_ob = self._process_orderbook(ob)
                
                # Cache for later use
                self.orderbook_cache[symbol] = processed_ob
                self.last_update[symbol] = datetime.now()
                
                return processed_ob
            else:
                # Simulated order book for testing
                return self._simulate_orderbook(symbol)
                
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def _process_orderbook(self, raw_ob: Dict) -> Dict:
        """Process raw order book into structured format."""
        bids = raw_ob.get('bids', [])
        asks = raw_ob.get('asks', [])
        
        if not bids or not asks:
            return {}
            
        # Best bid/ask
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        
        # Calculate spread
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000
        
        # Calculate depth
        bid_volume = sum(level[1] for level in bids)
        ask_volume = sum(level[1] for level in asks)
        
        # Mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Order book imbalance
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        return {
            'timestamp': datetime.now(),
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'bids': bids,
            'asks': asks
        }
    
    def _simulate_orderbook(self, symbol: str) -> Dict:
        """Simulate order book for testing."""
        # Simulate around a base price
        base_price = 100.0
        if symbol == 'BTC/USDT:USDT':
            base_price = 43000.0
        elif symbol == 'ETH/USDT:USDT':
            base_price = 2500.0
        
        # Add some randomness
        price_noise = np.random.normal(0, base_price * 0.001)
        mid_price = base_price + price_noise
        
        # Simulate spread (0.01% to 0.05%)
        spread = mid_price * np.random.uniform(0.0001, 0.0005)
        
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Simulate order book levels
        bids = []
        asks = []
        
        for i in range(10):
            # Bid levels going down
            bid_price = best_bid - i * spread * 0.1
            bid_volume = np.random.uniform(0.1, 10.0)
            bids.append([bid_price, bid_volume])
            
            # Ask levels going up
            ask_price = best_ask + i * spread * 0.1
            ask_volume = np.random.uniform(0.1, 10.0)
            asks.append([ask_price, ask_volume])
        
        return {
            'timestamp': datetime.now(),
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': spread,
            'spread_bps': (spread / mid_price) * 10000,
            'bid_volume': sum(level[1] for level in bids),
            'ask_volume': sum(level[1] for level in asks),
            'imbalance': np.random.uniform(-0.3, 0.3),
            'bids': bids,
            'asks': asks
        }
    
    def get_current_price(self, symbol: str) -> Dict:
        """Get current price from order book."""
        ob = self.get_current_orderbook(symbol)
        if ob:
            return {
                'bid': ob['best_bid'],
                'ask': ob['best_ask'],
                'last': ob['mid_price']
            }
        return {'bid': 0, 'ask': 0, 'last': 0}
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """Get order book (same as get_current_orderbook)."""
        return self.get_current_orderbook(symbol, limit)
    
    def get_market_depth(self, symbol: str, depth_usd: float = 1000) -> Dict:
        """
        Calculate market depth - how much you can buy/sell for given USD amount.
        
        Args:
            symbol: Trading symbol
            depth_usd: USD amount to calculate depth for
            
        Returns:
            Dict with depth analysis
        """
        ob = self.get_current_orderbook(symbol)
        if not ob:
            return {}
            
        # Calculate how much we can buy for depth_usd
        remaining_usd = depth_usd
        buy_quantity = 0
        avg_buy_price = 0
        
        for ask_price, ask_volume in ob['asks']:
            max_qty_at_level = remaining_usd / ask_price
            qty_to_buy = min(max_qty_at_level, ask_volume)
            
            buy_quantity += qty_to_buy
            avg_buy_price = ((avg_buy_price * (buy_quantity - qty_to_buy)) + 
                           (ask_price * qty_to_buy)) / buy_quantity
            
            remaining_usd -= qty_to_buy * ask_price
            
            if remaining_usd <= 0:
                break
        
        # Calculate how much we can sell for depth_usd worth
        target_quantity = depth_usd / ob['mid_price']
        remaining_qty = target_quantity
        sell_value = 0
        avg_sell_price = 0
        
        for bid_price, bid_volume in ob['bids']:
            qty_to_sell = min(remaining_qty, bid_volume)
            
            sell_value += qty_to_sell * bid_price
            avg_sell_price = ((avg_sell_price * (target_quantity - remaining_qty)) + 
                            (bid_price * qty_to_sell)) / (target_quantity - remaining_qty + qty_to_sell)
            
            remaining_qty -= qty_to_sell
            
            if remaining_qty <= 0:
                break
        
        return {
            'buy_quantity': buy_quantity,
            'avg_buy_price': avg_buy_price,
            'sell_value': sell_value,
            'avg_sell_price': avg_sell_price,
            'price_impact_buy': (avg_buy_price - ob['mid_price']) / ob['mid_price'],
            'price_impact_sell': (ob['mid_price'] - avg_sell_price) / ob['mid_price']
        }