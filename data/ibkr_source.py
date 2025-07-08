"""
IBKR data source implementation using ib_async.
Provides historical and real-time market data from Interactive Brokers.
"""
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import time

try:
    from ib_async import IB, Contract, BarData
    from ib_async.util import df
except ImportError:
    raise ImportError(
        "ib_async is required. Install with: pip install ib_async>=0.9.86"
    )

from data.base_data_source import DataSource
from data.ibkr_connection import IBKRConnectionManager
from config.ibkr_config import IBKRConfig, create_ibkr_config


class IBKRDataSource(DataSource):
    """
    Professional IBKR data source with caching and error handling.
    Supports both historical and real-time market data.
    """
    
    def __init__(self, config: Optional[IBKRConfig] = None):
        self.config = config or create_ibkr_config()
        self.connection = IBKRConnectionManager(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Data caching
        self.historical_cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
        # Real-time data
        self.active_subscriptions: Dict[str, Contract] = {}
        self.real_time_data: Dict[str, Dict] = {}
        
        # IBKR timeframe mapping
        self.timeframe_map = {
            '1m': '1 min',
            '5m': '5 mins',
            '15m': '15 mins',
            '30m': '30 mins',
            '1h': '1 hour',
            '2h': '2 hours',
            '4h': '4 hours',
            '1d': '1 day',
            '1w': '1 week',
            '1M': '1 month'
        }
    
    async def connect(self) -> bool:
        """Connect to IBKR"""
        return await self.connection.connect()
    
    async def disconnect(self):
        """Disconnect from IBKR"""
        await self.connection.disconnect()
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connection.is_connected()
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for IBKR"""
        # Remove any exchange suffixes and clean up
        symbol = symbol.split('.')[0].split(':')[0].upper()
        return symbol
    
    def _create_contract(self, symbol: str) -> Contract:
        """Create IBKR contract from symbol"""
        normalized = self._normalize_symbol(symbol)
        
        # Auto-detect contract type based on symbol
        if '/' in symbol:
            # Forex pair
            return self.connection.create_contract(normalized, 'FOREX')
        elif symbol.upper() in ['SPY', 'QQQ', 'IWM', 'DIA']:
            # ETFs
            return self.connection.create_contract(normalized, 'STK', 'ARCA')
        else:
            # Default to stock
            return self.connection.create_contract(normalized, 'STK')
    
    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """Generate cache key for historical data"""
        return f"{symbol}_{timeframe}_{start_date}_{end_date}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _convert_ib_bars_to_dataframe(self, bars: List[BarData]) -> pd.DataFrame:
        """Convert IBKR bar data to pandas DataFrame"""
        if not bars:
            return pd.DataFrame()
        
        # Convert to DataFrame using ib_async utility
        df_data = df(bars)
        
        if df_data.empty:
            return pd.DataFrame()
        
        # Ensure we have the required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df_data.columns for col in required_columns):
            self.logger.error(f"Missing required columns in IBKR data: {df_data.columns}")
            return pd.DataFrame()
        
        # Rename columns to match our standard format
        df_data = df_data.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Set datetime index
        if 'Date' in df_data.columns:
            df_data['Date'] = pd.to_datetime(df_data['Date'])
            df_data.set_index('Date', inplace=True)
        
        # Sort by date
        df_data.sort_index(inplace=True)
        
        return df_data
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                                timeframe: str = '1h') -> Optional[pd.DataFrame]:
        """
        Get historical data from IBKR
        
        Args:
            symbol: Symbol (e.g., 'AAPL', 'SPY')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format) 
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.is_connected():
            await self.connect()
        
        if not self.is_connected():
            self.logger.error("Not connected to IBKR")
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
        if cache_key in self.historical_cache and self._is_cache_valid(cache_key):
            self.logger.debug(f"Returning cached data for {symbol}")
            return self.historical_cache[cache_key].copy()
        
        try:
            # Apply rate limiting
            await self.connection.rate_limit()
            
            # Create contract
            contract = self._create_contract(symbol)
            
            # Qualify contract to get conId
            contracts = await self.connection.ib.qualifyContractsAsync(contract)
            if not contracts:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return None
            
            qualified_contract = contracts[0]
            
            # Map timeframe
            if timeframe not in self.timeframe_map:
                self.logger.error(f"Unsupported timeframe: {timeframe}")
                return None
            
            bar_size = self.timeframe_map[timeframe]
            
            # Calculate duration
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            duration_days = (end_dt - start_dt).days
            
            # IBKR duration format
            if duration_days <= 1:
                duration = "1 D"
            elif duration_days <= 7:
                duration = f"{duration_days} D"
            elif duration_days <= 365:
                duration = f"{duration_days} D"
            else:
                duration = f"{duration_days // 365} Y"
            
            self.logger.info(f"Requesting historical data for {symbol}: {duration} of {bar_size} bars")
            
            # Request historical data
            # Format end date properly for IBKR with UTC timezone (yyyymmdd-HH:mm:ss in UTC)
            end_dt_formatted = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d-23:59:59')
            
            bars = await self.connection.ib.reqHistoricalDataAsync(
                contract=qualified_contract,
                endDateTime=end_dt_formatted,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours
                formatDate=1,
                timeout=self.config.historical_data_timeout
            )
            
            if not bars:
                self.logger.warning(f"No historical data received for {symbol}")
                return None
            
            # Convert to DataFrame
            df = self._convert_ib_bars_to_dataframe(bars)
            
            if df.empty:
                self.logger.warning(f"Empty DataFrame for {symbol}")
                return None
            
            # Filter by date range - handle timezone-aware index
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Make start and end dates timezone-aware to match the index
            if df.index.tz is not None:
                start_dt = start_dt.tz_localize(df.index.tz)
                end_dt = end_dt.tz_localize(df.index.tz)
            
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Cache the result
            self.historical_cache[cache_key] = df.copy()
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            self.logger.info(f"Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current price for symbol
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Dict with price information or None if error
        """
        if not self.is_connected():
            await self.connect()
        
        if not self.is_connected():
            self.logger.error("Not connected to IBKR")
            return None
        
        try:
            # Apply rate limiting
            await self.connection.rate_limit()
            
            # Create contract
            contract = self._create_contract(symbol)
            
            # Qualify contract to get conId
            contracts = await self.connection.ib.qualifyContractsAsync(contract)
            if not contracts:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return None
            
            qualified_contract = contracts[0]
            
            # Request market data snapshot
            ticker = self.connection.ib.reqMktData(qualified_contract, '', True, False)
            
            # Wait for data to populate
            await asyncio.sleep(1.0)
            
            # Get the ticker from IB instance
            tickers = self.connection.ib.tickers()
            ticker = next((t for t in tickers if t.contract.symbol == qualified_contract.symbol), None)
            
            if ticker and ticker.last and ticker.last > 0:
                return {
                    'symbol': symbol,
                    'price': float(ticker.last),
                    'bid': float(ticker.bid) if ticker.bid and ticker.bid > 0 else float(ticker.last),
                    'ask': float(ticker.ask) if ticker.ask and ticker.ask > 0 else float(ticker.last),
                    'volume': int(ticker.volume) if ticker.volume else 0,
                    'timestamp': datetime.now()
                }
            else:
                self.logger.warning(f"No current price data for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def subscribe_real_time_data(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time data for symbols
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            bool: True if subscriptions successful
        """
        if not self.is_connected():
            await self.connect()
        
        if not self.is_connected():
            self.logger.error("Not connected to IBKR")
            return False
        
        try:
            for symbol in symbols:
                if symbol not in self.active_subscriptions:
                    contract = self._create_contract(symbol)
                    
                    # Request real-time data
                    ticker = self.connection.ib.reqMktData(contract, '', False, False)
                    
                    self.active_subscriptions[symbol] = contract
                    self.real_time_data[symbol] = {}
                    
                    self.logger.info(f"Subscribed to real-time data for {symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to real-time data: {e}")
            return False
    
    async def unsubscribe_real_time_data(self, symbols: List[str] = None):
        """
        Unsubscribe from real-time data
        
        Args:
            symbols: List of symbols to unsubscribe from (None = all)
        """
        if symbols is None:
            symbols = list(self.active_subscriptions.keys())
        
        for symbol in symbols:
            if symbol in self.active_subscriptions:
                contract = self.active_subscriptions[symbol]
                self.connection.ib.cancelMktData(contract)
                
                del self.active_subscriptions[symbol]
                if symbol in self.real_time_data:
                    del self.real_time_data[symbol]
                
                self.logger.info(f"Unsubscribed from real-time data for {symbol}")
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes"""
        return list(self.timeframe_map.keys())
    
    def get_name(self) -> str:
        """Get data source name"""
        return "IBKR"
    
    async def get_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Get order book data (Level II market data)
        
        Args:
            symbol: Symbol to get order book for
            
        Returns:
            Dict with bid/ask levels or None if not available
        """
        if not self.is_connected():
            await self.connect()
        
        if not self.is_connected():
            self.logger.error("Not connected to IBKR")
            return None
        
        try:
            # Apply rate limiting
            await self.connection.rate_limit()
            
            # Create contract
            contract = self._create_contract(symbol)
            
            # Qualify contract to get conId
            contracts = await self.connection.ib.qualifyContractsAsync(contract)
            if not contracts:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return None
            
            qualified_contract = contracts[0]
            
            # Request Level II market data (requires market data subscription)
            self.connection.ib.reqMktDepth(qualified_contract)
            
            # Wait for data to populate
            await asyncio.sleep(1.0)
            
            # Get the ticker from IB instance
            tickers = self.connection.ib.tickers()
            ticker = next((t for t in tickers if t.contract.symbol == qualified_contract.symbol), None)
            
            if ticker and hasattr(ticker, 'domBids') and hasattr(ticker, 'domAsks'):
                return {
                    'symbol': symbol,
                    'bids': [(level.price, level.size) for level in ticker.domBids[:5]],  # Top 5 bid levels
                    'asks': [(level.price, level.size) for level in ticker.domAsks[:5]],  # Top 5 ask levels
                    'timestamp': datetime.now()
                }
            else:
                self.logger.warning(f"No order book data available for {symbol}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Order book not available for {symbol}: {e}")
            # Return basic bid/ask from ticker data as fallback
            try:
                price_data = await self.get_current_price(symbol)
                if price_data:
                    return {
                        'symbol': symbol,
                        'bids': [(price_data['bid'], 0)],  # Basic bid without size
                        'asks': [(price_data['ask'], 0)],  # Basic ask without size
                        'timestamp': datetime.now()
                    }
            except:
                pass
            return None
    
    async def test_connection(self) -> bool:
        """Test IBKR connection"""
        try:
            success = await self.connect()
            if success:
                # Test basic functionality
                accounts = self.connection.get_managed_accounts()
                self.logger.info(f"Connected to IBKR with accounts: {accounts}")
                
                # Test a simple data request
                test_data = await self.get_current_price('SPY')
                if test_data:
                    self.logger.info(f"Test data request successful: SPY @ ${test_data['price']}")
                
            await self.disconnect()
            return success
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False


# Factory function
def create_ibkr_data_source(config: Optional[IBKRConfig] = None) -> IBKRDataSource:
    """Create IBKR data source with configuration"""
    return IBKRDataSource(config)


# Test script
async def main():
    """Test IBKR data source"""
    import os
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv('.env.ibkr.paper')
    
    # Create data source
    data_source = create_ibkr_data_source()
    
    print("🔌 Testing IBKR connection...")
    success = await data_source.test_connection()
    
    if success:
        print("✅ IBKR connection test passed!")
        
        # Test historical data
        print("\n📊 Testing historical data...")
        await data_source.connect()
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        df = await data_source.get_historical_data('SPY', start_date, end_date, '1h')
        if df is not None and not df.empty:
            print(f"✅ Retrieved {len(df)} bars for SPY")
            print(df.tail())
        else:
            print("❌ Failed to get historical data")
        
        # Test current price
        print("\n💰 Testing current price...")
        price = await data_source.get_current_price('SPY')
        if price:
            print(f"✅ SPY current price: ${price['price']:.2f}")
        else:
            print("❌ Failed to get current price")
        
        await data_source.disconnect()
    else:
        print("❌ IBKR connection test failed!")


if __name__ == "__main__":
    asyncio.run(main())