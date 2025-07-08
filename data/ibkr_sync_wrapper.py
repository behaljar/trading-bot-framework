"""
Synchronous wrapper for IBKR data source to maintain compatibility 
with existing synchronous paper trading framework.
"""
import asyncio
import logging
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

from data.base_data_source import DataSource
from data.ibkr_source import IBKRDataSource
from config.ibkr_config import IBKRConfig, create_ibkr_config


class IBKRSyncWrapper(DataSource):
    """
    Synchronous wrapper around IBKRDataSource for compatibility
    with existing synchronous trading frameworks.
    """
    
    def __init__(self, config: Optional[IBKRConfig] = None):
        self.config = config or create_ibkr_config()
        self.ibkr_source = IBKRDataSource(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Event loop management
        self.loop = None
        self.loop_running = False
        
        # Connection state
        self._connected = False
        
    def _get_event_loop(self):
        """Get or create event loop for async operations"""
        try:
            # Try to get the current loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return loop
        except RuntimeError:
            pass
        
        # Create new event loop if needed
        if self.loop is None or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        return self.loop
    
    def _run_async(self, coro):
        """Run async coroutine in sync context"""
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we need to run in a thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=60)  # 60 second timeout
                
        except RuntimeError:
            # No event loop running, we can create one
            loop = self._get_event_loop()
            try:
                return loop.run_until_complete(coro)
            except Exception as e:
                self.logger.error(f"Error running async operation: {e}")
                raise
    
    def connect(self) -> bool:
        """Connect to IBKR synchronously"""
        try:
            success = self._run_async(self.ibkr_source.connect())
            self._connected = success
            if success:
                self.logger.info("Successfully connected to IBKR")
            else:
                self.logger.error("Failed to connect to IBKR")
            return success
        except Exception as e:
            self.logger.error(f"Error connecting to IBKR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR synchronously"""
        try:
            if self._connected:
                self._run_async(self.ibkr_source.disconnect())
                self._connected = False
                self.logger.info("Disconnected from IBKR")
        except Exception as e:
            self.logger.error(f"Error disconnecting from IBKR: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self._connected and self.ibkr_source.is_connected()
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          timeframe: str = '1h') -> Optional[pd.DataFrame]:
        """
        Get historical data synchronously
        
        Args:
            symbol: Symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.is_connected():
            self.logger.info("Not connected to IBKR, attempting to connect...")
            if not self.connect():
                self.logger.error("Failed to connect to IBKR")
                return None
        
        try:
            df = self._run_async(
                self.ibkr_source.get_historical_data(symbol, start_date, end_date, timeframe)
            )
            
            if df is not None and not df.empty:
                self.logger.info(f"Retrieved {len(df)} bars for {symbol}")
                return df
            else:
                self.logger.warning(f"No data returned for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current price synchronously
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Dict with price information or None if error
        """
        if not self.is_connected():
            self.logger.info("Not connected to IBKR, attempting to connect...")
            if not self.connect():
                self.logger.error("Failed to connect to IBKR")
                return None
        
        try:
            price_data = self._run_async(self.ibkr_source.get_current_price(symbol))
            
            if price_data:
                self.logger.debug(f"Current price for {symbol}: ${price_data['price']:.2f}")
                return price_data
            else:
                self.logger.warning(f"No current price data for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes"""
        return self.ibkr_source.get_supported_timeframes()
    
    def get_name(self) -> str:
        """Get data source name"""
        return "IBKR (Sync)"
    
    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Get order book data (not implemented for IBKR sync wrapper)
        
        Args:
            symbol: Symbol to get order book for
            
        Returns:
            None - order book not implemented for sync wrapper
        """
        self.logger.warning("Order book not implemented for IBKR sync wrapper")
        return None
    
    def test_connection(self) -> bool:
        """Test IBKR connection synchronously"""
        try:
            success = self._run_async(self.ibkr_source.test_connection())
            return success
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.disconnect()
        except:
            pass


# Factory function for easy integration
def create_ibkr_sync_source(config: Optional[IBKRConfig] = None) -> IBKRSyncWrapper:
    """Create synchronous IBKR data source"""
    return IBKRSyncWrapper(config)


# Test the sync wrapper
def test_sync_wrapper():
    """Test the synchronous wrapper"""
    import os
    from dotenv import load_dotenv
    from datetime import timedelta
    
    # Load IBKR configuration
    load_dotenv('.env.ibkr.paper')
    
    print("🔌 Testing IBKR Synchronous Wrapper...")
    
    with create_ibkr_sync_source() as ibkr:
        print(f"✅ Connected: {ibkr.is_connected()}")
        
        # Test current price
        price = ibkr.get_current_price('SPY')
        if price:
            print(f"💰 SPY price: ${price['price']:.2f}")
        
        # Test historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        df = ibkr.get_historical_data('SPY', start_date, end_date, '1h')
        if df is not None and not df.empty:
            print(f"📊 Historical data: {len(df)} bars")
            print(df.tail(3))
        
        print("✅ Sync wrapper test completed!")


if __name__ == "__main__":
    test_sync_wrapper()