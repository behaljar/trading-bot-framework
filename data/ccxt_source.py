"""
CCXT data source implementation for cryptocurrencies
"""
import pandas as pd
import ccxt
from typing import List, Dict, Any
from datetime import datetime
import time
from .base_data_source import DataSource


class CCXTSource(DataSource):
    """CCXT data source for cryptocurrencies"""

    def __init__(self, exchange_name: str = "binance", api_key: str = "", api_secret: str = "", sandbox: bool = True):
        """
        Initialization of CCXT source

        Args:
            exchange_name: Exchange name (binance, coinbase, kraken, etc.)
            api_key: API key (optional for public data)
            api_secret: API secret (optional for public data)
            sandbox: Use testnet/sandbox
        """
        self.exchange_name = exchange_name

        try:
            # Dynamically create exchange instance
            exchange_class = getattr(ccxt, exchange_name)

            config = {
                'timeout': 30000,
                'enableRateLimit': True,  # Important for rate limiting
            }
            
            # Only add API credentials if they're provided and not placeholder values
            if api_key and api_key != "your_api_key_here" and api_key != "":
                config['apiKey'] = api_key
                config['secret'] = api_secret

            if sandbox:
                config['sandbox'] = True

            self.exchange = exchange_class(config)

            # Load markets
            self.markets = self.exchange.load_markets()

            print(f"CCXT {exchange_name} initialized with {len(self.markets)} markets")

        except Exception as e:
            raise Exception(f"Error initializing CCXT {exchange_name}: {e}")

    def get_available_symbols(self) -> List[str]:
        """Returns list of available symbols"""
        return list(self.markets.keys())

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for CCXT format (BTC/USDT)"""
        if '/' not in symbol:
            # Try to guess the correct format
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                return f"{base}/USDT"
            elif symbol.endswith('BTC'):
                base = symbol[:-3]
                return f"{base}/BTC"
            elif symbol.endswith('ETH'):
                base = symbol[:-3]
                return f"{base}/ETH"
        return symbol

    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
        """Downloads historical data via CCXT"""
        try:
            # Normalize symbol
            normalized_symbol = self.normalize_symbol(symbol)

            if normalized_symbol not in self.markets:
                available = ", ".join(list(self.markets.keys())[:10])
                raise ValueError(f"Symbol {normalized_symbol} is not available. Available: {available}...")

            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

            # Download data in chunks (CCXT has limits)
            all_ohlcv = []
            current_ts = start_ts

            while current_ts < end_ts:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        normalized_symbol,
                        timeframe,
                        since=current_ts,
                        limit=1000  # Most exchanges have a limit of 1000
                    )

                    if not ohlcv:
                        break

                    all_ohlcv.extend(ohlcv)

                    # Set next timestamp
                    current_ts = ohlcv[-1][0] + 1

                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)

                except ccxt.BaseError as e:
                    print(f"CCXT error downloading data: {e}")
                    break

            if not all_ohlcv:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Filter by end_date
            df = df[df.index <= end_date]

            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]

            return df

        except Exception as e:
            print(f"Error downloading CCXT data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Gets current price"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            ticker = self.exchange.fetch_ticker(normalized_symbol)
            return float(ticker['last'])
        except Exception as e:
            print(f"Error getting current price from CCXT for {symbol}: {e}")
            return 0.0

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Gets order book"""
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            order_book = self.exchange.fetch_order_book(normalized_symbol, limit)
            return {
                "bids": order_book['bids'][:limit],
                "asks": order_book['asks'][:limit],
                "timestamp": order_book['timestamp'],
                "symbol": normalized_symbol
            }
        except Exception as e:
            print(f"Error getting order book from CCXT for {symbol}: {e}")
            return {"bids": [], "asks": [], "timestamp": None}