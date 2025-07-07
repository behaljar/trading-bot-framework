"""
Basic configuration
"""
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file, ignoring comments
load_dotenv(override=True)

def get_env_value(key: str, default: str = "") -> str:
    """Get environment variable value, stripping inline comments"""
    value = os.getenv(key, default)
    if isinstance(value, str) and '#' in value:
        # Strip inline comments - everything after the first #
        value = value.split('#')[0].strip()
    return value

@dataclass
class TradingConfig:
    # Data settings
    data_source: str = get_env_value("DATA_SOURCE", "ccxt")  # yahoo, ccxt, csv
    exchange_name: str = get_env_value("EXCHANGE_NAME", "binance")  # For CCXT
    csv_data_directory: str = get_env_value("CSV_DATA_DIRECTORY", "data/csv")  # For CSV data source
    symbols: Optional[List[str]] = None
    timeframe: str = get_env_value("TIMEFRAME", "4h")
    use_sandbox: bool = get_env_value("USE_SANDBOX", "true").lower() == "true"  # For CCXT testnet

    # Strategy settings
    strategy_name: str = get_env_value("STRATEGY_NAME", "sma")
    strategy_params: Optional[Dict[str, Any]] = None

    # Risk management
    initial_capital: float = float(get_env_value("INITIAL_CAPITAL", "10000"))
    max_position_size: float = float(get_env_value("MAX_POSITION_SIZE", "0.1"))
    stop_loss_pct: float = float(get_env_value("STOP_LOSS_PCT", "0.05"))

    # Trading settings
    commission: float = float(get_env_value("COMMISSION", "0.001"))  # Crypto has lower fees
    slippage: float = float(get_env_value("SLIPPAGE", "0.0005"))

    # API keys (from environment variables)
    api_key: str = get_env_value("EXCHANGE_API_KEY", "")
    api_secret: str = get_env_value("EXCHANGE_API_SECRET", "")
    
    # Trading mode settings
    allow_short: bool = get_env_value("ALLOW_SHORT", "false").lower() == "true"
    trading_type: str = get_env_value("TRADING_TYPE", "spot")  # spot or future

    # Monitoring
    alert_email: str = get_env_value("ALERT_EMAIL", "")
    log_level: str = get_env_value("LOG_LEVEL", "INFO")

    def __post_init__(self) -> None:
        # Load symbols from environment variable (comma-separated)
        if self.symbols is None:
            symbols_env = get_env_value("SYMBOLS", "")
            if symbols_env:
                self.symbols = [s.strip() for s in symbols_env.split(",")]
            else:
                # Default symbols based on data source
                if self.data_source == "yahoo":
                    self.symbols = ["AAPL", "MSFT", "GOOGL"]
                else:  # CCXT
                    self.symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        # Load strategy parameters from environment variables
        if self.strategy_params is None:
            if self.strategy_name.lower() in ["breakout", "breakout_strategy"]:
                # Breakout strategy parameters
                entry_lookback = int(get_env_value("STRATEGY_ENTRY_LOOKBACK", "20"))
                exit_lookback = int(get_env_value("STRATEGY_EXIT_LOOKBACK", "10"))
                atr_period = int(get_env_value("STRATEGY_ATR_PERIOD", "14"))
                atr_multiplier = float(get_env_value("STRATEGY_ATR_MULTIPLIER", "2.0"))

                self.strategy_params = {
                    "entry_lookback": entry_lookback,
                    "exit_lookback": exit_lookback,
                    "atr_period": atr_period,
                    "atr_multiplier": atr_multiplier
                }
            else:
                # SMA and other strategy parameters
                short_window = int(get_env_value("STRATEGY_SHORT_WINDOW", "20"))
                long_window = int(get_env_value("STRATEGY_LONG_WINDOW", "50"))
                stop_loss_pct = float(get_env_value("STRATEGY_STOP_LOSS_PCT", "2.0"))
                take_profit_pct = float(get_env_value("STRATEGY_TAKE_PROFIT_PCT", "4.0"))
                
                self.strategy_params = {
                    "short_window": short_window, 
                    "long_window": long_window,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct
                }

def load_config() -> TradingConfig:
    """Load configuration from file"""
    return TradingConfig()