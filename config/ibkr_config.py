"""
IBKR-specific configuration management.
"""
import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class IBKRAccountType(Enum):
    """IBKR account types"""
    PAPER = "paper"
    LIVE = "live"


class IBKRMarketDataType(Enum):
    """IBKR market data types"""
    LIVE = 1      # Live market data (requires subscription)
    FROZEN = 2    # Frozen market data
    DELAYED = 3   # Delayed market data (15-20 min delay)
    DELAYED_FROZEN = 4  # Delayed frozen market data


@dataclass
class IBKRConfig:
    """IBKR connection and trading configuration"""
    
    # Connection settings
    host: str = "127.0.0.1"
    port: int = 7497  # 7496=live TWS, 7497=paper TWS, 4001=live Gateway, 4002=paper Gateway
    client_id: int = 1
    
    # Account settings
    account_type: IBKRAccountType = IBKRAccountType.PAPER
    account_id: Optional[str] = None
    
    # Market data settings
    market_data_type: IBKRMarketDataType = IBKRMarketDataType.DELAYED
    
    # Connection management
    connect_timeout: int = 10  # seconds
    read_timeout: int = 30     # seconds
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5   # seconds
    
    # Request settings
    max_requests_per_second: int = 50
    historical_data_timeout: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"
    
    @staticmethod
    def _clean_env_value(value: str) -> Optional[str]:
        """Clean environment variable value by removing inline comments"""
        # Import here to avoid circular import issues
        from utils.env_utils import clean_env_value
        return clean_env_value(value)
    
    @classmethod
    def from_env(cls) -> 'IBKRConfig':
        """Create configuration from environment variables"""
        
        # Determine account type and set default port
        account_type_str = cls._clean_env_value(os.getenv('IBKR_ACCOUNT_TYPE', 'paper')) or 'paper'
        account_type = IBKRAccountType.PAPER if account_type_str.lower() == 'paper' else IBKRAccountType.LIVE
        
        # Default ports based on account type
        default_port = 4002 if account_type == IBKRAccountType.PAPER else 7496
        
        # Market data type
        market_data_type_str = cls._clean_env_value(os.getenv('IBKR_MARKET_DATA_TYPE', '3')) or '3'
        market_data_type_int = int(market_data_type_str)
        market_data_type = IBKRMarketDataType(market_data_type_int)
        
        return cls(
            host=cls._clean_env_value(os.getenv('IBKR_HOST', '127.0.0.1')) or '127.0.0.1',
            port=int(cls._clean_env_value(os.getenv('IBKR_PORT', str(default_port))) or str(default_port)),
            client_id=int(cls._clean_env_value(os.getenv('IBKR_CLIENT_ID', '1')) or '1'),
            account_type=account_type,
            account_id=cls._clean_env_value(os.getenv('IBKR_ACCOUNT_ID')),
            market_data_type=market_data_type,
            connect_timeout=int(cls._clean_env_value(os.getenv('IBKR_CONNECT_TIMEOUT', '10')) or '10'),
            read_timeout=int(cls._clean_env_value(os.getenv('IBKR_READ_TIMEOUT', '30')) or '30'),
            auto_reconnect=(cls._clean_env_value(os.getenv('IBKR_AUTO_RECONNECT', 'true')) or 'true').lower() == 'true',
            max_reconnect_attempts=int(cls._clean_env_value(os.getenv('IBKR_MAX_RECONNECT_ATTEMPTS', '5')) or '5'),
            reconnect_delay=int(cls._clean_env_value(os.getenv('IBKR_RECONNECT_DELAY', '5')) or '5'),
            max_requests_per_second=int(cls._clean_env_value(os.getenv('IBKR_MAX_REQUESTS_PER_SECOND', '50')) or '50'),
            historical_data_timeout=int(cls._clean_env_value(os.getenv('IBKR_HISTORICAL_DATA_TIMEOUT', '60')) or '60'),
            log_level=cls._clean_env_value(os.getenv('IBKR_LOG_LEVEL', 'INFO')) or 'INFO'
        )
    
    @property
    def is_paper_trading(self) -> bool:
        """Check if this is paper trading configuration"""
        return self.account_type == IBKRAccountType.PAPER or self.port in [7497, 4002]
    
    @property
    def is_gateway(self) -> bool:
        """Check if connecting to IB Gateway (vs TWS)"""
        return self.port in [4001, 4002]
    
    def validate(self) -> None:
        """Validate configuration settings"""
        if self.port not in [7496, 7497, 4001, 4002]:
            raise ValueError(f"Invalid IBKR port: {self.port}. Must be 7496/7497 (TWS) or 4001/4002 (Gateway)")
        
        if self.client_id < 0 or self.client_id > 32:
            raise ValueError(f"Invalid client_id: {self.client_id}. Must be 0-32")
        
        if self.connect_timeout <= 0:
            raise ValueError(f"Invalid connect_timeout: {self.connect_timeout}. Must be > 0")
        
        if self.account_type == IBKRAccountType.LIVE and self.port in [7497, 4002]:
            raise ValueError("Live account type specified but using paper trading port")
        
        if self.account_type == IBKRAccountType.PAPER and self.port in [7496, 4001]:
            raise ValueError("Paper account type specified but using live trading port")
    
    def __str__(self) -> str:
        """String representation for logging"""
        return (f"IBKRConfig(host={self.host}, port={self.port}, "
                f"account_type={self.account_type.value}, "
                f"market_data_type={self.market_data_type.value}, "
                f"client_id={self.client_id})")


def create_ibkr_config() -> IBKRConfig:
    """Create and validate IBKR configuration"""
    config = IBKRConfig.from_env()
    config.validate()
    return config