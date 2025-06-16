"""
Logging utility
"""
import logging
import os
from datetime import datetime

def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Sets up logging for trading bot"""

    # Create logs folder
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Set up logger
    logger = logging.getLogger("TradingBot")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    log_filename = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger