"""
Strategy wrapper for integrating framework strategies with backtesting.py library.

This module provides a bridge between our custom trading framework strategies
and the backtesting.py library, allowing seamless integration and reuse of
existing strategy implementations.
"""

import pandas as pd
from typing import Dict, Any, Optional

try:
    from backtesting import Strategy
except ImportError:
    raise ImportError("backtesting library not found. Install it with: uv add backtesting")

from framework.utils.logger import setup_logger


class StrategyWrapper(Strategy):
    """
    Wrapper to adapt existing framework strategies to work with backtesting.py
    
    This allows reuse of existing strategy implementations without duplicating
    the trading logic. The wrapper handles data format conversion and integrates
    stop loss and take profit functionality from strategy signals.
    """
    
    # These will be set dynamically by create_wrapper_class()
    framework_strategy_class = None
    strategy_params = {}
    
    def init(self):
        """Initialize the wrapped strategy."""
        self.logger = setup_logger("INFO")
        
        # Create instance of framework strategy
        if self.__class__.framework_strategy_class is None:
            raise ValueError("framework_strategy_class must be set")
            
        self.strategy = self.__class__.framework_strategy_class(**self.__class__.strategy_params)
        
        # Convert backtesting.py data format to framework format
        # backtesting.py uses capitalized OHLCV, framework uses lowercase
        self.framework_data = pd.DataFrame({
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'close': self.data.Close,
            'volume': self.data.Volume
        }, index=self.data.index)
        
        # Generate signals for the entire dataset once
        self.signals = self.strategy.generate_signals(self.framework_data)
        
        # Track current position state for stop loss/take profit
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        self.logger.info(f"Initialized {self.strategy.get_description()}")
        
    def next(self):
        """Execute trading logic for current bar."""
        # Get current index
        current_idx = len(self.data) - 1
        
        # Skip if we don't have signals for this index yet
        if current_idx >= len(self.signals):
            return
            
        # Get current signal data
        current_signal_row = self.signals.iloc[current_idx]
        current_price = self.data.Close[-1]
        
        signal = current_signal_row.get('signal', 0)
        position_size = current_signal_row.get('position_size', 0.01)
        stop_loss = current_signal_row.get('stop_loss', None)
        take_profit = current_signal_row.get('take_profit', None)
        
        # Check stop loss and take profit for existing positions
        if self.position and (self.stop_loss_price or self.take_profit_price):
            should_exit = False
            exit_reason = ""
            
            # Check stop loss
            if (self.stop_loss_price and 
                ((self.position.is_long and current_price <= self.stop_loss_price) or
                 (self.position.is_short and current_price >= self.stop_loss_price))):
                should_exit = True
                exit_reason = f"Stop Loss at ${current_price:.2f}"
                
            # Check take profit  
            elif (self.take_profit_price and
                  ((self.position.is_long and current_price >= self.take_profit_price) or
                   (self.position.is_short and current_price <= self.take_profit_price))):
                should_exit = True
                exit_reason = f"Take Profit at ${current_price:.2f}"
                
            if should_exit:
                self.position.close()
                self.logger.info(f"EXIT: {exit_reason}")
                self._reset_position_tracking()
                return
        
        # Handle new signals
        if signal == 1 and not self.position:  # Buy signal
            # Place buy order
            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            
            # Track position details for our own stop loss/take profit logic
            self.entry_price = current_price
            self.stop_loss_price = stop_loss
            self.take_profit_price = take_profit
            
            self.logger.info(f"BUY at ${current_price:.2f} (size: {position_size:.1%})")
            if stop_loss:
                self.logger.info(f"Stop loss set at ${stop_loss:.2f}")
            if take_profit:
                self.logger.info(f"Take profit set at ${take_profit:.2f}")
                
        elif signal == -1 and self.position:  # Sell signal (close position)
            self.position.close()
            self.logger.info(f"SELL at ${current_price:.2f}")
            self._reset_position_tracking()
            
    def _reset_position_tracking(self):
        """Reset position tracking variables."""
        self.entry_price = None
        self.stop_loss_price = None  
        self.take_profit_price = None


def create_wrapper_class(strategy_cls, params: Dict[str, Any]):
    """
    Create a dynamic wrapper class for a framework strategy.
    
    Args:
        strategy_cls: The framework strategy class to wrap
        params: Parameters to pass to the strategy constructor
        
    Returns:
        A strategy class ready for use with backtesting.py
    """
    class DynamicWrapper(StrategyWrapper):
        framework_strategy_class = strategy_cls
        strategy_params = params
    
    return DynamicWrapper