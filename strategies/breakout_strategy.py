"""
High/Low Breakout Trend-Following Strategy
Entry: Price breaks above 20-period high
Exit: Price breaks below 10-period low
Stop Loss: 2 x ATR below entry price
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """High/Low Breakout Strategy with ATR-based stop loss"""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "entry_lookback": 20,  # Look back period for entry (high breakout)
            "exit_lookback": 10,   # Look back period for exit (low breakout)
            "atr_period": 14,      # ATR calculation period
            "atr_multiplier": 2.0, # Stop loss = entry - (ATR * multiplier)
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self.in_position = False
        self.entry_price = None

    def get_strategy_name(self) -> str:
        return f"Breakout_{self.params['entry_lookback']}_{self.params['exit_lookback']}"

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range calculation
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))
        
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.params['atr_period']).mean()
        
        return atr

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds breakout levels and ATR"""
        data = data.copy()
        
        # Calculate rolling highs and lows
        data[f'High_{self.params["entry_lookback"]}'] = data['High'].rolling(
            window=self.params['entry_lookback']
        ).max()
        
        data[f'Low_{self.params["exit_lookback"]}'] = data['Low'].rolling(
            window=self.params['exit_lookback']
        ).min()
        
        # Calculate ATR
        data['ATR'] = self.calculate_atr(data)
        
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates breakout signals with ATR-based stop loss"""
        data_with_indicators = self.add_indicators(data)
        
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        
        # Get indicators
        prev_high = data_with_indicators[f'High_{self.params["entry_lookback"]}'].shift(1)
        prev_low = data_with_indicators[f'Low_{self.params["exit_lookback"]}'].shift(1)
        atr = data_with_indicators['ATR']
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None
        
        # Track position state
        in_position = False
        entry_price = None
        stop_loss_price = None
        
        for i in range(len(data)):
            if i < max(self.params['entry_lookback'], self.params['exit_lookback'], self.params['atr_period']):
                continue
                
            current_close = close_prices.iloc[i]
            current_high = high_prices.iloc[i]
            current_low = low_prices.iloc[i]
            
            if not in_position:
                # Entry condition: price breaks above previous 20-period high
                if not pd.isna(prev_high.iloc[i]) and current_high > prev_high.iloc[i]:
                    result.loc[result.index[i], 'signal'] = Signal.BUY.value
                    entry_price = current_close
                    
                    # Set stop loss at 2 x ATR below entry
                    if not pd.isna(atr.iloc[i]):
                        stop_loss_price = entry_price - (atr.iloc[i] * self.params['atr_multiplier'])
                        result.loc[result.index[i], 'stop_loss'] = stop_loss_price
                    
                    in_position = True
                    
            else:  # in_position
                # Update stop loss for existing position
                if not pd.isna(atr.iloc[i]) and entry_price is not None:
                    stop_loss_price = entry_price - (atr.iloc[i] * self.params['atr_multiplier'])
                    result.loc[result.index[i], 'stop_loss'] = stop_loss_price
                
                # Exit condition 1: price goes below previous 10-period low
                if not pd.isna(prev_low.iloc[i]) and current_low < prev_low.iloc[i]:
                    result.loc[result.index[i], 'signal'] = Signal.SELL.value
                    in_position = False
                    entry_price = None
                    stop_loss_price = None
                    
                # Exit condition 2: stop loss hit
                elif stop_loss_price is not None and current_low <= stop_loss_price:
                    result.loc[result.index[i], 'signal'] = Signal.SELL.value
                    in_position = False
                    entry_price = None
                    stop_loss_price = None
        
        return result