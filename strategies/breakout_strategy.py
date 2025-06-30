"""
High/Low Breakout Trend-Following Strategy (Long and Short)
Long Entry: Price breaks above 20-period high
Long Exit: Price breaks below 10-period low
Short Entry: Price breaks below 20-period low
Short Exit: Price breaks above 10-period high
Stop Loss: 2 x ATR from entry price
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy, Signal

THRESHOLD = "trend_strength_threshold"


class BreakoutStrategy(BaseStrategy):
    """High/Low Breakout Strategy with ATR-based stop loss"""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "entry_lookback": 20,  # Look back period for entry (high/low breakout)
            "exit_lookback": 10,   # Look back period for exit (high/low breakout)
            "atr_period": 14,      # ATR calculation period
            "atr_multiplier": 4.0, # Stop loss = entry +/- (ATR * multiplier)
            "volume_roc_period": 5,  # Period for volume rate of change calculation
            "volume_roc_threshold": 0.5,  # Minimum volume ROC for entry (50% increase)
            "trend_period": 60,  # Period for medium-term trend calculation
            ("%s" % THRESHOLD): 0.10,  # Minimum trend strength (10% move required)
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self.position_type = None  # 'long', 'short', or None
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
        
        # Calculate rolling highs and lows for entries
        data[f'High_{self.params["entry_lookback"]}'] = data['High'].rolling(
            window=self.params['entry_lookback']
        ).max()
        
        data[f'Low_{self.params["entry_lookback"]}'] = data['Low'].rolling(
            window=self.params['entry_lookback']
        ).min()
        
        # Calculate rolling highs and lows for exits
        data[f'High_{self.params["exit_lookback"]}'] = data['High'].rolling(
            window=self.params['exit_lookback']
        ).max()
        
        data[f'Low_{self.params["exit_lookback"]}'] = data['Low'].rolling(
            window=self.params['exit_lookback']
        ).min()

        # Calculate Volume Rate of Change (ROC)
        # ROC = (Current Volume / Volume N periods ago) - 1
        volume_shift = data['Volume'].shift(self.params['volume_roc_period'])
        data['volume_roc'] = (data['Volume'] / volume_shift) - 1
        
        # Replace infinite values with NaN (in case of zero volume in past)
        data['volume_roc'] = data['volume_roc'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate medium-term trend (price change vs N periods ago)
        # Positive = uptrend, Negative = downtrend
        price_shift = data['Close'].shift(self.params['trend_period'])
        data['trend_roc'] = (data['Close'] - price_shift) / price_shift
        
        # Calculate ATR
        data['ATR'] = self.calculate_atr(data)
        
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates breakout signals with ATR-based stop loss for both long and short"""
        data_with_indicators = self.add_indicators(data)
        
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        
        # Get indicators
        prev_high_entry = data_with_indicators[f'High_{self.params["entry_lookback"]}'].shift(1)
        prev_low_entry = data_with_indicators[f'Low_{self.params["entry_lookback"]}'].shift(1)
        prev_high_exit = data_with_indicators[f'High_{self.params["exit_lookback"]}'].shift(1)
        prev_low_exit = data_with_indicators[f'Low_{self.params["exit_lookback"]}'].shift(1)
        atr = data_with_indicators['ATR']
        volume_roc = data_with_indicators['volume_roc']
        trend_roc = data_with_indicators['trend_roc']
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None
        
        # Track position state
        position_type = None  # 'long', 'short', or None
        entry_price = None
        stop_loss_price = None
        
        for i in range(len(data)):
            if i < max(self.params['entry_lookback'], self.params['exit_lookback'], self.params['atr_period']):
                continue
                
            current_close = close_prices.iloc[i]
            current_high = high_prices.iloc[i]
            current_low = low_prices.iloc[i]
            current_volume_roc = volume_roc.iloc[i]
            current_trend_roc = trend_roc.iloc[i]
            
            if position_type is None:
                # Long entry: price breaks above previous 20-period high with volume surge and strong uptrend
                if (not pd.isna(prev_high_entry.iloc[i]) and current_high > prev_high_entry.iloc[i] and
                    not pd.isna(current_volume_roc) and current_volume_roc >= self.params['volume_roc_threshold'] and
                    not pd.isna(current_trend_roc) and current_trend_roc >= self.params['trend_strength_threshold']):
                    result.loc[result.index[i], 'signal'] = Signal.BUY.value
                    entry_price = current_close
                    position_type = 'long'
                    
                    # Set stop loss at 2 x ATR below entry
                    if not pd.isna(atr.iloc[i]):
                        stop_loss_price = entry_price - (atr.iloc[i] * self.params['atr_multiplier'])
                        result.loc[result.index[i], 'stop_loss'] = stop_loss_price
                
                # Short entry: price breaks below previous 20-period low with volume surge and strong downtrend  
                elif (not pd.isna(prev_low_entry.iloc[i]) and current_low < prev_low_entry.iloc[i] and
                      not pd.isna(current_volume_roc) and current_volume_roc >= self.params['volume_roc_threshold'] and
                      not pd.isna(current_trend_roc) and current_trend_roc <= -self.params['trend_strength_threshold']):
                    result.loc[result.index[i], 'signal'] = Signal.SELL.value
                    entry_price = current_close
                    position_type = 'short'
                    
                    # Set stop loss at 2 x ATR above entry
                    if not pd.isna(atr.iloc[i]):
                        stop_loss_price = entry_price + (atr.iloc[i] * self.params['atr_multiplier'])
                        result.loc[result.index[i], 'stop_loss'] = stop_loss_price
                    
            elif position_type == 'long':
                # Update stop loss for long position
                if not pd.isna(atr.iloc[i]) and entry_price is not None:
                    stop_loss_price = entry_price - (atr.iloc[i] * self.params['atr_multiplier'])
                    result.loc[result.index[i], 'stop_loss'] = stop_loss_price
                
                # Exit condition 1: price goes below previous 10-period low
                if not pd.isna(prev_low_exit.iloc[i]) and current_low < prev_low_exit.iloc[i]:
                    result.loc[result.index[i], 'signal'] = Signal.SELL.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    
                # Exit condition 2: stop loss hit
                elif stop_loss_price is not None and current_low <= stop_loss_price:
                    result.loc[result.index[i], 'signal'] = Signal.SELL.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    
            elif position_type == 'short':
                # Update stop loss for short position
                if not pd.isna(atr.iloc[i]) and entry_price is not None:
                    stop_loss_price = entry_price + (atr.iloc[i] * self.params['atr_multiplier'])
                    result.loc[result.index[i], 'stop_loss'] = stop_loss_price
                
                # Exit condition 1: price goes above previous 10-period high
                if not pd.isna(prev_high_exit.iloc[i]) and current_high > prev_high_exit.iloc[i]:
                    result.loc[result.index[i], 'signal'] = Signal.BUY.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    
                # Exit condition 2: stop loss hit
                elif stop_loss_price is not None and current_high >= stop_loss_price:
                    result.loc[result.index[i], 'signal'] = Signal.BUY.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
        
        return result