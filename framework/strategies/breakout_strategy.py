import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """
    High/Low Breakout Trend-Following Strategy (Long and Short)
    
    Long Entry: Price breaks above 20-period high
    Long Exit: Price breaks below 10-period low
    Short Entry: Price breaks below 20-period low
    Short Exit: Price breaks above 10-period high
    Stop Loss: 2 x ATR from entry price
    """

    def __init__(self, entry_lookback: int = 20, exit_lookback: int = 10,
                 atr_period: int = 14, atr_multiplier: float = 2.0,
                 longterm_trend_period: int = 200, medium_trend_period: int = 50,
                 longterm_trend_threshold: float = 0.0, medium_trend_threshold: float = 0.02,
                 use_trend_filter: bool = True, volume_ma_period: int = 60,
                 relative_volume_threshold: float = 1.5, use_volume_filter: bool = True,
                 use_momentum_exit: bool = True, momentum_candle_threshold: float = 0.025,
                 momentum_volume_threshold: float = 2.0, momentum_volume_period: int = 20,
                 cooldown_periods: int = 4, position_size: float = 0.01):
        
        parameters = {
            "entry_lookback": entry_lookback,
            "exit_lookback": exit_lookback,
            "atr_period": atr_period,
            "atr_multiplier": atr_multiplier,
            "longterm_trend_period": longterm_trend_period,
            "medium_trend_period": medium_trend_period,
            "longterm_trend_threshold": longterm_trend_threshold,
            "medium_trend_threshold": medium_trend_threshold,
            "use_trend_filter": use_trend_filter,
            "volume_ma_period": volume_ma_period,
            "relative_volume_threshold": relative_volume_threshold,
            "use_volume_filter": use_volume_filter,
            "use_momentum_exit": use_momentum_exit,
            "momentum_candle_threshold": momentum_candle_threshold,
            "momentum_volume_threshold": momentum_volume_threshold,
            "momentum_volume_period": momentum_volume_period,
            "cooldown_periods": cooldown_periods,
            "position_size": position_size,
        }
        
        super().__init__("Breakout", parameters)
        self.params = parameters
        
    def get_strategy_name(self) -> str:
        return f"Breakout_{self.params['entry_lookback']}_{self.params['exit_lookback']}"

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
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
        data[f'high_{self.params["entry_lookback"]}'] = data['high'].rolling(
            window=self.params['entry_lookback']
        ).max()
        
        data[f'low_{self.params["entry_lookback"]}'] = data['low'].rolling(
            window=self.params['entry_lookback']
        ).min()
        
        # Calculate rolling highs and lows for exits
        data[f'high_{self.params["exit_lookback"]}'] = data['high'].rolling(
            window=self.params['exit_lookback']
        ).max()
        
        data[f'low_{self.params["exit_lookback"]}'] = data['low'].rolling(
            window=self.params['exit_lookback']
        ).min()
        
        # Calculate multi-timeframe trend filters
        # Long-term trend (major trend direction)
        longterm_price_shift = data['close'].shift(self.params['longterm_trend_period'])
        data['longterm_trend_roc'] = (data['close'] - longterm_price_shift) / longterm_price_shift
        
        # Medium trend (intermediate trend)
        medium_price_shift = data['close'].shift(self.params['medium_trend_period'])
        data['medium_trend_roc'] = (data['close'] - medium_price_shift) / medium_price_shift
        
        # Calculate relative volume (volume vs moving average)
        data['volume_ma'] = data['volume'].rolling(window=self.params['volume_ma_period']).mean()
        data['relative_volume'] = data['volume'] / data['volume_ma']
        
        # Calculate ATR
        data['atr'] = self.calculate_atr(data)
        
        # Calculate momentum exit indicators (big candles with high volume)
        if self.params['use_momentum_exit']:
            # Calculate candle size as percentage move
            data['candle_size_pct'] = abs(data['close'] - data['open']) / data['open']
            
            # Calculate volume moving average for comparison
            data['volume_ma_momentum'] = data['volume'].rolling(window=self.params['momentum_volume_period']).mean()
            data['volume_ratio_momentum'] = data['volume'] / data['volume_ma_momentum']
            
            # Momentum exhaustion signals:
            # For longs: big UP candle with high volume (selling climax after uptrend)
            data['long_momentum_exit'] = (
                (data['close'] > data['open']) &  # Bullish candle
                (data['candle_size_pct'] >= self.params['momentum_candle_threshold']) &  # Big candle
                (data['volume_ratio_momentum'] >= self.params['momentum_volume_threshold'])  # High volume
            )
            
            # For shorts: big DOWN candle with high volume (buying climax after downtrend)
            data['short_momentum_exit'] = (
                (data['close'] < data['open']) &  # Bearish candle
                (data['candle_size_pct'] >= self.params['momentum_candle_threshold']) &  # Big candle
                (data['volume_ratio_momentum'] >= self.params['momentum_volume_threshold'])  # High volume
            )
        
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates breakout signals with ATR-based stop loss for both long and short"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
            
        data_with_indicators = self.add_indicators(data)
        
        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        
        # Get indicators
        prev_high_entry = data_with_indicators[f'high_{self.params["entry_lookback"]}'].shift(1)
        prev_low_entry = data_with_indicators[f'low_{self.params["entry_lookback"]}'].shift(1)
        prev_high_exit = data_with_indicators[f'high_{self.params["exit_lookback"]}'].shift(1)
        prev_low_exit = data_with_indicators[f'low_{self.params["exit_lookback"]}'].shift(1)
        atr = data_with_indicators['atr']
        
        longterm_trend_roc = data_with_indicators['longterm_trend_roc']
        medium_trend_roc = data_with_indicators['medium_trend_roc']
        relative_volume = data_with_indicators['relative_volume']
        
        # Get momentum exit indicators if enabled
        long_momentum_exit = data_with_indicators.get('long_momentum_exit', None) if self.params['use_momentum_exit'] else None
        short_momentum_exit = data_with_indicators.get('short_momentum_exit', None) if self.params['use_momentum_exit'] else None
        
        # Initialize result DataFrame
        result = data.copy()
        result['signal'] = 0
        result['position_size'] = self.params['position_size']
        result['stop_loss'] = None
        result['take_profit'] = None
        
        # Track position state
        position_type = None  # 'long', 'short', or None
        entry_price = None
        stop_loss_price = None
        last_exit_bar = None  # Track when we last exited
        
        for i in range(len(data)):
            # Skip initial bars until we have enough data for all indicators
            min_bars_needed = max(
                self.params['entry_lookback'], 
                self.params['exit_lookback'], 
                self.params['atr_period'],
                self.params['longterm_trend_period'] if self.params['use_trend_filter'] else 0
            )
            if i < min_bars_needed:
                continue
                
            current_close = close_prices.iloc[i]
            current_high = high_prices.iloc[i]
            current_low = low_prices.iloc[i]
            current_longterm_trend = longterm_trend_roc.iloc[i]
            current_medium_trend = medium_trend_roc.iloc[i]
            current_relative_volume = relative_volume.iloc[i]
            current_long_momentum_exit = long_momentum_exit.iloc[i] if long_momentum_exit is not None else False
            current_short_momentum_exit = short_momentum_exit.iloc[i] if short_momentum_exit is not None else False
            
            if position_type is None:
                # Check cooldown period - wait after closing previous position
                cooldown_satisfied = True
                if last_exit_bar is not None:
                    bars_since_exit = i - last_exit_bar
                    cooldown_satisfied = bars_since_exit >= self.params['cooldown_periods']
                
                # Only consider new entries if cooldown period has passed
                if not cooldown_satisfied:
                    continue
                
                # Long entry: breakout + multi-timeframe trend alignment
                breakout_condition = not pd.isna(prev_high_entry.iloc[i]) and current_high > prev_high_entry.iloc[i]
                
                # Trend alignment conditions for longs
                longterm_trend_condition = True
                medium_trend_condition = True
                if self.params['use_trend_filter']:
                    # Long-term trend must be positive (uptrend)
                    longterm_trend_condition = (not pd.isna(current_longterm_trend) and 
                                              current_longterm_trend > self.params['longterm_trend_threshold'])
                    # Medium-term trend must be positive and strong enough
                    medium_trend_condition = (not pd.isna(current_medium_trend) and 
                                            current_medium_trend >= self.params['medium_trend_threshold'])
                
                # Relative volume condition (optional)
                volume_condition = True
                if self.params['use_volume_filter']:
                    volume_condition = (not pd.isna(current_relative_volume) and 
                                      current_relative_volume >= self.params['relative_volume_threshold'])
                
                if breakout_condition and longterm_trend_condition and medium_trend_condition and volume_condition:
                    result.iloc[i, result.columns.get_loc('signal')] = 1
                    entry_price = current_close
                    position_type = 'long'
                    
                    # Set stop loss at ATR below entry
                    if not pd.isna(atr.iloc[i]):
                        stop_loss_price = entry_price - (atr.iloc[i] * self.params['atr_multiplier'])
                        result.iloc[i, result.columns.get_loc('stop_loss')] = stop_loss_price
                
                # Short entry: breakout + multi-timeframe trend alignment
                else:
                    breakout_condition = not pd.isna(prev_low_entry.iloc[i]) and current_low < prev_low_entry.iloc[i]
                    
                    # Trend alignment conditions for shorts
                    longterm_trend_condition = True
                    medium_trend_condition = True
                    if self.params['use_trend_filter']:
                        # Long-term trend must be negative (downtrend)
                        longterm_trend_condition = (not pd.isna(current_longterm_trend) and 
                                                  current_longterm_trend < -self.params['longterm_trend_threshold'])
                        # Medium-term trend must be negative and strong enough
                        medium_trend_condition = (not pd.isna(current_medium_trend) and 
                                                current_medium_trend <= -self.params['medium_trend_threshold'])
                    
                    # Relative volume condition (same for shorts)
                    volume_condition = True
                    if self.params['use_volume_filter']:
                        volume_condition = (not pd.isna(current_relative_volume) and 
                                          current_relative_volume >= self.params['relative_volume_threshold'])
                    
                    if breakout_condition and longterm_trend_condition and medium_trend_condition and volume_condition:
                        result.iloc[i, result.columns.get_loc('signal')] = -1
                        entry_price = current_close
                        position_type = 'short'
                        
                        # Set stop loss at ATR above entry
                        if not pd.isna(atr.iloc[i]):
                            stop_loss_price = entry_price + (atr.iloc[i] * self.params['atr_multiplier'])
                            result.iloc[i, result.columns.get_loc('stop_loss')] = stop_loss_price
                    
            elif position_type == 'long':
                # Update stop loss for long position
                if not pd.isna(atr.iloc[i]) and entry_price is not None:
                    stop_loss_price = entry_price - (atr.iloc[i] * self.params['atr_multiplier'])
                    result.iloc[i, result.columns.get_loc('stop_loss')] = stop_loss_price
                
                # Exit condition 1: price goes below previous exit-period low
                if not pd.isna(prev_low_exit.iloc[i]) and current_low < prev_low_exit.iloc[i]:
                    result.iloc[i, result.columns.get_loc('signal')] = -1
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                    
                # Exit condition 2: stop loss hit
                elif stop_loss_price is not None and current_low <= stop_loss_price:
                    result.iloc[i, result.columns.get_loc('signal')] = -1
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                
                # Exit condition 3: momentum exhaustion (big up candle with high volume)
                elif self.params['use_momentum_exit'] and current_long_momentum_exit:
                    result.iloc[i, result.columns.get_loc('signal')] = -1
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                    
            elif position_type == 'short':
                # Update stop loss for short position
                if not pd.isna(atr.iloc[i]) and entry_price is not None:
                    stop_loss_price = entry_price + (atr.iloc[i] * self.params['atr_multiplier'])
                    result.iloc[i, result.columns.get_loc('stop_loss')] = stop_loss_price
                
                # Exit condition 1: price goes above previous exit-period high
                if not pd.isna(prev_high_exit.iloc[i]) and current_high > prev_high_exit.iloc[i]:
                    result.iloc[i, result.columns.get_loc('signal')] = 1
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                    
                # Exit condition 2: stop loss hit
                elif stop_loss_price is not None and current_high >= stop_loss_price:
                    result.iloc[i, result.columns.get_loc('signal')] = 1
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                
                # Exit condition 3: momentum exhaustion (big down candle with high volume)
                elif self.params['use_momentum_exit'] and current_short_momentum_exit:
                    result.iloc[i, result.columns.get_loc('signal')] = 1
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
        
        return result

    def get_description(self) -> str:
        """Return strategy description."""
        return (f"High/Low Breakout Strategy with {self.params['entry_lookback']}-period entry lookback "
                f"and {self.params['exit_lookback']}-period exit lookback. "
                f"Stop loss: {self.params['atr_multiplier']}x ATR. "
                f"Position size: {self.params['position_size'] * 100}%.")