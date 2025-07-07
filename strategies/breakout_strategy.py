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


class BreakoutStrategy(BaseStrategy):
    """High/Low Breakout Strategy with ATR-based stop loss"""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "entry_lookback": 30,  # Look back period for entry (high/low breakout)
            "exit_lookback": 20,   # Look back period for exit (high/low breakout)
            "atr_period": 14,      # ATR calculation period
            "atr_multiplier": 3.0, # Stop loss = entry +/- (ATR * multiplier)
            # Multi-timeframe trend filters
            "longterm_trend_period": 200,  # Long-term trend period (major trend)
            "medium_trend_period": 50,     # Medium-term trend period
            "longterm_trend_threshold": 0.0,  # 0% - just need positive/negative for direction
            "medium_trend_threshold": 0.02,   # 2% move required for medium trend
            "use_trend_filter": True,  # Enable/disable multi-timeframe trend filter
            # Volume filter
            "volume_ma_period": 60,    # Period for volume moving average
            "relative_volume_threshold": 1.5,  # Volume must be 1.5x average
            "use_volume_filter": True,
            # Momentum exit (big candles with high volume indicating trend exhaustion)
            "use_momentum_exit": True,  # Enable/disable momentum exit
            "momentum_candle_threshold": 0.025,  # Minimum candle size % for momentum signal (2.5%)
            "momentum_volume_threshold": 2.0,  # Volume must be 2x average for momentum exit
            "momentum_volume_period": 20,  # Period for volume moving average
            # Trade cooldown period
            "cooldown_periods": 4,  # Number of candles to wait after closing a position
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self.position_type = None  # 'long', 'short', or None
        self.entry_price = None
        self.last_exit_bar = None  # Track when we last exited a position
        self.min_bars_required = 2

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

        
        # Calculate multi-timeframe trend filters
        # Long-term trend (major trend direction)
        longterm_price_shift = data['Close'].shift(self.params['longterm_trend_period'])
        data['longterm_trend_roc'] = (data['Close'] - longterm_price_shift) / longterm_price_shift
        
        # Medium trend (intermediate trend)
        medium_price_shift = data['Close'].shift(self.params['medium_trend_period'])
        data['medium_trend_roc'] = (data['Close'] - medium_price_shift) / medium_price_shift
        
        # Calculate relative volume (volume vs moving average)
        data['volume_ma'] = data['Volume'].rolling(window=self.params['volume_ma_period']).mean()
        data['relative_volume'] = data['Volume'] / data['volume_ma']
        
        # Calculate ATR
        data['ATR'] = self.calculate_atr(data)
        
        
        # Calculate momentum exit indicators (big candles with high volume)
        if self.params['use_momentum_exit']:
            # Calculate candle size as percentage move
            data['candle_size_pct'] = abs(data['Close'] - data['Open']) / data['Open']
            
            # Calculate volume moving average for comparison
            data['volume_ma_momentum'] = data['Volume'].rolling(window=self.params['momentum_volume_period']).mean()
            data['volume_ratio_momentum'] = data['Volume'] / data['volume_ma_momentum']
            
            # Momentum exhaustion signals:
            # For longs: big UP candle with high volume (selling climax after uptrend)
            data['long_momentum_exit'] = (
                (data['Close'] > data['Open']) &  # Bullish candle
                (data['candle_size_pct'] >= self.params['momentum_candle_threshold']) &  # Big candle
                (data['volume_ratio_momentum'] >= self.params['momentum_volume_threshold'])  # High volume
            )
            
            # For shorts: big DOWN candle with high volume (buying climax after downtrend)
            data['short_momentum_exit'] = (
                (data['Close'] < data['Open']) &  # Bearish candle
                (data['candle_size_pct'] >= self.params['momentum_candle_threshold']) &  # Big candle
                (data['volume_ratio_momentum'] >= self.params['momentum_volume_threshold'])  # High volume
            )
        
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
        
        longterm_trend_roc = data_with_indicators['longterm_trend_roc']
        medium_trend_roc = data_with_indicators['medium_trend_roc']
        relative_volume = data_with_indicators['relative_volume']
        
        # Get momentum exit indicators if enabled
        long_momentum_exit = data_with_indicators.get('long_momentum_exit', None) if self.params['use_momentum_exit'] else None
        short_momentum_exit = data_with_indicators.get('short_momentum_exit', None) if self.params['use_momentum_exit'] else None
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
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
                    result.loc[result.index[i], 'signal'] = Signal.BUY.value
                    entry_price = current_close
                    position_type = 'long'
                    
                    # Set stop loss at 2 x ATR below entry
                    if not pd.isna(atr.iloc[i]):
                        stop_loss_price = entry_price - (atr.iloc[i] * self.params['atr_multiplier'])
                        result.loc[result.index[i], 'stop_loss'] = stop_loss_price
                
                # Short entry: breakout + multi-timeframe trend alignment
                elif True:  # Use elif to maintain structure
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
                    last_exit_bar = i
                    
                # Exit condition 2: stop loss hit
                elif stop_loss_price is not None and current_low <= stop_loss_price:
                    result.loc[result.index[i], 'signal'] = Signal.SELL.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                
                # Exit condition 3: momentum exhaustion (big up candle with high volume)
                elif self.params['use_momentum_exit'] and current_long_momentum_exit:
                    result.loc[result.index[i], 'signal'] = Signal.SELL.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                    
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
                    last_exit_bar = i
                    
                # Exit condition 2: stop loss hit
                elif stop_loss_price is not None and current_high >= stop_loss_price:
                    result.loc[result.index[i], 'signal'] = Signal.BUY.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
                
                # Exit condition 3: momentum exhaustion (big down candle with high volume)
                elif self.params['use_momentum_exit'] and current_short_momentum_exit:
                    result.loc[result.index[i], 'signal'] = Signal.BUY.value
                    position_type = None
                    entry_price = None
                    stop_loss_price = None
                    last_exit_bar = i
        
        return result