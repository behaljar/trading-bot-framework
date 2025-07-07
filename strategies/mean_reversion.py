"""
Implementation of mean reversion strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy, Signal

class MeanReversion(BaseStrategy):
    """ Mean Reversion Strategy"""
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "atr_period": 14,  # ATR calculation period
            "atr_multiplier": 2.0,  # Stop loss = entry +/- (ATR * multiplier)
            "lookback_period": 20,
            "rsi_period": 14,  # RSI calculation period
            "rsi_oversold": 30,  # RSI oversold threshold for long
            "rsi_overbought": 70,  # RSI overbought threshold for short
            "use_rsi_filter": True,
            "trend_period": 50,  # Medium-term trend lookback period
            "trend_threshold": 0.15,  # Max allowed trend strength (15% change)
            "adx_period": 14,  # ADX calculation period
            "adx_threshold": 35  # ADX threshold - avoid trading when ADX > this value
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self.position_type = None  # Track if we're in a LONG or SHORT position
        self.min_bars_required = 2

    def get_strategy_name(self) -> str:
        return f"ZScore_MeanRev_{self.params['lookback_period']}_{self.params['atr_multiplier']}"

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds indicators to the data"""
        data = data.copy()

        # Calculate ATR
        data['ATR'] = self.calculate_atr(data)

        data['SMA'] = data['Close'].rolling(
            window=self.params['lookback_period']
        ).mean()
        data['STD'] = data['Close'].rolling(
            window=self.params['lookback_period']
        ).std()

        data['Z-Score'] = (data['Close'] - data['SMA']) / data['STD']

        # Calculate RSI
        data['RSI'] = self.calculate_rsi(data, self.params['rsi_period'])

        # Calculate ADX
        data['ADX'] = self.calculate_adx(data, self.params['adx_period'])

        # Calculate medium-term trend
        data['Trend_Return'] = (data['Close'] / data['Close'].shift(self.params['trend_period']) - 1)
        data['Trend_Strength'] = data['Trend_Return'].abs()

        return data

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

    def calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        high = data['High']
        low = data['Low']
        close = data['Close']

        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)

        # +DM occurs when up_move > down_move and up_move > 0
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        # -DM occurs when down_move > up_move and down_move > 0
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

        # Calculate ATR (if not already available)
        atr = self.calculate_atr(data)

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

        # Calculate ADX as smoothed average of DX
        adx = dx.rolling(window=period).mean()

        return adx

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates signals based on Z-score levels"""
        data_with_indicators = self.add_indicators(data)

        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # Generate signals based on Z-score, RSI, and ADX
        # Short: Z-score > 2 AND RSI > 70 (overbought) AND ADX < threshold
        # Long: Z-score < -2 AND RSI < 30 (oversold) AND ADX < threshold
        # Exit when Z-score crosses 0
        for i in range(len(data_with_indicators)):
            if pd.notna(data_with_indicators['Z-Score'].iloc[i]) and pd.notna(data_with_indicators['ATR'].iloc[i]) and pd.notna(data_with_indicators['RSI'].iloc[i]) and pd.notna(data_with_indicators['ADX'].iloc[i]):
                z_score = data_with_indicators['Z-Score'].iloc[i]
                current_price = data_with_indicators['Close'].iloc[i]
                atr = data_with_indicators['ATR'].iloc[i]
                sma = data_with_indicators['SMA'].iloc[i]
                rsi = data_with_indicators['RSI'].iloc[i]
                adx = data_with_indicators['ADX'].iloc[i]

                # Check if we should exit existing position when Z-score crosses 0
                if self.position_type == 'LONG' and z_score >= 0:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.SELL.value
                    self.position_type = None
                elif self.position_type == 'SHORT' and z_score <= 0:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.BUY.value
                    self.position_type = None
                # New position signals with RSI confirmation and ADX filter
                elif z_score > 2 and (self.params['use_rsi_filter'] and rsi > self.params['rsi_overbought']) and adx < self.params['adx_threshold'] and self.position_type is None:  # Overbought - Short signal
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.SELL.value
                    result.iloc[i, result.columns.get_loc('stop_loss')] = current_price + (atr * self.params['atr_multiplier'])
                    # Take profit when Z-score crosses 0 (handled in exit logic above)
                    self.position_type = 'SHORT'
                elif z_score < -2 and (self.params['use_rsi_filter'] and rsi < self.params['rsi_oversold']) and adx < self.params['adx_threshold'] and self.position_type is None:  # Oversold - Long signal
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.BUY.value
                    result.iloc[i, result.columns.get_loc('stop_loss')] = current_price - (atr * self.params['atr_multiplier'])
                    # Take profit when Z-score crosses 0 (handled in exit logic above)
                    self.position_type = 'LONG'

        return result


# Alias for clarity
ZScoreStrategy = MeanReversion