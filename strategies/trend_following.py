"""
Implementation of trend-following strategies
"""
import pandas as pd
from typing import Dict, Any
from .base_strategy import BaseStrategy, Signal

class SMAStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "short_window": 20,
            "long_window": 40,
            "stop_loss_pct": 1.0,  # 2% stop loss
            "take_profit_pct": 3.0  # 4% take profit (2:1 RR ratio)
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self.min_bars_required = 2

    def get_strategy_name(self) -> str:
        return f"SMA_Cross_{self.params['short_window']}_{self.params['long_window']}"

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds SMA indicators"""
        data = data.copy()
        data[f'SMA_{self.params["short_window"]}'] = data['Close'].rolling(
            window=self.params['short_window']
        ).mean()
        data[f'SMA_{self.params["long_window"]}'] = data['Close'].rolling(
            window=self.params['long_window']
        ).mean()

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates signals with stop loss and take profit levels"""
        data_with_indicators = self.add_indicators(data)

        short_ma = data_with_indicators[f'SMA_{self.params["short_window"]}']
        long_ma = data_with_indicators[f'SMA_{self.params["long_window"]}']
        close_prices = data['Close']

        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # BUY signal: short MA crosses above long MA with volume confirmation
        buy_condition = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        result.loc[buy_condition, 'signal'] = Signal.BUY.value
        
        # For buy signals, set stop loss below entry and take profit above
        buy_indices = result[result['signal'] == Signal.BUY.value].index
        for idx in buy_indices:
            entry_price = close_prices[idx]
            result.loc[idx, 'stop_loss'] = entry_price * (1 - self.params['stop_loss_pct'] / 100)
            result.loc[idx, 'take_profit'] = entry_price * (1 + self.params['take_profit_pct'] / 100)

        # SELL signal: short MA crosses below long MA with volume confirmation
        sell_condition = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        result.loc[sell_condition, 'signal'] = Signal.SELL.value
            
        # For sell signals (short), set stop loss above entry and take profit below
        sell_indices = result[result['signal'] == Signal.SELL.value].index
        for idx in sell_indices:
            entry_price = close_prices[idx]
            result.loc[idx, 'stop_loss'] = entry_price * (1 + self.params['stop_loss_pct'] / 100)
            result.loc[idx, 'take_profit'] = entry_price * (1 - self.params['take_profit_pct'] / 100)

        return result