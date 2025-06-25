"""
Position-based SMA Strategy for Walk-Forward Testing
"""
import pandas as pd
from typing import Dict, Any
from .base_strategy import BaseStrategy, Signal

class SMAPositionStrategy(BaseStrategy):
    """SMA Strategy that maintains positions based on MA relationship"""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "short_window": 20,
            "long_window": 40,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def get_strategy_name(self) -> str:
        return f"SMA_Position_{self.params['short_window']}_{self.params['long_window']}"

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
        """Generates position-based signals"""
        data_with_indicators = self.add_indicators(data)

        short_ma = data_with_indicators[f'SMA_{self.params["short_window"]}']
        long_ma = data_with_indicators[f'SMA_{self.params["long_window"]}']
        close_prices = data['Close']

        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # Generate position signals based on MA relationship
        # When short MA is above long MA, we want to be long
        long_condition = (short_ma > long_ma) & short_ma.notna() & long_ma.notna()
        
        # When short MA is below long MA, we want to be short
        short_condition = (short_ma < long_ma) & short_ma.notna() & long_ma.notna()

        # Mark all periods where we should be long
        result.loc[long_condition, 'signal'] = Signal.BUY.value
        
        # Mark all periods where we should be short
        result.loc[short_condition, 'signal'] = Signal.SELL.value

        # Add stop loss and take profit for all non-hold signals
        for idx in result.index:
            if result.loc[idx, 'signal'] == Signal.BUY.value:
                entry_price = close_prices[idx]
                result.loc[idx, 'stop_loss'] = entry_price * (1 - self.params['stop_loss_pct'] / 100)
                result.loc[idx, 'take_profit'] = entry_price * (1 + self.params['take_profit_pct'] / 100)
            elif result.loc[idx, 'signal'] == Signal.SELL.value:
                entry_price = close_prices[idx]
                result.loc[idx, 'stop_loss'] = entry_price * (1 + self.params['stop_loss_pct'] / 100)
                result.loc[idx, 'take_profit'] = entry_price * (1 - self.params['take_profit_pct'] / 100)

        return result