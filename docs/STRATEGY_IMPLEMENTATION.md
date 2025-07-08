# 🎯 Strategy Implementation Guide

Complete guide for implementing custom trading strategies in the trading bot framework. Learn how to create, test, and deploy your own algorithmic trading strategies.

## 🚀 Quick Start

### 1. Create Your Strategy File
```bash
# Create new strategy file
touch strategies/my_custom_strategy.py
```

### 2. Basic Template
```python
"""
My Custom Trading Strategy
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy, Signal

class MyCustomStrategy(BaseStrategy):
    """Custom strategy description"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "param1": 20,
            "param2": 2.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self.min_bars_required = max(default_params.values())

    def get_strategy_name(self) -> str:
        return f"MyCustom_{self.params['param1']}_{self.params['param2']}"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        # Add your strategy logic here
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None
        return result
```

### 3. Test Your Strategy
```bash
# Test with paper trading
python scripts/run_paper_trading.py \
  --source yahoo \
  --symbols AAPL \
  --strategy my_custom \
  --timeframe 1h
```

## 🏗️ Architecture Overview

### BaseStrategy Class
All strategies must inherit from `BaseStrategy` and implement these methods:

```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from OHLCV data"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return unique strategy identifier"""
        pass
```

### Signal Types
```python
from .base_strategy import Signal

Signal.BUY    # 1  - Long position
Signal.SELL   # -1 - Short position  
Signal.HOLD   # 0  - No action
```

### Data Format
Input data is a pandas DataFrame with standardized columns:
- `Date` (index) - Timestamp
- `Open` - Opening price
- `High` - High price
- `Low` - Low price
- `Close` - Closing price
- `Volume` - Trading volume
- Additional features (if using CSV with preprocessed data)

## 📊 Strategy Implementation Patterns

### 1. Simple Moving Average Strategy
```python
class SMAStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "short_window": 20,
            "long_window": 50,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        data = data.copy()
        data['SMA_Short'] = data['Close'].rolling(
            window=self.params['short_window']
        ).mean()
        data['SMA_Long'] = data['Close'].rolling(
            window=self.params['long_window']
        ).mean()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals"""
        data_with_indicators = self.add_indicators(data)
        
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # Generate signals
        short_ma = data_with_indicators['SMA_Short']
        long_ma = data_with_indicators['SMA_Long']
        
        # Buy when short MA crosses above long MA
        buy_condition = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        result.loc[buy_condition, 'signal'] = Signal.BUY.value
        
        # Sell when short MA crosses below long MA
        sell_condition = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        result.loc[sell_condition, 'signal'] = Signal.SELL.value

        # Add stop loss and take profit levels
        for idx in result[result['signal'] != Signal.HOLD.value].index:
            entry_price = data.loc[idx, 'Close']
            if result.loc[idx, 'signal'] == Signal.BUY.value:
                result.loc[idx, 'stop_loss'] = entry_price * (1 - self.params['stop_loss_pct'] / 100)
                result.loc[idx, 'take_profit'] = entry_price * (1 + self.params['take_profit_pct'] / 100)
            else:  # SELL signal
                result.loc[idx, 'stop_loss'] = entry_price * (1 + self.params['stop_loss_pct'] / 100)
                result.loc[idx, 'take_profit'] = entry_price * (1 - self.params['take_profit_pct'] / 100)

        return result
```

### 2. RSI Mean Reversion Strategy
```python
class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "atr_period": 14,
            "atr_multiplier": 2.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        # Calculate indicators
        rsi = self.calculate_rsi(data, self.params['rsi_period'])
        atr = self.calculate_atr(data, self.params['atr_period'])
        
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # Generate signals
        for i in range(len(data)):
            if pd.notna(rsi.iloc[i]) and pd.notna(atr.iloc[i]):
                current_rsi = rsi.iloc[i]
                current_price = data['Close'].iloc[i]
                current_atr = atr.iloc[i]

                # Long signal: RSI oversold
                if current_rsi < self.params['oversold_threshold']:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.BUY.value
                    result.iloc[i, result.columns.get_loc('stop_loss')] = current_price - (current_atr * self.params['atr_multiplier'])
                    result.iloc[i, result.columns.get_loc('take_profit')] = current_price + (current_atr * self.params['atr_multiplier'] * 2)

                # Short signal: RSI overbought
                elif current_rsi > self.params['overbought_threshold']:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.SELL.value
                    result.iloc[i, result.columns.get_loc('stop_loss')] = current_price + (current_atr * self.params['atr_multiplier'])
                    result.iloc[i, result.columns.get_loc('take_profit')] = current_price - (current_atr * self.params['atr_multiplier'] * 2)

        return result
```

### 3. Breakout Strategy with Volume Confirmation
```python
class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "breakout_period": 20,
            "volume_ma_period": 20,
            "volume_multiplier": 1.5,
            "atr_period": 14,
            "atr_multiplier": 2.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals with volume confirmation"""
        # Calculate indicators
        high_breakout = data['High'].rolling(self.params['breakout_period']).max()
        low_breakout = data['Low'].rolling(self.params['breakout_period']).min()
        volume_ma = data['Volume'].rolling(self.params['volume_ma_period']).mean()
        atr = self.calculate_atr(data, self.params['atr_period'])
        
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # Generate signals
        for i in range(1, len(data)):
            if pd.notna(high_breakout.iloc[i]) and pd.notna(volume_ma.iloc[i]):
                current_price = data['Close'].iloc[i]
                prev_high = high_breakout.iloc[i-1]
                prev_low = low_breakout.iloc[i-1]
                current_volume = data['Volume'].iloc[i]
                avg_volume = volume_ma.iloc[i]

                # Volume confirmation
                volume_confirmed = current_volume > (avg_volume * self.params['volume_multiplier'])

                # Long breakout: price breaks above previous high with volume
                if current_price > prev_high and volume_confirmed:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.BUY.value
                    result.iloc[i, result.columns.get_loc('stop_loss')] = prev_low
                    result.iloc[i, result.columns.get_loc('take_profit')] = current_price + (current_price - prev_low) * 2

                # Short breakout: price breaks below previous low with volume
                elif current_price < prev_low and volume_confirmed:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.SELL.value
                    result.iloc[i, result.columns.get_loc('stop_loss')] = prev_high
                    result.iloc[i, result.columns.get_loc('take_profit')] = current_price - (prev_high - current_price) * 2

        return result
```

## 🔧 Advanced Features

### 1. State Management
```python
class StatefulStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.position_type = None  # 'long', 'short', or None
        self.entry_price = None
        self.bars_in_position = 0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Strategy with position state tracking"""
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            
            # Update position tracking
            if self.position_type is not None:
                self.bars_in_position += 1
            
            # Exit conditions
            if self.position_type == 'long':
                # Exit long position logic
                if self.should_exit_long(data, i):
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.SELL.value
                    self.position_type = None
                    self.entry_price = None
                    self.bars_in_position = 0
            
            elif self.position_type == 'short':
                # Exit short position logic
                if self.should_exit_short(data, i):
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.BUY.value
                    self.position_type = None
                    self.entry_price = None
                    self.bars_in_position = 0
            
            # Entry conditions (only when no position)
            elif self.position_type is None:
                if self.should_enter_long(data, i):
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.BUY.value
                    self.position_type = 'long'
                    self.entry_price = current_price
                    self.bars_in_position = 1
                
                elif self.should_enter_short(data, i):
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.SELL.value
                    self.position_type = 'short'
                    self.entry_price = current_price
                    self.bars_in_position = 1

        return result
```

### 2. Multi-Timeframe Analysis
```python
class MultiTimeframeStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "fast_period": 10,
            "slow_period": 30,
            "trend_period": 100
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def get_higher_timeframe_trend(self, data: pd.DataFrame) -> pd.Series:
        """Determine higher timeframe trend"""
        trend_ma = data['Close'].rolling(self.params['trend_period']).mean()
        trend_direction = pd.Series(0, index=data.index)
        trend_direction[data['Close'] > trend_ma] = 1  # Uptrend
        trend_direction[data['Close'] < trend_ma] = -1  # Downtrend
        return trend_direction

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with trend filter"""
        # Calculate indicators
        fast_ma = data['Close'].rolling(self.params['fast_period']).mean()
        slow_ma = data['Close'].rolling(self.params['slow_period']).mean()
        trend_direction = self.get_higher_timeframe_trend(data)
        
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # Only trade in direction of higher timeframe trend
        buy_condition = (
            (fast_ma > slow_ma) & 
            (fast_ma.shift(1) <= slow_ma.shift(1)) & 
            (trend_direction == 1)  # Only long in uptrend
        )
        
        sell_condition = (
            (fast_ma < slow_ma) & 
            (fast_ma.shift(1) >= slow_ma.shift(1)) & 
            (trend_direction == -1)  # Only short in downtrend
        )

        result.loc[buy_condition, 'signal'] = Signal.BUY.value
        result.loc[sell_condition, 'signal'] = Signal.SELL.value

        return result
```

### 3. Custom Indicators
```python
class CustomIndicatorStrategy(BaseStrategy):
    def calculate_custom_oscillator(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Custom oscillator combining RSI and Stochastic"""
        # RSI component
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        
        # Stochastic component
        lowest_low = data['Low'].rolling(period).min()
        highest_high = data['High'].rolling(period).max()
        stoch_k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        
        # Combined oscillator
        custom_osc = (rsi + stoch_k) / 2
        return custom_osc

    def calculate_trend_strength(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate trend strength indicator"""
        price_change = data['Close'].pct_change(period)
        volatility = data['Close'].rolling(period).std() / data['Close'].rolling(period).mean()
        trend_strength = price_change / volatility
        return trend_strength

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Use custom indicators for signal generation"""
        custom_osc = self.calculate_custom_oscillator(data, 14)
        trend_strength = self.calculate_trend_strength(data, 20)
        
        result = pd.DataFrame(index=data.index)
        result['signal'] = Signal.HOLD.value
        result['stop_loss'] = None
        result['take_profit'] = None

        # Signal logic using custom indicators
        for i in range(len(data)):
            if pd.notna(custom_osc.iloc[i]) and pd.notna(trend_strength.iloc[i]):
                osc_value = custom_osc.iloc[i]
                trend_val = trend_strength.iloc[i]
                
                # Long signal: oversold oscillator + positive trend
                if osc_value < 30 and trend_val > 0.02:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.BUY.value
                
                # Short signal: overbought oscillator + negative trend
                elif osc_value > 70 and trend_val < -0.02:
                    result.iloc[i, result.columns.get_loc('signal')] = Signal.SELL.value

        return result
```

## 🧪 Testing Your Strategy

### 1. Unit Testing
```python
# tests/test_my_strategy.py
import unittest
import pandas as pd
import numpy as np
from strategies.my_custom_strategy import MyCustomStrategy

class TestMyCustomStrategy(unittest.TestCase):
    def setUp(self):
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 101 + np.random.randn(100).cumsum(),
            'Low': 99 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        self.strategy = MyCustomStrategy()

    def test_strategy_initialization(self):
        """Test strategy initializes correctly"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.get_strategy_name(), "MyCustom_20_2.0")

    def test_generate_signals_returns_dataframe(self):
        """Test generate_signals returns proper DataFrame"""
        signals = self.strategy.generate_signals(self.sample_data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('stop_loss', signals.columns)
        self.assertIn('take_profit', signals.columns)

    def test_signal_values_are_valid(self):
        """Test signal values are within expected range"""
        signals = self.strategy.generate_signals(self.sample_data)
        signal_values = signals['signal'].dropna().unique()
        valid_signals = {-1, 0, 1}
        self.assertTrue(all(sig in valid_signals for sig in signal_values))

if __name__ == '__main__':
    unittest.main()
```

### 2. Paper Trading Testing
```bash
# Test with different data sources
python scripts/run_paper_trading.py \
  --source yahoo \
  --symbols AAPL MSFT \
  --strategy my_custom \
  --initial-capital 10000 \
  --timeframe 1h

# Test with crypto data
python scripts/run_paper_trading.py \
  --source ccxt \
  --symbols BTC/USDT ETH/USDT \
  --strategy my_custom \
  --timeframe 4h

# Test with IBKR data
python scripts/run_paper_trading.py \
  --source ibkr \
  --symbols SPY QQQ \
  --strategy my_custom \
  --timeframe 1h
```

### 3. Backtesting
```bash
# Run comprehensive backtest
python scripts/run_backtest.py \
  --strategy my_custom \
  --symbol AAPL \
  --start 2022-01-01 \
  --end 2023-12-31 \
  --timeframe 1h
```

## 📋 Strategy Integration Checklist

### ✅ Implementation Requirements
- [ ] Inherit from `BaseStrategy`
- [ ] Implement `generate_signals()` method
- [ ] Implement `get_strategy_name()` method
- [ ] Set `min_bars_required` appropriately
- [ ] Return DataFrame with required columns: `signal`, `stop_loss`, `take_profit`
- [ ] Handle edge cases (insufficient data, NaN values)
- [ ] Use proper parameter defaults and validation

### ✅ Testing Requirements
- [ ] Create unit tests for strategy logic
- [ ] Test with different market conditions (trending, ranging, volatile)
- [ ] Verify signal generation logic
- [ ] Test paper trading integration
- [ ] Run backtests on historical data
- [ ] Validate risk management (stop loss, position sizing)

### ✅ Documentation Requirements
- [ ] Add docstrings to all methods
- [ ] Document strategy parameters and their effects
- [ ] Include usage examples
- [ ] Explain strategy logic and market assumptions
- [ ] Document expected market conditions for strategy

### ✅ Production Readiness
- [ ] Strategy passes all unit tests
- [ ] Positive paper trading results
- [ ] Acceptable backtest performance
- [ ] Risk management properly implemented
- [ ] Strategy registered in main trading loop
- [ ] Configuration parameters documented

## 🔗 Adding Strategy to Framework

### 1. Register Strategy in Main Script
```python
# In main.py or strategy factory
from strategies.my_custom_strategy import MyCustomStrategy

STRATEGY_MAP = {
    'sma': SMAStrategy,
    'mean_reversion': MeanReversionStrategy,
    'breakout': BreakoutStrategy,
    'my_custom': MyCustomStrategy,  # Add your strategy here
}
```

### 2. Environment Configuration
```bash
# In .env file
STRATEGY_NAME=my_custom
STRATEGY_PARAM1=20
STRATEGY_PARAM2=2.0
STRATEGY_STOP_LOSS_PCT=2.0
STRATEGY_TAKE_PROFIT_PCT=4.0
```

### 3. Update Documentation
```python
# Update strategy list in README.md
| Strategy | Description | Best For |
|----------|-------------|----------|
| **My Custom** | Description of your strategy | Market conditions |
```

## 📚 Examples and Templates

### Strategy File Structure
```
strategies/
├── __init__.py
├── base_strategy.py          # Abstract base class
├── trend_following.py        # SMA crossover strategies
├── mean_reversion.py         # RSI and z-score strategies
├── breakout_strategy.py      # Breakout with volume confirmation
├── test_strategy.py          # Simple testing strategy
└── my_custom_strategy.py     # Your custom strategy
```

### Best Practices
1. **Keep It Simple**: Start with simple logic, add complexity gradually
2. **Validate Inputs**: Always check for sufficient data and handle edge cases
3. **Risk Management**: Always implement stop losses and position sizing
4. **Backtesting**: Test extensively on historical data before live trading
5. **Documentation**: Document your strategy logic and parameters
6. **Version Control**: Track strategy changes and performance over time

## 🚧 Common Pitfalls

### 1. Look-ahead Bias
```python
# ❌ Wrong - uses future data
future_high = data['High'].rolling(5).max().shift(-2)

# ✅ Correct - uses only past data
past_high = data['High'].rolling(5).max().shift(1)
```

### 2. Insufficient Data Validation
```python
# ❌ Wrong - doesn't handle NaN values
if data['RSI'].iloc[i] > 70:
    signal = Signal.SELL.value

# ✅ Correct - validates data first
if pd.notna(data['RSI'].iloc[i]) and data['RSI'].iloc[i] > 70:
    signal = Signal.SELL.value
```

### 3. Missing Risk Management
```python
# ❌ Wrong - no stop loss
result.loc[buy_condition, 'signal'] = Signal.BUY.value

# ✅ Correct - includes risk management
buy_indices = result[result['signal'] == Signal.BUY.value].index
for idx in buy_indices:
    entry_price = data.loc[idx, 'Close']
    result.loc[idx, 'stop_loss'] = entry_price * 0.98  # 2% stop loss
    result.loc[idx, 'take_profit'] = entry_price * 1.04  # 4% take profit
```

---

🎯 **Strategy Development** - Build, test, and deploy sophisticated trading algorithms with confidence and proper risk management.

> **Remember**: Always backtest thoroughly and start with paper trading before risking real capital. Past performance does not guarantee future results.