"""
Tests for SMA Strategy implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from framework.strategies.sma_strategy import SMAStrategy


class TestSMAStrategy:
    """Test cases for SMA Strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = SMAStrategy(
            short_window=5, 
            long_window=10, 
            position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )

    def create_test_data(self, length=50):
        """Create test OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
        
        # Create trending price data for testing crossovers
        base_price = 100.0
        trend = np.linspace(0, 10, length)
        noise = np.random.normal(0, 0.5, length)
        prices = base_price + trend + noise
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, length)
        }, index=dates)
        
        return data

    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.short_window == 5
        assert self.strategy.long_window == 10
        assert self.strategy.position_size == 0.1
        assert self.strategy.stop_loss_pct == 0.02
        assert self.strategy.take_profit_pct == 0.04

    def test_initialization_with_kwargs(self):
        """Test strategy initialization with keyword arguments."""
        strategy = SMAStrategy(
            short_window=20,
            long_window=50,
            position_size=0.05,
            stop_loss_pct=0.015,
            take_profit_pct=0.03
        )
        assert strategy.short_window == 20
        assert strategy.long_window == 50
        assert strategy.position_size == 0.05
        assert strategy.stop_loss_pct == 0.015
        assert strategy.take_profit_pct == 0.03

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        data = self.create_test_data(5)  # Less than long_window
        signals = self.strategy.generate_signals(data)
        
        # Should return signals but no trades due to insufficient data
        assert 'signal' in signals.columns
        assert 'position_size' in signals.columns
        assert signals['signal'].sum() == 0  # No signals due to insufficient data

    def test_generate_signals_basic_functionality(self):
        """Test basic signal generation functionality."""
        data = self.create_test_data(30)
        signals = self.strategy.generate_signals(data)
        
        # Check that all required columns are present
        required_columns = ['signal', 'position_size', 'stop_loss', 'take_profit', 'ma_short', 'ma_long']
        for col in required_columns:
            assert col in signals.columns
        
        # Check that moving averages are calculated
        assert not signals['ma_short'].isna().all()
        assert not signals['ma_long'].isna().all()
        
        # Check that early periods have no signals (insufficient MA data)
        assert signals['signal'].iloc[:10].sum() == 0

    def test_generate_signals_crossover_detection(self):
        """Test crossover signal detection."""
        # Create data with known crossover pattern
        dates = pd.date_range(start='2024-01-01', periods=20, freq='1h')
        
        # Prices that will create a clear golden cross
        prices_down = [100] * 8  # Flat for MA stabilization
        prices_up = list(range(101, 113))  # Rising trend for crossover
        close_prices = prices_down + prices_up
        
        data = pd.DataFrame({
            'open': close_prices,
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 20
        }, index=dates)
        
        strategy = SMAStrategy(short_window=3, long_window=5)
        signals = strategy.generate_signals(data)
        
        # Should have at least one buy signal when short MA crosses above long MA
        buy_signals = signals[signals['signal'] == 1]
        assert len(buy_signals) > 0

    def test_stop_loss_calculation_long(self):
        """Test stop loss calculation for long positions."""
        data = self.create_test_data(20)
        signals = self.strategy.generate_signals(data)
        
        # Find buy signals
        buy_signals = signals[signals['signal'] == 1]
        if len(buy_signals) > 0:
            for idx, row in buy_signals.iterrows():
                if pd.notna(row['stop_loss']):
                    # Stop loss should be 2% below entry price
                    expected_stop = row['close'] * (1 - 0.02)
                    assert abs(row['stop_loss'] - expected_stop) < 0.01

    def test_stop_loss_calculation_short(self):
        """Test stop loss calculation for short positions."""
        # Create data that triggers short signals
        dates = pd.date_range(start='2024-01-01', periods=20, freq='1h')
        
        # Prices that start high then decline (death cross pattern)
        prices_high = [120] * 8
        prices_down = list(range(119, 107, -1))
        close_prices = prices_high + prices_down
        
        data = pd.DataFrame({
            'open': close_prices,
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 20
        }, index=dates)
        
        strategy = SMAStrategy(short_window=3, long_window=5, stop_loss_pct=0.02)
        signals = strategy.generate_signals(data)
        
        # Find sell signals
        sell_signals = signals[signals['signal'] == -1]
        if len(sell_signals) > 0:
            for idx, row in sell_signals.iterrows():
                if pd.notna(row['stop_loss']):
                    # Stop loss should be 2% above entry price for short
                    expected_stop = row['close'] * (1 + 0.02)
                    assert abs(row['stop_loss'] - expected_stop) < 0.01

    def test_take_profit_calculation_long(self):
        """Test take profit calculation for long positions."""
        data = self.create_test_data(20)
        signals = self.strategy.generate_signals(data)
        
        # Find buy signals
        buy_signals = signals[signals['signal'] == 1]
        if len(buy_signals) > 0:
            for idx, row in buy_signals.iterrows():
                if pd.notna(row['take_profit']):
                    # Take profit should be 4% above entry price
                    expected_tp = row['close'] * (1 + 0.04)
                    assert abs(row['take_profit'] - expected_tp) < 0.01

    def test_take_profit_calculation_short(self):
        """Test take profit calculation for short positions."""
        # Create data that triggers short signals
        dates = pd.date_range(start='2024-01-01', periods=20, freq='1h')
        prices_high = [120] * 8
        prices_down = list(range(119, 107, -1))
        close_prices = prices_high + prices_down
        
        data = pd.DataFrame({
            'open': close_prices,
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 20
        }, index=dates)
        
        strategy = SMAStrategy(short_window=3, long_window=5, take_profit_pct=0.04)
        signals = strategy.generate_signals(data)
        
        # Find sell signals
        sell_signals = signals[signals['signal'] == -1]
        if len(sell_signals) > 0:
            for idx, row in sell_signals.iterrows():
                if pd.notna(row['take_profit']):
                    # Take profit should be 4% below entry price for short
                    expected_tp = row['close'] * (1 - 0.04)
                    assert abs(row['take_profit'] - expected_tp) < 0.01

    def test_no_stop_loss_or_take_profit(self):
        """Test strategy without stop loss or take profit."""
        strategy = SMAStrategy(
            short_window=5, 
            long_window=10, 
            stop_loss_pct=None, 
            take_profit_pct=None
        )
        data = self.create_test_data(20)
        signals = strategy.generate_signals(data)
        
        # All stop loss and take profit values should be NaN
        assert signals['stop_loss'].isna().all()
        assert signals['take_profit'].isna().all()

    def test_position_size_consistency(self):
        """Test that position size is consistent across signals."""
        data = self.create_test_data(30)
        signals = self.strategy.generate_signals(data)
        
        # All position sizes should be the same as configured
        non_zero_positions = signals[signals['signal'] != 0]['position_size']
        if len(non_zero_positions) > 0:
            assert (non_zero_positions == 0.1).all()

    def test_invalid_data_format(self):
        """Test handling of invalid data format."""
        # Missing required columns
        invalid_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(ValueError, match="Invalid data format"):
            self.strategy.generate_signals(invalid_data)

    def test_get_description(self):
        """Test strategy description."""
        desc = self.strategy.get_description()
        assert "Simple Moving Average Crossover Strategy" in desc
        assert "5-period short MA" in desc
        assert "10-period long MA" in desc
        assert "Position size: 10.0%" in desc
        assert "Stop loss: 2.0%" in desc
        assert "Take profit: 4.0%" in desc

    def test_get_strategy_name(self):
        """Test strategy name."""
        name = self.strategy.get_strategy_name()
        assert name == "SMA"

    def test_ma_calculation_accuracy(self):
        """Test that moving averages are calculated correctly."""
        # Create simple test data for easy verification
        dates = pd.date_range(start='2024-01-01', periods=15, freq='1h')
        close_prices = list(range(100, 115))  # 100, 101, 102, ..., 114
        
        data = pd.DataFrame({
            'open': close_prices,
            'high': [p + 0.5 for p in close_prices],
            'low': [p - 0.5 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 15
        }, index=dates)
        
        strategy = SMAStrategy(short_window=3, long_window=5)
        signals = strategy.generate_signals(data)
        
        # Verify MA calculations manually for a specific point
        # At index 4 (5th candle), we should have:
        # Short MA (3-period): mean of [102, 103, 104] = 103
        # Long MA (5-period): mean of [100, 101, 102, 103, 104] = 102
        if len(signals) > 4:
            assert abs(signals.iloc[4]['ma_short'] - 103.0) < 0.001
            assert abs(signals.iloc[4]['ma_long'] - 102.0) < 0.001

    def test_crossover_timing(self):
        """Test that crossover signals occur at the correct timing."""
        # Create data where we know exactly when crossover should happen
        dates = pd.date_range(start='2024-01-01', periods=25, freq='1h')
        
        # Design prices to create controlled crossover
        # First 10 periods: declining (short MA below long MA)
        # Last 15 periods: rising (short MA above long MA)
        decline_prices = list(range(120, 110, -1))  # 120, 119, ..., 111
        rise_prices = list(range(111, 126))          # 111, 112, ..., 125
        close_prices = decline_prices + rise_prices
        
        data = pd.DataFrame({
            'open': close_prices,
            'high': [p + 0.5 for p in close_prices],
            'low': [p - 0.5 for p in close_prices],
            'close': close_prices,
            'volume': [1000] * 25
        }, index=dates)
        
        strategy = SMAStrategy(short_window=3, long_window=6)
        signals = strategy.generate_signals(data)
        
        # Should have buy signals (1) when short MA crosses above long MA
        buy_signals = signals[signals['signal'] == 1]
        assert len(buy_signals) > 0
        
        # Verify crossover logic: short MA > long MA and was <= in previous period
        for idx, row in buy_signals.iterrows():
            idx_pos = signals.index.get_loc(idx)
            if idx_pos > 0:
                current_short = row['ma_short']
                current_long = row['ma_long']
                prev_short = signals.iloc[idx_pos - 1]['ma_short']
                prev_long = signals.iloc[idx_pos - 1]['ma_long']
                
                # Current: short > long
                assert current_short > current_long
                # Previous: short <= long
                assert prev_short <= prev_long