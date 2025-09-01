"""
Tests for FVG Strategy implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from framework.strategies.fvg_strategy import FVGStrategy


class TestFVGStrategy:
    """Test cases for FVG Strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = FVGStrategy(
            risk_reward_ratio=2.0,
            max_hold_hours=2,
            min_fvg_sensitivity=0.1,
            position_size=0.1,
            h1_lookback_candles=36,
            h4_lookback_candles=9
        )

    def create_test_data_m15(self, length=1000):
        """Create test M15 OHLCV data."""
        start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        dates = pd.date_range(start=start_date, periods=length, freq='15T')
        
        # Create realistic price movement
        base_price = 50000.0
        price_changes = np.random.normal(0, 100, length)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = max(prices[-1] + change, 1000)  # Prevent negative prices
            prices.append(new_price)
        
        data = pd.DataFrame({
            'open': [p * 0.9995 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.uniform(10, 100, length)
        }, index=dates)
        
        return data

    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.risk_reward_ratio == 2.0
        assert self.strategy.max_hold_hours == 2
        assert self.strategy.min_fvg_sensitivity == 0.1
        assert self.strategy.position_size == 0.1
        assert self.strategy.h1_lookback_candles == 36
        assert self.strategy.h4_lookback_candles == 9

    def test_initialization_with_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = FVGStrategy(
            risk_reward_ratio=3.0,
            max_hold_hours=4,
            min_fvg_sensitivity=0.2,
            position_size=0.05,
            h1_lookback_candles=48,
            h4_lookback_candles=12
        )
        assert strategy.risk_reward_ratio == 3.0
        assert strategy.max_hold_hours == 4
        assert strategy.min_fvg_sensitivity == 0.2
        assert strategy.position_size == 0.05
        assert strategy.h1_lookback_candles == 48
        assert strategy.h4_lookback_candles == 12

    def test_detect_timeframe_m15(self):
        """Test timeframe detection for M15 data."""
        data = self.create_test_data_m15(100)
        timeframe = self.strategy._detect_timeframe(data)
        assert timeframe == '15T'

    def test_detect_timeframe_h1(self):
        """Test timeframe detection for H1 data."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        data = pd.DataFrame({
            'open': range(100, 150),
            'high': range(101, 151),
            'low': range(99, 149),
            'close': range(100, 150),
            'volume': [1000] * 50
        }, index=dates)
        
        timeframe = self.strategy._detect_timeframe(data)
        assert timeframe == '1H'

    def test_detect_timeframe_insufficient_data(self):
        """Test timeframe detection with insufficient data."""
        data = pd.DataFrame({'close': [100]}, index=[datetime.now()])
        timeframe = self.strategy._detect_timeframe(data)
        assert timeframe == 'unknown'

    def test_is_execution_window(self):
        """Test execution window detection."""
        ny_tz = pytz.timezone('America/New_York')
        
        # Test London Open window (3:00-4:00 NY time)
        london_time = datetime(2024, 1, 15, 3, 30, tzinfo=ny_tz)  # Monday 3:30 AM NY
        assert self.strategy._is_execution_window(london_time) is True
        
        # Test NY Open window (10:00-11:00 NY time)
        ny_open_time = datetime(2024, 1, 15, 10, 15, tzinfo=ny_tz)  # Monday 10:15 AM NY
        assert self.strategy._is_execution_window(ny_open_time) is True
        
        # Test NY Afternoon window (14:00-15:00 NY time)
        ny_afternoon_time = datetime(2024, 1, 15, 14, 45, tzinfo=ny_tz)  # Monday 2:45 PM NY
        assert self.strategy._is_execution_window(ny_afternoon_time) is True
        
        # Test outside execution window
        outside_time = datetime(2024, 1, 15, 12, 0, tzinfo=ny_tz)  # Monday 12:00 PM NY
        assert self.strategy._is_execution_window(outside_time) is False

    def test_is_execution_window_utc_conversion(self):
        """Test execution window detection with UTC timestamp."""
        # UTC timestamp that corresponds to NY execution window
        utc_time = datetime(2024, 1, 15, 15, 30, tzinfo=pytz.UTC)  # 15:30 UTC = 10:30 NY (EST)
        assert self.strategy._is_execution_window(utc_time) is True
        
        # UTC timestamp outside execution window
        utc_time_outside = datetime(2024, 1, 15, 18, 0, tzinfo=pytz.UTC)  # 18:00 UTC = 13:00 NY
        assert self.strategy._is_execution_window(utc_time_outside) is False

    def test_is_weekend(self):
        """Test weekend detection."""
        ny_tz = pytz.timezone('America/New_York')
        
        # Saturday
        saturday = datetime(2024, 1, 13, 10, 0, tzinfo=ny_tz)
        assert self.strategy._is_weekend(saturday) is True
        
        # Sunday
        sunday = datetime(2024, 1, 14, 10, 0, tzinfo=ny_tz)
        assert self.strategy._is_weekend(sunday) is True
        
        # Monday (weekday)
        monday = datetime(2024, 1, 15, 10, 0, tzinfo=ny_tz)
        assert self.strategy._is_weekend(monday) is False
        
        # Friday (weekday)
        friday = datetime(2024, 1, 19, 10, 0, tzinfo=ny_tz)
        assert self.strategy._is_weekend(friday) is False

    def test_get_session_key(self):
        """Test session key generation."""
        ny_tz = pytz.timezone('America/New_York')
        
        # London Open window (3:00-4:00 NY) should be window 0
        london_time = datetime(2024, 1, 15, 3, 30, tzinfo=ny_tz)
        session_key = self.strategy._get_session_key(london_time)
        assert session_key == "2024-01-15_0"
        
        # NY Open window (10:00-11:00 NY) should be window 1
        ny_open_time = datetime(2024, 1, 15, 10, 30, tzinfo=ny_tz)
        session_key = self.strategy._get_session_key(ny_open_time)
        assert session_key == "2024-01-15_1"
        
        # NY Afternoon window (14:00-15:00 NY) should be window 2
        ny_afternoon_time = datetime(2024, 1, 15, 14, 30, tzinfo=ny_tz)
        session_key = self.strategy._get_session_key(ny_afternoon_time)
        assert session_key == "2024-01-15_2"

    def test_daily_open_filter_bullish(self):
        """Test daily open filter for bullish signals."""
        # Create daily data
        daily_dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        daily_data = pd.DataFrame({
            'open': [100, 105, 110, 115, 120],
            'high': [102, 107, 112, 117, 122],
            'low': [98, 103, 108, 113, 118],
            'close': [101, 106, 111, 116, 121],
            'volume': [1000] * 5
        }, index=daily_dates)
        
        # Test bullish FVG with price above daily open
        current_time = pd.Timestamp('2024-01-01 10:00:00')
        current_price = 102.0  # Above daily open (100)
        result = self.strategy._daily_open_filter(daily_data, current_time, current_price, 'bullish')
        assert result is True
        
        # Test bullish FVG with price below daily open
        current_price = 98.0  # Below daily open (100)
        result = self.strategy._daily_open_filter(daily_data, current_time, current_price, 'bullish')
        assert result is False

    def test_daily_open_filter_bearish(self):
        """Test daily open filter for bearish signals."""
        # Create daily data
        daily_dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        daily_data = pd.DataFrame({
            'open': [100, 105, 110, 115, 120],
            'high': [102, 107, 112, 117, 122],
            'low': [98, 103, 108, 113, 118],
            'close': [101, 106, 111, 116, 121],
            'volume': [1000] * 5
        }, index=daily_dates)
        
        # Test bearish FVG with price below daily open
        current_time = pd.Timestamp('2024-01-01 10:00:00')
        current_price = 98.0  # Below daily open (100)
        result = self.strategy._daily_open_filter(daily_data, current_time, current_price, 'bearish')
        assert result is True
        
        # Test bearish FVG with price above daily open
        current_price = 102.0  # Above daily open (100)
        result = self.strategy._daily_open_filter(daily_data, current_time, current_price, 'bearish')
        assert result is False

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        data = self.create_test_data_m15(50)  # Less than minimum required
        signals = self.strategy.generate_signals(data)
        
        # Should return data frame with zero signals
        assert 'signal' in signals.columns
        assert signals['signal'].sum() == 0

    def test_generate_signals_wrong_timeframe(self):
        """Test signal generation with wrong timeframe."""
        # Create H1 data instead of M15
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': range(100, 200),
            'high': range(101, 201),
            'low': range(99, 199),
            'close': range(100, 200),
            'volume': [1000] * 100
        }, index=dates)
        
        signals = self.strategy.generate_signals(data)
        
        # Should return dataframe with zero signals due to wrong timeframe
        assert signals['signal'].sum() == 0

    def test_generate_signals_basic_structure(self):
        """Test basic signal generation structure."""
        data = self.create_test_data_m15(500)
        signals = self.strategy.generate_signals(data)
        
        # Check that all required columns are present
        required_columns = ['signal', 'position_size', 'stop_loss', 'take_profit']
        for col in required_columns:
            assert col in signals.columns
        
        # Check data types and ranges
        assert signals['signal'].isin([0, 1, -1]).all()
        assert (signals['position_size'] >= 0).all()
        assert (signals['position_size'] <= 1).all()

    def test_is_fvg_unmitigated_bullish(self):
        """Test FVG mitigation detection for bullish FVG."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='15T')
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 10
        }, index=dates)
        
        # Mock FVG object
        class MockFVG:
            def __init__(self, fvg_type, bottom, top):
                self.fvg_type = fvg_type
                self.bottom = bottom
                self.top = top
        
        bullish_fvg = MockFVG('bullish', 102.0, 104.0)
        
        # Test unmitigated case (price doesn't go below bottom)
        result = self.strategy._is_fvg_unmitigated(data, bullish_fvg, 2, 8)
        assert result is True
        
        # Test mitigated case by modifying data
        data_mitigated = data.copy()
        data_mitigated.loc[data_mitigated.index[5], 'low'] = 101.0  # Below FVG bottom
        result = self.strategy._is_fvg_unmitigated(data_mitigated, bullish_fvg, 2, 8)
        assert result is False

    def test_is_fvg_unmitigated_bearish(self):
        """Test FVG mitigation detection for bearish FVG."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='15T')
        prices = [109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 10
        }, index=dates)
        
        # Mock FVG object
        class MockFVG:
            def __init__(self, fvg_type, bottom, top):
                self.fvg_type = fvg_type
                self.bottom = bottom
                self.top = top
        
        bearish_fvg = MockFVG('bearish', 104.0, 106.0)
        
        # Test unmitigated case (price doesn't go above top)
        result = self.strategy._is_fvg_unmitigated(data, bearish_fvg, 2, 8)
        assert result is True
        
        # Test mitigated case by modifying data
        data_mitigated = data.copy()
        data_mitigated.loc[data_mitigated.index[5], 'high'] = 107.0  # Above FVG top
        result = self.strategy._is_fvg_unmitigated(data_mitigated, bearish_fvg, 2, 8)
        assert result is False

    def test_has_price_touched_h4_fvg_basic(self):
        """Test basic H4 FVG touch detection."""
        m15_data = self.create_test_data_m15(200)
        
        # Create H4 data by resampling
        h4_data = m15_data.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Mock H4 FVG
        class MockFVG:
            def __init__(self, fvg_type, bottom, top, end_idx):
                self.fvg_type = fvg_type
                self.bottom = bottom
                self.top = top
                self.end_idx = end_idx
        
        # Create a bullish H4 FVG
        h4_fvgs = [MockFVG('bullish', 49900, 50100, 2)]
        
        # Test with a current index where we might have touched the FVG
        result = self.strategy._has_price_touched_h4_fvg(
            m15_data, h4_data, h4_fvgs, 100, 'bullish'
        )
        # Result depends on actual price data, just ensure it returns boolean
        assert isinstance(result, bool)

    def test_generate_signals_no_fvgs(self):
        """Test signal generation when no FVGs are detected."""
        # Create very stable price data (no gaps)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='15T')
        data = pd.DataFrame({
            'open': [50000] * 500,
            'high': [50001] * 500,
            'low': [49999] * 500,
            'close': [50000] * 500,
            'volume': [1000] * 500
        }, index=dates)
        
        signals = self.strategy.generate_signals(data)
        
        # Should have no signals due to no FVG detection
        assert signals['signal'].sum() == 0

    def test_session_trade_tracking(self):
        """Test that only one trade per session is allowed."""
        # This is more of an integration test that would require complex setup
        # For now, just test that session_trades dict is properly initialized
        assert isinstance(self.strategy.session_trades, dict)
        assert len(self.strategy.session_trades) == 0

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
        assert "FVG Strategy" in desc
        assert "M15 timeframe" in desc
        assert "H4 unmitigated FVG" in desc
        assert "R:R ratio 2.0:1" in desc
        assert "London Open" in desc
        assert "NY Open" in desc
        assert "Position size: 10.0%" in desc

    def test_get_strategy_name(self):
        """Test strategy name."""
        name = self.strategy.get_strategy_name()
        assert name == "fvg"

    def test_execution_windows_structure(self):
        """Test execution windows are properly configured."""
        assert len(self.strategy.execution_windows) == 3
        
        # Check London Open window
        assert (3, 4) in self.strategy.execution_windows
        
        # Check NY Open window
        assert (10, 11) in self.strategy.execution_windows
        
        # Check NY Afternoon window
        assert (14, 15) in self.strategy.execution_windows

    def test_fvg_detector_initialization(self):
        """Test that FVG detector is properly initialized."""
        assert hasattr(self.strategy, 'fvg_detector')
        assert self.strategy.fvg_detector is not None
        assert hasattr(self.strategy.fvg_detector, 'detect_fvgs')