"""
Tests for ICT Silver Bullet Strategy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from framework.strategies.ict_silver_bullet_strategy import ICTSilverBulletStrategy


class TestICTSilverBulletStrategy:
    
    @pytest.fixture
    def sample_1min_data(self):
        """Create sample 1-minute OHLCV data for testing."""
        # Create 7 days of 1-minute data
        start_date = datetime(2024, 1, 1, 0, 0)
        dates = []
        
        # Generate timestamps (skip weekends for realism)
        current_date = start_date
        while len(dates) < 7200:  # About 5 days of 1-minute data
            if current_date.weekday() < 5:  # Monday-Friday only
                dates.append(current_date)
            current_date += timedelta(minutes=1)
        
        dates = pd.to_datetime(dates).tz_localize('UTC')
        
        # Create realistic price data with some volatility
        np.random.seed(42)
        base_price = 50000.0
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Add session-based volatility
            hour = date.hour
            if 3 <= hour <= 4:  # London open - higher volatility
                volatility = 0.0015
            elif 10 <= hour <= 11:  # NY open - highest volatility  
                volatility = 0.002
            elif 14 <= hour <= 15:  # NY PM - medium volatility
                volatility = 0.0012
            else:
                volatility = 0.0005
            
            change = np.random.normal(0, volatility) * current_price
            current_price += change
            prices.append(current_price)
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            spread = price * 0.0001  # 0.01% spread
            
            open_price = prices[i-1] if i > 0 else price
            high = max(open_price, price) + abs(np.random.normal(0, spread))
            low = min(open_price, price) - abs(np.random.normal(0, spread))
            close = price
            volume = np.random.uniform(50, 500)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def strategy_default(self):
        """Create strategy instance with default parameters."""
        return ICTSilverBulletStrategy()
    
    @pytest.fixture
    def strategy_custom(self):
        """Create strategy instance with custom parameters."""
        return ICTSilverBulletStrategy(
            risk_reward_ratio=2.5,
            max_hold_hours=6,
            enable_london_open=True,
            enable_ny_open=True,
            enable_ny_pm=False,
            min_liquidity_strength=4,
            require_structure_shift=False,  # Disable for simpler testing
            max_trades_per_session=2
        )
    
    def test_strategy_initialization(self, strategy_default):
        """Test strategy initialization."""
        assert strategy_default.name == "ict_silver_bullet"
        assert strategy_default.risk_reward_ratio == 3.0
        assert strategy_default.max_hold_hours == 4
        assert strategy_default.position_size == 0.02
        assert strategy_default.min_liquidity_strength == 3
        
        # Check session windows
        assert 'london_open' in strategy_default.session_windows
        assert 'ny_open' in strategy_default.session_windows
        assert 'ny_pm' in strategy_default.session_windows
        assert strategy_default.session_windows['london_open'] == (3, 4)
        assert strategy_default.session_windows['ny_open'] == (10, 11)
        assert strategy_default.session_windows['ny_pm'] == (14, 15)
    
    def test_custom_initialization(self, strategy_custom):
        """Test strategy initialization with custom parameters."""
        assert strategy_custom.risk_reward_ratio == 2.5
        assert strategy_custom.max_hold_hours == 6
        assert strategy_custom.max_trades_per_session == 2
        assert 'ny_pm' not in strategy_custom.session_windows
    
    def test_data_validation(self, strategy_default, sample_1min_data):
        """Test data validation."""
        # Valid data
        assert strategy_default.validate_data(sample_1min_data) == True
        
        # Invalid data
        invalid_data = pd.DataFrame({'price': [1, 2, 3]})
        assert strategy_default.validate_data(invalid_data) == False
    
    def test_detect_timeframe(self, strategy_default, sample_1min_data):
        """Test timeframe detection."""
        timeframe = strategy_default._detect_timeframe(sample_1min_data)
        assert timeframe == '1min'
        
        # Test 5-minute data
        data_5min = sample_1min_data.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        timeframe_5min = strategy_default._detect_timeframe(data_5min)
        assert timeframe_5min == '5min'
    
    def test_session_detection(self, strategy_default):
        """Test session detection."""
        ny_tz = pytz.timezone('America/New_York')
        
        # Test different session times
        test_cases = [
            (datetime(2024, 1, 1, 3, 30), 'london_open'),
            (datetime(2024, 1, 1, 10, 30), 'ny_open'),
            (datetime(2024, 1, 1, 14, 30), 'ny_pm'),
            (datetime(2024, 1, 1, 8, 0), None),  # Outside session
            (datetime(2024, 1, 1, 22, 0), None),  # Outside session
        ]
        
        for dt, expected_session in test_cases:
            ny_time = ny_tz.localize(dt).astimezone(pytz.UTC)
            session = strategy_default._get_current_session(ny_time)
            assert session == expected_session
    
    def test_session_key_generation(self, strategy_default):
        """Test session key generation."""
        ny_tz = pytz.timezone('America/New_York')
        test_time = ny_tz.localize(datetime(2024, 1, 1, 10, 30)).astimezone(pytz.UTC)
        
        session_key = strategy_default._get_session_key(test_time, 'ny_open')
        assert session_key == "2024-01-01_ny_open"
    
    def test_trade_session_tracking(self, strategy_default):
        """Test session trade tracking."""
        session_key = "2024-01-01_ny_open"
        
        # Initially no trades
        assert not strategy_default._has_traded_in_session(session_key)
        
        # Mark as traded
        strategy_default._mark_session_traded(session_key)
        assert strategy_default._has_traded_in_session(session_key)
        
        # Check trade count
        assert strategy_default.session_trades[session_key] == 1
    
    def test_resample_to_higher_timeframes(self, strategy_default, sample_1min_data):
        """Test resampling to higher timeframes."""
        # Test 15-minute resampling
        m15_data = strategy_default._resample_to_15min(sample_1min_data)
        
        assert len(m15_data) > 0
        assert len(m15_data) < len(sample_1min_data)
        assert all(col in m15_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Test hourly resampling
        h1_data = strategy_default._resample_to_hourly(sample_1min_data)
        
        assert len(h1_data) > 0
        assert len(h1_data) < len(m15_data)
        assert all(col in h1_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_risk_reward_validation(self, strategy_default):
        """Test risk/reward ratio validation."""
        current_price = 50000.0
        
        # Valid bullish setup (3:1 R:R)
        stop_loss = 49900.0  # 100 points risk
        take_profit = 50300.0  # 300 points reward
        assert strategy_default._validate_risk_reward(
            current_price, stop_loss, take_profit, 'bullish'
        ) == True
        
        # Invalid bullish setup (insufficient R:R)
        take_profit_bad = 50150.0  # Only 150 points reward
        assert strategy_default._validate_risk_reward(
            current_price, stop_loss, take_profit_bad, 'bullish'
        ) == False
        
        # Valid bearish setup
        stop_loss = 50100.0  # 100 points risk
        take_profit = 49700.0  # 300 points reward
        assert strategy_default._validate_risk_reward(
            current_price, stop_loss, take_profit, 'bearish'
        ) == True
    
    def test_fvg_formation_detection(self, strategy_default):
        """Test FVG formation detection."""
        # Create mock data with clear FVG pattern
        data = pd.DataFrame({
            'open': [50000, 50050, 50200],
            'high': [50020, 50070, 50250],
            'low': [49980, 50030, 50180],
            'close': [50010, 50060, 50220],
            'volume': [100, 100, 100]
        })
        
        # Test bullish FVG detection (low[2] > high[0])
        fvg_info = strategy_default._check_fvg_formation(data, 2, 'bullish')
        
        assert fvg_info is not None
        assert fvg_info['type'] == 'bullish'
        assert fvg_info['top'] == 50180  # Low of third candle
        assert fvg_info['bottom'] == 50020  # High of first candle
        
        # Test bearish FVG (create opposite pattern)
        data_bearish = pd.DataFrame({
            'open': [50200, 50150, 50000],
            'high': [50220, 50170, 50020],
            'low': [50180, 50130, 49980],
            'close': [50190, 50140, 50010],
            'volume': [100, 100, 100]
        })
        
        fvg_info_bearish = strategy_default._check_fvg_formation(data_bearish, 2, 'bearish')
        
        assert fvg_info_bearish is not None
        assert fvg_info_bearish['type'] == 'bearish'
    
    def test_stop_loss_calculation(self, strategy_default):
        """Test stop loss calculation."""
        # Create mock data
        data = pd.DataFrame({
            'open': [50000, 50050, 50200],
            'high': [50020, 50070, 50250],
            'low': [49980, 50030, 50180],
            'close': [50010, 50060, 50220],
            'volume': [100, 100, 100]
        })
        
        # Test bullish stop loss (should be low of first candle)
        fvg_info = {'type': 'bullish'}
        stop_loss = strategy_default._calculate_stop_loss(data, 2, 'bullish', fvg_info)
        assert stop_loss == 49980  # Low of first candle (index 0)
        
        # Test bearish stop loss (should be high of first candle)
        stop_loss_bearish = strategy_default._calculate_stop_loss(data, 2, 'bearish', fvg_info)
        assert stop_loss_bearish == 50020  # High of first candle (index 0)
    
    def test_generate_signals_basic(self, strategy_custom, sample_1min_data):
        """Test basic signal generation."""
        # This is a basic test - the strategy may not generate signals 
        # with random data, but should not crash
        try:
            signals = strategy_custom.generate_signals(sample_1min_data)
            
            # Check output format
            assert len(signals) == len(sample_1min_data)
            assert 'signal' in signals.columns
            assert 'position_size' in signals.columns
            assert 'stop_loss' in signals.columns
            assert 'take_profit' in signals.columns
            
            # Check signal values are valid
            assert all(signals['signal'].isin([0, 1, -1]))
            assert all(signals['position_size'] >= 0)
            
            # Count signals
            buy_signals = (signals['signal'] == 1).sum()
            sell_signals = (signals['signal'] == -1).sum()
            
            # May have signals but not required with random data
            assert buy_signals >= 0
            assert sell_signals >= 0
            
        except Exception as e:
            # Strategy should handle edge cases gracefully
            pytest.fail(f"Strategy crashed with error: {e}")
    
    def test_generate_signals_insufficient_data(self, strategy_default):
        """Test signal generation with insufficient data."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'open': [50000, 50010],
            'high': [50020, 50030],
            'low': [49980, 49990],
            'close': [50010, 50020],
            'volume': [100, 100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1min', tz='UTC'))
        
        signals = strategy_default.generate_signals(small_data)
        
        # Should return all zeros due to insufficient data
        assert all(signals['signal'] == 0)
        assert all(signals['position_size'] == 0)
    
    def test_generate_signals_wrong_timeframe(self, strategy_default):
        """Test signal generation with wrong timeframe."""
        # Create hourly data (should not trade)
        hourly_data = pd.DataFrame({
            'open': [50000, 50100, 50200],
            'high': [50050, 50150, 50250],
            'low': [49950, 50050, 50150],
            'close': [50020, 50120, 50220],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC'))
        
        signals = strategy_default.generate_signals(hourly_data)
        
        # Should return all zeros for wrong timeframe
        assert all(signals['signal'] == 0)
    
    def test_strategy_description(self, strategy_default):
        """Test strategy description."""
        description = strategy_default.get_description()
        
        assert "ICT Silver Bullet Strategy" in description
        assert "R:R 3.0:1" in description
        assert "Min liquidity strength: 3" in description
        assert "Position size: 2.0%" in description
        assert "London Open" in description
        assert "Ny Open" in description
        assert "Ny Pm" in description
    
    def test_strategy_name(self, strategy_default):
        """Test strategy name."""
        name = strategy_default.get_strategy_name()
        assert name == "ict_silver_bullet"
    
    def test_liquidity_filtering(self, strategy_default):
        """Test liquidity level filtering."""
        from framework.strategies.detectors.ict_liquidity_detector import LiquidityLevel
        
        # Create mock liquidity levels
        liquidity_levels = [
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PDH',
                price=50000.0,
                strength=2,  # Below minimum
                is_swept=False
            ),
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PWH',
                price=50500.0,
                strength=4,  # Above minimum
                is_swept=False
            ),
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PDL',
                price=49500.0,
                strength=5,  # Above minimum
                is_swept=True  # But swept
            )
        ]
        
        filtered = strategy_default._filter_target_liquidity(liquidity_levels)
        
        # Should only keep PWH (strong and unswept)
        assert len(filtered) == 1
        assert filtered[0].level_type == 'PWH'
        assert filtered[0].strength >= strategy_default.min_liquidity_strength
        assert not filtered[0].is_swept


if __name__ == "__main__":
    pytest.main([__file__])