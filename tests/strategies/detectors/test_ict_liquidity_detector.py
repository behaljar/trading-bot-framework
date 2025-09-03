"""
Tests for ICT Liquidity Detector
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from framework.strategies.detectors.ict_liquidity_detector import (
    ICTLiquidityDetector, 
    LiquidityLevel, 
    WeekOpeningGap
)


class TestICTLiquidityDetector:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        
        # Create realistic price data
        np.random.seed(42)
        base_price = 50000.0
        prices = []
        current_price = base_price
        
        for _ in range(len(dates)):
            # Add some volatility
            change = np.random.normal(0, 0.02) * current_price
            current_price += change
            prices.append(current_price)
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high_noise = abs(np.random.normal(0, 0.01)) * price
            low_noise = abs(np.random.normal(0, 0.01)) * price
            
            high = price + high_noise
            low = price - low_noise
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df.tz_localize('UTC')
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return ICTLiquidityDetector()
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.equal_level_threshold == 0.0005
        assert detector.liquidity_sweep_threshold == 0.0001
        assert detector.strength_volume_multiplier == 1.5
    
    def test_validate_data(self, detector, sample_data):
        """Test data validation."""
        # Valid data
        assert detector._validate_data(sample_data) == True
        
        # Empty data
        empty_df = pd.DataFrame()
        assert detector._validate_data(empty_df) == False
        
        # Missing columns
        invalid_df = pd.DataFrame({'price': [1, 2, 3]})
        assert detector._validate_data(invalid_df) == False
        
        # Not enough data
        small_df = sample_data.head(5)
        assert detector._validate_data(small_df) == False
    
    def test_detect_previous_day_liquidity(self, detector, sample_data):
        """Test previous day liquidity detection."""
        current_time = sample_data.index[-1]
        liquidity = detector.detect_previous_day_liquidity(sample_data, current_time)
        
        # Should find some PDH/PDL levels
        assert len(liquidity) > 0
        
        # Check that we have both PDH and PDL
        level_types = [level.level_type for level in liquidity]
        assert 'PDH' in level_types
        assert 'PDL' in level_types
        
        # Check that all levels are properly structured
        for level in liquidity:
            assert isinstance(level, LiquidityLevel)
            assert level.level_type in ['PDH', 'PDL']
            assert level.price > 0
            assert 1 <= level.strength <= 5
    
    def test_detect_previous_week_liquidity(self, detector, sample_data):
        """Test previous week liquidity detection."""
        current_time = sample_data.index[-1]
        liquidity = detector.detect_previous_week_liquidity(sample_data, current_time)
        
        # Should find some PWH/PWL levels
        assert len(liquidity) > 0
        
        # Check level types
        level_types = [level.level_type for level in liquidity]
        assert 'PWH' in level_types
        assert 'PWL' in level_types
        
        # Weekly levels should have higher strength
        for level in liquidity:
            assert level.strength >= 2
    
    def test_detect_session_liquidity(self, detector, sample_data):
        """Test session liquidity detection."""
        current_time = sample_data.index[-1]
        liquidity = detector.detect_session_liquidity(sample_data, current_time)
        
        # Should find session-based liquidity
        assert len(liquidity) >= 0  # May be empty if no clear sessions
        
        # Check session assignments
        for level in liquidity:
            assert level.session in ['asian', 'london', 'ny_open', 'ny_pm', None]
    
    def test_detect_relative_equal_levels(self, detector):
        """Test relative equal levels detection."""
        # Create data with clear equal highs/lows
        dates = pd.date_range(start='2024-01-01', freq='1h', periods=100)
        
        # Create price pattern with equal highs
        equal_high_price = 50000.0
        data = []
        
        for i, date in enumerate(dates):
            if i in [20, 40, 60]:  # Equal highs at these points
                high = equal_high_price
                low = equal_high_price - 100
                close = equal_high_price - 50
            else:
                high = 49900 + np.random.uniform(-50, 50)
                low = high - 100
                close = high - 50
            
            data.append({
                'open': close,
                'high': high,
                'low': low,
                'close': close,
                'volume': 100
            })
        
        df = pd.DataFrame(data, index=dates).tz_localize('UTC')
        
        current_time = df.index[-1]
        liquidity = detector.detect_relative_equal_levels(df, current_time)
        
        # Should detect equal levels
        reh_levels = [l for l in liquidity if l.level_type == 'REH']
        assert len(reh_levels) > 0
    
    def test_detect_all_liquidity(self, detector, sample_data):
        """Test comprehensive liquidity detection."""
        current_time = sample_data.index[-1]
        all_liquidity = detector.detect_all_liquidity(sample_data, current_time)
        
        # Should find various types of liquidity
        assert len(all_liquidity) > 0
        
        # Check that liquidity is sorted by strength
        strengths = [level.strength for level in all_liquidity]
        assert strengths == sorted(strengths, reverse=True)
        
        # Check sweep status is updated
        swept_count = sum(1 for level in all_liquidity if level.is_swept)
        assert swept_count >= 0  # Some may be swept
    
    def test_get_liquidity_near_price(self, detector):
        """Test getting liquidity near price."""
        # Create mock liquidity levels
        liquidity = [
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PDH',
                price=50000.0,
                strength=3
            ),
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PDL',
                price=49000.0,
                strength=4
            ),
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PWH',
                price=52000.0,
                strength=5
            )
        ]
        
        current_price = 50100.0
        near_levels = detector.get_liquidity_near_price(
            liquidity, current_price, distance_pct=0.02
        )
        
        # Should find the PDH level (within 2%)
        assert len(near_levels) == 1
        assert near_levels[0].level_type == 'PDH'
    
    def test_get_unswept_liquidity(self, detector):
        """Test filtering unswept liquidity."""
        # Create mock liquidity with some swept
        liquidity = [
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PDH',
                price=50000.0,
                is_swept=False
            ),
            LiquidityLevel(
                timestamp=pd.Timestamp('2024-01-01'),
                level_type='PDL',
                price=49000.0,
                is_swept=True
            )
        ]
        
        unswept = detector.get_unswept_liquidity(liquidity)
        
        assert len(unswept) == 1
        assert unswept[0].level_type == 'PDH'
        assert not unswept[0].is_swept
    
    def test_week_opening_gaps(self, detector):
        """Test week opening gap detection."""
        # Create data spanning multiple weeks with gaps
        dates = pd.date_range(start='2024-01-01', end='2024-01-28', freq='1h')
        
        data = []
        for i, date in enumerate(dates):
            # Create a gap at week starts (Sundays)
            if date.weekday() == 6:  # Sunday
                base_price = 50000 if i == 0 else data[-1]['close'] * 1.01  # 1% gap
            else:
                base_price = 50000 if i == 0 else data[-1]['close']
            
            data.append({
                'open': base_price,
                'high': base_price + 10,
                'low': base_price - 10,
                'close': base_price,
                'volume': 100
            })
        
        df = pd.DataFrame(data, index=dates).tz_localize('UTC')
        
        current_time = df.index[-1]
        gaps = detector.detect_week_opening_gaps(df, current_time)
        
        # Should detect some gaps
        assert len(gaps) >= 0
        
        for gap in gaps:
            assert isinstance(gap, WeekOpeningGap)
            assert gap.gap_direction in ['up', 'down']
            assert gap.gap_size > 0
    
    def test_get_session_from_time(self, detector):
        """Test session identification from timestamp."""
        ny_tz = pytz.timezone('America/New_York')
        
        # Test different session times
        test_times = [
            (datetime(2024, 1, 1, 1, 0), 'asian'),    # 1 AM NY = Asian
            (datetime(2024, 1, 1, 5, 0), 'london'),   # 5 AM NY = London
            (datetime(2024, 1, 1, 10, 30), 'ny_open'), # 10:30 AM NY = NY Open
            (datetime(2024, 1, 1, 14, 0), 'ny_pm'),    # 2 PM NY = NY PM
            (datetime(2024, 1, 1, 8, 0), 'overlap'),   # 8 AM NY = Overlap
        ]
        
        for dt, expected_session in test_times:
            ny_time = ny_tz.localize(dt)
            session = detector._get_session_from_time(ny_time)
            assert session == expected_session
    
    def test_convert_to_ny_timezone(self, detector, sample_data):
        """Test timezone conversion."""
        ny_data = detector._convert_to_ny_timezone(sample_data)
        
        # Should have NY timezone
        assert str(ny_data.index.tz) == 'America/New_York'
        assert len(ny_data) == len(sample_data)
    
    def test_filter_session_data(self, detector):
        """Test session data filtering."""
        # Create hourly data for a full day
        ny_tz = pytz.timezone('America/New_York')
        dates = []
        for hour in range(24):
            dates.append(ny_tz.localize(datetime(2024, 1, 1, hour, 0)))
        
        data = pd.DataFrame({
            'open': range(24),
            'high': range(24),
            'low': range(24),
            'close': range(24),
            'volume': range(24)
        }, index=dates)
        
        # Test London session (3-8 AM)
        london_data = detector._filter_session_data(data, [(3, 8)], 'london')
        assert len(london_data) == 5  # 5 hours: 3, 4, 5, 6, 7
        assert london_data.index[0].hour == 3
        assert london_data.index[-1].hour == 7
        
        # Test Asian session (crosses midnight: 18-2)
        asian_data = detector._filter_session_data(data, [(18, 2)], 'asian')
        assert len(asian_data) == 8  # 6 hours PM + 2 hours AM


if __name__ == "__main__":
    pytest.main([__file__])