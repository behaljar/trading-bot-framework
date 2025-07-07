"""Tests for Candlestick Patterns detector utility."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.utils.candlestick_patterns import CandlestickPatternDetector


class TestCandlestickPatternDetector:
    """Test cases for CandlestickPatternDetector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
        np.random.seed(42)
        
        # Generate synthetic OHLCV data
        opens = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        # Create realistic OHLC relationships
        ohlc_data = []
        volumes = []
        
        for i, open_price in enumerate(opens):
            # Random close price
            close = open_price + np.random.randn() * 2
            
            # High and low based on open/close
            high = max(open_price, close) + abs(np.random.randn()) * 1
            low = min(open_price, close) - abs(np.random.randn()) * 1
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            ohlc_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            })
            
            volumes.append(np.random.randint(1000, 5000))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [d['open'] for d in ohlc_data],
            'high': [d['high'] for d in ohlc_data],
            'low': [d['low'] for d in ohlc_data],
            'close': [d['close'] for d in ohlc_data],
            'volume': volumes
        })
        
        return df
    
    @pytest.fixture
    def detector(self):
        """Create CandlestickPatternDetector instance."""
        return CandlestickPatternDetector()
    
    def test_initialization(self, detector):
        """Test CandlestickPatternDetector initialization."""
        assert len(detector.patterns) > 0
        assert 'hammer' in detector.patterns
        assert 'doji' in detector.patterns
        assert 'shooting_star' in detector.patterns
    
    def test_list_available_patterns(self, detector):
        """Test listing available patterns."""
        patterns = detector.list_available_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert 'hammer' in patterns
    
    def test_detect_patterns_all(self, detector, sample_data):
        """Test detecting all patterns."""
        detected = detector.detect_patterns(sample_data)
        assert isinstance(detected, dict)
        assert len(detected) > 0
        
        # Check that each pattern returns a DataFrame
        for pattern_name, result_df in detected.items():
            assert isinstance(result_df, pd.DataFrame)
            if not result_df.empty:
                assert 'open' in result_df.columns
                assert 'high' in result_df.columns
                assert 'low' in result_df.columns
                assert 'close' in result_df.columns
    
    def test_detect_patterns_specific(self, detector, sample_data):
        """Test detecting specific patterns."""
        patterns_to_test = ['hammer', 'doji', 'shooting_star']
        detected = detector.detect_patterns(sample_data, patterns=patterns_to_test)
        
        assert isinstance(detected, dict)
        assert set(detected.keys()) == set(patterns_to_test)
        
        for pattern_name in patterns_to_test:
            assert pattern_name in detected
            assert isinstance(detected[pattern_name], pd.DataFrame)
    
    def test_get_pattern_matches(self, detector, sample_data):
        """Test getting pattern match indices."""
        matches = detector.get_pattern_matches(sample_data, patterns=['hammer', 'doji'])
        
        assert isinstance(matches, dict)
        assert 'hammer' in matches
        assert 'doji' in matches
        
        for pattern_name, indices in matches.items():
            assert isinstance(indices, list)
            # All indices should be valid
            for idx in indices:
                assert 0 <= idx < len(sample_data)
    
    def test_hammer_pattern_specific(self, detector):
        """Test hammer pattern with specific data."""
        # Create data with a clear hammer pattern
        data = {
            'open': [100, 99, 98, 95, 96],
            'high': [101, 100, 99, 95.2, 97],
            'low': [99, 98, 97, 92, 95],    # Index 3 has long lower shadow
            'close': [99.5, 98.5, 97.5, 95.1, 96.5]  # Index 3 closes near high
        }
        
        df = pd.DataFrame(data)
        
        # Test detection
        detected = detector.detect_patterns(df, patterns=['hammer'])
        assert 'hammer' in detected
        assert isinstance(detected['hammer'], pd.DataFrame)
        
        # Test matches
        matches = detector.get_pattern_matches(df, patterns=['hammer'])
        assert 'hammer' in matches
        assert isinstance(matches['hammer'], list)
    
    def test_doji_pattern_specific(self, detector):
        """Test doji pattern with specific data."""
        # Create data with doji pattern
        data = {
            'open': [100, 99, 100, 101, 102],
            'high': [101, 100, 102, 103, 104],
            'low': [99, 98, 98, 99, 100],
            'close': [100, 99, 100.05, 101, 102]  # Index 2 has very small body (doji)
        }
        
        df = pd.DataFrame(data)
        
        detected = detector.detect_patterns(df, patterns=['doji'])
        assert 'doji' in detected
        assert isinstance(detected['doji'], pd.DataFrame)
    
    def test_engulfing_pattern_specific(self, detector):
        """Test engulfing pattern with specific data."""
        # Create data with potential engulfing pattern
        data = {
            'open': [100, 99, 98, 95, 93],
            'high': [101, 100, 99, 95.5, 97],
            'low': [99, 98, 97, 94, 92],
            'close': [99.5, 98.5, 97.5, 94.5, 96]  # Last candle might engulf previous
        }
        
        df = pd.DataFrame(data)
        
        detected = detector.detect_patterns(df, patterns=['bullish_engulfing'])
        assert 'bullish_engulfing' in detected
        assert isinstance(detected['bullish_engulfing'], pd.DataFrame)
    
    def test_empty_dataframe(self, detector):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            detector.detect_patterns(empty_df)
    
    def test_insufficient_data(self, detector):
        """Test with insufficient data."""
        small_df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5]
        })
        
        # Should handle gracefully
        detected = detector.detect_patterns(small_df, patterns=['hammer'])
        assert isinstance(detected, dict)
        assert 'hammer' in detected
    
    def test_invalid_pattern_name(self, detector, sample_data):
        """Test with invalid pattern name."""
        detected = detector.detect_patterns(sample_data, patterns=['invalid_pattern'])
        
        # Should return empty result for invalid pattern
        assert isinstance(detected, dict)
        # Invalid pattern should either not be in dict or be empty DataFrame
        if 'invalid_pattern' in detected:
            assert detected['invalid_pattern'].empty
    
    def test_missing_columns(self, detector):
        """Test with missing required columns."""
        incomplete_df = pd.DataFrame({
            'open': [100, 99],
            'high': [101, 100]
            # Missing 'low' and 'close'
        })
        
        with pytest.raises(Exception):
            detector.detect_patterns(incomplete_df)
    
    def test_pattern_results_structure(self, detector):
        """Test that pattern results have expected structure."""
        # Create simple test data
        df = pd.DataFrame({
            'open': [100, 99, 98, 97, 96],
            'high': [101, 100, 99, 98, 97],
            'low': [99, 98, 97, 96, 95],
            'close': [99.5, 98.5, 97.5, 96.5, 95.5]
        })
        
        detected = detector.detect_patterns(df, patterns=['hammer'])
        result_df = detected['hammer']
        
        if not result_df.empty:
            # Should have OHLC columns plus pattern column
            assert 'open' in result_df.columns
            assert 'high' in result_df.columns
            assert 'low' in result_df.columns
            assert 'close' in result_df.columns
            
            # Should have same number of rows as input
            assert len(result_df) == len(df)
            
            # Pattern column should contain boolean values
            pattern_col = result_df.columns[-1]  # Usually last column
            assert result_df[pattern_col].dtype == bool
    
    def test_consistency(self, detector, sample_data):
        """Test that results are consistent across multiple calls."""
        detected1 = detector.detect_patterns(sample_data, patterns=['hammer'])
        detected2 = detector.detect_patterns(sample_data, patterns=['hammer'])
        
        # Results should be identical
        if not detected1['hammer'].empty and not detected2['hammer'].empty:
            pd.testing.assert_frame_equal(detected1['hammer'], detected2['hammer'])