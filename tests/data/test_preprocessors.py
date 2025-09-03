import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from framework.data.preprocessors.base_preprocessor import BasePreprocessor
from framework.data.preprocessors.infinite_value_preprocessor import InfiniteValuePreprocessor
from framework.data.preprocessors.nan_value_preprocessor import NanValuePreprocessor
from framework.data.preprocessors.ohlc_validator_preprocessor import OhlcValidatorPreprocessor
from framework.data.preprocessors.data_quality_validator_preprocessor import DataQualityValidatorPreprocessor
from framework.data.preprocessors.time_gap_detector_preprocessor import TimeGapDetectorPreprocessor
from framework.data.preprocessors.preprocessor_manager import PreprocessorManager, create_default_manager


class TestInfiniteValuePreprocessor:
    def test_removes_infinite_values(self):
        # Create test data with infinite values
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'Open': [100, np.inf, 102, 103, 104],
            'High': [101, 102, 103, -np.inf, 105],
            'Low': [99, 100, 101, 102, np.inf],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 2000, np.inf, 4000, 5000]
        }, index=dates)
        
        preprocessor = InfiniteValuePreprocessor()
        result = preprocessor.process(df)
        
        # Check that infinite values are replaced with NaN
        assert np.isnan(result.iloc[1]['Open'])
        assert np.isnan(result.iloc[3]['High'])
        assert np.isnan(result.iloc[4]['Low'])
        assert np.isnan(result.iloc[2]['Volume'])
        
    def test_handles_empty_dataframe(self):
        df = pd.DataFrame()
        preprocessor = InfiniteValuePreprocessor()
        result = preprocessor.process(df)
        assert result.empty


class TestNanValuePreprocessor:
    def test_fills_nan_values(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, 104],
            'High': [101, 102, np.nan, 104, 105],
            'Low': [99, 100, 101, np.nan, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, np.nan],
            'Volume': [1000, 2000, 3000, np.nan, 5000]
        }, index=dates)
        
        preprocessor = NanValuePreprocessor()
        result = preprocessor.process(df)
        
        # Check that rows with NaN values are dropped (safest for trading data)
        assert not result.isnull().any().any()
        assert len(result) == 1  # Only one row without any NaN values (first row)
        assert result.iloc[0]['Open'] == 100
        assert result.iloc[0]['High'] == 101
        assert result.iloc[0]['Low'] == 99
        assert result.iloc[0]['Close'] == 100.5
        assert result.iloc[0]['Volume'] == 1000
        
    def test_handles_leading_nan(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'Open': [np.nan, np.nan, 102, 103, 104],
            'Close': [np.nan, 101.5, 102.5, 103.5, 104.5]
        }, index=dates)
        
        preprocessor = NanValuePreprocessor()
        result = preprocessor.process(df)
        
        # Check that leading NaN values are handled
        assert not result.isnull().any().any()


class TestOhlcValidatorPreprocessor:
    def test_fixes_invalid_ohlc(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [98, 102, 103, 104, 105],  # First value invalid (< open)
            'Low': [99, 100, 105, 102, 103],   # Third value invalid (> high)
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 2000, 3000, 4000, 5000]
        }, index=dates)
        
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(df)
        
        # Check that OHLC relationships are valid
        assert all(result['High'] >= result['Low'])
        assert all(result['High'] >= result['Open'])
        assert all(result['High'] >= result['Close'])
        assert all(result['Low'] <= result['Open'])
        assert all(result['Low'] <= result['Close'])
        
    def test_handles_missing_columns(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'Volume': [1000, 2000, 3000, 4000, 5000]
        }, index=dates)
        
        preprocessor = OhlcValidatorPreprocessor()
        result = preprocessor.process(df)
        
        # Should return unchanged when OHLC columns are missing
        assert result.equals(df)


class TestDataQualityValidatorPreprocessor:
    def test_removes_negative_prices(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'Open': [100, -101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 0, 103],  # Zero is also invalid
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 2000, 3000, 4000, 5000]
        }, index=dates)
        
        preprocessor = DataQualityValidatorPreprocessor()
        result = preprocessor.process(df)
        
        # Check that rows with non-positive prices are removed
        assert len(result) < len(df)
        assert all(result['Open'] > 0)
        assert all(result['Low'] > 0)
        
    def test_removes_duplicate_timestamps(self):
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5]
        }, index=dates)
        
        preprocessor = DataQualityValidatorPreprocessor()
        result = preprocessor.process(df)
        
        # Check that duplicates are removed
        assert not result.index.duplicated().any()
        assert len(result) == 4
        
    def test_sorts_by_timestamp(self):
        dates = pd.DatetimeIndex(['2024-01-04', '2024-01-02', '2024-01-01', '2024-01-03'])
        df = pd.DataFrame({
            'Open': [104, 102, 101, 103],
            'Close': [104.5, 102.5, 101.5, 103.5]
        }, index=dates)
        
        preprocessor = DataQualityValidatorPreprocessor()
        result = preprocessor.process(df)
        
        # Check that data is sorted
        assert result.index.is_monotonic_increasing


class TestTimeGapDetectorPreprocessor:
    def test_detects_time_gaps(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        
        # Create data with gaps
        dates = pd.DatetimeIndex([
            '2024-01-01 09:00:00',
            '2024-01-01 10:00:00',
            '2024-01-01 11:00:00',
            '2024-01-01 15:00:00',  # 4 hour gap
            '2024-01-01 16:00:00'
        ])
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5]
        }, index=dates)
        
        preprocessor = TimeGapDetectorPreprocessor(gap_threshold_minutes=60)
        result = preprocessor.process(df)
        
        # Check that data is unchanged (detector only reports)
        assert result.equals(df)
        
        # Check that gap was detected in logs
        assert "Found 1 time gaps" in caplog.text
        
    def test_detects_weekend_gaps(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        
        # Create data with weekend gap
        dates = pd.DatetimeIndex([
            '2024-01-05 16:00:00',  # Friday
            '2024-01-08 09:00:00',  # Monday
        ])
        df = pd.DataFrame({
            'Open': [100, 101],
            'Close': [100.5, 101.5]
        }, index=dates)
        
        preprocessor = TimeGapDetectorPreprocessor()
        result = preprocessor.process(df)
        
        # Check that weekend gap was detected
        assert "Found 1 time gaps" in caplog.text
        
    def test_infers_timeframe(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        
        # Create 1-hour data
        dates = pd.date_range('2024-01-01', periods=24, freq='1h')
        df = pd.DataFrame({
            'Open': range(100, 124),
            'Close': range(101, 125)
        }, index=dates)
        
        preprocessor = TimeGapDetectorPreprocessor()
        result = preprocessor.process(df)
        
        # Check that timeframe was detected
        assert "Detected timeframe: 1-hour" in caplog.text


class TestPreprocessorManager:
    def test_runs_all_preprocessors(self):
        # Create data with multiple issues
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'Open': [100, np.inf, -102, 103, 104],
            'High': [98, 102, 103, 104, 105],  # Invalid high < open
            'Low': [99, np.nan, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 2000, 3000, -4000, 5000]
        }, index=dates)
        
        manager = create_default_manager()
        result = manager.run(df)
        
        # Check that all issues are resolved
        assert not result.isnull().any().any()  # No NaN values
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()  # No inf values
        assert all(result['Open'] > 0)  # No negative prices
        assert all(result['Volume'] >= 0)  # No negative volumes
        assert all(result['High'] >= result['Low'])  # Valid OHLC relationships
        
    def test_add_remove_preprocessor(self):
        manager = PreprocessorManager()
        
        # Add preprocessors
        preprocessor1 = InfiniteValuePreprocessor()
        preprocessor2 = NanValuePreprocessor()
        
        manager.add_preprocessor(preprocessor1)
        manager.add_preprocessor(preprocessor2)
        
        assert len(manager.list_preprocessors()) == 2
        assert "InfiniteValuePreprocessor" in manager.list_preprocessors()
        assert "NanValuePreprocessor" in manager.list_preprocessors()
        
        # Remove preprocessor
        success = manager.remove_preprocessor("InfiniteValuePreprocessor")
        assert success
        assert len(manager.list_preprocessors()) == 1
        assert "InfiniteValuePreprocessor" not in manager.list_preprocessors()
        
        # Clear all
        manager.clear_preprocessors()
        assert len(manager.list_preprocessors()) == 0
        
    def test_handles_empty_dataframe(self):
        df = pd.DataFrame()
        manager = create_default_manager()
        result = manager.run(df)
        assert result.empty
        
    def test_handles_none_dataframe(self):
        manager = create_default_manager()
        result = manager.run(None)
        assert result is None