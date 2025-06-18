"""
Unit tests for trading strategies
"""
import unittest
import pandas as pd
import numpy as np
from strategies.trend_following import SMAStrategy
from strategies.base_strategy import Signal

class TestSMAStrategy(unittest.TestCase):
    """Tests for SMA strategy"""

    def setUp(self):
        """Prepare test data"""
        # Create trending data for SMA test
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = np.cumsum(np.random.randn(100)) + 100  # Random walk around 100

        self.test_data = pd.DataFrame({
            'Open': prices,
            'High': prices + np.random.rand(100),
            'Low': prices - np.random.rand(100),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

        self.strategy = SMAStrategy({"short_window": 20, "long_window": 50})

    def test_generate_signals(self):
        """Test SMA signal generation"""
        result = self.strategy.generate_signals(self.test_data)

        # Check that result is a DataFrame with the right shape
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that signal column exists and has valid values
        self.assertIn('signal', result.columns)
        self.assertTrue(all(signal in [Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value]
                          for signal in result['signal']))
        
        # Check that stop_loss and take_profit columns exist
        self.assertIn('stop_loss', result.columns)
        self.assertIn('take_profit', result.columns)

if __name__ == '__main__':
    unittest.main()