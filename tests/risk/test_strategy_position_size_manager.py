"""
Unit tests for Strategy Position Size Manager

Tests that the new risk manager correctly uses position sizes from strategies.
"""

import unittest
import sys
from pathlib import Path

# Add framework to path
test_dir = Path(__file__).parent
framework_dir = test_dir.parent
sys.path.append(str(framework_dir))

from framework.risk.strategy_position_size_manager import StrategyPositionSizeManager


class TestStrategyPositionSizeManager(unittest.TestCase):
    """Test suite for Strategy Position Size Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = StrategyPositionSizeManager()
    
    def test_initialization(self):
        """Test risk manager initialization."""
        # Default initialization
        rm = StrategyPositionSizeManager()
        self.assertEqual(rm.max_position_size, 1.0)
        self.assertEqual(rm.min_position_size, 0.001)
        self.assertTrue(rm.apply_safety_limits)
        
        # Custom initialization
        rm = StrategyPositionSizeManager(
            max_position_size=0.5,
            min_position_size=0.01,
            apply_safety_limits=False
        )
        self.assertEqual(rm.max_position_size, 0.5)
        self.assertEqual(rm.min_position_size, 0.01)
        self.assertFalse(rm.apply_safety_limits)
    
    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Max position size too high
        with self.assertRaises(ValueError):
            StrategyPositionSizeManager(max_position_size=3.0)
        
        # Min position size too low
        with self.assertRaises(ValueError):
            StrategyPositionSizeManager(min_position_size=0.0)
        
        # Min > Max
        with self.assertRaises(ValueError):
            StrategyPositionSizeManager(max_position_size=0.1, min_position_size=0.2)
    
    def test_use_strategy_position_size(self):
        """Test that manager uses strategy's position size."""
        # Valid strategy position size
        position_size = self.risk_manager.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.05
        )
        self.assertEqual(position_size, 0.05)
        
        # Different strategy position size
        position_size = self.risk_manager.calculate_position_size(
            signal=-1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.02
        )
        self.assertEqual(position_size, 0.02)
    
    def test_no_signal_returns_zero(self):
        """Test that no signal returns zero position size."""
        position_size = self.risk_manager.calculate_position_size(
            signal=0,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.05
        )
        self.assertEqual(position_size, 0.0)
    
    def test_no_strategy_position_size(self):
        """Test behavior when strategy doesn't provide position size."""
        # No strategy position size provided
        position_size = self.risk_manager.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0
        )
        self.assertEqual(position_size, 0.0)
        
        # Strategy position size is None
        position_size = self.risk_manager.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=None
        )
        self.assertEqual(position_size, 0.0)
        
        # Strategy position size is zero or negative
        position_size = self.risk_manager.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.0
        )
        self.assertEqual(position_size, 0.0)
    
    def test_safety_limits_enabled(self):
        """Test safety limits when enabled."""
        rm = StrategyPositionSizeManager(
            max_position_size=0.1,
            min_position_size=0.01,
            apply_safety_limits=True
        )
        
        # Position size above maximum - should be capped
        position_size = rm.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.2  # Above max of 0.1
        )
        self.assertEqual(position_size, 0.1)
        
        # Position size below minimum - should be rejected (return 0)
        position_size = rm.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.005  # Below min of 0.01
        )
        self.assertEqual(position_size, 0.0)
        
        # Position size within limits - should be unchanged
        position_size = rm.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.05  # Within limits
        )
        self.assertEqual(position_size, 0.05)
    
    def test_safety_limits_disabled(self):
        """Test behavior when safety limits are disabled."""
        rm = StrategyPositionSizeManager(
            max_position_size=0.1,
            min_position_size=0.01,
            apply_safety_limits=False
        )
        
        # Position size above maximum - should not be capped
        position_size = rm.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.2
        )
        self.assertEqual(position_size, 0.2)
        
        # Position size below minimum - should not be rejected
        position_size = rm.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.005
        )
        self.assertEqual(position_size, 0.005)
    
    def test_get_description(self):
        """Test description generation."""
        # With safety limits
        rm = StrategyPositionSizeManager(apply_safety_limits=True)
        description = rm.get_description()
        self.assertIn("Strategy Position Size Manager", description)
        self.assertIn("min=", description)
        self.assertIn("max=", description)
        
        # Without safety limits
        rm = StrategyPositionSizeManager(apply_safety_limits=False)
        description = rm.get_description()
        self.assertIn("Strategy Position Size Manager", description)
        self.assertIn("no limits", description)
    
    def test_kwargs_handling(self):
        """Test that additional kwargs are handled gracefully."""
        position_size = self.risk_manager.calculate_position_size(
            signal=1,
            current_price=100.0,
            equity=10000.0,
            strategy_position_size=0.05,
            stop_loss=95.0,  # Additional kwargs should be ignored
            take_profit=110.0,
            some_other_param="ignored"
        )
        self.assertEqual(position_size, 0.05)


if __name__ == '__main__':
    unittest.main()