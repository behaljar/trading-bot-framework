"""
Tests for risk manager implementations.
"""

import pytest
import numpy as np
from framework.risk.fixed_risk_manager import FixedRiskManager
from framework.risk.fixed_position_size_manager import FixedPositionSizeManager


class TestFixedRiskManager:
    """Test cases for FixedRiskManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.risk_manager = FixedRiskManager(risk_percent=0.01, default_stop_distance=0.02)

    def test_initialization_valid_params(self):
        """Test initialization with valid parameters."""
        manager = FixedRiskManager(risk_percent=0.02, default_stop_distance=0.03)
        assert manager.risk_percent == 0.02
        assert manager.default_stop_distance == 0.03

    def test_initialization_invalid_risk_percent(self):
        """Test initialization with invalid risk percentage."""
        with pytest.raises(ValueError, match="Risk percent must be between 0 and 0.1"):
            FixedRiskManager(risk_percent=0.15)
        
        with pytest.raises(ValueError, match="Risk percent must be between 0 and 0.1"):
            FixedRiskManager(risk_percent=-0.01)

    def test_initialization_invalid_stop_distance(self):
        """Test initialization with invalid stop distance."""
        with pytest.raises(ValueError, match="Default stop distance must be between 0 and 0.5"):
            FixedRiskManager(default_stop_distance=0.6)
        
        with pytest.raises(ValueError, match="Default stop distance must be between 0 and 0.5"):
            FixedRiskManager(default_stop_distance=-0.01)

    def test_calculate_position_size_no_signal(self):
        """Test position size calculation with no signal."""
        size = self.risk_manager.calculate_position_size(
            signal=0, current_price=100.0, equity=10000.0
        )
        assert size == 0.0

    def test_calculate_position_size_long_with_stop_loss(self):
        """Test position size calculation for long position with stop loss."""
        # 1% risk on $10,000 = $100 risk
        # Stop loss at $95 (risk $5 per share) -> 100/5 = 20 shares
        # Since units >= 1.0, returns absolute unit count
        size = self.risk_manager.calculate_position_size(
            signal=1, current_price=100.0, equity=10000.0, stop_loss=95.0
        )
        expected_units = 20  # Absolute units
        assert size == expected_units

    def test_calculate_position_size_short_with_stop_loss(self):
        """Test position size calculation for short position with stop loss."""
        # 1% risk on $10,000 = $100 risk
        # Stop loss at $105 (risk $5 per share) -> 100/5 = 20 shares
        # Since units >= 1.0, returns absolute unit count
        size = self.risk_manager.calculate_position_size(
            signal=-1, current_price=100.0, equity=10000.0, stop_loss=105.0
        )
        expected_units = 20  # Absolute units
        assert size == expected_units

    def test_calculate_position_size_large_units(self):
        """Test position size calculation when units >= 1.0."""
        # 1% risk on $100,000 = $1000 risk
        # Stop loss at $95 (risk $5 per share) -> 1000/5 = 200 shares (>= 1.0)
        # Should return absolute unit count
        size = self.risk_manager.calculate_position_size(
            signal=1, current_price=100.0, equity=100000.0, stop_loss=95.0
        )
        assert size == 200  # Absolute units

    def test_calculate_position_size_invalid_stop_loss(self):
        """Test position size calculation with invalid stop loss."""
        # Stop loss above current price for long position
        size = self.risk_manager.calculate_position_size(
            signal=1, current_price=100.0, equity=10000.0, stop_loss=105.0
        )
        assert size == 0.0

        # Stop loss below current price for short position
        size = self.risk_manager.calculate_position_size(
            signal=-1, current_price=100.0, equity=10000.0, stop_loss=95.0
        )
        assert size == 0.0

    def test_calculate_position_size_small_fractional_case(self):
        """Test position size calculation that returns fraction (< 1 unit)."""
        # Use parameters that result in < 1.0 units
        # 1% risk on $10,000 = $100 risk
        # Stop loss at $1 (risk $99 per share) -> 100/99 = 1.01 shares (>= 1.0, so returns units)
        # Let's use smaller risk to get < 1 unit
        small_risk_manager = FixedRiskManager(risk_percent=0.001, default_stop_distance=0.02)  # 0.1% risk
        # 0.1% risk on $10,000 = $10 risk
        # Stop loss at $95 (risk $5 per share) -> 10/5 = 2 shares (>= 1.0)
        # Let's use even smaller risk
        tiny_risk_manager = FixedRiskManager(risk_percent=0.0001, default_stop_distance=0.02)  # 0.01% risk
        # 0.01% risk on $10,000 = $1 risk  
        # Stop loss at $95 (risk $5 per share) -> 1/5 = 0.2 shares (< 1.0)
        # Position value = 0.2 * $100 = $20
        # Position fraction = $20 / $10000 = 0.002 = 0.2%
        size = tiny_risk_manager.calculate_position_size(
            signal=1, current_price=100.0, equity=10000.0, stop_loss=95.0
        )
        expected_fraction = 0.002  # 0.2% of equity
        assert abs(size - expected_fraction) < 0.0001

    def test_calculate_position_size_no_stop_loss(self):
        """Test position size calculation without stop loss."""
        # Should use default stop distance: risk_percent / default_stop_distance
        # 0.01 / 0.02 = 0.5 = 50%
        size = self.risk_manager.calculate_position_size(
            signal=1, current_price=100.0, equity=10000.0
        )
        expected_fraction = 0.01 / 0.02  # 0.5 = 50%
        assert abs(size - expected_fraction) < 0.001

    def test_get_description(self):
        """Test description string."""
        desc = self.risk_manager.get_description()
        assert "Fixed Risk Manager" in desc
        assert "risk=1.0%" in desc
        assert "default_stop=2.0%" in desc


class TestFixedPositionSizeManager:
    """Test cases for FixedPositionSizeManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = FixedPositionSizeManager(position_size=0.1)

    def test_initialization_valid_params(self):
        """Test initialization with valid parameters."""
        manager = FixedPositionSizeManager(position_size=0.05)
        assert manager.position_size == 0.05

    def test_initialization_invalid_position_size(self):
        """Test initialization with invalid position size."""
        with pytest.raises(ValueError, match="Position size must be between 0 and 1"):
            FixedPositionSizeManager(position_size=1.5)
        
        with pytest.raises(ValueError, match="Position size must be between 0 and 1"):
            FixedPositionSizeManager(position_size=-0.1)

    def test_calculate_position_size_no_signal(self):
        """Test position size calculation with no signal."""
        size = self.manager.calculate_position_size(
            signal=0, current_price=100.0, equity=10000.0
        )
        assert size == 0.0

    def test_calculate_position_size_long_signal(self):
        """Test position size calculation for long signal."""
        size = self.manager.calculate_position_size(
            signal=1, current_price=100.0, equity=10000.0, stop_loss=95.0
        )
        assert size == 0.1

    def test_calculate_position_size_short_signal(self):
        """Test position size calculation for short signal."""
        size = self.manager.calculate_position_size(
            signal=-1, current_price=100.0, equity=10000.0, stop_loss=105.0
        )
        assert size == 0.1

    def test_calculate_position_size_ignores_parameters(self):
        """Test that fixed manager ignores stop loss and take profit."""
        size1 = self.manager.calculate_position_size(
            signal=1, current_price=100.0, equity=10000.0
        )
        size2 = self.manager.calculate_position_size(
            signal=1, current_price=200.0, equity=50000.0, 
            stop_loss=180.0, take_profit=250.0
        )
        assert size1 == size2 == 0.1

    def test_get_description(self):
        """Test description string."""
        desc = self.manager.get_description()
        assert "Fixed Position Size Manager" in desc
        assert "size=10.0%" in desc