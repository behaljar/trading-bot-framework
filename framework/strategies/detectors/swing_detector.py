"""
Swing Highs/Lows Detector

This module implements swing point detection for identifying local highs and lows
in price series data. A swing high is a local maximum that is higher than a specified
number of bars to its left and right. A swing low is a local minimum that is lower
than a specified number of bars to its left and right.
"""

import numpy as np
from typing import Optional, Union, List
import pandas as pd


class SwingDetector:
    """
    Detector for swing highs and lows in price series data.
    
    This class provides methods to detect swing points (local extrema) in time series data.
    Swing points are useful for identifying potential support and resistance levels,
    trend changes, and key price levels.
    """
    
    def __init__(self, sensitivity: int = 3):
        """Initialize the swing detector.
        
        Args:
            sensitivity: Number of candles around swing point to check (default: 3)
                        For swing high: sensitivity candles before must not be higher
                        For swing low: sensitivity candles before must not be lower
        """
        self.sensitivity = sensitivity
    
    def swing_high(self, 
                   series: Union[List[float], np.ndarray, pd.Series], 
                   left_bars: Optional[int] = None, 
                   right_bars: Optional[int] = None) -> Optional[float]:
        """
        Detect swing high in a price series using sensitivity-based detection.
        
        A swing high is identified when the current bar's value is higher than
        'sensitivity' bars to the left and 'sensitivity' bars to the right.
        
        Args:
            series: Price series data (typically high prices)
            left_bars: Number of bars to the left (uses sensitivity if None)
            right_bars: Number of bars to the right (uses sensitivity if None)
            
        Returns:
            The swing high value if found, None otherwise
        """
        # Use sensitivity as default if not specified
        left_bars = left_bars or self.sensitivity
        right_bars = right_bars or self.sensitivity
        
        if not self._validate_inputs(series, left_bars, right_bars):
            return None
            
        # Convert to numpy array for easier manipulation
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        swing_range = left_bars + right_bars
        
        # Need enough data points
        if len(data) < swing_range + 1:
            return None
            
        # Check if we have enough data at the left edge
        if len(data) <= swing_range or np.isnan(data[-(swing_range + 1)]):
            return None
            
        # The potential swing point is at position 'right_bars' from the end
        possible_swing_high = data[-(right_bars + 1)]
        
        # Get the range of values to check
        array_of_values = []
        for bar_index in range(swing_range + 1):
            array_of_values.append(data[-(swing_range + 1) + bar_index])
            
        # Find the maximum value and its position
        max_value = max(array_of_values)
        max_index = array_of_values.index(max_value)
        
        # Calculate how many bars from the right the maximum is
        swing_high_right_bars = len(array_of_values) - max_index - 1
        
        # Return the swing high if it's at the correct position
        return possible_swing_high if swing_high_right_bars == right_bars else None
    
    def swing_low(self, 
                  series: Union[List[float], np.ndarray, pd.Series], 
                  left_bars: Optional[int] = None, 
                  right_bars: Optional[int] = None) -> Optional[float]:
        """
        Detect swing low in a price series using sensitivity-based detection.
        
        A swing low is identified when the current bar's value is lower than
        'sensitivity' bars to the left and 'sensitivity' bars to the right.
        
        Args:
            series: Price series data (typically low prices)
            left_bars: Number of bars to the left (uses sensitivity if None)
            right_bars: Number of bars to the right (uses sensitivity if None)
            
        Returns:
            The swing low value if found, None otherwise
        """
        # Use sensitivity as default if not specified
        left_bars = left_bars or self.sensitivity
        right_bars = right_bars or self.sensitivity
        
        if not self._validate_inputs(series, left_bars, right_bars):
            return None
            
        # Convert to numpy array for easier manipulation
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        swing_range = left_bars + right_bars
        
        # Need enough data points
        if len(data) < swing_range + 1:
            return None
            
        # Check if we have enough data at the left edge
        if len(data) <= swing_range or np.isnan(data[-(swing_range + 1)]):
            return None
            
        # The potential swing point is at position 'right_bars' from the end
        possible_swing_low = data[-(right_bars + 1)]
        
        # Get the range of values to check
        array_of_values = []
        for bar_index in range(swing_range + 1):
            array_of_values.append(data[-(swing_range + 1) + bar_index])
            
        # Find the minimum value and its position
        min_value = min(array_of_values)
        min_index = array_of_values.index(min_value)
        
        # Calculate how many bars from the right the minimum is
        swing_low_right_bars = len(array_of_values) - min_index - 1
        
        # Return the swing low if it's at the correct position
        return possible_swing_low if swing_low_right_bars == right_bars else None
    
    def find_all_swing_highs(self, 
                            series: Union[List[float], np.ndarray, pd.Series], 
                            left_bars: Optional[int] = None, 
                            right_bars: Optional[int] = None) -> List[tuple]:
        """
        Find all swing highs in a price series.
        
        Args:
            series: Price series data
            left_bars: Number of bars to the left (uses sensitivity if None)
            right_bars: Number of bars to the right (uses sensitivity if None)
            
        Returns:
            List of tuples (index, value) for each swing high found
        """
        # Use sensitivity as default if not specified
        left_bars = left_bars or self.sensitivity
        right_bars = right_bars or self.sensitivity
        
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        swing_highs = []
        swing_range = left_bars + right_bars
        
        # Start from the first possible swing position
        # We need left_bars + right_bars total bars, plus the swing bar itself
        for i in range(left_bars, len(data) - right_bars):
            # Create a subset for this potential swing
            subset = data[i - left_bars:i + right_bars + 1]
            swing_high = self.swing_high(subset, left_bars, right_bars)
            
            if swing_high is not None:
                swing_highs.append((i, swing_high))
                
        return swing_highs
    
    def find_all_swing_lows(self, 
                           series: Union[List[float], np.ndarray, pd.Series], 
                           left_bars: Optional[int] = None, 
                           right_bars: Optional[int] = None) -> List[tuple]:
        """
        Find all swing lows in a price series.
        
        Args:
            series: Price series data
            left_bars: Number of bars to the left (uses sensitivity if None)
            right_bars: Number of bars to the right (uses sensitivity if None)
            
        Returns:
            List of tuples (index, value) for each swing low found
        """
        # Use sensitivity as default if not specified
        left_bars = left_bars or self.sensitivity
        right_bars = right_bars or self.sensitivity
        
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        swing_lows = []
        swing_range = left_bars + right_bars
        
        # Start from the first possible swing position
        # We need left_bars + right_bars total bars, plus the swing bar itself
        for i in range(left_bars, len(data) - right_bars):
            # Create a subset for this potential swing
            subset = data[i - left_bars:i + right_bars + 1]
            swing_low = self.swing_low(subset, left_bars, right_bars)
            
            if swing_low is not None:
                swing_lows.append((i, swing_low))
                
        return swing_lows
    
    
    def _validate_inputs(self, series, left_bars: int, right_bars: int) -> bool:
        """
        Validate input parameters.
        
        Args:
            series: Price series data
            left_bars: Number of left bars
            right_bars: Number of right bars
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if series is None or len(series) == 0:
            return False
            
        if left_bars <= 0 or right_bars <= 0:
            return False
            
        return True