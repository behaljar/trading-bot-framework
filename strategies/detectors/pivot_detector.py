"""
Pivot Highs/Lows Detector

This module implements pivot point detection for identifying local highs and lows
in price series data. A pivot high is a local maximum that is higher than a specified
number of bars to its left and right. A pivot low is a local minimum that is lower
than a specified number of bars to its left and right.
"""

import numpy as np
from typing import Optional, Union, List
import pandas as pd


class PivotDetector:
    """
    Detector for pivot highs and lows in price series data.
    
    This class provides methods to detect pivot points (local extrema) in time series data.
    Pivot points are useful for identifying potential support and resistance levels,
    trend changes, and key price levels.
    """
    
    def __init__(self):
        """Initialize the pivot detector."""
        pass
    
    def pivot_high(self, 
                   series: Union[List[float], np.ndarray, pd.Series], 
                   left_bars: int, 
                   right_bars: int) -> Optional[float]:
        """
        Detect pivot high in a price series.
        
        A pivot high is identified when the current bar's value is higher than
        'left_bars' bars to the left and 'right_bars' bars to the right.
        
        Args:
            series: Price series data (typically high prices)
            left_bars: Number of bars to the left that must be lower
            right_bars: Number of bars to the right that must be lower
            
        Returns:
            The pivot high value if found, None otherwise
        """
        if not self._validate_inputs(series, left_bars, right_bars):
            return None
            
        # Convert to numpy array for easier manipulation
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        pivot_range = left_bars + right_bars
        
        # Need enough data points
        if len(data) < pivot_range + 1:
            return None
            
        # Check if we have enough data at the left edge
        if len(data) <= pivot_range or np.isnan(data[-(pivot_range + 1)]):
            return None
            
        # The potential pivot point is at position 'right_bars' from the end
        possible_pivot_high = data[-(right_bars + 1)]
        
        # Get the range of values to check
        array_of_values = []
        for bar_index in range(pivot_range + 1):
            array_of_values.append(data[-(pivot_range + 1) + bar_index])
            
        # Find the maximum value and its position
        max_value = max(array_of_values)
        max_index = array_of_values.index(max_value)
        
        # Calculate how many bars from the right the maximum is
        pivot_high_right_bars = len(array_of_values) - max_index - 1
        
        # Return the pivot high if it's at the correct position
        return possible_pivot_high if pivot_high_right_bars == right_bars else None
    
    def pivot_low(self, 
                  series: Union[List[float], np.ndarray, pd.Series], 
                  left_bars: int, 
                  right_bars: int) -> Optional[float]:
        """
        Detect pivot low in a price series.
        
        A pivot low is identified when the current bar's value is lower than
        'left_bars' bars to the left and 'right_bars' bars to the right.
        
        Args:
            series: Price series data (typically low prices)
            left_bars: Number of bars to the left that must be higher
            right_bars: Number of bars to the right that must be higher
            
        Returns:
            The pivot low value if found, None otherwise
        """
        if not self._validate_inputs(series, left_bars, right_bars):
            return None
            
        # Convert to numpy array for easier manipulation
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        pivot_range = left_bars + right_bars
        
        # Need enough data points
        if len(data) < pivot_range + 1:
            return None
            
        # Check if we have enough data at the left edge
        if len(data) <= pivot_range or np.isnan(data[-(pivot_range + 1)]):
            return None
            
        # The potential pivot point is at position 'right_bars' from the end
        possible_pivot_low = data[-(right_bars + 1)]
        
        # Get the range of values to check
        array_of_values = []
        for bar_index in range(pivot_range + 1):
            array_of_values.append(data[-(pivot_range + 1) + bar_index])
            
        # Find the minimum value and its position
        min_value = min(array_of_values)
        min_index = array_of_values.index(min_value)
        
        # Calculate how many bars from the right the minimum is
        pivot_low_right_bars = len(array_of_values) - min_index - 1
        
        # Return the pivot low if it's at the correct position
        return possible_pivot_low if pivot_low_right_bars == right_bars else None
    
    def find_all_pivot_highs(self, 
                            series: Union[List[float], np.ndarray, pd.Series], 
                            left_bars: int, 
                            right_bars: int) -> List[tuple]:
        """
        Find all pivot highs in a price series.
        
        Args:
            series: Price series data
            left_bars: Number of bars to the left that must be lower
            right_bars: Number of bars to the right that must be lower
            
        Returns:
            List of tuples (index, value) for each pivot high found
        """
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        pivot_highs = []
        pivot_range = left_bars + right_bars
        
        # Start from the first possible pivot position
        # We need left_bars + right_bars total bars, plus the pivot bar itself
        for i in range(left_bars, len(data) - right_bars):
            # Create a subset for this potential pivot
            subset = data[i - left_bars:i + right_bars + 1]
            pivot_high = self.pivot_high(subset, left_bars, right_bars)
            
            if pivot_high is not None:
                pivot_highs.append((i, pivot_high))
                
        return pivot_highs
    
    def find_all_pivot_lows(self, 
                           series: Union[List[float], np.ndarray, pd.Series], 
                           left_bars: int, 
                           right_bars: int) -> List[tuple]:
        """
        Find all pivot lows in a price series.
        
        Args:
            series: Price series data
            left_bars: Number of bars to the left that must be higher
            right_bars: Number of bars to the right that must be higher
            
        Returns:
            List of tuples (index, value) for each pivot low found
        """
        if isinstance(series, pd.Series):
            data = series.values
        else:
            data = np.array(series)
            
        pivot_lows = []
        pivot_range = left_bars + right_bars
        
        # Start from the first possible pivot position
        # We need left_bars + right_bars total bars, plus the pivot bar itself
        for i in range(left_bars, len(data) - right_bars):
            # Create a subset for this potential pivot
            subset = data[i - left_bars:i + right_bars + 1]
            pivot_low = self.pivot_low(subset, left_bars, right_bars)
            
            if pivot_low is not None:
                pivot_lows.append((i, pivot_low))
                
        return pivot_lows
    
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