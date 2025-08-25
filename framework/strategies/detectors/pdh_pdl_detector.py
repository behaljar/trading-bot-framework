"""
Previous Day/Week High/Low Detector

This module implements detection of previous day/week high/low levels with support
for tracking "unclaimed" levels (levels that haven't been tested or broken by 
subsequent price action).

The detector identifies:
- Previous Day High (PDH) / Previous Day Low (PDL)
- Previous Week High (PWH) / Previous Week Low (PWL)
- Tracks whether levels have been "claimed" (tested/broken) or remain "unclaimed"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class HighLowLevel:
    """
    Represents a high/low level from previous day/week.
    
    Attributes:
        timestamp: When this level was established
        level_type: 'PDH', 'PDL', 'PWH', 'PWL'
        price: The actual price level
        period_start: Start of the period (day/week)
        period_end: End of the period (day/week)
        validity_start: When this level becomes valid (after period ends)
        validity_end: When this level expires (1 day for daily, 1 week for weekly)
        is_claimed: Whether this level has been tested/broken
        claim_timestamp: When the level was first claimed (if applicable)
        claim_price: Price at which level was claimed
        touch_count: Number of times price has approached this level
        is_valid: Whether this level is currently valid based on time
    """
    timestamp: pd.Timestamp
    level_type: str
    price: float
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    validity_start: pd.Timestamp
    validity_end: pd.Timestamp
    is_claimed: bool = False
    claim_timestamp: Optional[pd.Timestamp] = None
    claim_price: Optional[float] = None
    touch_count: int = 0
    is_valid: bool = True


class PDHPDLDetector:
    """
    Detector for Previous Day/Week High/Low levels with unclaimed tracking.
    
    This detector identifies significant high/low levels from previous periods
    and tracks whether they remain "unclaimed" (untested by price action).
    """
    
    def __init__(self, 
                 touch_threshold: float = 0.001,  # 0.1% threshold for level touches
                 break_threshold: float = 0.0005): # 0.05% threshold for level breaks
        """
        Initialize the PDH/PDL detector.
        
        Args:
            touch_threshold: Percentage threshold for considering a level "touched"
            break_threshold: Percentage threshold for considering a level "broken"
        """
        self.touch_threshold = touch_threshold
        self.break_threshold = break_threshold
    
    def detect_daily_levels(self, 
                          data: pd.DataFrame, 
                          current_date: Optional[pd.Timestamp] = None,
                          unclaimed_only: bool = False) -> List[HighLowLevel]:
        """
        Detect previous day high/low levels.
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            current_date: Current date for analysis (uses last date if None)
            unclaimed_only: If True, only return unclaimed levels
            
        Returns:
            List of HighLowLevel objects for daily levels
        """
        if not self._validate_data(data):
            return []
        
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        if current_date is None:
            current_date = data.index[-1]
        
        # Group data by day
        daily_groups = data.groupby(data.index.date)
        levels = []
        
        # Get unique dates and sort them
        dates = sorted(daily_groups.groups.keys())
        
        # Process each day (except the current day)
        current_day = current_date.date()
        for date in dates:
            if date >= current_day:
                continue  # Skip current and future days
                
            day_data = daily_groups.get_group(date)
            
            # Calculate PDH and PDL for this day
            pdh_price = day_data['high'].max()
            pdl_price = day_data['low'].min()
            
            # Find timestamps where these levels occurred
            pdh_timestamp = day_data[day_data['high'] == pdh_price].index[0]
            pdl_timestamp = day_data[day_data['low'] == pdl_price].index[0]
            
            period_start = day_data.index[0]
            period_end = day_data.index[-1]
            
            # Daily levels are valid for 1 day after the period ends
            validity_start = period_end
            validity_end = period_end + timedelta(days=1)
            
            # Create PDH level
            pdh_level = HighLowLevel(
                timestamp=pdh_timestamp,
                level_type='PDH',
                price=pdh_price,
                period_start=period_start,
                period_end=period_end,
                validity_start=validity_start,
                validity_end=validity_end
            )
            
            # Create PDL level
            pdl_level = HighLowLevel(
                timestamp=pdl_timestamp,
                level_type='PDL',
                price=pdl_price,
                period_start=period_start,
                period_end=period_end,
                validity_start=validity_start,
                validity_end=validity_end
            )
            
            # Check validity and claim status
            self._update_level_validity(pdh_level, current_date)
            self._update_level_validity(pdl_level, current_date)
            
            # Check if levels have been claimed by subsequent price action
            future_data = data[data.index > period_end]
            if not future_data.empty:
                self._update_level_status(pdh_level, future_data)
                self._update_level_status(pdl_level, future_data)
            
            levels.extend([pdh_level, pdl_level])
        
        # Filter for unclaimed only if requested
        if unclaimed_only:
            levels = [level for level in levels if not level.is_claimed]
        
        return levels
    
    def detect_weekly_levels(self, 
                           data: pd.DataFrame,
                           current_date: Optional[pd.Timestamp] = None,
                           unclaimed_only: bool = False) -> List[HighLowLevel]:
        """
        Detect previous week high/low levels.
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            current_date: Current date for analysis (uses last date if None)
            unclaimed_only: If True, only return unclaimed levels
            
        Returns:
            List of HighLowLevel objects for weekly levels
        """
        if not self._validate_data(data):
            return []
        
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        if current_date is None:
            current_date = data.index[-1]
        
        # Group data by week (Monday to Sunday)
        weekly_groups = data.groupby(pd.Grouper(freq='W-MON'))
        levels = []
        
        # Process each complete week (except the current week)
        current_week = pd.Timestamp(current_date.date() - timedelta(days=current_date.weekday()))
        
        for week_start, week_data in weekly_groups:
            if week_data.empty or week_start >= current_week:
                continue  # Skip current and future weeks
            
            # Calculate PWH and PWL for this week
            pwh_price = week_data['high'].max()
            pwl_price = week_data['low'].min()
            
            # Find timestamps where these levels occurred
            pwh_timestamp = week_data[week_data['high'] == pwh_price].index[0]
            pwl_timestamp = week_data[week_data['low'] == pwl_price].index[0]
            
            period_start = week_data.index[0]
            period_end = week_data.index[-1]
            
            # Weekly levels are valid for 1 week after the period ends
            validity_start = period_end
            validity_end = period_end + timedelta(weeks=1)
            
            # Create PWH level
            pwh_level = HighLowLevel(
                timestamp=pwh_timestamp,
                level_type='PWH',
                price=pwh_price,
                period_start=period_start,
                period_end=period_end,
                validity_start=validity_start,
                validity_end=validity_end
            )
            
            # Create PWL level
            pwl_level = HighLowLevel(
                timestamp=pwl_timestamp,
                level_type='PWL',
                price=pwl_price,
                period_start=period_start,
                period_end=period_end,
                validity_start=validity_start,
                validity_end=validity_end
            )
            
            # Check validity and claim status
            self._update_level_validity(pwh_level, current_date)
            self._update_level_validity(pwl_level, current_date)
            
            # Check if levels have been claimed by subsequent price action
            future_data = data[data.index > period_end]
            if not future_data.empty:
                self._update_level_status(pwh_level, future_data)
                self._update_level_status(pwl_level, future_data)
            
            levels.extend([pwh_level, pwl_level])
        
        # Filter for unclaimed only if requested
        if unclaimed_only:
            levels = [level for level in levels if not level.is_claimed]
        
        return levels
    
    def detect_all_levels(self,
                         data: pd.DataFrame,
                         current_date: Optional[pd.Timestamp] = None,
                         unclaimed_only: bool = False) -> List[HighLowLevel]:
        """
        Detect both daily and weekly levels.
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            current_date: Current date for analysis
            unclaimed_only: If True, only return unclaimed levels
            
        Returns:
            Combined list of daily and weekly levels
        """
        daily_levels = self.detect_daily_levels(data, current_date, unclaimed_only)
        weekly_levels = self.detect_weekly_levels(data, current_date, unclaimed_only)
        
        # Combine and sort by timestamp
        all_levels = daily_levels + weekly_levels
        all_levels.sort(key=lambda x: x.timestamp)
        
        return all_levels
    
    def _update_level_validity(self, level: HighLowLevel, current_date: pd.Timestamp):
        """
        Update the validity status of a level based on current date.
        
        Args:
            level: HighLowLevel object to update
            current_date: Current date to check validity against
        """
        level.is_valid = (level.validity_start <= current_date <= level.validity_end)
    
    def _update_level_status(self, level: HighLowLevel, future_data: pd.DataFrame):
        """
        Update the claim status of a level based on future price action.
        
        Args:
            level: HighLowLevel object to update
            future_data: DataFrame with price data after the level period
        """
        if level.level_type in ['PDH', 'PWH']:
            # For highs, check if price has broken above
            touches = future_data[
                (future_data['high'] >= level.price * (1 - self.touch_threshold)) &
                (future_data['high'] <= level.price * (1 + self.touch_threshold))
            ]
            breaks = future_data[future_data['high'] > level.price * (1 + self.break_threshold)]
            
        else:  # PDL, PWL
            # For lows, check if price has broken below
            touches = future_data[
                (future_data['low'] <= level.price * (1 + self.touch_threshold)) &
                (future_data['low'] >= level.price * (1 - self.touch_threshold))
            ]
            breaks = future_data[future_data['low'] < level.price * (1 - self.break_threshold)]
        
        # Update touch count
        level.touch_count = len(touches)
        
        # Check if level has been broken (claimed)
        if not breaks.empty:
            level.is_claimed = True
            level.claim_timestamp = breaks.index[0]
            if level.level_type in ['PDH', 'PWH']:
                level.claim_price = breaks.iloc[0]['high']
            else:
                level.claim_price = breaks.iloc[0]['low']
        
        # If not broken but touched multiple times, might still be considered claimed
        elif level.touch_count >= 3:  # Threshold for multiple touches
            level.is_claimed = True
            level.claim_timestamp = touches.index[0]
            if level.level_type in ['PDH', 'PWH']:
                level.claim_price = touches.iloc[0]['high']
            else:
                level.claim_price = touches.iloc[0]['low']
    
    def get_levels_near_price(self, 
                             levels: List[HighLowLevel],
                             current_price: float,
                             distance_pct: float = 0.02) -> List[HighLowLevel]:
        """
        Get levels that are within a certain percentage distance from current price.
        
        Args:
            levels: List of HighLowLevel objects
            current_price: Current market price
            distance_pct: Maximum distance as percentage (default: 2%)
            
        Returns:
            List of levels near current price
        """
        near_levels = []
        
        for level in levels:
            distance = abs(level.price - current_price) / current_price
            if distance <= distance_pct:
                near_levels.append(level)
        
        # Sort by distance from current price
        near_levels.sort(key=lambda x: abs(x.price - current_price))
        
        return near_levels
    
    def get_levels_summary(self, levels: List[HighLowLevel]) -> Dict:
        """
        Get summary statistics for detected levels.
        
        Args:
            levels: List of HighLowLevel objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not levels:
            return {
                'total_levels': 0,
                'unclaimed_levels': 0,
                'daily_levels': 0,
                'weekly_levels': 0,
                'pdh_count': 0,
                'pdl_count': 0,
                'pwh_count': 0,
                'pwl_count': 0
            }
        
        unclaimed_count = sum(1 for level in levels if not level.is_claimed)
        daily_count = sum(1 for level in levels if level.level_type in ['PDH', 'PDL'])
        weekly_count = sum(1 for level in levels if level.level_type in ['PWH', 'PWL'])
        
        type_counts = {}
        for level_type in ['PDH', 'PDL', 'PWH', 'PWL']:
            type_counts[f'{level_type.lower()}_count'] = sum(1 for level in levels if level.level_type == level_type)
        
        return {
            'total_levels': len(levels),
            'unclaimed_levels': unclaimed_count,
            'daily_levels': daily_count,
            'weekly_levels': weekly_count,
            **type_counts
        }
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or len(data) < 2:
            return False
        
        required_columns = ['high', 'low', 'open', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check if index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        
        # Check for NaN values in required columns
        if data[required_columns].isnull().any().any():
            return False
        
        return True
    
    def filter_by_age(self, 
                     levels: List[HighLowLevel],
                     max_age_days: int) -> List[HighLowLevel]:
        """
        Filter levels by maximum age.
        
        Args:
            levels: List of HighLowLevel objects
            max_age_days: Maximum age in days
            
        Returns:
            Filtered list of levels
        """
        cutoff_date = pd.Timestamp.now() - timedelta(days=max_age_days)
        return [level for level in levels if level.timestamp >= cutoff_date]
    
    def get_resistance_levels(self, levels: List[HighLowLevel]) -> List[HighLowLevel]:
        """Get only resistance levels (PDH, PWH)."""
        return [level for level in levels if level.level_type in ['PDH', 'PWH']]
    
    def get_support_levels(self, levels: List[HighLowLevel]) -> List[HighLowLevel]:
        """Get only support levels (PDL, PWL)."""
        return [level for level in levels if level.level_type in ['PDL', 'PWL']]
    
    def get_valid_levels(self, levels: List[HighLowLevel]) -> List[HighLowLevel]:
        """Get only currently valid levels (within their validity period)."""
        return [level for level in levels if level.is_valid]
    
    def get_valid_unclaimed_levels(self, levels: List[HighLowLevel]) -> List[HighLowLevel]:
        """Get levels that are both valid and unclaimed."""
        return [level for level in levels if level.is_valid and not level.is_claimed]