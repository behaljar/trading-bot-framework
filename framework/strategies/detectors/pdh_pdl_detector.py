"""
Previous Day/Week High/Low Detector

This module implements a simple detector for previous day and week high/low levels.
It returns only the latest PDH/PDL and PWH/PWL with their break status.
"""

import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class HighLowLevel:
    """
    Represents a high/low level from previous day/week/4-hour.
    
    Attributes:
        level_type: 'PDH', 'PDL', 'PWH', 'PWL', 'P4H', 'P4L'
        price: The actual price level
        date: The date when this level was established
        is_broken: Whether this level has been broken by subsequent price action
        break_time: When the level was broken (if applicable)
    """
    level_type: str
    price: float
    date: pd.Timestamp
    is_broken: bool = False
    break_time: Optional[pd.Timestamp] = None


class PDHPDLDetector:
    """
    Simple detector for Previous Day/Week/4-Hour High/Low levels.
    
    Returns only the latest previous day, week, and 4-hour high/low levels
    along with whether they have been broken.
    """
    
    def __init__(self):
        """Initialize the PDH/PDL detector."""
        pass
    
    def detect_levels(self, data: pd.DataFrame) -> Dict[str, Optional[HighLowLevel]]:
        """
        Detect the latest previous day, week, and 4-hour high/low levels.
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            
        Returns:
            Dictionary with keys 'PDH', 'PDL', 'PWH', 'PWL', 'P4H', 'P4L' containing HighLowLevel objects
        """
        if not self._validate_data(data):
            return {'PDH': None, 'PDL': None, 'PWH': None, 'PWL': None, 'P4H': None, 'P4L': None}
        
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Get the latest date in the data
        latest_date = data.index[-1]
        
        # Detect daily levels
        pdh, pdl = self._detect_daily_levels(data, latest_date)
        
        # Detect weekly levels
        pwh, pwl = self._detect_weekly_levels(data, latest_date)
        
        # Detect 4-hour levels
        p4h, p4l = self._detect_4hour_levels(data, latest_date)
        
        return {
            'PDH': pdh,
            'PDL': pdl,
            'PWH': pwh,
            'PWL': pwl,
            'P4H': p4h,
            'P4L': p4l
        }
    
    def _detect_daily_levels(self, data: pd.DataFrame, latest_date: pd.Timestamp) -> tuple[Optional[HighLowLevel], Optional[HighLowLevel]]:
        """
        Detect the previous day high and low.
        
        Args:
            data: DataFrame with OHLCV data
            latest_date: The latest date in the data
            
        Returns:
            Tuple of (PDH, PDL) HighLowLevel objects
        """
        # Resample to daily frequency
        daily_data = data.resample('D').agg({
            'High': 'max',
            'Low': 'min',
            'Open': 'first',
            'Close': 'last'
        }).dropna()
        
        if len(daily_data) < 2:
            return None, None
        
        # Get the last complete day (not today)
        current_day = latest_date.normalize()
        
        # Find the most recent complete day
        complete_days = daily_data[daily_data.index < current_day]
        
        if complete_days.empty:
            return None, None
        
        # Get the previous day's data
        prev_day = complete_days.iloc[-1]
        prev_day_date = complete_days.index[-1]
        
        # Create PDH level
        pdh = HighLowLevel(
            level_type='PDH',
            price=prev_day['High'],
            date=prev_day_date,
            is_broken=False
        )
        
        # Create PDL level
        pdl = HighLowLevel(
            level_type='PDL',
            price=prev_day['Low'],
            date=prev_day_date,
            is_broken=False
        )
        
        # Check if levels have been broken by subsequent price action
        # Future data is all data after the previous day ended
        future_data = data[data.index.normalize() > prev_day_date]
        
        if not future_data.empty:
            # Check PDH break (price went above)
            if (future_data['High'] > pdh.price).any():
                pdh.is_broken = True
                pdh.break_time = future_data[future_data['High'] > pdh.price].index[0]
            
            # Check PDL break (price went below)
            if (future_data['Low'] < pdl.price).any():
                pdl.is_broken = True
                pdl.break_time = future_data[future_data['Low'] < pdl.price].index[0]
        
        return pdh, pdl
    
    def _detect_weekly_levels(self, data: pd.DataFrame, latest_date: pd.Timestamp) -> tuple[Optional[HighLowLevel], Optional[HighLowLevel]]:
        """
        Detect the previous week high and low.
        
        Args:
            data: DataFrame with OHLCV data
            latest_date: The latest date in the data
            
        Returns:
            Tuple of (PWH, PWL) HighLowLevel objects
        """
        # Resample to weekly frequency (week ending on Sunday)
        weekly_data = data.resample('W').agg({
            'High': 'max',
            'Low': 'min',
            'Open': 'first',
            'Close': 'last'
        }).dropna()
        
        if len(weekly_data) < 2:
            return None, None
        
        # Get the current week start (Monday)
        current_week_start = latest_date.normalize() - pd.Timedelta(days=latest_date.weekday())
        
        # Find the most recent complete week
        complete_weeks = weekly_data[weekly_data.index < current_week_start]
        
        if complete_weeks.empty:
            return None, None
        
        # Get the previous week's data
        prev_week = complete_weeks.iloc[-1]
        prev_week_date = complete_weeks.index[-1]
        
        # Create PWH level
        pwh = HighLowLevel(
            level_type='PWH',
            price=prev_week['High'],
            date=prev_week_date,
            is_broken=False
        )
        
        # Create PWL level
        pwl = HighLowLevel(
            level_type='PWL',
            price=prev_week['Low'],
            date=prev_week_date,
            is_broken=False
        )
        
        # Check if levels have been broken by subsequent price action
        # For weekly levels, check data after the week ended
        week_end = prev_week_date
        future_data = data[data.index > week_end]
        
        if not future_data.empty:
            # Check PWH break (price went above)
            if (future_data['High'] > pwh.price).any():
                pwh.is_broken = True
                pwh.break_time = future_data[future_data['High'] > pwh.price].index[0]
            
            # Check PWL break (price went below)
            if (future_data['Low'] < pwl.price).any():
                pwl.is_broken = True
                pwl.break_time = future_data[future_data['Low'] < pwl.price].index[0]
        
        return pwh, pwl
    
    def _detect_4hour_levels(self, data: pd.DataFrame, latest_date: pd.Timestamp) -> tuple[Optional[HighLowLevel], Optional[HighLowLevel]]:
        """
        Detect the previous 4-hour high and low.
        
        Args:
            data: DataFrame with OHLCV data
            latest_date: The latest date in the data
            
        Returns:
            Tuple of (P4H, P4L) HighLowLevel objects
        """
        # Resample to 4-hour frequency
        four_hour_data = data.resample('4h').agg({
            'High': 'max',
            'Low': 'min',
            'Open': 'first',
            'Close': 'last'
        }).dropna()
        
        if len(four_hour_data) < 2:
            return None, None
        
        # Get the current 4-hour session start
        # 4H sessions start at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        latest_hour = latest_date.hour
        current_4h_start_hour = (latest_hour // 4) * 4
        current_4h_start = latest_date.replace(hour=current_4h_start_hour, minute=0, second=0, microsecond=0)
        
        # Find the most recent complete 4-hour session
        complete_4hours = four_hour_data[four_hour_data.index < current_4h_start]
        
        if complete_4hours.empty:
            return None, None
        
        # Get the previous 4-hour session's data
        prev_4hour = complete_4hours.iloc[-1]
        prev_4hour_date = complete_4hours.index[-1]
        
        # Create P4H level
        p4h = HighLowLevel(
            level_type='P4H',
            price=prev_4hour['High'],
            date=prev_4hour_date,
            is_broken=False
        )
        
        # Create P4L level
        p4l = HighLowLevel(
            level_type='P4L',
            price=prev_4hour['Low'],
            date=prev_4hour_date,
            is_broken=False
        )
        
        # Check if levels have been broken by subsequent price action
        # For 4-hour levels, check data after the 4-hour session ended
        future_data = data[data.index > prev_4hour_date]
        
        if not future_data.empty:
            # Check P4H break (price went above)
            if (future_data['High'] > p4h.price).any():
                p4h.is_broken = True
                p4h.break_time = future_data[future_data['High'] > p4h.price].index[0]
            
            # Check P4L break (price went below)
            if (future_data['Low'] < p4l.price).any():
                p4l.is_broken = True
                p4l.break_time = future_data[future_data['Low'] < p4l.price].index[0]
        
        return p4h, p4l
    
    def get_summary(self, levels: Dict[str, Optional[HighLowLevel]]) -> Dict:
        """
        Get a summary of the detected levels.
        
        Args:
            levels: Dictionary of detected levels
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            'pdh_price': levels['PDH'].price if levels['PDH'] else None,
            'pdh_broken': levels['PDH'].is_broken if levels['PDH'] else None,
            'pdl_price': levels['PDL'].price if levels['PDL'] else None,
            'pdl_broken': levels['PDL'].is_broken if levels['PDL'] else None,
            'pwh_price': levels['PWH'].price if levels['PWH'] else None,
            'pwh_broken': levels['PWH'].is_broken if levels['PWH'] else None,
            'pwl_price': levels['PWL'].price if levels['PWL'] else None,
            'pwl_broken': levels['PWL'].is_broken if levels['PWL'] else None,
            'p4h_price': levels['P4H'].price if levels['P4H'] else None,
            'p4h_broken': levels['P4H'].is_broken if levels['P4H'] else None,
            'p4l_price': levels['P4L'].price if levels['P4L'] else None,
            'p4l_broken': levels['P4L'].is_broken if levels['P4L'] else None,
        }
        
        # Add price ranges
        if levels['PDH'] and levels['PDL']:
            summary['daily_range'] = levels['PDH'].price - levels['PDL'].price
            summary['daily_range_pct'] = (summary['daily_range'] / levels['PDL'].price) * 100
        
        if levels['PWH'] and levels['PWL']:
            summary['weekly_range'] = levels['PWH'].price - levels['PWL'].price
            summary['weekly_range_pct'] = (summary['weekly_range'] / levels['PWL'].price) * 100
        
        if levels['P4H'] and levels['P4L']:
            summary['4hour_range'] = levels['P4H'].price - levels['P4L'].price
            summary['4hour_range_pct'] = (summary['4hour_range'] / levels['P4L'].price) * 100
        
        return summary
    
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
        
        required_columns = ['High', 'Low', 'Open', 'Close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check if index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        
        # Check for NaN values in required columns
        if data[required_columns].isnull().any().any():
            return False
        
        return True