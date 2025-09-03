"""
ICT Liquidity Detector

This module implements detection of ICT (Inner Circle Trader) liquidity concepts
including various types of liquidity pools that are commonly targeted in price action.

Types of liquidity detected:
1. Previous day high/low (PDH/PDL) liquidity
2. Previous session high/low liquidity
3. Established high/low on timeframes
4. Previous week high/low (PWH/PWL) liquidity
5. Week opening gaps (liquidity zones)
6. Relative equal highs/lows
7. Session high/low levels
8. Asian session high/low levels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import pytz


@dataclass
class LiquidityLevel:
    """
    Represents an ICT liquidity level.
    
    Attributes:
        timestamp: When this level was established
        level_type: Type of liquidity ('PDH', 'PDL', 'PWH', 'PWL', 'REH', 'REL', etc.)
        price: The actual price level
        strength: Liquidity strength (1-5, with 5 being strongest)
        session: Trading session when established ('london', 'ny_open', 'ny_pm', 'asian')
        is_swept: Whether this liquidity has been swept
        sweep_timestamp: When the liquidity was swept
        touches: Number of times price has tested this level
        volume_profile: Volume associated with this level (if available)
    """
    timestamp: pd.Timestamp
    level_type: str
    price: float
    strength: int = 3
    session: Optional[str] = None
    is_swept: bool = False
    sweep_timestamp: Optional[pd.Timestamp] = None
    touches: int = 0
    volume_profile: float = 0.0


@dataclass
class WeekOpeningGap:
    """
    Represents a week opening gap (Sunday open vs Friday close).
    
    Attributes:
        week_start: Start of the week
        friday_close: Friday's closing price
        sunday_open: Sunday's opening price
        gap_size: Size of the gap (absolute value)
        gap_direction: 'up' for gap up, 'down' for gap down
        is_filled: Whether the gap has been filled
        fill_timestamp: When the gap was filled
    """
    week_start: pd.Timestamp
    friday_close: float
    sunday_open: float
    gap_size: float
    gap_direction: str
    is_filled: bool = False
    fill_timestamp: Optional[pd.Timestamp] = None


class ICTLiquidityDetector:
    """
    Detector for ICT liquidity concepts and patterns.
    """
    
    def __init__(self, 
                 equal_level_threshold: float = 0.0005,  # 0.05% for relative equal levels
                 liquidity_sweep_threshold: float = 0.0001,  # 0.01% for liquidity sweeps
                 strength_volume_multiplier: float = 1.5):
        """
        Initialize the ICT liquidity detector.
        
        Args:
            equal_level_threshold: Threshold for considering levels "equal"
            liquidity_sweep_threshold: Threshold for considering liquidity "swept"
            strength_volume_multiplier: Volume multiplier for strength calculation
        """
        self.equal_level_threshold = equal_level_threshold
        self.liquidity_sweep_threshold = liquidity_sweep_threshold
        self.strength_volume_multiplier = strength_volume_multiplier
        self.ny_tz = pytz.timezone('America/New_York')
        self.london_tz = pytz.timezone('Europe/London')
    
    def detect_all_liquidity(self, data: pd.DataFrame, 
                           current_time: Optional[pd.Timestamp] = None) -> List[LiquidityLevel]:
        """
        Detect all types of ICT liquidity.
        
        Args:
            data: DataFrame with OHLCV data
            current_time: Current timestamp for analysis
            
        Returns:
            List of all detected liquidity levels
        """
        if not self._validate_data(data):
            return []
        
        if current_time is None:
            current_time = data.index[-1]
        
        all_liquidity = []
        
        # Detect various types of liquidity
        all_liquidity.extend(self.detect_previous_day_liquidity(data, current_time))
        all_liquidity.extend(self.detect_previous_week_liquidity(data, current_time))
        all_liquidity.extend(self.detect_session_liquidity(data, current_time))
        all_liquidity.extend(self.detect_relative_equal_levels(data, current_time))
        all_liquidity.extend(self.detect_timeframe_highs_lows(data, current_time))
        
        # Update sweep status for all levels
        self._update_liquidity_sweeps(all_liquidity, data, current_time)
        
        # Sort by strength and timestamp
        all_liquidity.sort(key=lambda x: (x.strength, x.timestamp), reverse=True)
        
        return all_liquidity
    
    def detect_previous_day_liquidity(self, data: pd.DataFrame, 
                                    current_time: pd.Timestamp) -> List[LiquidityLevel]:
        """Detect previous day high/low liquidity pools."""
        liquidity = []
        
        # Group data by day
        daily_groups = data.groupby(data.index.date)
        current_date = current_time.date()
        
        # Get previous trading days
        dates = sorted([d for d in daily_groups.groups.keys() if d < current_date])
        
        # Look at last 5 trading days for PDH/PDL liquidity
        for date in dates[-5:]:
            day_data = daily_groups.get_group(date)
            
            pdh_price = day_data['high'].max()
            pdl_price = day_data['low'].min()
            
            pdh_timestamp = day_data[day_data['high'] == pdh_price].index[0]
            pdl_timestamp = day_data[day_data['low'] == pdl_price].index[0]
            
            # Calculate strength based on volume and range
            day_range = pdh_price - pdl_price
            avg_volume = day_data['volume'].mean() if 'volume' in day_data.columns else 1.0
            strength = min(5, max(1, int(3 + (avg_volume / day_data['volume'].median() - 1))))
            
            liquidity.append(LiquidityLevel(
                timestamp=pdh_timestamp,
                level_type='PDH',
                price=pdh_price,
                strength=strength,
                session=self._get_session_from_time(pdh_timestamp)
            ))
            
            liquidity.append(LiquidityLevel(
                timestamp=pdl_timestamp,
                level_type='PDL',
                price=pdl_price,
                strength=strength,
                session=self._get_session_from_time(pdl_timestamp)
            ))
        
        return liquidity
    
    def detect_previous_week_liquidity(self, data: pd.DataFrame,
                                     current_time: pd.Timestamp) -> List[LiquidityLevel]:
        """Detect previous week high/low liquidity pools."""
        liquidity = []
        
        # Group data by week (Monday to Sunday)
        weekly_groups = data.groupby(pd.Grouper(freq='W-MON'))
        
        # Ensure current_time has timezone info
        if current_time.tz is None:
            current_time = current_time.tz_localize('UTC')
        
        current_week = pd.Timestamp(current_time.date() - timedelta(days=current_time.weekday())).tz_localize(current_time.tz)
        
        # Look at last 3 weeks for PWH/PWL liquidity
        valid_weeks = [(week_start, week_data) for week_start, week_data in weekly_groups 
                      if not week_data.empty and week_start < current_week]
        
        for week_start, week_data in valid_weeks[-3:]:
            pwh_price = week_data['high'].max()
            pwl_price = week_data['low'].min()
            
            pwh_timestamp = week_data[week_data['high'] == pwh_price].index[0]
            pwl_timestamp = week_data[week_data['low'] == pwl_price].index[0]
            
            # Weekly levels have higher strength
            avg_volume = week_data['volume'].mean() if 'volume' in week_data.columns else 1.0
            strength = min(5, max(2, int(4 + (avg_volume / week_data['volume'].median() - 1))))
            
            liquidity.append(LiquidityLevel(
                timestamp=pwh_timestamp,
                level_type='PWH',
                price=pwh_price,
                strength=strength,
                session=self._get_session_from_time(pwh_timestamp)
            ))
            
            liquidity.append(LiquidityLevel(
                timestamp=pwl_timestamp,
                level_type='PWL',
                price=pwl_price,
                strength=strength,
                session=self._get_session_from_time(pwl_timestamp)
            ))
        
        return liquidity
    
    def detect_session_liquidity(self, data: pd.DataFrame,
                               current_time: pd.Timestamp) -> List[LiquidityLevel]:
        """Detect session-specific liquidity (Asian, London, NY sessions)."""
        liquidity = []
        
        # Define session times in NY timezone
        session_times = {
            'asian': [(18, 2)],  # 6 PM - 2 AM NY time (crosses midnight)
            'london': [(3, 8)],   # 3 AM - 8 AM NY time
            'ny_open': [(9, 11)], # 9 AM - 11 AM NY time
            'ny_pm': [(12, 16)]   # 12 PM - 4 PM NY time
        }
        
        # Convert to NY timezone
        ny_data = self._convert_to_ny_timezone(data)
        
        # Group by date to process each trading day
        daily_groups = ny_data.groupby(ny_data.index.date)
        current_date = current_time.date()
        
        # Look at last 3 trading days
        dates = sorted([d for d in daily_groups.groups.keys() if d < current_date])
        
        for date in dates[-3:]:
            day_data = daily_groups.get_group(date)
            
            for session_name, time_ranges in session_times.items():
                session_data = self._filter_session_data(day_data, time_ranges, session_name)
                
                if not session_data.empty:
                    session_high = session_data['high'].max()
                    session_low = session_data['low'].min()
                    
                    session_high_time = session_data[session_data['high'] == session_high].index[0]
                    session_low_time = session_data[session_data['low'] == session_low].index[0]
                    
                    # Asian session liquidity is considered strongest
                    base_strength = 4 if session_name == 'asian' else 3
                    
                    liquidity.extend([
                        LiquidityLevel(
                            timestamp=session_high_time,
                            level_type=f'{session_name.upper()}_HIGH',
                            price=session_high,
                            strength=base_strength,
                            session=session_name
                        ),
                        LiquidityLevel(
                            timestamp=session_low_time,
                            level_type=f'{session_name.upper()}_LOW',
                            price=session_low,
                            strength=base_strength,
                            session=session_name
                        )
                    ])
        
        return liquidity
    
    def detect_relative_equal_levels(self, data: pd.DataFrame,
                                   current_time: pd.Timestamp,
                                   lookback_candles: int = 50) -> List[LiquidityLevel]:
        """Detect relative equal highs and lows."""
        liquidity = []
        
        if len(data) < lookback_candles:
            return liquidity
        
        # Use recent data for analysis
        recent_data = data.tail(lookback_candles)
        
        # Find relative equal highs
        high_levels = self._find_relative_equal_levels(
            recent_data['high'], 'REH', self.equal_level_threshold
        )
        
        # Find relative equal lows
        low_levels = self._find_relative_equal_levels(
            recent_data['low'], 'REL', self.equal_level_threshold
        )
        
        liquidity.extend(high_levels)
        liquidity.extend(low_levels)
        
        return liquidity
    
    def detect_timeframe_highs_lows(self, data: pd.DataFrame,
                                  current_time: pd.Timestamp,
                                  lookback_candles: int = 100) -> List[LiquidityLevel]:
        """Detect established highs/lows on current timeframe."""
        liquidity = []
        
        if len(data) < lookback_candles:
            return liquidity
        
        # Use recent data
        recent_data = data.tail(lookback_candles)
        
        # Find swing highs and lows with different window sizes
        for window in [5, 10, 20]:
            swing_highs = self._find_swing_levels(recent_data, 'high', window, 'SWING_HIGH')
            swing_lows = self._find_swing_levels(recent_data, 'low', window, 'SWING_LOW')
            
            liquidity.extend(swing_highs)
            liquidity.extend(swing_lows)
        
        return liquidity
    
    def detect_week_opening_gaps(self, data: pd.DataFrame,
                               current_time: pd.Timestamp) -> List[WeekOpeningGap]:
        """Detect week opening gaps (Sunday open vs Friday close)."""
        gaps = []
        
        # Group by week
        weekly_groups = data.groupby(pd.Grouper(freq='W-MON'))
        
        previous_week_close = None
        
        for week_start, week_data in weekly_groups:
            if week_data.empty:
                continue
            
            week_open = week_data.iloc[0]['open']
            week_close = week_data.iloc[-1]['close']
            
            # Check for gap from previous week
            if previous_week_close is not None:
                gap_size = abs(week_open - previous_week_close)
                gap_direction = 'up' if week_open > previous_week_close else 'down'
                
                # Only consider significant gaps (> 0.1%)
                if gap_size / previous_week_close > 0.001:
                    gap = WeekOpeningGap(
                        week_start=week_start,
                        friday_close=previous_week_close,
                        sunday_open=week_open,
                        gap_size=gap_size,
                        gap_direction=gap_direction
                    )
                    
                    # Check if gap was filled during the week
                    if gap_direction == 'up':
                        fill_candles = week_data[week_data['low'] <= previous_week_close]
                    else:
                        fill_candles = week_data[week_data['high'] >= previous_week_close]
                    
                    if not fill_candles.empty:
                        gap.is_filled = True
                        gap.fill_timestamp = fill_candles.index[0]
                    
                    gaps.append(gap)
            
            previous_week_close = week_close
        
        return gaps
    
    def get_liquidity_near_price(self, liquidity_levels: List[LiquidityLevel],
                               current_price: float,
                               distance_pct: float = 0.02) -> List[LiquidityLevel]:
        """Get liquidity levels near current price."""
        near_levels = []
        
        for level in liquidity_levels:
            distance = abs(level.price - current_price) / current_price
            if distance <= distance_pct:
                near_levels.append(level)
        
        # Sort by distance and strength
        near_levels.sort(key=lambda x: (abs(x.price - current_price), -x.strength))
        
        return near_levels
    
    def get_unswept_liquidity(self, liquidity_levels: List[LiquidityLevel]) -> List[LiquidityLevel]:
        """Get only unswept liquidity levels."""
        return [level for level in liquidity_levels if not level.is_swept]
    
    def get_liquidity_by_session(self, liquidity_levels: List[LiquidityLevel],
                               session: str) -> List[LiquidityLevel]:
        """Get liquidity levels from specific session."""
        return [level for level in liquidity_levels if level.session == session]
    
    def get_strongest_liquidity(self, liquidity_levels: List[LiquidityLevel],
                              min_strength: int = 4) -> List[LiquidityLevel]:
        """Get only the strongest liquidity levels."""
        return [level for level in liquidity_levels if level.strength >= min_strength]
    
    def _find_relative_equal_levels(self, price_series: pd.Series, 
                                  level_type: str, 
                                  threshold: float) -> List[LiquidityLevel]:
        """Find relative equal highs or lows."""
        levels = []
        
        # Find local extremes first
        if 'REH' in level_type:
            extremes = self._find_local_maxima(price_series, window=3)
        else:
            extremes = self._find_local_minima(price_series, window=3)
        
        # Group extremes that are within threshold of each other
        extreme_groups = []
        for idx, price in extremes:
            added_to_group = False
            
            for group in extreme_groups:
                group_avg = sum(p for _, p in group) / len(group)
                if abs(price - group_avg) / group_avg <= threshold:
                    group.append((idx, price))
                    added_to_group = True
                    break
            
            if not added_to_group:
                extreme_groups.append([(idx, price)])
        
        # Create liquidity levels for groups with multiple extremes
        for group in extreme_groups:
            if len(group) >= 2:  # At least 2 equal levels
                avg_price = sum(p for _, p in group) / len(group)
                latest_timestamp = max(price_series.index[idx] for idx, _ in group)
                strength = min(5, max(2, len(group)))  # Strength based on number of equal levels
                
                levels.append(LiquidityLevel(
                    timestamp=latest_timestamp,
                    level_type=level_type,
                    price=avg_price,
                    strength=strength,
                    touches=len(group)
                ))
        
        return levels
    
    def _find_swing_levels(self, data: pd.DataFrame, column: str, 
                          window: int, level_type: str) -> List[LiquidityLevel]:
        """Find swing highs or lows."""
        levels = []
        
        if column == 'high':
            extremes = self._find_local_maxima(data[column], window)
        else:
            extremes = self._find_local_minima(data[column], window)
        
        for idx, price in extremes:
            timestamp = data.index[idx]
            strength = max(1, min(5, 6 - window // 5))  # Smaller windows = higher strength
            
            levels.append(LiquidityLevel(
                timestamp=timestamp,
                level_type=level_type,
                price=price,
                strength=strength,
                session=self._get_session_from_time(timestamp)
            ))
        
        return levels
    
    def _find_local_maxima(self, series: pd.Series, window: int) -> List[Tuple[int, float]]:
        """Find local maxima in a price series."""
        maxima = []
        
        for i in range(window, len(series) - window):
            if all(series.iloc[i] >= series.iloc[j] for j in range(i - window, i + window + 1)):
                maxima.append((i, series.iloc[i]))
        
        return maxima
    
    def _find_local_minima(self, series: pd.Series, window: int) -> List[Tuple[int, float]]:
        """Find local minima in a price series."""
        minima = []
        
        for i in range(window, len(series) - window):
            if all(series.iloc[i] <= series.iloc[j] for j in range(i - window, i + window + 1)):
                minima.append((i, series.iloc[i]))
        
        return minima
    
    def _update_liquidity_sweeps(self, liquidity_levels: List[LiquidityLevel],
                               data: pd.DataFrame, current_time: pd.Timestamp):
        """Update sweep status for all liquidity levels."""
        for level in liquidity_levels:
            # Only check data after the level was established
            future_data = data[data.index > level.timestamp]
            
            if future_data.empty:
                continue
            
            # Check if liquidity was swept
            if 'HIGH' in level.level_type or level.level_type in ['PDH', 'PWH', 'REH']:
                # For highs, check if price swept above
                sweeps = future_data[future_data['high'] > level.price * (1 + self.liquidity_sweep_threshold)]
            else:
                # For lows, check if price swept below
                sweeps = future_data[future_data['low'] < level.price * (1 - self.liquidity_sweep_threshold)]
            
            if not sweeps.empty:
                level.is_swept = True
                level.sweep_timestamp = sweeps.index[0]
            
            # Count touches (price came close but didn't sweep)
            if 'HIGH' in level.level_type or level.level_type in ['PDH', 'PWH', 'REH']:
                touches = future_data[
                    (future_data['high'] >= level.price * (1 - 0.001)) &
                    (future_data['high'] <= level.price * (1 + self.liquidity_sweep_threshold))
                ]
            else:
                touches = future_data[
                    (future_data['low'] <= level.price * (1 + 0.001)) &
                    (future_data['low'] >= level.price * (1 - self.liquidity_sweep_threshold))
                ]
            
            level.touches = len(touches)
    
    def _convert_to_ny_timezone(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data to NY timezone."""
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        
        ny_data = data.copy()
        ny_data.index = data.index.tz_convert(self.ny_tz)
        
        return ny_data
    
    def _filter_session_data(self, data: pd.DataFrame, time_ranges: List[Tuple[int, int]],
                           session_name: str) -> pd.DataFrame:
        """Filter data for specific session times."""
        session_data = pd.DataFrame()
        
        for start_hour, end_hour in time_ranges:
            if start_hour > end_hour:  # Crosses midnight (like Asian session)
                # Handle overnight sessions
                morning_data = data[data.index.hour < end_hour]
                evening_data = data[data.index.hour >= start_hour]
                session_data = pd.concat([session_data, morning_data, evening_data])
            else:
                # Regular session within same day
                mask = (data.index.hour >= start_hour) & (data.index.hour < end_hour)
                session_data = pd.concat([session_data, data[mask]])
        
        return session_data.sort_index()
    
    def _get_session_from_time(self, timestamp) -> str:
        """Determine which session a timestamp belongs to."""
        # Convert to pandas Timestamp if it's a datetime
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        ny_time = timestamp.astimezone(self.ny_tz)
        hour = ny_time.hour
        
        if (hour >= 18) or (hour < 2):
            return 'asian'
        elif 3 <= hour < 8:
            return 'london'
        elif 9 <= hour < 11:
            return 'ny_open'
        elif 12 <= hour < 16:
            return 'ny_pm'
        else:
            return 'overlap'
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        if data is None or len(data) < 10:
            return False
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        
        return True