"""
Change of Character (ChoCh) Detector

This module implements Change of Character detection using swing highs and lows.
A Change of Character occurs when:
- Bullish ChoCh: Price breaks a swing high after forming a higher low pattern
- Bearish ChoCh: Price breaks a swing low after forming a lower high pattern
"""

import pandas as pd
from typing import List, Dict
from .swing_detector import SwingDetector


class ChOChDetector:
    """
    Change of Character detector that identifies trend reversals based on swing points.
    
    The detector works by:
    1. Finding all swing highs and lows in the price data
    2. Identifying higher low/lower high patterns
    3. Detecting when price breaks previous swing levels after these patterns
    4. Validating patterns using only historical data to prevent lookahead bias
    """
    
    def __init__(self, swing_sensitivity: int = 3, max_swing_lookback: int = 10):
        """Initialize the ChoCh detector.
        
        Args:
            swing_sensitivity: Sensitivity for swing detection (default: 3)
            max_swing_lookback: Maximum number of recent swing points to consider (default: 10)
        """
        self.swing_detector = SwingDetector(sensitivity=swing_sensitivity)
        self.swing_sensitivity = swing_sensitivity
        self.max_swing_lookback = max_swing_lookback
    
    def detect_choch_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all Change of Character events in the given price data.
        
        Args:
            df: DataFrame with OHLC data (must have 'High', 'Low', 'Close' columns)
            
        Returns:
            DataFrame with ChoCh events including timestamp, type, and price level
        """
        if not self._validate_data(df):
            return pd.DataFrame(columns=['timestamp', 'choch_type', 'price', 'swing_broken'])
        
        # Find all swing points
        all_swing_highs = self._find_swing_points_with_timestamps(df, 'High', is_high=True)
        all_swing_lows = self._find_swing_points_with_timestamps(df, 'Low', is_high=False)
        
        # Detect ChoCh events
        bullish_events = self._detect_bullish_choch(df, all_swing_highs, all_swing_lows)
        bearish_events = self._detect_bearish_choch(df, all_swing_highs, all_swing_lows)
        
        choch_events = bullish_events + bearish_events
        
        # Filter overlapping patterns and return sorted results
        if choch_events:
            choch_events = self._filter_overlapping_choch(choch_events)
            result_df = pd.DataFrame(choch_events)
            result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        else:
            result_df = pd.DataFrame(columns=['timestamp', 'choch_type', 'price', 'swing_broken'])
        
        return result_df
    
    def _find_swing_points_with_timestamps(self, df: pd.DataFrame, column: str, is_high: bool) -> List[Dict]:
        """Find swing points and return with timestamps."""
        if is_high:
            swing_points = self.swing_detector.find_all_swing_highs(df[column])
        else:
            swing_points = self.swing_detector.find_all_swing_lows(df[column])
        
        swing_data = []
        for idx, value in swing_points:
            swing_data.append({
                'index': idx,
                'timestamp': df.index[idx],
                'price': value,
                'type': 'high' if is_high else 'low'
            })
        
        return swing_data
    
    def _detect_bullish_choch(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """
        Detect bullish ChoCh events.
        Pattern: SH -> SL -> HL -> Break SH (ChoCh)
        """
        choch_events = []
        
        if len(swing_lows) < 2 or len(swing_highs) < 1:
            return choch_events
        
        swing_lows_sorted = sorted(swing_lows, key=lambda x: x['timestamp'])
        swing_highs_sorted = sorted(swing_highs, key=lambda x: x['timestamp'])
        used_swing_highs = set()
        
        for i in range(1, len(swing_lows_sorted)):
            current_low = swing_lows_sorted[i]  # Higher low (HL)
            previous_low = swing_lows_sorted[i-1]  # Previous low (SL)
            
            if current_low['price'] > previous_low['price']:
                # Find swing high that occurred before the previous low
                choch_target_high = None
                for high in reversed(swing_highs_sorted):
                    if high['timestamp'] < previous_low['timestamp']:
                        if high['timestamp'] not in used_swing_highs:
                            choch_target_high = high
                            break
                
                if choch_target_high is None:
                    continue
                
                # Look for price breaking above the target swing high
                for j in range(current_low['index'] + 1, len(df)):
                    current_candle_timestamp = df.index[j]
                    
                    # Check for intermediate highs that invalidate the pattern
                    intermediate_highs_invalidate = False
                    for high in swing_highs_sorted:
                        if (high['timestamp'] > choch_target_high['timestamp'] and 
                            high['timestamp'] < current_candle_timestamp and
                            high['price'] > choch_target_high['price']):
                            intermediate_highs_invalidate = True
                            break
                    
                    if intermediate_highs_invalidate:
                        break
                    
                    if df['Close'].iloc[j] > choch_target_high['price']:
                        choch_events.append({
                            'timestamp': current_candle_timestamp,
                            'choch_type': 'bullish', 
                            'price': df['Close'].iloc[j],
                            'swing_broken': choch_target_high['price']
                        })
                        used_swing_highs.add(choch_target_high['timestamp'])
                        break
        
        return choch_events
    
    def _detect_bearish_choch(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """
        Detect bearish ChoCh events.
        Pattern: SL -> SH -> LH -> Break SL (ChoCh)
        """
        choch_events = []
        
        if len(swing_highs) < 2 or len(swing_lows) < 1:
            return choch_events
        
        swing_highs_sorted = sorted(swing_highs, key=lambda x: x['timestamp'])
        swing_lows_sorted = sorted(swing_lows, key=lambda x: x['timestamp'])
        used_swing_lows = set()
        
        for i in range(1, len(swing_highs_sorted)):
            current_high = swing_highs_sorted[i]  # Lower high (LH)
            previous_high = swing_highs_sorted[i-1]  # Previous high (SH)
            
            if current_high['price'] < previous_high['price']:
                # Find swing low that occurred before the previous high
                choch_target_low = None
                for low in reversed(swing_lows_sorted):
                    if low['timestamp'] < previous_high['timestamp']:
                        if low['timestamp'] not in used_swing_lows:
                            choch_target_low = low
                            break
                
                if choch_target_low is None:
                    continue
                
                # Look for price breaking below the target swing low
                for j in range(current_high['index'] + 1, len(df)):
                    current_candle_timestamp = df.index[j]
                    
                    # Check for intermediate lows that invalidate the pattern
                    intermediate_lows_invalidate = False
                    for low in swing_lows_sorted:
                        if (low['timestamp'] > choch_target_low['timestamp'] and 
                            low['timestamp'] < current_candle_timestamp and
                            low['price'] < choch_target_low['price']):
                            intermediate_lows_invalidate = True
                            break
                    
                    if intermediate_lows_invalidate:
                        break
                    
                    if df['Close'].iloc[j] < choch_target_low['price']:
                        choch_events.append({
                            'timestamp': current_candle_timestamp,
                            'choch_type': 'bearish',
                            'price': df['Close'].iloc[j],
                            'swing_broken': choch_target_low['price']
                        })
                        used_swing_lows.add(choch_target_low['timestamp'])
                        break
        
        return choch_events
    
    def _filter_overlapping_choch(self, choch_events: List[Dict]) -> List[Dict]:
        """Filter out weaker overlapping ChoCh events."""
        if not choch_events:
            return choch_events
        
        sorted_events = sorted(choch_events, key=lambda x: x['timestamp'])
        filtered_events = []
        
        for i, current_event in enumerate(sorted_events):
            should_keep = True
            
            for j in range(i + 1, min(i + 11, len(sorted_events))):
                next_event = sorted_events[j]
                
                if current_event['choch_type'] != next_event['choch_type']:
                    continue
                
                time_diff_minutes = (next_event['timestamp'] - current_event['timestamp']).total_seconds() / 60
                
                if time_diff_minutes <= 10:
                    if current_event['choch_type'] == 'bullish':
                        if next_event['price'] > current_event['price']:
                            should_keep = False
                            break
                    else:
                        if next_event['price'] < current_event['price']:
                            should_keep = False
                            break
            
            if should_keep:
                filtered_events.append(current_event)
        
        return filtered_events
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data."""
        if df is None or df.empty:
            return False
        
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            return False
        
        if len(df) < self.swing_sensitivity * 4:
            return False
        
        return True