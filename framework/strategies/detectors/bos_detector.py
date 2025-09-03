"""
Break of Structure (BoS) Detector

This module implements Break of Structure detection using swing highs and lows.
A Break of Structure occurs when:
- Bullish BoS: Price makes a higher low and then breaks the previous swing high
- Bearish BoS: Price makes a lower high and then breaks the previous swing low
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from .swing_detector import SwingDetector


class BoSDetector:
    """
    Break of Structure detector that identifies trend changes based on swing points.
    
    The detector works by:
    1. Finding all swing highs and lows in the price data
    2. Identifying higher lows and lower highs
    3. Detecting when price breaks previous swing levels
    """
    
    def __init__(self, swing_sensitivity: int = 3):
        """Initialize the BoS detector.
        
        Args:
            swing_sensitivity: Sensitivity for swing detection (default: 3)
        """
        self.swing_detector = SwingDetector(sensitivity=swing_sensitivity)
        self.swing_sensitivity = swing_sensitivity
    
    def detect_bos_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all Break of Structure events in the given price data.
        
        Args:
            df: DataFrame with OHLC data (must have 'High', 'Low', 'Close' columns)
            
        Returns:
            DataFrame with BoS events including timestamp, type, and price level
        """
        if not self._validate_data(df):
            return pd.DataFrame(columns=['timestamp', 'bos_type', 'price', 'swing_broken'])
        
        # Find all swing points
        swing_highs = self._find_swing_points_with_timestamps(df, 'High', is_high=True)
        swing_lows = self._find_swing_points_with_timestamps(df, 'Low', is_high=False)
        
        # Detect BoS events
        bos_events = []
        
        # Detect bullish BoS events
        bullish_bos = self._detect_bullish_bos(df, swing_highs, swing_lows)
        bos_events.extend(bullish_bos)
        
        # Detect bearish BoS events
        bearish_bos = self._detect_bearish_bos(df, swing_highs, swing_lows)
        bos_events.extend(bearish_bos)
        
        # Create result DataFrame
        if bos_events:
            result_df = pd.DataFrame(bos_events)
            result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        else:
            result_df = pd.DataFrame(columns=['timestamp', 'bos_type', 'price', 'swing_broken'])
        
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
    
    def _detect_bullish_bos(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """
        Detect bullish BoS events: Higher low followed by break of previous swing high.
        """
        bos_events = []
        
        if len(swing_lows) < 2 or len(swing_highs) < 1:
            return bos_events
        
        # Sort swing points by timestamp
        swing_lows_sorted = sorted(swing_lows, key=lambda x: x['timestamp'])
        swing_highs_sorted = sorted(swing_highs, key=lambda x: x['timestamp'])
        
        # Look for higher lows
        for i in range(1, len(swing_lows_sorted)):
            current_low = swing_lows_sorted[i]
            previous_low = swing_lows_sorted[i-1]
            
            # Check if current low is higher than previous low
            if current_low['price'] > previous_low['price']:
                # Find the most recent swing high before this higher low
                relevant_high = None
                for high in reversed(swing_highs_sorted):
                    if high['timestamp'] < current_low['timestamp']:
                        relevant_high = high
                        break
                
                if relevant_high is None:
                    continue
                
                # Look for price breaking above this swing high after the higher low
                start_idx = current_low['index']
                for j in range(start_idx + 1, len(df)):
                    if df['High'].iloc[j] > relevant_high['price']:
                        # Bullish BoS detected
                        bos_events.append({
                            'timestamp': df.index[j],
                            'bos_type': 'bullish',
                            'price': df['High'].iloc[j],
                            'swing_broken': relevant_high['price']
                        })
                        break
        
        return bos_events
    
    def _detect_bearish_bos(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """
        Detect bearish BoS events: Lower high followed by break of previous swing low.
        """
        bos_events = []
        
        if len(swing_highs) < 2 or len(swing_lows) < 1:
            return bos_events
        
        # Sort swing points by timestamp
        swing_highs_sorted = sorted(swing_highs, key=lambda x: x['timestamp'])
        swing_lows_sorted = sorted(swing_lows, key=lambda x: x['timestamp'])
        
        # Look for lower highs
        for i in range(1, len(swing_highs_sorted)):
            current_high = swing_highs_sorted[i]
            previous_high = swing_highs_sorted[i-1]
            
            # Check if current high is lower than previous high
            if current_high['price'] < previous_high['price']:
                # Find the most recent swing low before this lower high
                relevant_low = None
                for low in reversed(swing_lows_sorted):
                    if low['timestamp'] < current_high['timestamp']:
                        relevant_low = low
                        break
                
                if relevant_low is None:
                    continue
                
                # Look for price breaking below this swing low after the lower high
                start_idx = current_high['index']
                for j in range(start_idx + 1, len(df)):
                    if df['Low'].iloc[j] < relevant_low['price']:
                        # Bearish BoS detected
                        bos_events.append({
                            'timestamp': df.index[j],
                            'bos_type': 'bearish',
                            'price': df['Low'].iloc[j],
                            'swing_broken': relevant_low['price']
                        })
                        break
        
        return bos_events
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data."""
        if df is None or df.empty:
            return False
        
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            return False
        
        if len(df) < self.swing_sensitivity * 4:  # Need enough data for swing detection
            return False
        
        return True