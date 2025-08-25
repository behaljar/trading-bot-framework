"""
Break of Structure (BoS) Detector

This module implements Break of Structure detection for identifying market structure shifts.
BoS occurs when price breaks a significant swing high or swing low, indicating a potential
change in market direction.

Market Structure Types:
- HL (Higher Low) -> HH (Higher High): Bullish structure break
- LH (Lower High) -> LL (Lower Low): Bearish structure break
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from enum import Enum

from .pivot_detector import PivotDetector


class StructureType(Enum):
    """Enumeration for market structure types."""
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    UNKNOWN = "UNKNOWN"


class BoSType(Enum):
    """Enumeration for Break of Structure types."""
    BULLISH = "BULLISH"  # HL -> HH
    BEARISH = "BEARISH"  # LH -> LL
    NONE = "NONE"


class BoSDetector:
    """
    Detector for Break of Structure in price series data.
    
    This class identifies when market structure changes from one pattern to another,
    specifically detecting transitions from HL to HH (bullish BoS) or LH to LL (bearish BoS).
    """
    
    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        """
        Initialize the BoS detector.
        
        Args:
            left_bars: Number of bars to the left for pivot detection sensitivity
            right_bars: Number of bars to the right for pivot detection sensitivity
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.pivot_detector = PivotDetector()
        
    def detect_structure_breaks(self, 
                              high_series: Union[pd.Series, np.ndarray],
                              low_series: Union[pd.Series, np.ndarray],
                              min_structure_points: int = 2) -> pd.DataFrame:
        """
        Detect Break of Structure patterns in price data.
        
        Args:
            high_series: Series of high prices
            low_series: Series of low prices
            min_structure_points: Minimum number of structure points needed for BoS detection
            
        Returns:
            DataFrame with BoS detection results
        """
        # Convert to pandas Series if needed
        if not isinstance(high_series, pd.Series):
            high_series = pd.Series(high_series)
        if not isinstance(low_series, pd.Series):
            low_series = pd.Series(low_series)
            
        # Ensure same index
        if not high_series.index.equals(low_series.index):
            high_series.index = range(len(high_series))
            low_series.index = range(len(low_series))
        
        # Find all pivot points
        pivot_highs = self.pivot_detector.find_all_pivot_highs(
            high_series, self.left_bars, self.right_bars
        )
        pivot_lows = self.pivot_detector.find_all_pivot_lows(
            low_series, self.left_bars, self.right_bars
        )
        
        # Create structure points DataFrame
        structure_points = self._create_structure_points(
            pivot_highs, pivot_lows, high_series.index
        )
        
        if len(structure_points) < min_structure_points:
            return self._create_empty_result(high_series.index)
        
        # Detect structure patterns
        structure_patterns = self._detect_structure_patterns(structure_points)
        
        # Identify BoS events
        bos_events = self._identify_bos_events(structure_patterns)
        
        # Create result DataFrame
        result = self._create_result_dataframe(
            high_series.index, structure_points, bos_events
        )
        
        return result
    
    def _create_structure_points(self, 
                               pivot_highs: List[Tuple], 
                               pivot_lows: List[Tuple],
                               index: pd.Index) -> pd.DataFrame:
        """Create a DataFrame of all structure points (highs and lows)."""
        structure_points = []
        
        # Add pivot highs
        for idx, value in pivot_highs:
            structure_points.append({
                'index': idx,
                'timestamp': index[idx] if hasattr(index, '__getitem__') else idx,
                'value': value,
                'type': 'high'
            })
        
        # Add pivot lows
        for idx, value in pivot_lows:
            structure_points.append({
                'index': idx,
                'timestamp': index[idx] if hasattr(index, '__getitem__') else idx,
                'value': value,
                'type': 'low'
            })
        
        # Sort by index
        structure_points.sort(key=lambda x: x['index'])
        
        return pd.DataFrame(structure_points)
    
    def _detect_structure_patterns(self, structure_points: pd.DataFrame) -> List[Dict]:
        """
        Detect structure patterns (HH, HL, LH, LL) in consecutive structure points.
        
        Args:
            structure_points: DataFrame with structure points
            
        Returns:
            List of structure pattern dictionaries
        """
        if len(structure_points) < 2:
            return []
        
        patterns = []
        
        # Group by type and analyze sequences
        highs = structure_points[structure_points['type'] == 'high'].copy()
        lows = structure_points[structure_points['type'] == 'low'].copy()
        
        # Detect patterns in highs
        for i in range(1, len(highs)):
            prev_high = highs.iloc[i-1]
            curr_high = highs.iloc[i]
            
            if curr_high['value'] > prev_high['value']:
                pattern_type = StructureType.HIGHER_HIGH
            else:
                pattern_type = StructureType.LOWER_HIGH
                
            patterns.append({
                'start_index': prev_high['index'],
                'end_index': curr_high['index'],
                'start_value': prev_high['value'],
                'end_value': curr_high['value'],
                'type': pattern_type,
                'point_type': 'high'
            })
        
        # Detect patterns in lows
        for i in range(1, len(lows)):
            prev_low = lows.iloc[i-1]
            curr_low = lows.iloc[i]
            
            if curr_low['value'] > prev_low['value']:
                pattern_type = StructureType.HIGHER_LOW
            else:
                pattern_type = StructureType.LOWER_LOW
                
            patterns.append({
                'start_index': prev_low['index'],
                'end_index': curr_low['index'],
                'start_value': prev_low['value'],
                'end_value': curr_low['value'],
                'type': pattern_type,
                'point_type': 'low'
            })
        
        # Sort by end index
        patterns.sort(key=lambda x: x['end_index'])
        
        return patterns
    
    def _identify_bos_events(self, structure_patterns: List[Dict]) -> List[Dict]:
        """
        Identify Break of Structure events from structure patterns.
        
        Args:
            structure_patterns: List of structure pattern dictionaries
            
        Returns:
            List of BoS event dictionaries
        """
        bos_events = []
        
        if len(structure_patterns) < 2:  # Need at least 2 patterns for BoS
            return bos_events
        
        # Separate high and low patterns
        high_patterns = [p for p in structure_patterns if p['point_type'] == 'high']
        low_patterns = [p for p in structure_patterns if p['point_type'] == 'low']
        
        # Look for HL -> HH (Bullish BoS)
        # Need: Previous HL in lows, followed by HH in highs
        if len(high_patterns) >= 1 and len(low_patterns) >= 1:
            # Find HL patterns in lows
            for i, low_pattern in enumerate(low_patterns):
                if low_pattern['type'] == StructureType.HIGHER_LOW:
                    # Look for subsequent HH in highs that occurs after this HL
                    for high_pattern in high_patterns:
                        if (high_pattern['type'] == StructureType.HIGHER_HIGH and
                            high_pattern['end_index'] > low_pattern['end_index']):
                            
                            bos_events.append({
                                'index': high_pattern['end_index'],
                                'type': BoSType.BULLISH,
                                'from_pattern': low_pattern['type'],
                                'to_pattern': high_pattern['type'],
                                'break_level': high_pattern['end_value'],
                                'confidence': self._calculate_confidence(low_pattern, high_pattern)
                            })
                            break  # Only take the first HH after HL
        
            # Find LH -> LL (Bearish BoS)
            # Need: Previous LH in highs, followed by LL in lows
            for i, high_pattern in enumerate(high_patterns):
                if high_pattern['type'] == StructureType.LOWER_HIGH:
                    # Look for subsequent LL in lows that occurs after this LH
                    for low_pattern in low_patterns:
                        if (low_pattern['type'] == StructureType.LOWER_LOW and
                            low_pattern['end_index'] > high_pattern['end_index']):
                            
                            bos_events.append({
                                'index': low_pattern['end_index'],
                                'type': BoSType.BEARISH,
                                'from_pattern': high_pattern['type'],
                                'to_pattern': low_pattern['type'],
                                'break_level': low_pattern['end_value'],
                                'confidence': self._calculate_confidence(high_pattern, low_pattern)
                            })
                            break  # Only take the first LL after LH
        
        # Sort by index and remove duplicates
        bos_events.sort(key=lambda x: x['index'])
        
        return bos_events
    
    def _calculate_confidence(self, prev_pattern: Dict, curr_pattern: Dict) -> float:
        """
        Calculate confidence level for BoS detection.
        
        Args:
            prev_pattern: Previous structure pattern
            curr_pattern: Current structure pattern
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5
        
        # Increase confidence based on the magnitude of the break
        if curr_pattern['point_type'] == 'high':
            # For highs, bigger break above previous high = higher confidence
            price_diff = curr_pattern['end_value'] - prev_pattern['end_value']
            if price_diff > 0:
                confidence += min(0.3, price_diff / prev_pattern['end_value'] * 10)
        else:
            # For lows, bigger break below previous low = higher confidence
            price_diff = prev_pattern['end_value'] - curr_pattern['end_value']
            if price_diff > 0:
                confidence += min(0.3, price_diff / prev_pattern['end_value'] * 10)
        
        # Increase confidence based on time separation (more separation = higher confidence)
        time_diff = curr_pattern['end_index'] - prev_pattern['end_index']
        if time_diff > self.left_bars + self.right_bars:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _create_result_dataframe(self, 
                               index: pd.Index,
                               structure_points: pd.DataFrame,
                               bos_events: List[Dict]) -> pd.DataFrame:
        """Create the final result DataFrame with BoS detection results."""
        result = pd.DataFrame(index=index)
        
        # Initialize columns
        result['bos_type'] = BoSType.NONE.value
        result['bos_confidence'] = 0.0
        result['break_level'] = np.nan
        result['is_pivot_high'] = False
        result['is_pivot_low'] = False
        result['pivot_value'] = np.nan
        result['structure_type'] = StructureType.UNKNOWN.value
        
        # Mark pivot points
        for _, point in structure_points.iterrows():
            idx = point['index']
            if idx < len(result):
                if point['type'] == 'high':
                    result.at[idx, 'is_pivot_high'] = True
                else:
                    result.at[idx, 'is_pivot_low'] = True
                result.at[idx, 'pivot_value'] = point['value']
        
        # Mark BoS events
        for event in bos_events:
            idx = event['index']
            if idx < len(result):
                result.at[idx, 'bos_type'] = event['type'].value
                result.at[idx, 'bos_confidence'] = event['confidence']
                result.at[idx, 'break_level'] = event['break_level']
                result.at[idx, 'structure_type'] = f"{event['from_pattern'].value}->{event['to_pattern'].value}"
        
        return result
    
    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create an empty result DataFrame when insufficient data."""
        result = pd.DataFrame(index=index)
        result['bos_type'] = BoSType.NONE.value
        result['bos_confidence'] = 0.0
        result['break_level'] = np.nan
        result['is_pivot_high'] = False
        result['is_pivot_low'] = False
        result['pivot_value'] = np.nan
        result['structure_type'] = StructureType.UNKNOWN.value
        
        return result
    
    def get_latest_bos(self, 
                      high_series: Union[pd.Series, np.ndarray],
                      low_series: Union[pd.Series, np.ndarray],
                      lookback_periods: int = 100) -> Optional[Dict]:
        """
        Get the most recent Break of Structure event.
        
        Args:
            high_series: Series of high prices
            low_series: Series of low prices
            lookback_periods: Number of periods to look back for BoS detection
            
        Returns:
            Dictionary with latest BoS information, None if no BoS found
        """
        # Limit data to lookback periods
        if len(high_series) > lookback_periods:
            high_series = high_series[-lookback_periods:]
            low_series = low_series[-lookback_periods:]
        
        bos_result = self.detect_structure_breaks(high_series, low_series)
        
        # Find the most recent BoS event
        bos_events = bos_result[bos_result['bos_type'] != BoSType.NONE.value]
        
        if len(bos_events) == 0:
            return None
        
        latest_bos = bos_events.iloc[-1]
        
        return {
            'type': latest_bos['bos_type'],
            'confidence': latest_bos['bos_confidence'],
            'break_level': latest_bos['break_level'],
            'structure_type': latest_bos['structure_type'],
            'index': latest_bos.name
        }