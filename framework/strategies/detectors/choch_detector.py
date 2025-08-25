"""
Change of Character (CHoCH) Detector

This module implements Change of Character detection for identifying shifts in market behavior.
CHoCH occurs when the market changes its character from bullish to bearish or vice versa,
indicating a potential trend reversal or change in momentum.

CHoCH Pattern Logic:
- Bullish CHoCH: Higher High after the last Low that was detected before a previous High
- Bearish CHoCH: Lower Low after the last High that was detected before a previous Low
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from enum import Enum

from .pivot_detector import PivotDetector


class CHoCHType(Enum):
    """Enumeration for Change of Character types."""
    BULLISH = "BULLISH"  # HH after significant LL
    BEARISH = "BEARISH"  # LL after significant HH
    NONE = "NONE"


class MarketCharacter(Enum):
    """Enumeration for market character states."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH" 
    NEUTRAL = "NEUTRAL"


class CHoCHDetector:
    """
    Detector for Change of Character in price series data.
    
    This class identifies when market character changes from bullish to bearish or vice versa,
    by detecting significant structural changes in swing highs and lows patterns.
    """
    
    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        """
        Initialize the CHoCH detector.
        
        Args:
            left_bars: Number of bars to the left for pivot detection sensitivity
            right_bars: Number of bars to the right for pivot detection sensitivity
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.pivot_detector = PivotDetector()
        
    def detect_character_changes(self, 
                               high_series: Union[pd.Series, np.ndarray],
                               low_series: Union[pd.Series, np.ndarray],
                               min_pivot_points: int = 3) -> pd.DataFrame:
        """
        Detect Change of Character patterns in price data.
        
        Args:
            high_series: Series of high prices
            low_series: Series of low prices
            min_pivot_points: Minimum number of pivot points needed for CHoCH detection
            
        Returns:
            DataFrame with CHoCH detection results
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
        
        if len(pivot_highs) + len(pivot_lows) < min_pivot_points:
            return self._create_empty_result(high_series.index)
        
        # Create chronological pivot sequence
        pivot_sequence = self._create_pivot_sequence(pivot_highs, pivot_lows)
        
        # Detect CHoCH events
        choch_events = self._identify_choch_events(pivot_sequence)
        
        # Create result DataFrame
        result = self._create_result_dataframe(
            high_series.index, pivot_highs, pivot_lows, choch_events
        )
        
        return result
    
    def _create_pivot_sequence(self, pivot_highs: List[Tuple], pivot_lows: List[Tuple]) -> List[Dict]:
        """Create a chronological sequence of pivot points."""
        pivot_sequence = []
        
        # Add pivot highs
        for idx, value in pivot_highs:
            pivot_sequence.append({
                'index': idx,
                'value': value,
                'type': 'high',
                'timestamp': idx
            })
        
        # Add pivot lows
        for idx, value in pivot_lows:
            pivot_sequence.append({
                'index': idx,
                'value': value,
                'type': 'low',
                'timestamp': idx
            })
        
        # Sort by chronological order (index)
        pivot_sequence.sort(key=lambda x: x['index'])
        
        return pivot_sequence
    
    def _identify_choch_events(self, pivot_sequence: List[Dict]) -> List[Dict]:
        """
        Identify Change of Character events from pivot sequence.
        
        CHoCH Logic (as per user requirements):
        - Bullish CHoCH: Higher high than the last high detected BEFORE the lowest low detected
        - Bearish CHoCH: Lower low than the last low detected BEFORE the highest high detected
        
        Args:
            pivot_sequence: Chronologically ordered pivot points
            
        Returns:
            List of CHoCH event dictionaries
        """
        choch_events = []
        
        if len(pivot_sequence) < 4:  # Need at least 4 points for CHoCH pattern
            return choch_events
        
        # Extract highs and lows separately with their indices
        highs = [(p['index'], p['value']) for p in pivot_sequence if p['type'] == 'high']
        lows = [(p['index'], p['value']) for p in pivot_sequence if p['type'] == 'low']
        
        # Bullish CHoCH: HH than the last high detected before the lowest low
        for i in range(len(highs)):
            current_high_idx, current_high_value = highs[i]
            
            # Find the lowest low that occurred before this high
            lowest_low_before = None
            lowest_low_idx = None
            for low_idx, low_value in lows:
                if low_idx < current_high_idx:
                    if lowest_low_before is None or low_value < lowest_low_before:
                        lowest_low_before = low_value
                        lowest_low_idx = low_idx
            
            if lowest_low_before is not None:
                # Find the last high that occurred before the lowest low
                last_high_before_ll = None
                for high_idx, high_value in highs:
                    if high_idx < lowest_low_idx:
                        if last_high_before_ll is None or high_idx > last_high_before_ll[0]:
                            last_high_before_ll = (high_idx, high_value)
                
                # Check if current high is higher than the last high before the lowest low
                if (last_high_before_ll is not None and 
                    current_high_value > last_high_before_ll[1]):
                    
                    choch_events.append({
                        'index': current_high_idx,
                        'type': CHoCHType.BULLISH,
                        'level': current_high_value,
                        'previous_high': last_high_before_ll[1],
                        'triggering_low': lowest_low_before,
                        'confidence': self._calculate_choch_confidence(
                            current_high_value, last_high_before_ll[1], 
                            {'value': lowest_low_before}, 'bullish'
                        )
                    })
        
        # Bearish CHoCH: LL than the last low detected before the highest high  
        for i in range(len(lows)):
            current_low_idx, current_low_value = lows[i]
            
            # Find the highest high that occurred before this low
            highest_high_before = None
            highest_high_idx = None
            for high_idx, high_value in highs:
                if high_idx < current_low_idx:
                    if highest_high_before is None or high_value > highest_high_before:
                        highest_high_before = high_value
                        highest_high_idx = high_idx
            
            if highest_high_before is not None:
                # Find the last low that occurred before the highest high
                last_low_before_hh = None
                for low_idx, low_value in lows:
                    if low_idx < highest_high_idx:
                        if last_low_before_hh is None or low_idx > last_low_before_hh[0]:
                            last_low_before_hh = (low_idx, low_value)
                
                # Check if current low is lower than the last low before the highest high
                if (last_low_before_hh is not None and 
                    current_low_value < last_low_before_hh[1]):
                    
                    choch_events.append({
                        'index': current_low_idx,
                        'type': CHoCHType.BEARISH,
                        'level': current_low_value,
                        'previous_low': last_low_before_hh[1],
                        'triggering_high': highest_high_before,
                        'confidence': self._calculate_choch_confidence(
                            current_low_value, last_low_before_hh[1],
                            {'value': highest_high_before}, 'bearish'
                        )
                    })
        
        # Sort events by index and remove duplicates
        choch_events.sort(key=lambda x: x['index'])
        
        # Remove duplicate events at the same index (keep the one with higher confidence)
        filtered_events = []
        seen_indices = set()
        for event in choch_events:
            if event['index'] not in seen_indices:
                filtered_events.append(event)
                seen_indices.add(event['index'])
            else:
                # Replace if higher confidence
                for j, existing_event in enumerate(filtered_events):
                    if (existing_event['index'] == event['index'] and 
                        event['confidence'] > existing_event['confidence']):
                        filtered_events[j] = event
                        break
        
        return filtered_events
    
    def _calculate_choch_confidence(self, current_level: float, previous_level: float, 
                                  triggering_pivot: Optional[Dict], direction: str) -> float:
        """
        Calculate confidence level for CHoCH detection.
        
        Args:
            current_level: Current pivot level
            previous_level: Previous significant level  
            triggering_pivot: The pivot that triggered the character change
            direction: 'bullish' or 'bearish'
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on the magnitude of the break
        if direction == 'bullish':
            price_diff = current_level - previous_level
            if price_diff > 0:
                confidence += min(0.3, price_diff / previous_level * 10)
        else:  # bearish
            price_diff = previous_level - current_level
            if price_diff > 0:
                confidence += min(0.3, price_diff / previous_level * 10)
        
        # Increase confidence if there's a clear triggering pivot
        if triggering_pivot is not None:
            confidence += 0.1
        
        # Cap confidence at 1.0
        return min(1.0, confidence)
    
    def _create_result_dataframe(self, 
                               index: pd.Index,
                               pivot_highs: List[Tuple],
                               pivot_lows: List[Tuple],
                               choch_events: List[Dict]) -> pd.DataFrame:
        """Create the final result DataFrame with CHoCH detection results."""
        result = pd.DataFrame(index=index)
        
        # Initialize columns
        result['choch_type'] = CHoCHType.NONE.value
        result['choch_confidence'] = 0.0
        result['choch_level'] = np.nan
        result['is_pivot_high'] = False
        result['is_pivot_low'] = False
        result['pivot_value'] = np.nan
        result['market_character'] = MarketCharacter.NEUTRAL.value
        result['previous_level'] = np.nan
        result['triggering_level'] = np.nan
        
        # Mark pivot points
        for idx, value in pivot_highs:
            if idx < len(result):
                result.at[idx, 'is_pivot_high'] = True
                result.at[idx, 'pivot_value'] = value
        
        for idx, value in pivot_lows:
            if idx < len(result):
                result.at[idx, 'is_pivot_low'] = True
                result.at[idx, 'pivot_value'] = value
        
        # Mark CHoCH events
        for event in choch_events:
            idx = event['index']
            if idx < len(result):
                result.at[idx, 'choch_type'] = event['type'].value
                result.at[idx, 'choch_confidence'] = event['confidence']
                result.at[idx, 'choch_level'] = event['level']
                result.at[idx, 'market_character'] = (MarketCharacter.BULLISH.value 
                                                     if event['type'] == CHoCHType.BULLISH 
                                                     else MarketCharacter.BEARISH.value)
                
                # Store reference levels
                if event['type'] == CHoCHType.BULLISH:
                    result.at[idx, 'previous_level'] = event['previous_high']
                    result.at[idx, 'triggering_level'] = event['triggering_low']
                else:
                    result.at[idx, 'previous_level'] = event['previous_low'] 
                    result.at[idx, 'triggering_level'] = event['triggering_high']
        
        return result
    
    def _create_empty_result(self, index: pd.Index) -> pd.DataFrame:
        """Create an empty result DataFrame when insufficient data."""
        result = pd.DataFrame(index=index)
        result['choch_type'] = CHoCHType.NONE.value
        result['choch_confidence'] = 0.0
        result['choch_level'] = np.nan
        result['is_pivot_high'] = False
        result['is_pivot_low'] = False
        result['pivot_value'] = np.nan
        result['market_character'] = MarketCharacter.NEUTRAL.value
        result['previous_level'] = np.nan
        result['triggering_level'] = np.nan
        
        return result
    
    def get_latest_choch(self, 
                        high_series: Union[pd.Series, np.ndarray],
                        low_series: Union[pd.Series, np.ndarray],
                        lookback_periods: int = 100) -> Optional[Dict]:
        """
        Get the most recent Change of Character event.
        
        Args:
            high_series: Series of high prices
            low_series: Series of low prices
            lookback_periods: Number of periods to look back for CHoCH detection
            
        Returns:
            Dictionary with latest CHoCH information, None if no CHoCH found
        """
        # Limit data to lookback periods
        if len(high_series) > lookback_periods:
            high_series = high_series[-lookback_periods:]
            low_series = low_series[-lookback_periods:]
        
        choch_result = self.detect_character_changes(high_series, low_series)
        
        # Find the most recent CHoCH event
        choch_events = choch_result[choch_result['choch_type'] != CHoCHType.NONE.value]
        
        if len(choch_events) == 0:
            return None
        
        latest_choch = choch_events.iloc[-1]
        
        return {
            'type': latest_choch['choch_type'],
            'confidence': latest_choch['choch_confidence'],
            'level': latest_choch['choch_level'],
            'market_character': latest_choch['market_character'],
            'previous_level': latest_choch['previous_level'],
            'triggering_level': latest_choch['triggering_level'],
            'index': latest_choch.name
        }
    
    def get_current_market_character(self, 
                                   high_series: Union[pd.Series, np.ndarray],
                                   low_series: Union[pd.Series, np.ndarray],
                                   lookback_periods: int = 50) -> MarketCharacter:
        """
        Determine the current market character based on recent CHoCH events.
        
        Args:
            high_series: Series of high prices
            low_series: Series of low prices
            lookback_periods: Number of periods to analyze for character determination
            
        Returns:
            Current market character (BULLISH, BEARISH, or NEUTRAL)
        """
        latest_choch = self.get_latest_choch(high_series, low_series, lookback_periods)
        
        if latest_choch is None:
            return MarketCharacter.NEUTRAL
        
        # Return the market character from the latest CHoCH event
        if latest_choch['type'] == CHoCHType.BULLISH.value:
            return MarketCharacter.BULLISH
        elif latest_choch['type'] == CHoCHType.BEARISH.value:
            return MarketCharacter.BEARISH
        else:
            return MarketCharacter.NEUTRAL