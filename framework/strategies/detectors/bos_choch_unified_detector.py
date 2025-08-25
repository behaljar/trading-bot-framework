"""
Unified Break of Structure (BoS) and Change of Character (CHoCH) Detector

This module implements precise BoS and CHoCH detection using 4-point swing patterns
and level relationship analysis, following proven market structure logic.

BoS (Break of Structure):
- Bullish BoS: Pattern [-1,1,-1,1] with levels: L1 < L3 < L2 < L4
- Bearish BoS: Pattern [1,-1,1,-1] with levels: L1 > L3 > L2 > L4

CHoCH (Change of Character):  
- Bullish CHoCH: Pattern [-1,1,-1,1] with levels: L4 > L2 > L1 > L3
- Bearish CHoCH: Pattern [1,-1,1,-1] with levels: L4 < L2 < L1 < L3

All patterns require break confirmation and handle invalidation properly.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Union
from enum import Enum

from .pivot_detector import PivotDetector


class StructureEventType(Enum):
    """Enumeration for structure event types."""
    BULLISH_BOS = "BULLISH_BOS"
    BEARISH_BOS = "BEARISH_BOS"
    BULLISH_CHOCH = "BULLISH_CHOCH"
    BEARISH_CHOCH = "BEARISH_CHOCH"
    NONE = "NONE"


class BoSCHoCHDetector:
    """
    Unified detector for Break of Structure (BoS) and Change of Character (CHoCH).
    
    This detector implements precise 4-point swing pattern recognition with
    level relationship analysis and break confirmation logic.
    """
    
    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        """
        Initialize the unified BoS/CHoCH detector.
        
        Args:
            left_bars: Number of bars to the left for pivot detection sensitivity
            right_bars: Number of bars to the right for pivot detection sensitivity
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.pivot_detector = PivotDetector()
        
    def detect_bos_choch(self, 
                        ohlc_data: pd.DataFrame,
                        close_break: bool = True) -> pd.DataFrame:
        """
        Detect BoS and CHoCH patterns in OHLC data.
        
        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            close_break: If True, use close prices for break confirmation, else use high/low
            
        Returns:
            DataFrame with columns:
            - bos: 1 for bullish BoS, -1 for bearish BoS, NaN otherwise
            - choch: 1 for bullish CHoCH, -1 for bearish CHoCH, NaN otherwise
            - level: The level that was broken
            - broken_index: Index where the level was broken
        """
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in ohlc_data.columns for col in required_columns):
            raise ValueError(f"OHLC data must contain columns: {required_columns}")
        
        # Find swing highs and lows
        swing_highs_lows = self._find_swing_highs_lows(ohlc_data)
        
        # Initialize result arrays
        data_len = len(ohlc_data)
        bos = np.zeros(data_len, dtype=np.int32)
        choch = np.zeros(data_len, dtype=np.int32) 
        level = np.zeros(data_len, dtype=np.float32)
        
        # Track swing pattern for analysis
        level_order = []
        highs_lows_order = []
        last_positions = []
        
        # Process each swing point
        for i in range(len(swing_highs_lows)):
            swing_point = swing_highs_lows.iloc[i]
            
            if not pd.isna(swing_point['high_low']):
                level_order.append(swing_point['level'])
                highs_lows_order.append(int(swing_point['high_low']))
                
                # Need at least 4 swing points for pattern analysis
                if len(level_order) >= 4:
                    pattern_index = last_positions[-2]  # Index of 3rd swing point
                    
                    # Check for Bullish BoS pattern
                    if (np.array_equal(highs_lows_order[-4:], [-1, 1, -1, 1]) and
                        self._check_bullish_bos_levels(level_order[-4:])):
                        bos[pattern_index] = 1
                        level[pattern_index] = level_order[-3]  # Break level is the previous high
                    
                    # Check for Bearish BoS pattern
                    elif (np.array_equal(highs_lows_order[-4:], [1, -1, 1, -1]) and
                          self._check_bearish_bos_levels(level_order[-4:])):
                        bos[pattern_index] = -1
                        level[pattern_index] = level_order[-3]  # Break level is the previous low
                    
                    # Check for Bullish CHoCH pattern
                    elif (np.array_equal(highs_lows_order[-4:], [-1, 1, -1, 1]) and
                          self._check_bullish_choch_levels(level_order[-4:])):
                        choch[pattern_index] = 1
                        level[pattern_index] = level_order[-3]  # Break level is the previous high
                    
                    # Check for Bearish CHoCH pattern
                    elif (np.array_equal(highs_lows_order[-4:], [1, -1, 1, -1]) and
                          self._check_bearish_choch_levels(level_order[-4:])):
                        choch[pattern_index] = -1
                        level[pattern_index] = level_order[-3]  # Break level is the previous low
                
                last_positions.append(i)
        
        # Confirm breaks and handle invalidation
        broken_index = self._confirm_breaks_and_invalidate(
            ohlc_data, bos, choch, level, close_break
        )
        
        # Create result DataFrame
        result = pd.DataFrame(index=ohlc_data.index)
        result['bos'] = pd.Series(np.where(bos != 0, bos, np.nan), index=ohlc_data.index, name='BOS')
        result['choch'] = pd.Series(np.where(choch != 0, choch, np.nan), index=ohlc_data.index, name='CHOCH')
        result['level'] = pd.Series(np.where(level != 0, level, np.nan), index=ohlc_data.index, name='Level')
        result['broken_index'] = pd.Series(np.where(broken_index != 0, broken_index, np.nan), 
                                         index=ohlc_data.index, name='BrokenIndex')
        
        return result
    
    def _find_swing_highs_lows(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Find swing highs and lows in OHLC data.
        
        Returns:
            DataFrame with columns: ['high_low', 'level'] where:
            - high_low: 1 for swing high, -1 for swing low, NaN otherwise
            - level: The price level of the swing point
        """
        # Find pivot points
        pivot_highs = self.pivot_detector.find_all_pivot_highs(
            ohlc_data['high'], self.left_bars, self.right_bars
        )
        pivot_lows = self.pivot_detector.find_all_pivot_lows(
            ohlc_data['low'], self.left_bars, self.right_bars
        )
        
        # Initialize result arrays
        data_len = len(ohlc_data)
        high_low = np.full(data_len, np.nan)
        levels = np.full(data_len, np.nan)
        
        # Mark swing highs
        for idx, value in pivot_highs:
            if idx < data_len:
                high_low[idx] = 1
                levels[idx] = value
        
        # Mark swing lows  
        for idx, value in pivot_lows:
            if idx < data_len:
                high_low[idx] = -1
                levels[idx] = value
        
        return pd.DataFrame({
            'high_low': high_low,
            'level': levels
        })
    
    def _check_bullish_bos_levels(self, levels: list) -> bool:
        """
        Check if levels match bullish BoS pattern: L1 < L3 < L2 < L4
        
        Args:
            levels: List of 4 consecutive swing levels [L1, L2, L3, L4]
            
        Returns:
            True if levels match bullish BoS pattern
        """
        return (levels[0] < levels[2] < levels[1] < levels[3])
    
    def _check_bearish_bos_levels(self, levels: list) -> bool:
        """
        Check if levels match bearish BoS pattern: L1 > L3 > L2 > L4
        
        Args:
            levels: List of 4 consecutive swing levels [L1, L2, L3, L4]
            
        Returns:
            True if levels match bearish BoS pattern
        """
        return (levels[0] > levels[2] > levels[1] > levels[3])
    
    def _check_bullish_choch_levels(self, levels: list) -> bool:
        """
        Check if levels match bullish CHoCH pattern: L4 > L2 > L1 > L3
        
        Args:
            levels: List of 4 consecutive swing levels [L1, L2, L3, L4]
            
        Returns:
            True if levels match bullish CHoCH pattern
        """
        return (levels[3] > levels[1] > levels[0] > levels[2])
    
    def _check_bearish_choch_levels(self, levels: list) -> bool:
        """
        Check if levels match bearish CHoCH pattern: L4 < L2 < L1 < L3
        
        Args:
            levels: List of 4 consecutive swing levels [L1, L2, L3, L4]
            
        Returns:
            True if levels match bearish CHoCH pattern
        """
        return (levels[3] < levels[1] < levels[0] < levels[2])
    
    def _confirm_breaks_and_invalidate(self, 
                                     ohlc_data: pd.DataFrame,
                                     bos: np.ndarray, 
                                     choch: np.ndarray, 
                                     level: np.ndarray,
                                     close_break: bool) -> np.ndarray:
        """
        Confirm breaks and invalidate unbroken or overlapping patterns.
        
        Args:
            ohlc_data: OHLC price data
            bos: Array of BoS signals
            choch: Array of CHoCH signals  
            level: Array of break levels
            close_break: Use close prices for break confirmation
            
        Returns:
            Array of broken indices
        """
        data_len = len(ohlc_data)
        broken_index = np.zeros(data_len, dtype=np.int32)
        
        # Find all signal indices
        signal_indices = np.where(np.logical_or(bos != 0, choch != 0))[0]
        
        for i in signal_indices:
            # Determine break condition based on signal direction
            if bos[i] == 1 or choch[i] == 1:  # Bullish signal - look for upward break
                price_series = ohlc_data['close'] if close_break else ohlc_data['high']
                mask = price_series[i + 2:] > level[i]
            elif bos[i] == -1 or choch[i] == -1:  # Bearish signal - look for downward break
                price_series = ohlc_data['close'] if close_break else ohlc_data['low']
                mask = price_series[i + 2:] < level[i]
            else:
                continue
            
            # Check if level was broken
            if np.any(mask):
                # Find first break occurrence
                break_offset = np.argmax(mask)
                break_index = i + 2 + break_offset
                broken_index[i] = break_index
                
                # Invalidate overlapping unbroken signals
                self._invalidate_overlapping_signals(
                    signal_indices, i, break_index, bos, choch, level, broken_index
                )
        
        # Remove unbroken signals
        unbroken_mask = np.logical_and(
            np.logical_or(bos != 0, choch != 0),
            broken_index == 0
        )
        unbroken_indices = np.where(unbroken_mask)[0]
        
        for idx in unbroken_indices:
            bos[idx] = 0
            choch[idx] = 0
            level[idx] = 0
        
        return broken_index
    
    def _invalidate_overlapping_signals(self, 
                                      signal_indices: np.ndarray,
                                      current_idx: int,
                                      break_index: int,
                                      bos: np.ndarray,
                                      choch: np.ndarray, 
                                      level: np.ndarray,
                                      broken_index: np.ndarray):
        """
        Invalidate signals that started before current signal but broke after it.
        
        This handles the case where an earlier signal's break is invalidated by
        a later signal that breaks first.
        """
        for other_idx in signal_indices:
            if (other_idx < current_idx and 
                broken_index[other_idx] >= break_index and
                broken_index[other_idx] != 0):
                # Invalidate the earlier signal
                bos[other_idx] = 0
                choch[other_idx] = 0
                level[other_idx] = 0
                broken_index[other_idx] = 0
    
    def get_latest_signal(self, 
                         ohlc_data: pd.DataFrame,
                         close_break: bool = True,
                         lookback_periods: int = 100) -> Optional[Dict]:
        """
        Get the most recent BoS or CHoCH signal.
        
        Args:
            ohlc_data: OHLC price data
            close_break: Use close prices for break confirmation
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with latest signal information, None if no signals found
        """
        # Limit data to lookback periods
        if len(ohlc_data) > lookback_periods:
            ohlc_data = ohlc_data.tail(lookback_periods).copy()
        
        result = self.detect_bos_choch(ohlc_data, close_break)
        
        # Find the most recent signal
        bos_signals = result[~pd.isna(result['bos'])]
        choch_signals = result[~pd.isna(result['choch'])]
        
        all_signals = []
        
        for idx, row in bos_signals.iterrows():
            all_signals.append({
                'index': idx,
                'type': 'BULLISH_BOS' if row['bos'] == 1 else 'BEARISH_BOS',
                'level': row['level'],
                'broken_index': row['broken_index']
            })
        
        for idx, row in choch_signals.iterrows():
            all_signals.append({
                'index': idx,
                'type': 'BULLISH_CHOCH' if row['choch'] == 1 else 'BEARISH_CHOCH', 
                'level': row['level'],
                'broken_index': row['broken_index']
            })
        
        if not all_signals:
            return None
        
        # Sort by index and return the most recent
        all_signals.sort(key=lambda x: x['index'])
        return all_signals[-1]
    
    def get_market_structure_bias(self, 
                                 ohlc_data: pd.DataFrame,
                                 close_break: bool = True,
                                 lookback_periods: int = 50) -> str:
        """
        Determine current market structure bias based on recent BoS/CHoCH signals.
        
        Args:
            ohlc_data: OHLC price data
            close_break: Use close prices for break confirmation
            lookback_periods: Number of periods to analyze
            
        Returns:
            Market bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        latest_signal = self.get_latest_signal(ohlc_data, close_break, lookback_periods)
        
        if latest_signal is None:
            return 'NEUTRAL'
        
        signal_type = latest_signal['type']
        
        if signal_type in ['BULLISH_BOS', 'BULLISH_CHOCH']:
            return 'BULLISH'
        elif signal_type in ['BEARISH_BOS', 'BEARISH_CHOCH']:
            return 'BEARISH'
        else:
            return 'NEUTRAL'