"""
Fair Value Gap (FVG) Detector

This module implements Fair Value Gap detection for identifying imbalances in price action.
A Fair Value Gap occurs when there is a gap between the high of one candle and the low of 
the candle two periods later (or vice versa), indicating an imbalance that the market
may return to fill.

An FVG is formed when:
- Bullish FVG: high[i-1] < low[i+1] (gap between previous high and next low)
- Bearish FVG: low[i-1] > high[i+1] (gap between previous low and next high)

The detector includes sensitivity filtering based on gap size relative to candle size
and merging logic for consecutive FVGs.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class FVG:
    """
    Represents a Fair Value Gap.
    
    Attributes:
        start_idx: Index where the FVG starts
        end_idx: Index where the FVG ends  
        top: Upper boundary of the gap
        bottom: Lower boundary of the gap
        fvg_type: 'bullish' or 'bearish'
        gap_size: Size of the gap (top - bottom)
        sensitivity_ratio: Gap size relative to the middle candle size
        is_merged: Whether this FVG was created by merging others
    """
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    fvg_type: str
    gap_size: float
    sensitivity_ratio: float
    is_merged: bool = False


class FVGDetector:
    """
    Detector for Fair Value Gaps in OHLCV price data.
    
    This class identifies FVGs based on price gaps and provides filtering
    based on sensitivity (gap size relative to candle size) and merging
    of consecutive FVGs.
    """
    
    def __init__(self, min_sensitivity: float = 0.1):
        """
        Initialize the FVG detector.
        
        Args:
            min_sensitivity: Minimum ratio of gap size to middle candle size
                           to consider a valid FVG (default: 0.1 = 10%)
        """
        self.min_sensitivity = min_sensitivity
    
    def detect_fvgs(self, 
                    data: pd.DataFrame, 
                    merge_consecutive: bool = True) -> List[FVG]:
        """
        Detect all Fair Value Gaps in the given OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
            merge_consecutive: Whether to merge consecutive FVGs
            
        Returns:
            List of FVG objects sorted by start index
        """
        if not self._validate_data(data):
            return []
        
        fvgs = []
        
        # Need at least 3 candles to detect an FVG
        for i in range(1, len(data) - 1):
            # Check for bullish FVG: gap between prev high and next low
            bullish_fvg = self._detect_bullish_fvg(data, i)
            if bullish_fvg:
                fvgs.append(bullish_fvg)
            
            # Check for bearish FVG: gap between prev low and next high
            bearish_fvg = self._detect_bearish_fvg(data, i)
            if bearish_fvg:
                fvgs.append(bearish_fvg)
        
        # Sort by start index
        fvgs.sort(key=lambda x: x.start_idx)
        
        # Merge consecutive FVGs if requested
        if merge_consecutive and fvgs:
            fvgs = self._merge_consecutive_fvgs(fvgs)
        
        return fvgs
    
    def _detect_bullish_fvg(self, data: pd.DataFrame, i: int) -> Optional[FVG]:
        """
        Detect bullish FVG at position i.
        
        A bullish FVG occurs when high[i-1] < low[i+1], creating an upward gap.
        """
        prev_high = data.iloc[i-1]['high']
        curr_open = data.iloc[i]['open']
        curr_close = data.iloc[i]['close']
        curr_high = data.iloc[i]['high']
        curr_low = data.iloc[i]['low']
        next_low = data.iloc[i+1]['low']
        
        # Check if there's a gap (bullish FVG condition)
        if prev_high < next_low:
            gap_top = next_low
            gap_bottom = prev_high
            gap_size = gap_top - gap_bottom
            
            # Calculate middle candle size for sensitivity check
            middle_candle_size = curr_high - curr_low
            if middle_candle_size == 0:
                return None
            
            sensitivity_ratio = gap_size / middle_candle_size
            
            # Apply sensitivity filter
            if sensitivity_ratio >= self.min_sensitivity:
                return FVG(
                    start_idx=i-1,
                    end_idx=i+1,
                    top=gap_top,
                    bottom=gap_bottom,
                    fvg_type='bullish',
                    gap_size=gap_size,
                    sensitivity_ratio=sensitivity_ratio
                )
        
        return None
    
    def _detect_bearish_fvg(self, data: pd.DataFrame, i: int) -> Optional[FVG]:
        """
        Detect bearish FVG at position i.
        
        A bearish FVG occurs when low[i-1] > high[i+1], creating a downward gap.
        """
        prev_low = data.iloc[i-1]['low']
        curr_open = data.iloc[i]['open']
        curr_close = data.iloc[i]['close']
        curr_high = data.iloc[i]['high']
        curr_low = data.iloc[i]['low']
        next_high = data.iloc[i+1]['high']
        
        # Check if there's a gap (bearish FVG condition)
        if prev_low > next_high:
            gap_top = prev_low
            gap_bottom = next_high
            gap_size = gap_top - gap_bottom
            
            # Calculate middle candle size for sensitivity check
            middle_candle_size = curr_high - curr_low
            if middle_candle_size == 0:
                return None
            
            sensitivity_ratio = gap_size / middle_candle_size
            
            # Apply sensitivity filter
            if sensitivity_ratio >= self.min_sensitivity:
                return FVG(
                    start_idx=i-1,
                    end_idx=i+1,
                    top=gap_top,
                    bottom=gap_bottom,
                    fvg_type='bearish',
                    gap_size=gap_size,
                    sensitivity_ratio=sensitivity_ratio
                )
        
        return None
    
    def _merge_consecutive_fvgs(self, fvgs: List[FVG]) -> List[FVG]:
        """
        Merge consecutive FVGs of the same type that overlap or are adjacent.
        
        Args:
            fvgs: List of FVG objects sorted by start_idx
            
        Returns:
            List of FVG objects with consecutive ones merged
        """
        if len(fvgs) <= 1:
            return fvgs
        
        merged_fvgs = []
        current_fvg = fvgs[0]
        
        for i in range(1, len(fvgs)):
            next_fvg = fvgs[i]
            
            # Check if FVGs can be merged:
            # 1. Same type (bullish/bearish)
            # 2. Adjacent or overlapping indices
            # 3. Overlapping price ranges
            if (current_fvg.fvg_type == next_fvg.fvg_type and
                self._can_merge_fvgs(current_fvg, next_fvg)):
                
                # Merge the FVGs
                current_fvg = self._merge_two_fvgs(current_fvg, next_fvg)
            else:
                # Can't merge, add current and move to next
                merged_fvgs.append(current_fvg)
                current_fvg = next_fvg
        
        # Add the last FVG
        merged_fvgs.append(current_fvg)
        
        return merged_fvgs
    
    def _can_merge_fvgs(self, fvg1: FVG, fvg2: FVG) -> bool:
        """
        Check if two FVGs can be merged.
        
        Args:
            fvg1: First FVG
            fvg2: Second FVG (should have start_idx >= fvg1.start_idx)
            
        Returns:
            True if FVGs can be merged, False otherwise
        """
        # Check if they're adjacent or overlapping in time
        indices_overlap = fvg2.start_idx <= fvg1.end_idx + 1
        
        # Check if their price ranges overlap
        if fvg1.fvg_type == 'bullish':
            # For bullish FVGs, check if ranges overlap
            price_overlap = (fvg1.bottom <= fvg2.top and fvg2.bottom <= fvg1.top)
        else:
            # For bearish FVGs, check if ranges overlap
            price_overlap = (fvg1.bottom <= fvg2.top and fvg2.bottom <= fvg1.top)
        
        return indices_overlap and price_overlap
    
    def _merge_two_fvgs(self, fvg1: FVG, fvg2: FVG) -> FVG:
        """
        Merge two FVGs into one.
        
        Args:
            fvg1: First FVG
            fvg2: Second FVG
            
        Returns:
            Merged FVG
        """
        # Combine index ranges
        start_idx = min(fvg1.start_idx, fvg2.start_idx)
        end_idx = max(fvg1.end_idx, fvg2.end_idx)
        
        # Combine price ranges
        top = max(fvg1.top, fvg2.top)
        bottom = min(fvg1.bottom, fvg2.bottom)
        gap_size = top - bottom
        
        # Use the higher sensitivity ratio
        sensitivity_ratio = max(fvg1.sensitivity_ratio, fvg2.sensitivity_ratio)
        
        return FVG(
            start_idx=start_idx,
            end_idx=end_idx,
            top=top,
            bottom=bottom,
            fvg_type=fvg1.fvg_type,  # Same type since we checked
            gap_size=gap_size,
            sensitivity_ratio=sensitivity_ratio,
            is_merged=True
        )
    
    def get_active_fvgs(self, 
                       fvgs: List[FVG], 
                       current_price: float) -> List[FVG]:
        """
        Get FVGs that haven't been filled by current price.
        
        Args:
            fvgs: List of FVG objects
            current_price: Current market price
            
        Returns:
            List of unfilled FVGs
        """
        active_fvgs = []
        
        for fvg in fvgs:
            # FVG is still active if price hasn't completely filled the gap
            if fvg.bottom <= current_price <= fvg.top:
                # Price is within the gap - partially filled but still active
                active_fvgs.append(fvg)
            elif fvg.fvg_type == 'bullish' and current_price < fvg.bottom:
                # Bullish FVG not yet reached
                active_fvgs.append(fvg)
            elif fvg.fvg_type == 'bearish' and current_price > fvg.top:
                # Bearish FVG not yet reached
                active_fvgs.append(fvg)
        
        return active_fvgs
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or len(data) < 3:
            return False
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for NaN values in required columns
        if data[required_columns].isnull().any().any():
            return False
        
        return True
    
    def filter_by_sensitivity(self, 
                             fvgs: List[FVG], 
                             min_sensitivity: float) -> List[FVG]:
        """
        Filter FVGs by sensitivity ratio.
        
        Args:
            fvgs: List of FVG objects
            min_sensitivity: Minimum sensitivity ratio
            
        Returns:
            Filtered list of FVGs
        """
        return [fvg for fvg in fvgs if fvg.sensitivity_ratio >= min_sensitivity]
    
    def get_fvgs_summary(self, fvgs: List[FVG]) -> Dict:
        """
        Get summary statistics for detected FVGs.
        
        Args:
            fvgs: List of FVG objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not fvgs:
            return {
                'total_fvgs': 0,
                'bullish_fvgs': 0,
                'bearish_fvgs': 0,
                'merged_fvgs': 0,
                'avg_gap_size': 0,
                'avg_sensitivity': 0
            }
        
        bullish_count = sum(1 for fvg in fvgs if fvg.fvg_type == 'bullish')
        bearish_count = sum(1 for fvg in fvgs if fvg.fvg_type == 'bearish')
        merged_count = sum(1 for fvg in fvgs if fvg.is_merged)
        
        avg_gap_size = np.mean([fvg.gap_size for fvg in fvgs])
        avg_sensitivity = np.mean([fvg.sensitivity_ratio for fvg in fvgs])
        
        return {
            'total_fvgs': len(fvgs),
            'bullish_fvgs': bullish_count,
            'bearish_fvgs': bearish_count,
            'merged_fvgs': merged_count,
            'avg_gap_size': avg_gap_size,
            'avg_sensitivity': avg_sensitivity
        }