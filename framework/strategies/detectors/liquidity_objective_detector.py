"""
Liquidity Objective Detector

This module combines all available detectors to identify potential liquidity objectives
based on trade direction. It finds unclaimed levels that could serve as targets:
- Unclaimed swing highs/lows
- Previous day/week high/low levels (unbroken)
- Fair Value Gaps (unfilled)

The detector analyzes the current market structure and provides ranked targets
in the specified trade direction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .swing_detector import SwingDetector
from .pdh_pdl_detector import PDHPDLDetector, HighLowLevel
from .fvg_detector import FVGDetector, FVG


class TradeDirection(Enum):
    """Trade direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class LiquidityObjective:
    """
    Represents a liquidity objective (potential target).
    
    Attributes:
        price: The target price level
        level_type: Type of level ('swing_high', 'swing_low', 'pdh', 'pdl', 'pwh', 'pwl', 'p4h', 'p4l', 'fvg')
        direction: Direction this objective serves ('bullish' or 'bearish')
        distance_pct: Distance from current price as percentage
        priority: Priority ranking (1=highest, lower numbers = higher priority)
        is_claimed: Whether this objective has been reached/claimed
        confidence: Confidence score (0.0-1.0) based on level strength
        details: Additional details about the objective
    """
    price: float
    level_type: str
    direction: str
    distance_pct: float
    priority: int
    is_claimed: bool = False
    confidence: float = 0.0
    details: Dict = None


class LiquidityObjectiveDetector:
    """
    Detector that combines all available detectors to find liquidity objectives.
    
    This detector analyzes:
    1. Swing highs/lows that haven't been broken
    2. Previous day/week/4-hour highs/lows that are unbroken
    3. Fair Value Gaps that haven't been filled
    
    It then ranks these based on distance, confidence, and trade direction.
    """
    
    def __init__(self, 
                 swing_sensitivity: int = 3,
                 fvg_min_sensitivity: float = 0.1,
                 max_objectives: int = 10):
        """
        Initialize the liquidity objective detector.
        
        Args:
            swing_sensitivity: Sensitivity for swing detection
            fvg_min_sensitivity: Minimum FVG sensitivity ratio (default: 0.1 = 10%)
            max_objectives: Maximum number of objectives to return
        """
        self.swing_detector = SwingDetector(sensitivity=swing_sensitivity)
        self.pdh_pdl_detector = PDHPDLDetector()
        self.fvg_detector = FVGDetector(min_sensitivity=fvg_min_sensitivity)
        self.max_objectives = max_objectives
        
    def detect_objectives(self, 
                         data: pd.DataFrame, 
                         trade_direction: Union[TradeDirection, str],
                         current_price: Optional[float] = None,
                         max_distance_pct: float = 0.10) -> List[LiquidityObjective]:
        """
        Detect liquidity objectives in the specified trade direction.
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            trade_direction: Direction to look for objectives ('bullish' or 'bearish')
            current_price: Current market price (uses last close if None)
            max_distance_pct: Maximum distance from current price (10% default)
            
        Returns:
            List of LiquidityObjective objects sorted by priority
        """
        if not self._validate_data(data):
            return []
        
        # Normalize trade direction
        if isinstance(trade_direction, str):
            trade_direction = TradeDirection(trade_direction.lower())
            
        if current_price is None:
            current_price = data['Close'].iloc[-1]
        
        objectives = []
        
        # 1. Get swing-based objectives
        swing_objectives = self._get_swing_objectives(data, trade_direction, current_price, max_distance_pct)
        objectives.extend(swing_objectives)
        
        # 2. Get PDH/PDL objectives
        pdh_pdl_objectives = self._get_pdh_pdl_objectives(data, trade_direction, current_price, max_distance_pct)
        objectives.extend(pdh_pdl_objectives)
        
        # 3. Get FVG objectives
        fvg_objectives = self._get_fvg_objectives(data, trade_direction, current_price, max_distance_pct)
        objectives.extend(fvg_objectives)
        
        # 4. Remove duplicates and rank objectives
        objectives = self._rank_and_filter_objectives(objectives, current_price)
        
        return objectives[:self.max_objectives]
    
    def _get_swing_objectives(self, 
                            data: pd.DataFrame, 
                            trade_direction: TradeDirection,
                            current_price: float,
                            max_distance_pct: float) -> List[LiquidityObjective]:
        """Get objectives from swing highs/lows."""
        objectives = []
        
        if trade_direction == TradeDirection.BULLISH:
            # Look for swing highs above current price
            swing_highs = self.swing_detector.find_all_swing_highs(data['High'])
            
            for idx, price in swing_highs:
                if price > current_price:
                    distance_pct = (price - current_price) / current_price
                    
                    if distance_pct <= max_distance_pct:
                        # Check if this swing high is still unclaimed
                        future_data = data.iloc[idx+1:]
                        is_claimed = not future_data.empty and (future_data['High'] > price).any()
                        
                        if not is_claimed:
                            confidence = self._calculate_swing_confidence(data, idx, True)
                            objectives.append(LiquidityObjective(
                                price=price,
                                level_type='swing_high',
                                direction='bullish',
                                distance_pct=distance_pct,
                                priority=0,  # Will be set later
                                is_claimed=False,
                                confidence=confidence,
                                details={'swing_index': idx, 'timestamp': data.index[idx]}
                            ))
        
        else:  # BEARISH
            # Look for swing lows below current price
            swing_lows = self.swing_detector.find_all_swing_lows(data['Low'])
            
            for idx, price in swing_lows:
                if price < current_price:
                    distance_pct = (current_price - price) / current_price
                    
                    if distance_pct <= max_distance_pct:
                        # Check if this swing low is still unclaimed
                        future_data = data.iloc[idx+1:]
                        is_claimed = not future_data.empty and (future_data['Low'] < price).any()
                        
                        if not is_claimed:
                            confidence = self._calculate_swing_confidence(data, idx, False)
                            objectives.append(LiquidityObjective(
                                price=price,
                                level_type='swing_low',
                                direction='bearish',
                                distance_pct=distance_pct,
                                priority=0,
                                is_claimed=False,
                                confidence=confidence,
                                details={'swing_index': idx, 'timestamp': data.index[idx]}
                            ))
        
        return objectives
    
    def _get_pdh_pdl_objectives(self, 
                               data: pd.DataFrame, 
                               trade_direction: TradeDirection,
                               current_price: float,
                               max_distance_pct: float) -> List[LiquidityObjective]:
        """Get objectives from PDH/PDL levels."""
        objectives = []
        levels = self.pdh_pdl_detector.detect_levels(data)
        
        level_mapping = {
            TradeDirection.BULLISH: [('PDH', levels['PDH']), ('PWH', levels['PWH']), ('P4H', levels['P4H'])],
            TradeDirection.BEARISH: [('PDL', levels['PDL']), ('PWL', levels['PWL']), ('P4L', levels['P4L'])]
        }
        
        for level_name, level in level_mapping[trade_direction]:
            if level is None:
                continue
                
            if trade_direction == TradeDirection.BULLISH and level.price > current_price:
                distance_pct = (level.price - current_price) / current_price
            elif trade_direction == TradeDirection.BEARISH and level.price < current_price:
                distance_pct = (current_price - level.price) / current_price
            else:
                continue  # Level is in wrong direction
            
            if distance_pct <= max_distance_pct and not level.is_broken:
                # Set confidence based on level type: 4H > Daily > Weekly
                if level_name in ['P4H', 'P4L']:
                    confidence = 0.9  # 4-hour levels highest confidence (most recent)
                elif level_name in ['PDH', 'PDL']:
                    confidence = 0.8  # Daily levels high confidence
                else:  # PWH, PWL
                    confidence = 0.6  # Weekly levels lower confidence
                
                objectives.append(LiquidityObjective(
                    price=level.price,
                    level_type=level_name.lower(),
                    direction=trade_direction.value,
                    distance_pct=distance_pct,
                    priority=0,
                    is_claimed=level.is_broken,
                    confidence=confidence,
                    details={'level_date': level.date, 'level_type': level.level_type}
                ))
        
        return objectives
    
    def _get_fvg_objectives(self, 
                           data: pd.DataFrame, 
                           trade_direction: TradeDirection,
                           current_price: float,
                           max_distance_pct: float) -> List[LiquidityObjective]:
        """Get objectives from Fair Value Gaps."""
        objectives = []
        fvgs = self.fvg_detector.detect_fvgs(data)
        
        for fvg in fvgs:
            # Determine target price (top for bullish, bottom for bearish)
            if trade_direction == TradeDirection.BULLISH and fvg.fvg_type == 'bullish':
                target_price = fvg.top
                if target_price > current_price:
                    distance_pct = (target_price - current_price) / current_price
                else:
                    continue
            elif trade_direction == TradeDirection.BEARISH and fvg.fvg_type == 'bearish':
                target_price = fvg.bottom
                if target_price < current_price:
                    distance_pct = (current_price - target_price) / current_price
                else:
                    continue
            else:
                continue  # FVG doesn't match trade direction
            
            if distance_pct <= max_distance_pct:
                confidence = min(0.7, fvg.sensitivity_ratio)  # Cap at 0.7, scale with gap size
                
                objectives.append(LiquidityObjective(
                    price=target_price,
                    level_type='fvg',
                    direction=trade_direction.value,
                    distance_pct=distance_pct,
                    priority=0,
                    is_claimed=False,  # FVG detector would need to implement fill detection
                    confidence=confidence,
                    details={
                        'fvg_type': fvg.fvg_type,
                        'gap_size': fvg.gap_size,
                        'start_idx': fvg.start_idx,
                        'end_idx': fvg.end_idx
                    }
                ))
        
        return objectives
    
    def _calculate_swing_confidence(self, data: pd.DataFrame, swing_idx: int, is_high: bool) -> float:
        """Calculate confidence score for a swing level."""
        if swing_idx < self.swing_detector.sensitivity or swing_idx >= len(data) - self.swing_detector.sensitivity:
            return 0.5  # Edge case, moderate confidence
        
        swing_price = data['High'].iloc[swing_idx] if is_high else data['Low'].iloc[swing_idx]
        
        # Look at surrounding bars to assess swing strength
        start_idx = max(0, swing_idx - self.swing_detector.sensitivity * 2)
        end_idx = min(len(data), swing_idx + self.swing_detector.sensitivity * 2)
        surrounding_data = data.iloc[start_idx:end_idx]
        
        if is_high:
            # For highs, check how much higher this swing is compared to surrounding highs
            other_highs = surrounding_data['High'].drop(surrounding_data.index[swing_idx - start_idx])
            if len(other_highs) > 0:
                price_diff = (swing_price - other_highs.max()) / swing_price
                confidence = min(1.0, 0.5 + price_diff * 10)  # Scale difference to confidence
            else:
                confidence = 0.5
        else:
            # For lows, check how much lower this swing is compared to surrounding lows
            other_lows = surrounding_data['Low'].drop(surrounding_data.index[swing_idx - start_idx])
            if len(other_lows) > 0:
                price_diff = (other_lows.min() - swing_price) / swing_price
                confidence = min(1.0, 0.5 + price_diff * 10)  # Scale difference to confidence
            else:
                confidence = 0.5
        
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
    
    def _rank_and_filter_objectives(self, 
                                   objectives: List[LiquidityObjective], 
                                   current_price: float) -> List[LiquidityObjective]:
        """Rank objectives by priority and remove duplicates."""
        if not objectives:
            return []
        
        # Remove near-duplicate objectives (within 0.1% of each other)
        filtered_objectives = []
        for obj in objectives:
            is_duplicate = False
            for existing in filtered_objectives:
                price_diff_pct = abs(obj.price - existing.price) / current_price
                if price_diff_pct < 0.001:  # Within 0.1%
                    # Keep the one with higher confidence
                    if obj.confidence > existing.confidence:
                        filtered_objectives.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_objectives.append(obj)
        
        # Calculate priority scores and sort
        for i, obj in enumerate(filtered_objectives):
            # Priority score combines distance (closer is better) and confidence (higher is better)
            distance_score = 1.0 - (obj.distance_pct * 2)  # Closer gets higher score
            confidence_score = obj.confidence
            
            # Weight the scores
            priority_score = (distance_score * 0.4) + (confidence_score * 0.6)
            obj.priority = int((1.0 - priority_score) * 100)  # Lower number = higher priority
        
        # Sort by priority (lower number = higher priority)
        filtered_objectives.sort(key=lambda x: x.priority)
        
        return filtered_objectives
    
    def get_summary(self, objectives: List[LiquidityObjective], current_price: float) -> Dict:
        """
        Get summary of detected liquidity objectives.
        
        Args:
            objectives: List of detected objectives
            current_price: Current market price
            
        Returns:
            Dictionary with summary statistics
        """
        if not objectives:
            return {
                'total_objectives': 0,
                'avg_distance_pct': 0,
                'closest_objective': None,
                'highest_confidence': None,
                'level_types': {}
            }
        
        level_type_counts = {}
        distances = []
        confidences = []
        
        for obj in objectives:
            level_type_counts[obj.level_type] = level_type_counts.get(obj.level_type, 0) + 1
            distances.append(obj.distance_pct)
            confidences.append(obj.confidence)
        
        closest_obj = min(objectives, key=lambda x: x.distance_pct)
        highest_confidence_obj = max(objectives, key=lambda x: x.confidence)
        
        return {
            'total_objectives': len(objectives),
            'avg_distance_pct': np.mean(distances) * 100,
            'closest_objective': {
                'price': closest_obj.price,
                'distance_pct': closest_obj.distance_pct * 100,
                'level_type': closest_obj.level_type
            },
            'highest_confidence': {
                'price': highest_confidence_obj.price,
                'confidence': highest_confidence_obj.confidence,
                'level_type': highest_confidence_obj.level_type
            },
            'level_types': level_type_counts,
            'avg_confidence': np.mean(confidences)
        }
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        if data is None or len(data) < 10:
            return False
        
        required_columns = ['High', 'Low', 'Open', 'Close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        
        return True