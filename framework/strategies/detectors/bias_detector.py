#!/usr/bin/env python3
"""
Bias Detector

 This is a three-candle pattern detector that determines market bias based on candle relationships:

  Core Logic:

  - C1: Candle from 2 periods ago (T-2)
  - C2: Previous candle (T-1)
  - C3: Current candle (T)

  Bias Determination Rules (based on C2 vs C1):

  1. BULLISH: C2 closes above C1's high
    - Target: C2's high must be claimed by C3
  2. BEARISH: C2 closes below C1's low
    - Target: C2's low must be claimed by C3
  3. Inside Bar (IB): C2 closes inside C1's range with multiple subcases:
    - If C2 high > C1 high → BEARISH (until C2 low claimed)
    - If C2 low < C1 low → BULLISH (until C2 high claimed)
    - If C2 completely inside C1 → NO_BIAS
    - Otherwise → Inherit previous bias (IB)

"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass


class BiasType(Enum):
    """Enumeration for bias types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    IB = "ib"  # Inside Bar / Consolidation
    NO_BIAS = "no_bias"


@dataclass
class BiasSignal:
    """Data class representing a bias signal"""
    timestamp: pd.Timestamp
    bias: BiasType
    c1_high: float
    c1_low: float
    c2_high: float
    c2_low: float
    c2_close: float
    target_high: Optional[float] = None  # C2 high for bullish bias
    target_low: Optional[float] = None   # C2 low for bearish bias
    validation: Optional[bool] = None    # True if bias validated by C3
    validation_reason: Optional[str] = None


class BiasDetector:
    """
    Bias Detector based on three-candle relationships
    
    Logic:
    C1 = Candle at T-2 (two candles ago)
    C2 = Candle at T-1 (previous candle)  
    C3 = Candle at T (current candle)
    
    Bias Rules based on C2 position relative to C1:
    1. C2 close above C1: BULLISH (until C2 high claimed)
    2. C2 close below C1: BEARISH (until C2 low claimed)
    3. C2 close inside C1: Multiple sub-cases
    
    Validation Rules:
    - Bullish bias: C3 high > C2 high = True
    - Bearish bias: C3 low < C2 low = True
    """
    
    def __init__(self, lookback_periods: int = 100):
        """
        Initialize bias detector
        
        Args:
            lookback_periods: Number of periods to look back for bias history
        """
        self.lookback_periods = lookback_periods
        self.bias_history: List[BiasSignal] = []
    
    def _determine_bias(self, c1_high: float, c1_low: float, c1_close: float,
                       c2_high: float, c2_low: float, c2_close: float,
                       previous_bias: BiasType = BiasType.NO_BIAS) -> Tuple[BiasType, Optional[float], Optional[float]]:
        """
        Determine bias based on candle relationships
        
        Args:
            c1_high, c1_low, c1_close: C1 candle OHLC
            c2_high, c2_low, c2_close: C2 candle OHLC  
            previous_bias: Previous bias for IB cases
            
        Returns:
            Tuple of (bias_type, target_high, target_low)
        """
        
        # Case 1: C2 close above C1 range - BULLISH
        if c2_close > c1_high:
            return BiasType.BULLISH, c2_high, None
            
        # Case 2: C2 close below C1 range - BEARISH  
        elif c2_close < c1_low:
            return BiasType.BEARISH, None, c2_low
            
        # Case 3: C2 close inside C1 range - Multiple subcases
        else:
            # Subcase 3a: C2 high above C1 high - BEARISH until C2 low claimed
            if c2_high > c1_high:
                return BiasType.BEARISH, None, c2_low
                
            # Subcase 3b: C2 low below C1 low - BULLISH until C2 high claimed  
            elif c2_low < c1_low:
                return BiasType.BULLISH, c2_high, None
                
            # Subcase 3c: C2 completely inside C1 - NO_BIAS
            elif c1_high > c2_high and c1_low < c2_low:
                return BiasType.NO_BIAS, None, None
                
            # Subcase 3d: Other inside cases - Copy previous bias (IB)
            else:
                if previous_bias in [BiasType.BULLISH, BiasType.BEARISH]:
                    # Maintain previous bias targets
                    if previous_bias == BiasType.BULLISH:
                        return BiasType.IB, c2_high, None
                    else:
                        return BiasType.IB, None, c2_low
                else:
                    return BiasType.IB, None, None
    
    def _validate_bias(self, bias_signal: BiasSignal, c3_high: float, c3_low: float) -> Tuple[bool, str]:
        """
        Validate bias signal against C3 candle behavior
        
        Args:
            bias_signal: The bias signal to validate
            c3_high, c3_low: C3 candle high and low
            
        Returns:
            Tuple of (validation_result, reason)
        """
        if bias_signal.bias == BiasType.BULLISH:
            if c3_high > bias_signal.c2_high:
                return True, f"C3 high ({c3_high:.4f}) > C2 high ({bias_signal.c2_high:.4f})"
            else:
                return False, f"C3 high ({c3_high:.4f}) <= C2 high ({bias_signal.c2_high:.4f})"
                
        elif bias_signal.bias == BiasType.BEARISH:
            if c3_low < bias_signal.c2_low:
                return True, f"C3 low ({c3_low:.4f}) < C2 low ({bias_signal.c2_low:.4f})"
            else:
                return False, f"C3 low ({c3_low:.4f}) >= C2 low ({bias_signal.c2_low:.4f})"
                
        else:
            # IB and NO_BIAS cases - no validation criteria
            return None, "No validation criteria for this bias type"
    
    def detect_bias_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect bias signals for entire dataset
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with bias signals and validation results
        """
        if len(data) < 3:
            raise ValueError("Need at least 3 candles for bias detection")
        
        results = []
        previous_bias = BiasType.NO_BIAS
        
        # Process from index 2 onwards (need C1, C2 to determine C3 bias)
        for i in range(2, len(data)):
            # Get candle data
            c1_idx = i - 2  # T-2
            c2_idx = i - 1  # T-1  
            c3_idx = i      # T (current)
            
            c1_data = data.iloc[c1_idx]
            c2_data = data.iloc[c2_idx]
            c3_data = data.iloc[c3_idx]
            
            # Determine bias based on C1 and C2
            bias_type, target_high, target_low = self._determine_bias(
                c1_data['High'], c1_data['Low'], c1_data['Close'],
                c2_data['High'], c2_data['Low'], c2_data['Close'],
                previous_bias
            )
            
            # Create bias signal
            bias_signal = BiasSignal(
                timestamp=data.index[c3_idx],
                bias=bias_type,
                c1_high=c1_data['High'],
                c1_low=c1_data['Low'],
                c2_high=c2_data['High'],
                c2_low=c2_data['Low'],
                c2_close=c2_data['Close'],
                target_high=target_high,
                target_low=target_low
            )
            
            # Validate bias if applicable
            if bias_type in [BiasType.BULLISH, BiasType.BEARISH]:
                validation, reason = self._validate_bias(bias_signal, c3_data['High'], c3_data['Low'])
                bias_signal.validation = validation
                bias_signal.validation_reason = reason
            
            results.append({
                'timestamp': bias_signal.timestamp,
                'bias': bias_signal.bias.value,
                'c1_high': bias_signal.c1_high,
                'c1_low': bias_signal.c1_low,
                'c2_high': bias_signal.c2_high,
                'c2_low': bias_signal.c2_low,
                'c2_close': bias_signal.c2_close,
                'c3_high': c3_data['High'],
                'c3_low': c3_data['Low'],
                'target_high': bias_signal.target_high,
                'target_low': bias_signal.target_low,
                'validation': bias_signal.validation,
                'validation_reason': bias_signal.validation_reason
            })
            
            # Update previous bias for next iteration
            if bias_type != BiasType.IB:
                previous_bias = bias_type
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)
        
        return result_df
    
    def calculate_accuracy(self, bias_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate accuracy statistics for bias predictions
        
        Args:
            bias_df: DataFrame from detect_bias_signals()
            
        Returns:
            Dictionary with accuracy statistics
        """
        if bias_df.empty:
            return {"error": "No bias signals to evaluate"}
        
        # Filter for validated bias signals only
        validated_signals = bias_df[bias_df['validation'].notna()]
        
        if validated_signals.empty:
            return {"error": "No validated bias signals found"}
        
        # Calculate overall accuracy
        total_signals = len(validated_signals)
        correct_signals = len(validated_signals[validated_signals['validation'] == True])
        overall_accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # Calculate accuracy by bias type
        bullish_signals = validated_signals[validated_signals['bias'] == 'bullish']
        bearish_signals = validated_signals[validated_signals['bias'] == 'bearish']
        
        bullish_accuracy = 0
        bearish_accuracy = 0
        
        if len(bullish_signals) > 0:
            bullish_correct = len(bullish_signals[bullish_signals['validation'] == True])
            bullish_accuracy = (bullish_correct / len(bullish_signals)) * 100
            
        if len(bearish_signals) > 0:
            bearish_correct = len(bearish_signals[bearish_signals['validation'] == True])
            bearish_accuracy = (bearish_correct / len(bearish_signals)) * 100
        
        # Bias type distribution
        bias_counts = bias_df['bias'].value_counts()
        
        return {
            'total_signals': len(bias_df),
            'validated_signals': total_signals,
            'correct_predictions': correct_signals,
            'overall_accuracy_pct': round(overall_accuracy, 2),
            'bullish_signals': len(bullish_signals),
            'bullish_accuracy_pct': round(bullish_accuracy, 2),
            'bearish_signals': len(bearish_signals), 
            'bearish_accuracy_pct': round(bearish_accuracy, 2),
            'bias_distribution': bias_counts.to_dict(),
            'sample_validations': validated_signals[['bias', 'validation', 'validation_reason']].head(10).to_dict('records')
        }
    
    def get_current_bias(self, data: pd.DataFrame) -> Optional[BiasSignal]:
        """
        Get the current bias for the most recent candle
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            BiasSignal for the most recent period or None
        """
        if len(data) < 3:
            return None
            
        bias_df = self.detect_bias_signals(data)
        if bias_df.empty:
            return None
            
        # Get most recent bias
        latest = bias_df.iloc[-1]
        
        return BiasSignal(
            timestamp=latest.name,
            bias=BiasType(latest['bias']),
            c1_high=latest['c1_high'],
            c1_low=latest['c1_low'],
            c2_high=latest['c2_high'],
            c2_low=latest['c2_low'],
            c2_close=latest['c2_close'],
            target_high=latest['target_high'],
            target_low=latest['target_low'],
            validation=latest['validation'],
            validation_reason=latest['validation_reason']
        )