"""Simple wrapper for candlestick patterns detection."""
import pandas as pd
from typing import Optional, Dict, List
from candlestick import candlestick
from framework.utils.logger import setup_logger

logger = setup_logger()


class CandlestickPatternDetector:
    """Simple wrapper for candlestick pattern detection."""
    
    def __init__(self):
        """Initialize Candlestick Pattern Detector."""
        # Available patterns from the library
        self.patterns = {
            'hammer': candlestick.hammer,
            'hanging_man': candlestick.hanging_man,
            'inverted_hammer': candlestick.inverted_hammer,
            'shooting_star': candlestick.shooting_star,
            'doji': candlestick.doji,
            'gravestone_doji': candlestick.gravestone_doji,
            'dragonfly_doji': candlestick.dragonfly_doji,
            'doji_star': candlestick.doji_star,
            'bullish_engulfing': candlestick.bullish_engulfing,
            'bearish_engulfing': candlestick.bearish_engulfing,
            'bullish_harami': candlestick.bullish_harami,
            'bearish_harami': candlestick.bearish_harami,
            'piercing_pattern': candlestick.piercing_pattern,
            'dark_cloud_cover': candlestick.dark_cloud_cover,
            'morning_star': candlestick.morning_star,
            'morning_star_doji': candlestick.morning_star_doji,
            'rain_drop': candlestick.rain_drop,
            'rain_drop_doji': candlestick.rain_drop_doji,
            'star': candlestick.star
        }
    
    def detect_patterns(self, 
                       df: pd.DataFrame,
                       patterns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Detect candlestick patterns in the data.
        
        Args:
            df: DataFrame with OHLCV data (must have open, high, low, close columns)
            patterns: List of pattern names to detect. If None, detect all patterns
            
        Returns:
            Dictionary with pattern names as keys and DataFrames with detection results as values
        """
        if patterns is None:
            patterns = list(self.patterns.keys())
        
        # Prepare OHLC data
        ohlc_df = df[['open', 'high', 'low', 'close']].copy()
        
        detected_patterns = {}
        
        for pattern_name in patterns:
            if pattern_name not in self.patterns:
                logger.warning(f"Unknown pattern: {pattern_name}")
                continue
                
            try:
                pattern_func = self.patterns[pattern_name]
                result_df = pattern_func(ohlc_df)
                detected_patterns[pattern_name] = result_df
                
            except Exception as e:
                logger.error(f"Error detecting pattern {pattern_name}: {e}")
                detected_patterns[pattern_name] = pd.DataFrame()
        
        return detected_patterns
    
    def get_pattern_matches(self, 
                           df: pd.DataFrame,
                           patterns: Optional[List[str]] = None) -> Dict[str, List[int]]:
        """
        Get indices where patterns are detected.
        
        Args:
            df: DataFrame with OHLCV data
            patterns: List of pattern names to detect
            
        Returns:
            Dictionary with pattern names as keys and lists of matching indices as values
        """
        detected = self.detect_patterns(df, patterns)
        matches = {}
        
        for pattern_name, result_df in detected.items():
            if result_df.empty:
                matches[pattern_name] = []
                continue
            
            # Find the pattern column (usually the last column)
            pattern_col = result_df.columns[-1]
            
            # Get indices where pattern is True
            pattern_indices = []
            for i, is_pattern in enumerate(result_df[pattern_col]):
                if is_pattern:
                    pattern_indices.append(i)
            
            matches[pattern_name] = pattern_indices
        
        return matches
    
    def list_available_patterns(self) -> List[str]:
        """Get list of available patterns."""
        return list(self.patterns.keys())