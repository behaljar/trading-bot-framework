"""Volume Profile utility for trading strategies."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import volprofile
from framework.utils.logger import setup_logger

logger = setup_logger()


class VolumeProfileAnalyzer:
    """Analyzes volume profile data for trading strategies."""
    
    def __init__(self, 
                 n_bins: int = 20,
                 range_type: str = 'auto',
                 custom_range: Optional[Tuple[float, float]] = None):
        """
        Initialize Volume Profile Analyzer.
        
        Args:
            n_bins: Number of price bins for volume profile
            range_type: 'auto' for automatic range or 'custom' for custom range
            custom_range: (min_price, max_price) tuple if range_type is 'custom'
        """
        self.n_bins = n_bins
        self.range_type = range_type
        self.custom_range = custom_range
        
    def calculate_volume_profile(self, 
                                df: pd.DataFrame,
                                price_col: str = 'close',
                                volume_col: str = 'volume') -> Dict[str, Union[pd.DataFrame, float]]:
        """
        Calculate volume profile for given price and volume data.
        
        Args:
            df: DataFrame with price and volume data
            price_col: Column name for price data
            volume_col: Column name for volume data
            
        Returns:
            Dictionary containing:
                - profile: DataFrame with price bins and volume
                - poc: Point of Control (price with highest volume)
                - vah: Value Area High
                - val: Value Area Low
                - value_area_volume_pct: Percentage of volume in value area
        """
        try:
            # Prepare data
            prices = df[price_col].values
            volumes = df[volume_col].values
            
            # Determine price range
            if self.range_type == 'custom' and self.custom_range:
                min_price, max_price = self.custom_range
            else:
                min_price, max_price = prices.min(), prices.max()
            
            # Create DataFrame with required columns for volprofile
            vp_input_df = pd.DataFrame({
                'price': prices,
                'volume': volumes
            })
            
            # Create volume profile using volprofile library
            vp_result = volprofile.getVP(vp_input_df, nBins=self.n_bins)
            
            # Extract profile data - use midpoint of price range
            profile_df = pd.DataFrame({
                'price': (vp_result['minPrice'] + vp_result['maxPrice']) / 2,
                'volume': vp_result['aggregateVolume']
            })
            
            # Calculate key levels
            total_volume = profile_df['volume'].sum()
            
            # Find POC (Point of Control - price with highest volume)
            poc_idx = profile_df['volume'].idxmax()
            poc = profile_df.loc[poc_idx, 'price']
            
            # Calculate Value Area (70% of volume)
            target_volume = total_volume * 0.7
            
            # Sort by volume descending
            sorted_profile = profile_df.sort_values('volume', ascending=False)
            cumulative_volume = 0
            value_area_prices = []
            
            for idx, row in sorted_profile.iterrows():
                cumulative_volume += row['volume']
                value_area_prices.append(row['price'])
                if cumulative_volume >= target_volume:
                    break
            
            val = min(value_area_prices)  # Value Area Low
            vah = max(value_area_prices)  # Value Area High
            
            # Calculate value area volume percentage
            total_volume = profile_df['volume'].sum()
            value_area_mask = (profile_df['price'] >= val) & (profile_df['price'] <= vah)
            value_area_volume = profile_df.loc[value_area_mask, 'volume'].sum()
            value_area_volume_pct = (value_area_volume / total_volume) * 100 if total_volume > 0 else 0
            
            return {
                'profile': profile_df,
                'poc': poc,
                'vah': vah,
                'val': val,
                'value_area_volume_pct': value_area_volume_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            raise
            
    def identify_support_resistance(self, 
                                   profile_df: pd.DataFrame,
                                   volume_threshold_pct: float = 5.0) -> List[float]:
        """
        Identify support/resistance levels from volume profile.
        
        Args:
            profile_df: DataFrame with price and volume columns
            volume_threshold_pct: Minimum volume percentage to consider as S/R level
            
        Returns:
            List of price levels that act as support/resistance
        """
        total_volume = profile_df['volume'].sum()
        if total_volume == 0:
            return []
            
        # Find high volume nodes
        profile_df['volume_pct'] = (profile_df['volume'] / total_volume) * 100
        high_volume_nodes = profile_df[profile_df['volume_pct'] >= volume_threshold_pct]
        
        return high_volume_nodes['price'].tolist()
        
    def calculate_volume_weighted_levels(self,
                                       df: pd.DataFrame,
                                       lookback_periods: int = 20) -> Dict[str, float]:
        """
        Calculate volume-weighted price levels over a lookback period.
        
        Args:
            df: DataFrame with OHLCV data
            lookback_periods: Number of periods to look back
            
        Returns:
            Dictionary with VWAP and other volume-weighted metrics
        """
        # Get recent data
        recent_df = df.tail(lookback_periods).copy()
        
        # Calculate VWAP
        typical_price = (recent_df['high'] + recent_df['low'] + recent_df['close']) / 3
        vwap = (typical_price * recent_df['volume']).sum() / recent_df['volume'].sum()
        
        # Calculate volume-weighted high and low
        vw_high = (recent_df['high'] * recent_df['volume']).sum() / recent_df['volume'].sum()
        vw_low = (recent_df['low'] * recent_df['volume']).sum() / recent_df['volume'].sum()
        
        # Calculate volume-weighted standard deviation
        vw_variance = ((typical_price - vwap) ** 2 * recent_df['volume']).sum() / recent_df['volume'].sum()
        vw_std = np.sqrt(vw_variance)
        
        return {
            'vwap': vwap,
            'vw_high': vw_high,
            'vw_low': vw_low,
            'vw_std': vw_std,
            'vwap_upper_band': vwap + 2 * vw_std,
            'vwap_lower_band': vwap - 2 * vw_std
        }
        
    def analyze_volume_distribution(self, 
                                  df: pd.DataFrame,
                                  current_price: float) -> Dict[str, Union[float, str]]:
        """
        Analyze volume distribution relative to current price.
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            
        Returns:
            Dictionary with analysis results
        """
        vp_data = self.calculate_volume_profile(df)
        profile_df = vp_data['profile']
        
        # Calculate volume above and below current price
        above_mask = profile_df['price'] > current_price
        below_mask = profile_df['price'] < current_price
        
        total_volume = profile_df['volume'].sum()
        volume_above = profile_df.loc[above_mask, 'volume'].sum()
        volume_below = profile_df.loc[below_mask, 'volume'].sum()
        
        # Determine market bias
        if total_volume > 0:
            volume_above_pct = (volume_above / total_volume) * 100
            volume_below_pct = (volume_below / total_volume) * 100
            
            if volume_above_pct > 60:
                bias = 'bearish'
            elif volume_below_pct > 60:
                bias = 'bullish'
            else:
                bias = 'neutral'
        else:
            volume_above_pct = 0
            volume_below_pct = 0
            bias = 'neutral'
            
        # Calculate distance from key levels
        poc_distance_pct = ((current_price - vp_data['poc']) / vp_data['poc']) * 100
        vah_distance_pct = ((current_price - vp_data['vah']) / vp_data['vah']) * 100
        val_distance_pct = ((current_price - vp_data['val']) / vp_data['val']) * 100
        
        return {
            'volume_above_pct': volume_above_pct,
            'volume_below_pct': volume_below_pct,
            'market_bias': bias,
            'poc_distance_pct': poc_distance_pct,
            'vah_distance_pct': vah_distance_pct,
            'val_distance_pct': val_distance_pct,
            'in_value_area': vp_data['val'] <= current_price <= vp_data['vah']
        }
        
    def get_trading_signals(self,
                          df: pd.DataFrame,
                          current_price: float,
                          volume_threshold: float = 0.7) -> Dict[str, Union[str, float]]:
        """
        Generate trading signals based on volume profile analysis.
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            volume_threshold: Threshold for value area volume percentage
            
        Returns:
            Dictionary with signal and reasoning
        """
        try:
            vp_data = self.calculate_volume_profile(df)
            analysis = self.analyze_volume_distribution(df, current_price)
            vw_levels = self.calculate_volume_weighted_levels(df)
            
            signal = 'HOLD'
            confidence = 0.0
            reasons = []
            
            # Check if in value area
            if analysis['in_value_area']:
                # Price within value area - look for breakout
                if current_price > vp_data['poc'] and analysis['market_bias'] == 'bullish':
                    signal = 'BUY'
                    confidence = 0.6
                    reasons.append("Price above POC with bullish volume distribution")
                elif current_price < vp_data['poc'] and analysis['market_bias'] == 'bearish':
                    signal = 'SELL'
                    confidence = 0.6
                    reasons.append("Price below POC with bearish volume distribution")
            else:
                # Price outside value area
                if current_price > vp_data['vah']:
                    # Above value area high
                    if analysis['volume_above_pct'] > 30:
                        signal = 'BUY'
                        confidence = 0.7
                        reasons.append("Breakout above VAH with volume confirmation")
                    else:
                        signal = 'SELL'
                        confidence = 0.5
                        reasons.append("Price extended above VAH without volume support")
                elif current_price < vp_data['val']:
                    # Below value area low
                    if analysis['volume_below_pct'] > 30:
                        signal = 'SELL'
                        confidence = 0.7
                        reasons.append("Breakdown below VAL with volume confirmation")
                    else:
                        signal = 'BUY'
                        confidence = 0.5
                        reasons.append("Price oversold below VAL without volume pressure")
                        
            # Add VWAP confirmation
            if current_price > vw_levels['vwap'] and signal == 'BUY':
                confidence += 0.1
                reasons.append("Price above VWAP confirms bullish bias")
            elif current_price < vw_levels['vwap'] and signal == 'SELL':
                confidence += 0.1
                reasons.append("Price below VWAP confirms bearish bias")
                
            # Check for extreme conditions
            if current_price > vw_levels['vwap_upper_band']:
                if signal == 'BUY':
                    confidence -= 0.2
                    reasons.append("Price at upper VWAP band - overbought warning")
            elif current_price < vw_levels['vwap_lower_band']:
                if signal == 'SELL':
                    confidence -= 0.2
                    reasons.append("Price at lower VWAP band - oversold warning")
                    
            return {
                'signal': signal,
                'confidence': min(max(confidence, 0.0), 1.0),
                'reasons': reasons,
                'poc': vp_data['poc'],
                'vah': vp_data['vah'],
                'val': vp_data['val'],
                'vwap': vw_levels['vwap'],
                'value_area_volume_pct': vp_data['value_area_volume_pct']
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasons': [f"Error in analysis: {str(e)}"],
                'poc': 0.0,
                'vah': 0.0,
                'val': 0.0,
                'vwap': 0.0,
                'value_area_volume_pct': 0.0
            }