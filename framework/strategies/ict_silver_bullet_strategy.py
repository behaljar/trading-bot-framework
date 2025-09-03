"""
ICT Silver Bullet Strategy

This strategy implements the ICT (Inner Circle Trader) Silver Bullet methodology
which focuses on trading during specific session windows when liquidity is most
likely to be targeted by institutional players.

Key Components:
1. Session-based trading windows (London Open, NY AM, NY PM)
2. Liquidity identification and sweep detection
3. Market structure shift confirmation
4. Fair Value Gap (FVG) entries
5. Risk management with proper R:R ratios

Trading Process:
1. Identify unclaimed liquidity levels
2. Wait for liquidity sweep during session window
3. Look for market structure shift (break of structure)
4. Enter on FVG formation in direction of structure shift
5. Target opposing liquidity with proper risk/reward
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
from typing import Dict, List, Optional, Tuple
from .base_strategy import BaseStrategy
from .detectors.fvg_detector import FVGDetector
from .detectors.ict_liquidity_detector import ICTLiquidityDetector, LiquidityLevel
from .detectors.bos_choch_unified_detector import BoSCHoCHDetector


class ICTSilverBulletStrategy(BaseStrategy):
    """
    ICT Silver Bullet Strategy implementation.
    """
    
    def __init__(self,
                 # Core parameters
                 risk_reward_ratio: float = 3.5,
                 max_hold_hours: int = 2,
                 position_size: float = 0.02,
                 
                 # Session windows
                 enable_london_open: bool = False,
                 enable_ny_open: bool = True,
                 enable_ny_pm: bool = False,
                 
                 # Liquidity parameters
                 min_liquidity_strength: int = 2,
                 liquidity_sweep_confirmation: bool = True,
                 max_liquidity_distance_pct: float = 0.05,  # 5% max distance to liquidity
                 
                 # Market structure parameters
                 require_structure_shift: bool = True,
                 structure_confirmation_candles: int = 2,
                 
                 # FVG parameters
                 min_fvg_size_pct: float = 0.0004,  # 0.04% minimum FVG size
                 fvg_confirmation_method: str = 'close',  # 'close' or 'wick'
                 
                 # Risk management
                 max_trades_per_session: int = 1,
                 use_trailing_stop: bool = False,
                 trailing_stop_pct: float = 0.01,
                 
                 **kwargs):
        """Initialize ICT Silver Bullet Strategy."""
        super().__init__("ict_silver_bullet", kwargs)
        
        # Core parameters
        self.risk_reward_ratio = risk_reward_ratio
        self.max_hold_hours = max_hold_hours
        self.position_size = position_size
        
        # Session configuration
        self.enable_london_open = enable_london_open
        self.enable_ny_open = enable_ny_open
        self.enable_ny_pm = enable_ny_pm
        
        # Liquidity parameters
        self.min_liquidity_strength = min_liquidity_strength
        self.liquidity_sweep_confirmation = liquidity_sweep_confirmation
        self.max_liquidity_distance_pct = max_liquidity_distance_pct
        
        # Market structure parameters
        self.require_structure_shift = require_structure_shift
        self.structure_confirmation_candles = structure_confirmation_candles
        
        # FVG parameters
        self.min_fvg_size_pct = min_fvg_size_pct
        self.fvg_confirmation_method = fvg_confirmation_method
        
        # Risk management
        self.max_trades_per_session = max_trades_per_session
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        
        # Initialize detectors
        self.fvg_detector = FVGDetector(min_sensitivity=min_fvg_size_pct)
        self.liquidity_detector = ICTLiquidityDetector()
        self.structure_detector = BoSCHoCHDetector(left_bars=3, right_bars=3)
        
        # Session windows in NY timezone (EST)
        self.session_windows = {}
        if enable_london_open:
            self.session_windows['london_open'] = (3, 4)
        if enable_ny_open:
            self.session_windows['ny_open'] = (10, 11)
        if enable_ny_pm:
            self.session_windows['ny_pm'] = (14, 15)
        
        # Track trades per session
        self.session_trades = {}
        self.ny_tz = pytz.timezone('America/New_York')
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ICT Silver Bullet trading signals."""
        
        if not self.validate_data(data):
            raise ValueError("Invalid input data format")
        
        # Initialize signal columns
        signals_df = data.copy()
        signals_df['signal'] = 0
        signals_df['position_size'] = 0.0
        signals_df['stop_loss'] = np.nan
        signals_df['take_profit'] = np.nan
        signals_df['exit_reason'] = ''
        
        # Need sufficient data for analysis
        min_required_candles = 200
        if len(data) < min_required_candles:
            return signals_df
        
        # Detect timeframe
        timeframe = self._detect_timeframe(data)
        if timeframe not in ['1min', '3min', '5min']:
            return signals_df  # Only trade on lower timeframes
        
        # Resample to higher timeframes for context
        h1_data = self._resample_to_hourly(data)
        m15_data = self._resample_to_15min(data)
        
        # Ensure data has timezone info for liquidity detection
        h1_data_tz = h1_data.copy()
        if h1_data_tz.index.tz is None:
            h1_data_tz.index = h1_data_tz.index.tz_localize('UTC')
        
        # Data may be normalized by backtesting.py but patterns/ratios are preserved
        
        # Use simplified session-based liquidity like the Pine Script
        # Instead of complex liquidity detection, use session highs/lows
        session_liquidity = self._detect_session_liquidity_levels(data)
        
        if not session_liquidity:
            return signals_df
        
        # Main trading loop
        session_checks = 0
        setup_checks = 0
        signals_found = 0
        
        for i in range(min_required_candles, len(data)):
            current_time = data.index[i]
            
            # Ensure current_time has timezone info
            if current_time.tz is None:
                current_time = current_time.tz_localize('UTC')
                
            current_price = data.iloc[i]['close']
            
            # Check if we're in a trading session
            session_name = self._get_current_session(current_time)
            if not session_name:
                continue
                
            session_checks += 1
            
            # Check if we've already traded in this session
            session_key = self._get_session_key(current_time, session_name)
            if self._has_traded_in_session(session_key):
                continue
            
            # Check for Silver Bullet setup (simplified approach)
            setup_checks += 1
            signal_info = self._check_simplified_silver_bullet_setup(
                data, i, current_time, current_price, session_liquidity
            )
            
            if signal_info:
                signals_found += 1
                
                # Apply signal using .loc indexing like other strategies
                signals_df.loc[signals_df.index[i], 'signal'] = signal_info['signal']
                signals_df.loc[signals_df.index[i], 'position_size'] = self.position_size
                signals_df.loc[signals_df.index[i], 'stop_loss'] = signal_info['stop_loss']
                signals_df.loc[signals_df.index[i], 'take_profit'] = signal_info['take_profit']
                
                # Mark session as traded
                self._mark_session_traded(session_key)
        
        return signals_df
    
    def _check_silver_bullet_setup(self, data: pd.DataFrame, m15_data: pd.DataFrame,
                                  current_idx: int, current_time: pd.Timestamp,
                                  current_price: float, target_liquidity: List[LiquidityLevel]) -> Optional[Dict]:
        """
        Check for complete Silver Bullet setup.
        
        Process:
        1. Identify nearby liquidity
        2. Check if liquidity was recently swept
        3. Look for market structure shift
        4. Confirm FVG formation
        5. Validate entry conditions
        """
        
        # Step 1: Find nearby liquidity levels
        nearby_liquidity = self.liquidity_detector.get_liquidity_near_price(
            target_liquidity, current_price, self.max_liquidity_distance_pct
        )
        
        if not nearby_liquidity:
            return None
        
        # Step 2: Check for recent liquidity sweeps
        recent_sweeps = self._check_recent_liquidity_sweeps(
            nearby_liquidity, data, current_idx, lookback_candles=10
        )
        
        if not recent_sweeps:
            return None
        
        # Step 3: Determine trade direction based on swept liquidity  
        trade_direction = self._determine_trade_direction(recent_sweeps)
        if not trade_direction:
            return None
        
        # Step 4: Check for market structure shift
        if self.require_structure_shift:
            structure_shift = self._check_market_structure_shift(
                m15_data, current_time, trade_direction
            )
            if not structure_shift:
                return None
        
        # Step 5: Look for FVG formation
        fvg_info = self._check_fvg_formation(data, current_idx, trade_direction)
        if not fvg_info:
            return None
        
        # Step 6: Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(data, current_idx, trade_direction, fvg_info)
        take_profit = self._calculate_take_profit(
            current_price, stop_loss, trade_direction, nearby_liquidity
        )
        
        if not self._validate_risk_reward(current_price, stop_loss, take_profit, trade_direction):
            return None
        
        # Return signal information
        return {
            'signal': 1 if trade_direction == 'bullish' else -1,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'liquidity_swept': recent_sweeps[0].level_type,
            'fvg_type': fvg_info['type']
        }
    
    def _filter_target_liquidity(self, liquidity_levels: List[LiquidityLevel]) -> List[LiquidityLevel]:
        """Filter liquidity levels for trading targets."""
        target_liquidity = []
        
        for level in liquidity_levels:
            # Only consider strong, unswept liquidity
            if (level.strength >= self.min_liquidity_strength and 
                not level.is_swept):
                target_liquidity.append(level)
        
        return target_liquidity
    
    def _detect_session_liquidity_levels(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect session-based liquidity levels like the Pine Script.
        Uses the previous hour's high/low within each session as liquidity.
        """
        session_liquidity = {}
        
        # Convert to NY timezone for session analysis
        ny_data = data.copy()
        if ny_data.index.tz is None:
            ny_data.index = ny_data.index.tz_localize('UTC')
        ny_data.index = ny_data.index.tz_convert(self.ny_tz)
        
        # Define session hours
        session_hours = {
            'london_open': (3, 4),
            'ny_open': (10, 11), 
            'ny_pm': (14, 15)
        }
        
        for session_name, (start_hour, end_hour) in session_hours.items():
            if not getattr(self, f"enable_{session_name}", True):
                continue
                
            # Find session data - use time attribute for better compatibility
            try:
                session_mask = (ny_data.index.hour >= start_hour) & (ny_data.index.hour < end_hour)
            except AttributeError:
                # Fallback for compatibility issues
                session_mask = ny_data.index.to_series().dt.hour.between(start_hour, end_hour - 1)
            session_data = ny_data[session_mask]
            
            if len(session_data) > 0:
                # Get the session high/low as liquidity levels
                session_high = session_data['high'].max()
                session_low = session_data['low'].min()
                
                session_liquidity[session_name] = {
                    'high': session_high,
                    'low': session_low,
                    'start_time': session_data.index[0],
                    'end_time': session_data.index[-1]
                }
        
        return session_liquidity
    
    def _check_simplified_silver_bullet_setup(self, data: pd.DataFrame, current_idx: int,
                                            current_time: pd.Timestamp, current_price: float,
                                            session_liquidity: Dict[str, Dict]) -> Optional[Dict]:
        """
        Simplified Silver Bullet setup check inspired by Pine Script logic.
        
        Process:
        1. Check if we broke above/below session liquidity 
        2. Look for FVG formation in opposite direction
        3. Wait for retracement into FVG
        """
        
        if current_idx < 5:  # Minimal history required
            return None
        
        # Check recent price action for liquidity breaks - very short lookback for more opportunities
        lookback = min(10, current_idx)
        recent_data = data.iloc[current_idx - lookback:current_idx + 1]
        
        # Determine current session
        current_session = self._get_current_session(current_time)
        if not current_session or current_session not in session_liquidity:
            return None
            
        session_levels = session_liquidity[current_session]
        
        # More lenient liquidity sweep detection - look for moves TOWARD session levels
        # rather than requiring clean breaks beyond them
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        
        trade_direction = None
        swept_level = None
        
        # Much more lenient - trade based on position in session range
        session_range = session_levels['high'] - session_levels['low']
        session_mid = (session_levels['high'] + session_levels['low']) / 2
        
        # Trade when price is in outer thirds of session range with some momentum
        upper_third = session_levels['low'] + (session_range * 0.67)
        lower_third = session_levels['low'] + (session_range * 0.33)
        
        # Check for stronger momentum and pullback conditions
        if current_price > upper_third:
            # Check if we had upward momentum followed by pullback
            if recent_high > current_price * 1.003 and current_price < recent_high * 0.997:
                trade_direction = 'bearish'
                swept_level = session_levels['high']
        elif current_price < lower_third:
            # Check if we had downward momentum followed by pullback
            if recent_low < current_price * 0.997 and current_price > recent_low * 1.003:
                trade_direction = 'bullish'
                swept_level = session_levels['low']
        
        if not trade_direction:
            return None
        
        # Look for FVG formation in the expected direction (optional for more trades)
        fvg_info = self._check_fvg_formation(data, current_idx, trade_direction)
        if not fvg_info:
            # Create minimal FVG for entry
            fvg_info = {'type': trade_direction, 'size': 0.0001}
        
        # Calculate stop loss and take profit (ATR-based like Pine Script)
        atr = self._calculate_atr(data, current_idx, period=14)
        
        if trade_direction == 'bullish':
            stop_loss = current_price - (atr * 2.5)  # More conservative than Pine Script
            take_profit = current_price + (abs(current_price - stop_loss) * self.risk_reward_ratio)
            signal = 1
        else:
            stop_loss = current_price + (atr * 2.5)
            take_profit = current_price - (abs(stop_loss - current_price) * self.risk_reward_ratio)
            signal = -1
        
        return {
            'signal': signal,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'swept_level': swept_level,
            'fvg_type': fvg_info['type'],
            'atr': atr
        }
    
    def _calculate_atr(self, data: pd.DataFrame, current_idx: int, period: int = 14) -> float:
        """Calculate Average True Range."""
        if current_idx < period:
            return data['high'].iloc[:current_idx + 1].mean() - data['low'].iloc[:current_idx + 1].mean()
        
        # Get recent data for ATR calculation
        recent_data = data.iloc[current_idx - period + 1:current_idx + 1]
        
        true_ranges = []
        for i in range(1, len(recent_data)):
            high_low = recent_data.iloc[i]['high'] - recent_data.iloc[i]['low']
            high_close_prev = abs(recent_data.iloc[i]['high'] - recent_data.iloc[i-1]['close'])
            low_close_prev = abs(recent_data.iloc[i]['low'] - recent_data.iloc[i-1]['close'])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.01
    
    def _check_recent_liquidity_sweeps(self, liquidity_levels: List[LiquidityLevel],
                                     data: pd.DataFrame, current_idx: int,
                                     lookback_candles: int = 10) -> List[LiquidityLevel]:
        """Check for liquidity sweeps in recent candles."""
        recent_sweeps = []
        
        if current_idx < lookback_candles:
            return recent_sweeps
        
        # Look at recent price action
        lookback_start = max(0, current_idx - lookback_candles)
        recent_data = data.iloc[lookback_start:current_idx + 1]
        
        for level in liquidity_levels:
            # Check if liquidity was swept in recent candles
            if self._was_liquidity_swept_recently(level, recent_data):
                recent_sweeps.append(level)
        
        return recent_sweeps
    
    def _was_liquidity_swept_recently(self, level: LiquidityLevel, 
                                    recent_data: pd.DataFrame) -> bool:
        """
        Check if liquidity level was swept in recent data.
        
        A sweep is when price moves beyond a liquidity level and then reverses,
        indicating that liquidity has been taken and the smart money may now
        move price in the opposite direction.
        """
        level_price = level.price
        sweep_threshold = 0.0001  # 0.01% threshold for sweep detection
        
        if 'HIGH' in level.level_type or level.level_type in ['PDH', 'PWH', 'REH']:
            # For high liquidity (resistance), check if price swept above
            sweep_candles = recent_data[recent_data['high'] > level_price * (1 + sweep_threshold)]
            
            if sweep_candles.empty:
                return False
            
            # Check for reversal after sweep (price should come back down)
            sweep_idx = sweep_candles.index[0]
            candles_after_sweep = recent_data[recent_data.index > sweep_idx]
            
            if candles_after_sweep.empty:
                return False
            
            # Look for bearish reversal (lower closes after the sweep)
            sweep_high = sweep_candles.iloc[0]['high']
            reversal_candles = candles_after_sweep[candles_after_sweep['close'] < sweep_high * 0.999]
            
            return not reversal_candles.empty
            
        else:
            # For low liquidity (support), check if price swept below
            sweep_candles = recent_data[recent_data['low'] < level_price * (1 - sweep_threshold)]
            
            if sweep_candles.empty:
                return False
            
            # Check for reversal after sweep (price should come back up)
            sweep_idx = sweep_candles.index[0]
            candles_after_sweep = recent_data[recent_data.index > sweep_idx]
            
            if candles_after_sweep.empty:
                return False
            
            # Look for bullish reversal (higher closes after the sweep)
            sweep_low = sweep_candles.iloc[0]['low']
            reversal_candles = candles_after_sweep[candles_after_sweep['close'] > sweep_low * 1.001]
            
            return not reversal_candles.empty
    
    def _determine_trade_direction(self, swept_liquidity: List[LiquidityLevel]) -> Optional[str]:
        """
        Determine trade direction based on swept liquidity.
        
        ICT Concept: After liquidity is swept, smart money will often move price
        in the opposite direction to where retail traders would expect.
        """
        if not swept_liquidity:
            return None
        
        # Prioritize by strength and recency - strongest, most recent sweeps are most significant
        sorted_sweeps = sorted(swept_liquidity, key=lambda x: x.strength, reverse=True)
        primary_sweep = sorted_sweeps[0]
        
        # If buy-side liquidity (highs/resistance) was swept, expect bearish move
        # This is because stops above resistance have been taken, now smart money sells
        if ('HIGH' in primary_sweep.level_type or 
            primary_sweep.level_type in ['PDH', 'PWH', 'REH', 'SWING_HIGH']):
            return 'bearish'
        
        # If sell-side liquidity (lows/support) was swept, expect bullish move  
        # This is because stops below support have been taken, now smart money buys
        elif ('LOW' in primary_sweep.level_type or 
              primary_sweep.level_type in ['PDL', 'PWL', 'REL', 'SWING_LOW']):
            return 'bullish'
        
        return None
    
    def _check_market_structure_shift(self, m15_data: pd.DataFrame,
                                    current_time: pd.Timestamp,
                                    direction: str) -> bool:
        """
        Check for market structure shift using simplified swing point analysis.
        
        A structure shift occurs when price breaks beyond recent swing highs/lows,
        indicating a potential change in market direction.
        """
        if len(m15_data) < 10:
            return True  # Not enough data, allow trade
        
        # Use recent 15-20 candles for structure analysis
        lookback_candles = min(20, len(m15_data))
        recent_data = m15_data.tail(lookback_candles)
        
        if direction == 'bullish':
            # For bullish direction, check if we've broken above recent swing highs
            return self._check_bullish_structure_shift(recent_data)
        else:
            # For bearish direction, check if we've broken below recent swing lows
            return self._check_bearish_structure_shift(recent_data)
    
    def _check_bullish_structure_shift(self, data: pd.DataFrame) -> bool:
        """Check for bullish structure shift (breaking above recent swing highs)."""
        if len(data) < 5:
            return True
        
        # Find recent swing highs (local maxima)
        swing_highs = []
        for i in range(2, len(data) - 2):
            current_high = data.iloc[i]['high']
            if (current_high > data.iloc[i-1]['high'] and 
                current_high > data.iloc[i-2]['high'] and
                current_high > data.iloc[i+1]['high'] and 
                current_high > data.iloc[i+2]['high']):
                swing_highs.append(current_high)
        
        if not swing_highs:
            return True  # No clear swings, allow trade
        
        # Check if recent price has broken above the highest swing high
        recent_high = data['high'].tail(3).max()
        highest_swing = max(swing_highs)
        
        return recent_high > highest_swing * 1.0005  # 0.05% buffer for break
    
    def _check_bearish_structure_shift(self, data: pd.DataFrame) -> bool:
        """Check for bearish structure shift (breaking below recent swing lows)."""
        if len(data) < 5:
            return True
        
        # Find recent swing lows (local minima)
        swing_lows = []
        for i in range(2, len(data) - 2):
            current_low = data.iloc[i]['low']
            if (current_low < data.iloc[i-1]['low'] and 
                current_low < data.iloc[i-2]['low'] and
                current_low < data.iloc[i+1]['low'] and 
                current_low < data.iloc[i+2]['low']):
                swing_lows.append(current_low)
        
        if not swing_lows:
            return True  # No clear swings, allow trade
        
        # Check if recent price has broken below the lowest swing low
        recent_low = data['low'].tail(3).min()
        lowest_swing = min(swing_lows)
        
        return recent_low < lowest_swing * 0.9995  # 0.05% buffer for break
    
    def _check_fvg_formation(self, data: pd.DataFrame, current_idx: int,
                           direction: str) -> Optional[Dict]:
        """Check for FVG formation at current candle."""
        if current_idx < 2:
            return None
        
        # Check for 3-candle FVG pattern
        c1_high = data.iloc[current_idx - 2]['high']
        c1_low = data.iloc[current_idx - 2]['low']
        c2_high = data.iloc[current_idx - 1]['high']
        c2_low = data.iloc[current_idx - 1]['low']
        c3_high = data.iloc[current_idx]['high']
        c3_low = data.iloc[current_idx]['low']
        
        # Check for bullish FVG
        if direction == 'bullish' and c3_low > c1_high:
            gap_size = (c3_low - c1_high) / data.iloc[current_idx]['close']
            if gap_size >= self.min_fvg_size_pct:
                return {
                    'type': 'bullish',
                    'top': c3_low,
                    'bottom': c1_high,
                    'size': gap_size
                }
        
        # Check for bearish FVG
        elif direction == 'bearish' and c3_high < c1_low:
            gap_size = (c1_low - c3_high) / data.iloc[current_idx]['close']
            if gap_size >= self.min_fvg_size_pct:
                return {
                    'type': 'bearish',
                    'top': c1_low,
                    'bottom': c3_high,
                    'size': gap_size
                }
        
        return None
    
    def _calculate_stop_loss(self, data: pd.DataFrame, current_idx: int,
                           direction: str, fvg_info: Dict) -> float:
        """Calculate stop loss based on FVG formation."""
        if direction == 'bullish':
            # Stop below the low of the first candle in FVG formation
            return data.iloc[current_idx - 2]['low']
        else:
            # Stop above the high of the first candle in FVG formation
            return data.iloc[current_idx - 2]['high']
    
    def _calculate_take_profit(self, current_price: float, stop_loss: float,
                             direction: str, nearby_liquidity: List[LiquidityLevel]) -> float:
        """Calculate take profit based on risk/reward ratio or opposing liquidity."""
        
        # Calculate risk distance
        if direction == 'bullish':
            risk_distance = current_price - stop_loss
            basic_tp = current_price + (risk_distance * self.risk_reward_ratio)
        else:
            risk_distance = stop_loss - current_price
            basic_tp = current_price - (risk_distance * self.risk_reward_ratio)
        
        # Try to target opposing liquidity if available
        opposing_liquidity = self._find_opposing_liquidity(
            nearby_liquidity, current_price, direction
        )
        
        if opposing_liquidity:
            liquidity_tp = opposing_liquidity.price
            
            # Use liquidity target if it provides better R:R than minimum
            if direction == 'bullish':
                liquidity_rr = (liquidity_tp - current_price) / (current_price - stop_loss)
            else:
                liquidity_rr = (current_price - liquidity_tp) / (stop_loss - current_price)
            
            if liquidity_rr >= self.risk_reward_ratio:
                return liquidity_tp
        
        return basic_tp
    
    def _find_opposing_liquidity(self, liquidity_levels: List[LiquidityLevel],
                               current_price: float, direction: str) -> Optional[LiquidityLevel]:
        """Find opposing liquidity level to target."""
        opposing_levels = []
        
        for level in liquidity_levels:
            if direction == 'bullish':
                # Look for resistance levels above current price
                if (level.price > current_price and
                    ('HIGH' in level.level_type or level.level_type in ['PDH', 'PWH', 'REH'])):
                    opposing_levels.append(level)
            else:
                # Look for support levels below current price
                if (level.price < current_price and
                    ('LOW' in level.level_type or level.level_type in ['PDL', 'PWL', 'REL'])):
                    opposing_levels.append(level)
        
        if opposing_levels:
            # Return the strongest opposing level
            opposing_levels.sort(key=lambda x: x.strength, reverse=True)
            return opposing_levels[0]
        
        return None
    
    def _validate_risk_reward(self, current_price: float, stop_loss: float,
                            take_profit: float, direction: str) -> bool:
        """Validate risk/reward ratio meets minimum requirements."""
        if direction == 'bullish':
            risk = current_price - stop_loss
            reward = take_profit - current_price
        else:
            risk = stop_loss - current_price
            reward = current_price - take_profit
        
        if risk <= 0:
            return False
        
        rr_ratio = reward / risk
        return rr_ratio >= self.risk_reward_ratio
    
    def _get_current_session(self, timestamp) -> Optional[str]:
        """Get current trading session."""
        # Convert to pandas Timestamp if it's a datetime
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
            
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        ny_time = timestamp.astimezone(self.ny_tz)
        hour = ny_time.hour
        
        for session_name, (start_hour, end_hour) in self.session_windows.items():
            if start_hour <= hour < end_hour:
                return session_name
        
        return None
    
    def _get_session_key(self, timestamp, session_name: str) -> str:
        """Get unique session key for trade tracking."""
        # Convert to pandas Timestamp if it's a datetime
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
            
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        ny_time = timestamp.astimezone(self.ny_tz)
        return f"{ny_time.date()}_{session_name}"
    
    def _has_traded_in_session(self, session_key: str) -> bool:
        """Check if we've already traded in this session."""
        return self.session_trades.get(session_key, 0) >= self.max_trades_per_session
    
    def _mark_session_traded(self, session_key: str):
        """Mark that we've traded in this session."""
        self.session_trades[session_key] = self.session_trades.get(session_key, 0) + 1
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect timeframe of data."""
        if len(data) < 2:
            return 'unknown'
        
        time_diff = data.index[1] - data.index[0]
        
        if time_diff == timedelta(minutes=1):
            return '1min'
        elif time_diff == timedelta(minutes=3):
            return '3min'
        elif time_diff == timedelta(minutes=5):
            return '5min'
        elif time_diff == timedelta(minutes=15):
            return '15min'
        else:
            return 'unknown'
    
    def _resample_to_15min(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample data to 15-minute timeframe."""
        return data.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    def _resample_to_hourly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample data to hourly timeframe."""
        return data.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    def get_description(self) -> str:
        """Return strategy description."""
        sessions = [name.replace('_', ' ').title() for name in self.session_windows.keys()]
        
        return (f"ICT Silver Bullet Strategy: Trade liquidity sweeps during key sessions "
                f"({', '.join(sessions)}). R:R {self.risk_reward_ratio}:1, "
                f"Min liquidity strength: {self.min_liquidity_strength}, "
                f"Position size: {self.position_size * 100}%.")
    
    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return self.name