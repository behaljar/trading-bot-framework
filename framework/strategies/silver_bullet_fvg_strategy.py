"""
Silver Bullet FVG Strategy

Advanced FVG-based trading strategy for 1-minute charts with multi-timeframe FVG analysis.
This strategy looks for high-probability entries based on liquidity claiming or FVG interactions
across multiple timeframes.

Chart Timeframe: 1-minute
FVG Analysis: 15-minute timeframe only (resampled from 1min data)

Execution Windows (NY Time):
- 03:00 - 04:00 (London Open)
- 10:00 - 11:00 (NY Open)
- 14:00 - 15:00 (NY Afternoon)

Entry Conditions (Step 1 - Wait for ONE of these):
1. Liquidity grab on previous H4/Daily/Weekly High/Low
   - Price breaks and claims a significant previous level
   - Creates liquidity grab setup
   
2. Unmitigated part of 15min FVG touch
   - Price touches the unmitigated portion of a 15-minute Fair Value Gap
   - Sets up for potential reversal from the gap

NOTHING ELSE. Other FVG timeframes will be used later.

The strategy waits patiently during each session for one of these setups to occur.
"""

import pandas as pd
import numpy as np
import pytz
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from .base_strategy import BaseStrategy
from .detectors.fvg_detector import FVGDetector
from .detectors.pdh_pdl_detector import PDHPDLDetector
from .detectors.liquidity_objective_detector import LiquidityObjectiveDetector, TradeDirection
from .detectors.bos_detector import BoSDetector
from .detectors.choch_detector import ChOChDetector
from .detectors.swing_detector import SwingDetector


class SilverBulletFVGStrategy(BaseStrategy):
    """
    Advanced Silver Bullet FVG strategy for 1-minute charts with multi-timeframe FVG analysis.
    Analyzes FVGs across 15min, 1h, 4h, and daily timeframes resampled from 1-minute data.
    """
    
    def __init__(self, 
                 risk_reward_ratio: float = 2.0,
                 stop_loss_pct: float = 0.02,
                 position_size: float = 0.1,
                 min_fvg_sensitivity: float = 0.1,
                 max_trades_per_session: int = 1,
                 **kwargs):
        """
        Initialize Silver Bullet FVG strategy.
        
        Args:
            risk_reward_ratio: Risk/Reward ratio for take profit (default: 2.0)
            stop_loss_pct: Stop loss as percentage of entry price (default: 0.02 = 2%)
            position_size: Position size as fraction of equity (default: 0.1)
            min_fvg_sensitivity: Minimum FVG sensitivity ratio (default: 0.1)
            max_trades_per_session: Maximum trades per execution window (default: 1)
        """
        super().__init__("silver_bullet_fvg", kwargs)
        
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_pct = stop_loss_pct
        self.position_size = position_size
        self.min_fvg_sensitivity = min_fvg_sensitivity
        self.max_trades_per_session = max_trades_per_session
        
        # Initialize detectors
        self.fvg_detector = FVGDetector(min_sensitivity=min_fvg_sensitivity)
        self.pdh_pdl_detector = PDHPDLDetector()
        self.bos_detector = BoSDetector()
        self.choch_detector = ChOChDetector()
        self.swing_detector = SwingDetector(sensitivity=3)
        
        # FVG analysis settings - ONLY 15min for now
        self.fvg_timeframe = '15min'
        
        # Execution windows in NY timezone (24-hour format)
        self.execution_windows = [
            (3, 4),   # London Open: 03:00 - 04:00 NY
            (10, 11), # NY Open: 10:00 - 11:00 NY
            (14, 15)  # NY Afternoon: 14:00 - 15:00 NY
        ]
        
        # Track trades per session
        self.session_trades = {}
        
        # Track setup state within session
        self.session_setup_triggered = {}  # session_key: {'triggered': bool, 'direction': str, 'trigger_candle': int}
        
        # Cache for performance optimization
        self.cache = {
            'm15_data': None,
            'm15_fvgs': [],
            'h1_data': None,
            'h1_fvgs': [],
            'h4_data': None, 
            'h4_fvgs': [],
            'daily_data': None,
            'daily_fvgs': [],
            'pdh_pdl_levels': None,
            'last_cache_update': -1  # Last processed index
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Silver Bullet setups.
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            
        Returns:
            DataFrame with trading signals and position management columns
        """
        if not self.validate_data(data):
            raise ValueError("Invalid input data format")
        
        # Initialize signal columns (avoid full copy for performance)
        signals_df = pd.DataFrame(index=data.index)
        signals_df['signal'] = 0
        signals_df['position_size'] = 0.0
        signals_df['stop_loss'] = np.nan
        signals_df['take_profit'] = np.nan
        signals_df['tag'] = ''
        
        # Only copy OHLCV columns if needed by the wrapper
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                signals_df[col] = data[col]
        
        # Need sufficient data for analysis
        if len(data) < 50:
            return signals_df
        
        # Pre-compute execution window mask to avoid per-candle timezone conversions
        ny_tz = pytz.timezone('America/New_York')
        if data.index.tz is None:
            ny_index = data.index.tz_localize('UTC').tz_convert(ny_tz)
        else:
            ny_index = data.index.tz_convert(ny_tz)
        
        # Create execution window mask
        execution_mask = np.zeros(len(data), dtype=bool)
        for start_hour, end_hour in self.execution_windows:
            execution_mask |= (ny_index.hour >= start_hour) & (ny_index.hour < end_hour)
        
        # Convert to numpy arrays for faster access
        high_values = data['High'].to_numpy()
        low_values = data['Low'].to_numpy()
        close_values = data['Close'].to_numpy()
        
        # Pre-filter to only execution window indices for maximum efficiency  
        execution_indices = [i for i in range(10, len(data)) if execution_mask[i]]
        
        # Process only candles within execution windows
        for i in tqdm(execution_indices, desc="🔍 Analyzing signals", unit="candles", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} candles [{elapsed}<{remaining}]", 
                     miniters=100):  # Update progress less frequently
                
            current_time = data.index[i]
            
            # Update cache if needed (only every 15 minutes to improve performance)
            self._update_cache(data, i)
            
            # Get session key for tracking
            session_key = self._get_session_key(current_time)
            
            # Get previous H4/D/W H/L levels from cache
            levels = self.cache['pdh_pdl_levels'] if self.cache['pdh_pdl_levels'] else {}
            
            prev_4h_high = levels.get('P4H')
            prev_4h_low = levels.get('P4L') 
            prev_daily_high = levels.get('PDH')
            prev_daily_low = levels.get('PDL')
            prev_weekly_high = levels.get('PWH')
            prev_weekly_low = levels.get('PWL')
            
            # Filter to only unclaimed levels
            unclaimed_levels = []
            for level in [prev_4h_high, prev_4h_low, prev_daily_high, prev_daily_low, prev_weekly_high, prev_weekly_low]:
                if level and not level.is_broken:
                    unclaimed_levels.append(level)
            
            # Get M15 FVGs from cache (much faster!)
            m15_data = self.cache['m15_data']
            all_m15_fvgs = self.cache['m15_fvgs']
            
            # Filter to only unmitigated FVGs
            unmitigated_m15_fvgs = []
            for fvg in all_m15_fvgs:
                if self._is_m15_fvg_unmitigated(m15_data, fvg, data.index[i]):
                    unmitigated_m15_fvgs.append(fvg)
            
            current_high = high_values[i]
            current_low = low_values[i]
            
            # Check if setup already triggered in this session
            session_setup = self.session_setup_triggered.get(session_key, {'triggered': False, 'direction': None, 'trigger_candle': None})
            
            # Step 1: Check if price claims liq levels OR touches M15 unmitigated FVG (only if not already triggered)
            if not session_setup['triggered']:
                setup_triggered = False
                setup_direction = None
                setup_reason = None
                
                # Check liquidity claims
                for level in unclaimed_levels:
                    if level.level_type in ['P4H', 'PDH', 'PWH'] and current_high > level.price:
                        setup_triggered = True
                        setup_direction = 'bearish'  # After claiming high, expect bearish move
                        setup_reason = f"Claimed {level.level_type} @ ${level.price:.2f}"
                        break
                    elif level.level_type in ['P4L', 'PDL', 'PWL'] and current_low < level.price:
                        setup_triggered = True
                        setup_direction = 'bullish'  # After claiming low, expect bullish move
                        setup_reason = f"Claimed {level.level_type} @ ${level.price:.2f}"
                        break
                
                # Check M15 FVG touches
                if not setup_triggered:
                    for fvg in unmitigated_m15_fvgs:
                        if fvg.fvg_type == 'bullish' and current_low <= fvg.top and current_low >= fvg.bottom:
                            setup_triggered = True
                            setup_direction = 'bullish'  # Touch bullish FVG, expect bounce up
                            setup_reason = f"M15 Bullish FVG touch @ ${float(fvg.top)*1e8:.2f}-${float(fvg.bottom)*1e8:.2f}"
                            break
                        elif fvg.fvg_type == 'bearish' and current_high >= fvg.bottom and current_high <= fvg.top:
                            setup_triggered = True
                            setup_direction = 'bearish'  # Touch bearish FVG, expect bounce down
                            setup_reason = f"M15 Bearish FVG touch @ ${float(fvg.top)*1e8:.2f}-${float(fvg.bottom)*1e8:.2f}"
                            break
                
                # If setup triggered, save it for this session
                if setup_triggered:
                    self.session_setup_triggered[session_key] = {
                        'triggered': True, 
                        'direction': setup_direction, 
                        'trigger_candle': i,
                        'reason': setup_reason
                    }
            
            # Step 2: If setup was triggered in this session, check for M15 FVG, BoS, or ChoCh confirmation
            if self.session_setup_triggered.get(session_key, {}).get('triggered', False):
                setup_direction = self.session_setup_triggered[session_key]['direction']
                
                # Check if we already have a trade in this session
                if self.session_trades.get(session_key, 0) >= self.max_trades_per_session:
                    continue
                
                confirmation_signal, confirmation_type = self._check_confirmation(data, i, setup_direction)
                
                if confirmation_signal != 0:
                    # Find first liquidity objective as target
                    target_result = self._find_liquidity_target(data, i, confirmation_signal, unclaimed_levels, unmitigated_m15_fvgs)
                    
                    if target_result is not None:
                        target_price, target_type, target_details = target_result
                        # Open trade
                        current_price = close_values[i]
                        stop_loss, take_profit = self._calculate_trade_levels(current_price, confirmation_signal, target_price, data, i, confirmation_type)
                        
                        if stop_loss is not None and take_profit is not None:
                            # Get confirmation details
                            confirmation_details = self._get_confirmation_details(data, i, confirmation_type, current_price, stop_loss, take_profit)
                            
                            # Create enhanced tag with all details
                            setup_reason = self.session_setup_triggered[session_key].get('reason', 'Unknown setup')
                            current_time_str = data.index[i].strftime('%H:%M')
                            
                            trade_tag = f"{setup_reason} • {confirmation_details} • Target: {target_type} ${target_price*1e8:.2f} ({target_details}) • {current_time_str}"
                            
                            # Record the trade
                            signals_df.iloc[i, signals_df.columns.get_loc('signal')] = confirmation_signal
                            signals_df.iloc[i, signals_df.columns.get_loc('position_size')] = self.position_size
                            signals_df.iloc[i, signals_df.columns.get_loc('stop_loss')] = stop_loss
                            signals_df.iloc[i, signals_df.columns.get_loc('take_profit')] = take_profit
                            signals_df.iloc[i, signals_df.columns.get_loc('tag')] = trade_tag
                            
                            # Track session trade
                            self.session_trades[session_key] = self.session_trades.get(session_key, 0) + 1

        
        return signals_df
    
    def _is_execution_window(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp falls within execution windows (NY timezone).
        """
        ny_tz = pytz.timezone('America/New_York')
        
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        ny_time = timestamp.astimezone(ny_tz)
        current_hour = ny_time.hour
        
        for start_hour, end_hour in self.execution_windows:
            if start_hour <= current_hour < end_hour:
                return True
        
        return False
    
    def _get_session_key(self, timestamp: pd.Timestamp) -> str:
        """
        Get session key for tracking trades and setups per session.
        """
        ny_tz = pytz.timezone('America/New_York')
        
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
            
        ny_time = timestamp.astimezone(ny_tz)
        current_hour = ny_time.hour
        
        # Determine which window we're in
        window_id = 0
        for i, (start_hour, end_hour) in enumerate(self.execution_windows):
            if start_hour <= current_hour < end_hour:
                window_id = i
                break
        
        return f"{ny_time.date()}_{window_id}"
    
    def _is_m15_fvg_unmitigated(self, m15_data: pd.DataFrame, fvg, current_time: pd.Timestamp) -> bool:
        """
        Check if M15 FVG is still unmitigated (hasn't been fully filled).
        """
        # Look at M15 candles from after FVG formation until current time
        fvg_end_time = m15_data.index[fvg.end_idx]
        
        # Get M15 candles after FVG formation
        future_candles = m15_data[m15_data.index > fvg_end_time]
        if future_candles.empty:
            return True  # No future candles to check
        
        for _, candle in future_candles.iterrows():
            candle_high = candle['High']
            candle_low = candle['Low']
            
            # Check if FVG has been fully mitigated
            if fvg.fvg_type == 'bullish':
                # Bullish FVG is mitigated if price fully fills the gap
                if candle_low <= fvg.bottom:
                    return False
            elif fvg.fvg_type == 'bearish':
                # Bearish FVG is mitigated if price fully fills the gap  
                if candle_high >= fvg.top:
                    return False
        
        return True
    
    def _check_confirmation(self, data: pd.DataFrame, current_idx: int, setup_direction: str) -> Tuple[int, str]:
        """
        Check for M1 FVG, BoS, or ChoCh confirmation.
        
        Returns:
            Tuple of (signal, confirmation_type)
            signal: 1 for bullish confirmation, -1 for bearish confirmation, 0 for no confirmation
            confirmation_type: 'fvg', 'bos', 'choch', or 'none'
        """
        if current_idx < 5:  # Need some lookback
            return 0, 'none'
        
        # Check for M1 FVG formation
        m1_fvg_signal = self._check_m1_fvg_confirmation(data, current_idx, setup_direction)
        if m1_fvg_signal != 0:
            return m1_fvg_signal, 'fvg'
        
        # Check for BoS (Break of Structure)  
        bos_signal = self._check_bos_confirmation(data, current_idx, setup_direction)
        if bos_signal != 0:
            return bos_signal, 'bos'
            
        # Check for ChoCh (Change of Character)
        choch_signal = self._check_choch_confirmation(data, current_idx, setup_direction)
        if choch_signal != 0:
            return choch_signal, 'choch'
        
        return 0, 'none'
    
    def _check_m1_fvg_confirmation(self, data: pd.DataFrame, current_idx: int, setup_direction: str) -> int:
        """
        Check if current candle or recent candles form M1 FVG in expected direction.
        """
        # Look at last few candles for FVG formation
        lookback = min(5, current_idx)
        recent_data = data.iloc[current_idx-lookback:current_idx+1]
        
        # Detect FVGs in recent 1-minute data
        recent_fvgs = self.fvg_detector.detect_fvgs(recent_data, merge_consecutive=False)
        
        for fvg in recent_fvgs:
            # Check if FVG direction matches setup direction
            if setup_direction == 'bullish' and fvg.fvg_type == 'bullish':
                return 1  # Bullish M1 FVG confirms bullish setup
            elif setup_direction == 'bearish' and fvg.fvg_type == 'bearish':
                return -1  # Bearish M1 FVG confirms bearish setup
        
        return 0
    
    def _check_bos_confirmation(self, data: pd.DataFrame, current_idx: int, setup_direction: str) -> int:
        """
        Check for Break of Structure (BoS) confirmation.
        Uses more candles for context but event must happen on current/previous candle.
        """
        if current_idx < 20:  # Need sufficient lookback for BoS detection
            return 0
        
        # Get more data for context but focus on recent events
        lookback = min(50, current_idx)
        context_data = data.iloc[current_idx-lookback:current_idx+1]
        
        # Detect BoS in context data
        bos_signals = self.bos_detector.detect_bos_events(context_data)
        
        if not bos_signals.empty:
            # Get the original index mapping
            context_start_idx = current_idx - lookback
            
            # Check if BoS event happened on current or previous candle
            for i in range(len(bos_signals)-1, -1, -1):  # Check from most recent
                bos_signal = bos_signals.iloc[i]
                bos_original_idx = context_start_idx + bos_signals.index[i]
                
                # Event must be on current or previous candle
                if current_idx - 1 <= bos_original_idx <= current_idx:
                    # Confirm if BoS direction matches setup direction
                    if setup_direction == 'bullish' and bos_signal['bos_direction'] == 'bullish':
                        return 1  # Bullish BoS confirms bullish setup
                    elif setup_direction == 'bearish' and bos_signal['bos_direction'] == 'bearish':
                        return -1  # Bearish BoS confirms bearish setup
        
        return 0
    
    def _check_choch_confirmation(self, data: pd.DataFrame, current_idx: int, setup_direction: str) -> int:
        """
        Check for Change of Character (ChoCh) confirmation.
        Uses more candles for context but event must happen on current/previous candle.
        """
        if current_idx < 20:  # Need sufficient lookback for ChoCh detection
            return 0
        
        # Get more data for context but focus on recent events
        lookback = min(50, current_idx)
        context_data = data.iloc[current_idx-lookback:current_idx+1]
        
        # Detect ChoCh in context data
        choch_signals = self.choch_detector.detect_choch_events(context_data)
        
        if not choch_signals.empty:
            # Get the original index mapping
            context_start_idx = current_idx - lookback
            
            # Check if ChoCh event happened on current or previous candle
            for i in range(len(choch_signals)-1, -1, -1):  # Check from most recent
                choch_signal = choch_signals.iloc[i]
                choch_original_idx = context_start_idx + choch_signals.index[i]
                
                # Event must be on current or previous candle
                if current_idx - 1 <= choch_original_idx <= current_idx:
                    # Confirm if ChoCh direction matches setup direction
                    if setup_direction == 'bullish' and choch_signal['choch_direction'] == 'bullish':
                        return 1  # Bullish ChoCh confirms bullish setup
                    elif setup_direction == 'bearish' and choch_signal['choch_direction'] == 'bearish':
                        return -1  # Bearish ChoCh confirms bearish setup
        
        return 0
    
    def _find_liquidity_target(self, data: pd.DataFrame, current_idx: int, signal: int, 
                              unclaimed_levels: list, unmitigated_m15_fvgs: list) -> Optional[float]:
        """
        Find first liquidity objective as target:
        - Unclaimed prev H4/D/W high-low
        - Unmitigated H1/H4/D FVG  
        - Unclaimed M15 swing high-low
        """
        current_price = data['Close'].iloc[current_idx]
        targets = []
        
        # 1. Unclaimed H4/D/W levels
        for level in unclaimed_levels:
            if signal == 1 and level.level_type in ['P4H', 'PDH', 'PWH'] and level.price > current_price:
                targets.append(('level', level.price, abs(level.price - current_price)))
            elif signal == -1 and level.level_type in ['P4L', 'PDL', 'PWL'] and level.price < current_price:
                targets.append(('level', level.price, abs(current_price - level.price)))
        
        # 2. Unmitigated M15 FVGs (use as targets in opposite direction)
        for fvg in unmitigated_m15_fvgs:
            if signal == 1 and fvg.fvg_type == 'bearish' and fvg.top > current_price:
                targets.append(('fvg', fvg.top, abs(fvg.top - current_price)))
            elif signal == -1 and fvg.fvg_type == 'bullish' and fvg.bottom < current_price:
                targets.append(('fvg', fvg.bottom, abs(current_price - fvg.bottom)))
        
        # 3. Higher timeframe FVGs (H1, H4, Daily)
        htf_targets = self._get_htf_fvg_targets(data, current_idx, signal, current_price)
        targets.extend(htf_targets)
        
        # 4. M15 swing high-low targets
        m15_swing_targets = self._get_m15_swing_targets(data, current_idx, signal, current_price)
        targets.extend(m15_swing_targets)
        
        # Return closest target with details
        if targets:
            targets.sort(key=lambda x: x[2])  # Sort by distance
            closest_target = targets[0]
            target_type, target_price, distance = closest_target
            
            # Create target details string
            distance_pct = (distance / current_price) * 100
            target_details = f"{distance_pct:.1f}% away"
            
            return target_price, target_type, target_details  # Return (price, target_type, details)
        
        return None
    
    def _calculate_trade_levels(self, current_price: float, signal: int, target_price: float, 
                               data: pd.DataFrame, current_idx: int, confirmation_type: str) -> tuple:
        """
        Calculate stop loss and take profit levels.
        SL: Below/above 1st candle of M1 FVG (if formed) OR below/above last swing high-low
        """
        stop_loss = None
        
        if confirmation_type == 'fvg':
            # Find the M1 FVG and use first candle for stop loss
            stop_loss = self._get_m1_fvg_stop_loss(data, current_idx, signal)
        
        # If no FVG or FVG SL not found, use swing high-low
        if stop_loss is None:
            stop_loss = self._get_swing_stop_loss(data, current_idx, signal)
        
        # Fallback to percentage-based if swing SL not found
        if stop_loss is None:
            if signal == 1:  # Long
                stop_loss = current_price * (1 - self.stop_loss_pct)
            elif signal == -1:  # Short
                stop_loss = current_price * (1 + self.stop_loss_pct)
            else:
                return None, None
        
        # Calculate take profit
        if signal == 1:  # Long
            risk_distance = current_price - stop_loss
            if risk_distance <= 0:  # Invalid stop loss
                return None, None
            rr_take_profit = current_price + (risk_distance * self.risk_reward_ratio)
            take_profit = min(target_price, rr_take_profit)
            
            # Check minimum 1:1 RR ratio
            actual_reward = take_profit - current_price
            if actual_reward < risk_distance:  # Less than 1:1 RR
                return None, None
                
        elif signal == -1:  # Short
            risk_distance = stop_loss - current_price
            if risk_distance <= 0:  # Invalid stop loss
                return None, None
            rr_take_profit = current_price - (risk_distance * self.risk_reward_ratio)
            take_profit = max(target_price, rr_take_profit)
            
            # Check minimum 1:1 RR ratio
            actual_reward = current_price - take_profit
            if actual_reward < risk_distance:  # Less than 1:1 RR
                return None, None
                
        else:
            return None, None
        
        return stop_loss, take_profit
    
    def _get_m1_fvg_stop_loss(self, data: pd.DataFrame, current_idx: int, signal: int) -> Optional[float]:
        """
        Get stop loss based on first candle of M1 FVG formation.
        """
        # Look for recent FVG formation
        lookback = min(5, current_idx)
        recent_data = data.iloc[current_idx-lookback:current_idx+1]
        
        recent_fvgs = self.fvg_detector.detect_fvgs(recent_data, merge_consecutive=False)
        
        if recent_fvgs:
            # Get the most recent FVG
            latest_fvg = recent_fvgs[-1]
            fvg_start_idx = current_idx - lookback + latest_fvg.start_idx
            
            if signal == 1:  # Long - SL below first candle low
                return data.iloc[fvg_start_idx]['Low']
            elif signal == -1:  # Short - SL above first candle high
                return data.iloc[fvg_start_idx]['High']
        
        return None
    
    def _get_swing_stop_loss(self, data: pd.DataFrame, current_idx: int, signal: int) -> Optional[float]:
        """
        Get stop loss based on last swing high-low.
        """
        if current_idx < 10:
            return None
        
        # Look for swing points in recent data
        lookback = min(50, current_idx)
        swing_data = data.iloc[current_idx-lookback:current_idx+1]
        
        if signal == 1:  # Long - find last swing low
            swing_lows = self.swing_detector.find_all_swing_lows(swing_data['Low'])
            if swing_lows:
                # Get the most recent swing low
                _, last_swing_low = swing_lows[-1]
                return last_swing_low
                
        elif signal == -1:  # Short - find last swing high
            swing_highs = self.swing_detector.find_all_swing_highs(swing_data['High'])
            if swing_highs:
                # Get the most recent swing high
                _, last_swing_high = swing_highs[-1]
                return last_swing_high
        
        return None
    
    def _get_htf_fvg_targets(self, data: pd.DataFrame, current_idx: int, signal: int, current_price: float) -> list:
        """
        Get higher timeframe FVG targets (H1, H4, Daily) from cache.
        """
        targets = []
        
        # H1 FVGs from cache
        try:
            h1_data = self.cache['h1_data']
            h1_fvgs = self.cache['h1_fvgs']
            
            if h1_data is not None and h1_fvgs:
                for fvg in h1_fvgs:
                    if self._is_htf_fvg_unmitigated(h1_data, fvg, data.index[current_idx]):
                        if signal == 1 and fvg.fvg_type == 'bearish' and fvg.top > current_price:
                            targets.append(('h1_fvg', fvg.top, abs(fvg.top - current_price)))
                        elif signal == -1 and fvg.fvg_type == 'bullish' and fvg.bottom < current_price:
                            targets.append(('h1_fvg', fvg.bottom, abs(current_price - fvg.bottom)))
        except Exception:
            pass
        
        # H4 FVGs from cache
        try:
            h4_data = self.cache['h4_data']
            h4_fvgs = self.cache['h4_fvgs']
            
            if h4_data is not None and h4_fvgs:
                for fvg in h4_fvgs:
                    if self._is_htf_fvg_unmitigated(h4_data, fvg, data.index[current_idx]):
                        if signal == 1 and fvg.fvg_type == 'bearish' and fvg.top > current_price:
                            targets.append(('h4_fvg', fvg.top, abs(fvg.top - current_price)))
                        elif signal == -1 and fvg.fvg_type == 'bullish' and fvg.bottom < current_price:
                            targets.append(('h4_fvg', fvg.bottom, abs(current_price - fvg.bottom)))
        except Exception:
            pass
        
        # Daily FVGs from cache
        try:
            daily_data = self.cache['daily_data']
            daily_fvgs = self.cache['daily_fvgs']
            
            if daily_data is not None and daily_fvgs:
                for fvg in daily_fvgs:
                    if self._is_htf_fvg_unmitigated(daily_data, fvg, data.index[current_idx]):
                        if signal == 1 and fvg.fvg_type == 'bearish' and fvg.top > current_price:
                            targets.append(('daily_fvg', fvg.top, abs(fvg.top - current_price)))
                        elif signal == -1 and fvg.fvg_type == 'bullish' and fvg.bottom < current_price:
                            targets.append(('daily_fvg', fvg.bottom, abs(current_price - fvg.bottom)))
        except Exception:
            pass
        
        return targets
    
    def _get_m15_swing_targets(self, data: pd.DataFrame, current_idx: int, signal: int, current_price: float) -> list:
        """
        Get M15 swing high/low targets from cached M15 data.
        """
        targets = []
        
        try:
            # Use cached M15 data for swing detection
            m15_data = self.cache['m15_data']
            
            if m15_data is not None and len(m15_data) > 10:
                # Find swing highs and lows
                swing_highs = self.swing_detector.find_all_swing_highs(m15_data['High'])
                swing_lows = self.swing_detector.find_all_swing_lows(m15_data['Low'])
                
                # For long signals, look for swing highs above current price
                if signal == 1:
                    for idx, price in swing_highs:
                        if price > current_price:
                            # Check if swing high is still unclaimed
                            future_data = m15_data.iloc[idx+1:]
                            is_claimed = not future_data.empty and (future_data['High'] > price).any()
                            if not is_claimed:
                                targets.append(('m15_swing_high', price, abs(price - current_price)))
                
                # For short signals, look for swing lows below current price
                elif signal == -1:
                    for idx, price in swing_lows:
                        if price < current_price:
                            # Check if swing low is still unclaimed
                            future_data = m15_data.iloc[idx+1:]
                            is_claimed = not future_data.empty and (future_data['Low'] < price).any()
                            if not is_claimed:
                                targets.append(('m15_swing_low', price, abs(current_price - price)))
        
        except Exception:
            pass
        
        return targets
    
    def _is_htf_fvg_unmitigated(self, htf_data: pd.DataFrame, fvg, current_time: pd.Timestamp) -> bool:
        """
        Check if higher timeframe FVG is still unmitigated.
        """
        try:
            # Look at HTF candles from after FVG formation until current time
            fvg_end_time = htf_data.index[fvg.end_idx]
            
            # Get HTF candles after FVG formation
            future_candles = htf_data[htf_data.index > fvg_end_time]
            if future_candles.empty:
                return True
            
            for _, candle in future_candles.iterrows():
                candle_high = candle['High']
                candle_low = candle['Low']
                
                # Check if FVG has been fully mitigated
                if fvg.fvg_type == 'bullish':
                    if candle_low <= fvg.bottom:
                        return False
                elif fvg.fvg_type == 'bearish':
                    if candle_high >= fvg.top:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def get_description(self) -> str:
        """Return strategy description."""
        return (f"Silver Bullet FVG Strategy: Advanced setups during execution windows - "
                f"Level claims (H4/D/W H/L) + FVG touch setups, "
                f"R:R ratio {self.risk_reward_ratio}:1, stop loss {self.stop_loss_pct*100}%, "
                f"max {self.max_trades_per_session} trade(s) per session, "
                f"execution windows: London Open (03:00-04:00), NY Open (10:00-11:00), "
                f"NY Afternoon (14:00-15:00) NY time. Position size: {self.position_size * 100}%.")
    
    def get_strategy_name(self) -> str:
        """Return strategy name for identification."""
        return self.name
    
    def _update_cache(self, data: pd.DataFrame, current_idx: int):
        """
        Update cached resampled data and FVGs incrementally.
        Only recalculates when new significant data is available.
        """
        # Skip if we've already processed this index recently
        if current_idx <= self.cache['last_cache_update']:
            return
            
        # Only update cache when we cross actual time boundaries (more efficient)
        current_time = data.index[current_idx]
        
        # Check if we've crossed a 15-minute boundary since last update
        if self.cache['last_cache_update'] != -1:
            last_time = data.index[self.cache['last_cache_update']]
            # Only update if we've crossed into a new 15-minute window
            if (current_time.minute // 15) == (last_time.minute // 15) and current_time.hour == last_time.hour:
                return
        
        # Look back 5 days for analysis
        lookback_candles = min(5 * 24 * 60, current_idx)  
        lookback_start = max(0, current_idx - lookback_candles)
        
        # Update M15 data and FVGs
        try:
            self.cache['m15_data'] = data.iloc[lookback_start:current_idx+1].resample('15min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            if len(self.cache['m15_data']) > 3:
                self.cache['m15_fvgs'] = self.fvg_detector.detect_fvgs(self.cache['m15_data'], merge_consecutive=True)
            else:
                self.cache['m15_fvgs'] = []
        except Exception:
            self.cache['m15_fvgs'] = []
        
        # Update H1 data and FVGs  
        try:
            self.cache['h1_data'] = data.iloc[lookback_start:current_idx+1].resample('1h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            if len(self.cache['h1_data']) > 3:
                self.cache['h1_fvgs'] = self.fvg_detector.detect_fvgs(self.cache['h1_data'], merge_consecutive=True)
            else:
                self.cache['h1_fvgs'] = []
        except Exception:
            self.cache['h1_fvgs'] = []
        
        # Update H4 data and FVGs
        try:
            self.cache['h4_data'] = data.iloc[lookback_start:current_idx+1].resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            if len(self.cache['h4_data']) > 3:
                self.cache['h4_fvgs'] = self.fvg_detector.detect_fvgs(self.cache['h4_data'], merge_consecutive=True)
            else:
                self.cache['h4_fvgs'] = []
        except Exception:
            self.cache['h4_fvgs'] = []
        
        # Update Daily data and FVGs
        try:
            self.cache['daily_data'] = data.iloc[lookback_start:current_idx+1].resample('1D').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            
            if len(self.cache['daily_data']) > 3:
                self.cache['daily_fvgs'] = self.fvg_detector.detect_fvgs(self.cache['daily_data'], merge_consecutive=True)
            else:
                self.cache['daily_fvgs'] = []
        except Exception:
            self.cache['daily_fvgs'] = []
        
        # Update PDH/PDL levels
        try:
            self.cache['pdh_pdl_levels'] = self.pdh_pdl_detector.detect_levels(data.iloc[:current_idx+1])
        except Exception:
            self.cache['pdh_pdl_levels'] = None
            
        # Mark cache as updated
        self.cache['last_cache_update'] = current_idx
    
    def _get_confirmation_details(self, data: pd.DataFrame, current_idx: int, confirmation_type: str, 
                                current_price: float, stop_loss: float, take_profit: float) -> str:
        """
        Get clean, readable confirmation information for trade tags.
        """
        # Calculate risk-reward ratio first
        rr_ratio = 0
        if stop_loss and take_profit:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            rr_ratio = reward / risk if risk > 0 else 0
        
        # Create clean confirmation strings
        if confirmation_type == 'fvg':
            return f"M1 FVG RR:{rr_ratio:.1f}"
            
        elif confirmation_type == 'bos':
            return f"BoS RR:{rr_ratio:.1f}"
            
        elif confirmation_type == 'choch':
            return f"ChoCh RR:{rr_ratio:.1f}"
        
        return f"Entry RR:{rr_ratio:.1f}"