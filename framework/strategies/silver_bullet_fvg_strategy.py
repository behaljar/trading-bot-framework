"""
Silver Bullet FVG Strategy - 1M + 15M Multi-Timeframe

Entry Rules:
- Price in M15 FVG + M1 FVG prints in same direction
- NY execution windows: 03:00-04:00, 10:00-11:00, 14:00-15:00
- One trade per session window
- Smart Money Concepts filters (configurable for optimization)

Exit Rules:
- Stop Loss: Low/High of first candle in M1 FVG formation
- Take Profit: Configurable Risk/Reward ratio (default 2:1)
- Time-based exit: 2 hours maximum
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
import hashlib
from .base_strategy import BaseStrategy
from .detectors.fvg_detector import FVGDetector
from .detectors.swing_detector import SwingDetector
from .detectors.bos_choch_unified_detector import BoSCHoCHDetector


class SilverBulletFVGStrategy(BaseStrategy):
    """
    Silver Bullet FVG Strategy using M15 FVG context + M1 FVG trigger.
    """
    
    def __init__(self, 
                 # Core FVG parameters
                 risk_reward_ratio: float = 2.0,
                 max_hold_hours: int = 2,
                 min_fvg_sensitivity: float = 0.1,
                 position_size: float = 0.01,
                 m15_lookback_candles: int = 144,
                 
                 # Session control (for optimization)
                 enable_london_open: bool = True,
                 enable_ny_open: bool = True, 
                 enable_ny_afternoon: bool = True,
                 
                 # Smart Money Concepts filters (for optimization)
                 use_swing_hl_filter: bool = False,
                 swing_length: int = 5,
                 require_swing_high_for_short: bool = True,
                 require_swing_low_for_long: bool = True,
                 
                 use_bos_choch_filter: bool = False,
                 require_bos_for_entry: bool = True,
                 require_choch_for_entry: bool = False,
                 bos_lookback_candles: int = 10,
                 
                 use_liquidity_filter: bool = False,
                 require_liquidity_sweep: bool = True,
                 liquidity_lookback_candles: int = 20,
                 
                 **kwargs):
        """Initialize Silver Bullet FVG strategy."""
        super().__init__("silver_bullet_fvg", kwargs)
        
        self.risk_reward_ratio = risk_reward_ratio
        self.max_hold_hours = max_hold_hours
        self.min_fvg_sensitivity = min_fvg_sensitivity
        self.position_size = position_size
        self.m15_lookback_candles = m15_lookback_candles
        
        # Session control
        self.enable_london_open = enable_london_open
        self.enable_ny_open = enable_ny_open
        self.enable_ny_afternoon = enable_ny_afternoon
        
        # SMC filters
        self.use_swing_hl_filter = use_swing_hl_filter
        self.swing_length = swing_length
        self.require_swing_high_for_short = require_swing_high_for_short
        self.require_swing_low_for_long = require_swing_low_for_long
        
        self.use_bos_choch_filter = use_bos_choch_filter
        self.require_bos_for_entry = require_bos_for_entry
        self.require_choch_for_entry = require_choch_for_entry
        self.bos_lookback_candles = bos_lookback_candles
        
        self.use_liquidity_filter = use_liquidity_filter
        self.require_liquidity_sweep = require_liquidity_sweep
        self.liquidity_lookback_candles = liquidity_lookback_candles
        
        # Initialize detectors
        self.fvg_detector = FVGDetector(min_sensitivity=min_fvg_sensitivity)
        self.swing_detector = SwingDetector()
        self.bos_choch_detector = BoSCHoCHDetector(left_bars=swing_length, right_bars=swing_length)
        
        # Execution windows in NY timezone (24-hour format)
        self.execution_windows = []
        if enable_london_open:
            self.execution_windows.append((3, 4))   # London Open
        if enable_ny_open:
            self.execution_windows.append((10, 11)) # NY Open
        if enable_ny_afternoon:
            self.execution_windows.append((14, 15)) # NY Afternoon
        
        # Track trades per session to enforce one trade per window
        self.session_trades = {}
        
        # Caching for performance optimization
        self._cache = {}
        self._timeframe = None
        self._ny_tz = pytz.timezone('America/New_York')
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate FVG-based trading signals on 1-minute timeframe (optimized)."""
        
        if not self.validate_data(data):
            raise ValueError("Invalid input data format")
        
        # Initialize signal columns
        signals_df = data.copy()
        signals_df['signal'] = 0
        signals_df['position_size'] = 0.0
        signals_df['stop_loss'] = np.nan
        signals_df['take_profit'] = np.nan
        
        # Need sufficient data for analysis
        min_required = self.m15_lookback_candles * 15
        if len(data) < min_required:
            return signals_df
        
        # Cache timeframe detection
        if self._timeframe is None:
            self._timeframe = self._detect_timeframe(data)
        if self._timeframe != '1min':
            return signals_df
        
        # Generate data fingerprint for caching
        data_fingerprint = self._get_data_fingerprint(data)
        
        # Cache M15 resample and FVG detection
        cache_key_m15 = f"m15_data_{data_fingerprint}"
        cache_key_fvgs = f"m15_fvgs_{data_fingerprint}"
        
        if cache_key_m15 in self._cache:
            m15_data = self._cache[cache_key_m15]
            m15_fvgs = self._cache[cache_key_fvgs]
        else:
            # Resample to M15 for context FVGs
            m15_data = data.resample('15min').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Detect M15 FVGs for context
            m15_fvgs = self.fvg_detector.detect_fvgs(m15_data, merge_consecutive=True)
            
            # Cache results
            self._cache[cache_key_m15] = m15_data
            self._cache[cache_key_fvgs] = m15_fvgs
        
        if not m15_fvgs:
            return signals_df
        
        # Convert timezone once and create boolean masks
        ny_timestamps = self._get_ny_timestamps(data.index)
        in_window_mask = self._get_execution_window_mask(ny_timestamps)
        weekend_mask = self._get_weekend_mask(ny_timestamps)
        
        # Detect all M1 FVGs at once (for 3-candle patterns)
        m1_fvg_indices = self._detect_all_m1_fvgs(data)
        if len(m1_fvg_indices) == 0:
            return signals_df
        
        # Extract OHLC arrays for fast access
        open_arr = data['Open'].values
        high_arr = data['High'].values
        low_arr = data['Low'].values
        close_arr = data['Close'].values
        
        # Precompute session keys for vectorized one-trade-per-session
        session_keys = self._compute_session_keys(ny_timestamps, in_window_mask)
        
        # Track signals to apply one-trade-per-session rule
        valid_signals = []
        
        # Process only M1 FVG completion indices
        for idx, fvg_info in m1_fvg_indices:
            # Skip if before minimum required data
            if idx < min_required:
                continue
            
            # Check basic filters using masks
            if not in_window_mask[idx] or weekend_mask[idx]:
                continue
            
            current_price = close_arr[idx]
            current_time = data.index[idx]
            
            # Check if price is in any active M15 FVG
            active_m15_fvg = self._find_active_m15_fvg_fast(
                m15_fvgs, m15_data, current_price, current_time, idx
            )
            if not active_m15_fvg:
                continue
            
            # Check if M1 and M15 FVGs are in same direction
            if fvg_info['type'] != active_m15_fvg.fvg_type:
                continue
            
            # Apply Smart Money Concepts filters (if enabled)
            if not self._check_smc_filters_fast(data, m15_data, idx, current_time, fvg_info['type']):
                continue
            
            # Generate signal based on FVG type
            if fvg_info['type'] == 'bullish':
                signal = 1
                stop_loss = low_arr[idx-2]  # Low of first candle in M1 FVG
            else:  # bearish
                signal = -1
                stop_loss = high_arr[idx-2]  # High of first candle in M1 FVG
            
            # Calculate take profit based on risk/reward ratio
            if signal == 1:
                risk_distance = current_price - stop_loss
                if risk_distance <= 0:
                    continue
                take_profit = current_price + (risk_distance * self.risk_reward_ratio)
            else:
                risk_distance = stop_loss - current_price
                if risk_distance <= 0:
                    continue
                take_profit = current_price - (risk_distance * self.risk_reward_ratio)
            
            # Store valid signal
            valid_signals.append({
                'idx': idx,
                'signal': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'session_key': session_keys[idx]
            })
        
        # Apply one-trade-per-session rule vectorized
        if valid_signals:
            seen_sessions = set()
            for sig in valid_signals:
                if sig['session_key'] and sig['session_key'] not in seen_sessions:
                    # Apply signal
                    idx = sig['idx']
                    signals_df.iloc[idx, signals_df.columns.get_loc('signal')] = sig['signal']
                    signals_df.iloc[idx, signals_df.columns.get_loc('position_size')] = self.position_size
                    signals_df.iloc[idx, signals_df.columns.get_loc('stop_loss')] = sig['stop_loss']
                    signals_df.iloc[idx, signals_df.columns.get_loc('take_profit')] = sig['take_profit']
                    
                    seen_sessions.add(sig['session_key'])
        
        return signals_df
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect the timeframe of the data based on index frequency."""
        if len(data) < 2:
            return 'unknown'
        
        time_diffs = data.index[1:] - data.index[:-1]
        most_common_diff = pd.Series(time_diffs).mode().iloc[0]
        
        if most_common_diff == timedelta(minutes=1):
            return '1min'
        elif most_common_diff == timedelta(minutes=15):
            return '15min'
        else:
            return 'unknown'
    
    def _is_execution_window(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp falls within execution windows (NY timezone)."""
        ny_tz = pytz.timezone('America/New_York')
        
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        ny_time = timestamp.astimezone(ny_tz)
        current_hour = ny_time.hour
        
        # Check if current hour falls within any execution window
        for start_hour, end_hour in self.execution_windows:
            if start_hour <= current_hour < end_hour:
                return True
        
        return False
    
    def _is_weekend(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp falls on weekend (Saturday/Sunday)."""
        ny_tz = pytz.timezone('America/New_York')
        
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
            
        ny_time = timestamp.astimezone(ny_tz)
        return ny_time.weekday() >= 5
    
    def _has_traded_in_session(self, timestamp: pd.Timestamp) -> bool:
        """Check if we've already traded in this session."""
        session_key = self._get_session_key(timestamp)
        return session_key in self.session_trades
    
    def _mark_session_traded(self, timestamp: pd.Timestamp):
        """Mark current session as having traded."""
        session_key = self._get_session_key(timestamp)
        if session_key:
            self.session_trades[session_key] = True
    
    def _get_session_key(self, timestamp: pd.Timestamp) -> str:
        """Get session key for tracking trades per session."""
        ny_tz = pytz.timezone('America/New_York')
        
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
            
        ny_time = timestamp.astimezone(ny_tz)
        current_hour = ny_time.hour
        
        # Determine which window we're in
        window_id = None
        for i, (start_hour, end_hour) in enumerate(self.execution_windows):
            if start_hour <= current_hour < end_hour:
                window_id = i
                break
        
        if window_id is not None:
            return f"{ny_time.date()}_{window_id}"
        return None
    
    def _get_data_fingerprint(self, data: pd.DataFrame) -> str:
        """Generate a fingerprint for the data to use as cache key."""
        # Use index range and length to create a unique fingerprint
        fingerprint_str = f"{data.index[0]}_{data.index[-1]}_{len(data)}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    
    def _get_ny_timestamps(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Convert timestamps to NY timezone once."""
        if index.tz is None:
            index = index.tz_localize('UTC')
        return np.array([ts.astimezone(self._ny_tz) for ts in index])
    
    def _get_execution_window_mask(self, ny_timestamps: np.ndarray) -> np.ndarray:
        """Create boolean mask for execution windows."""
        mask = np.zeros(len(ny_timestamps), dtype=bool)
        for ts_idx, ts in enumerate(ny_timestamps):
            hour = ts.hour
            for start_hour, end_hour in self.execution_windows:
                if start_hour <= hour < end_hour:
                    mask[ts_idx] = True
                    break
        return mask
    
    def _get_weekend_mask(self, ny_timestamps: np.ndarray) -> np.ndarray:
        """Create boolean mask for weekends."""
        return np.array([ts.weekday() >= 5 for ts in ny_timestamps])
    
    def _detect_all_m1_fvgs(self, data: pd.DataFrame) -> list:
        """Detect all M1 FVGs at once and return their completion indices."""
        m1_fvg_indices = []
        
        # Need at least 3 candles for FVG pattern
        if len(data) < 3:
            return m1_fvg_indices
        
        # Get arrays for fast access
        high_arr = data['High'].values
        low_arr = data['Low'].values
        
        # Check each potential 3-candle FVG pattern
        for i in range(2, len(data)):
            # Check for bullish FVG: low[i] > high[i-2]
            if low_arr[i] > high_arr[i-2]:
                m1_fvg_indices.append((i, {
                    'type': 'bullish',
                    'top': low_arr[i],
                    'bottom': high_arr[i-2]
                }))
            # Check for bearish FVG: high[i] < low[i-2]
            elif high_arr[i] < low_arr[i-2]:
                m1_fvg_indices.append((i, {
                    'type': 'bearish',
                    'top': low_arr[i-2],
                    'bottom': high_arr[i]
                }))
        
        return m1_fvg_indices
    
    def _compute_session_keys(self, ny_timestamps: np.ndarray, in_window_mask: np.ndarray) -> list:
        """Precompute session keys for all timestamps."""
        session_keys = [None] * len(ny_timestamps)
        
        for i, ts in enumerate(ny_timestamps):
            if not in_window_mask[i]:
                continue
            
            hour = ts.hour
            window_id = None
            for j, (start_hour, end_hour) in enumerate(self.execution_windows):
                if start_hour <= hour < end_hour:
                    window_id = j
                    break
            
            if window_id is not None:
                session_keys[i] = f"{ts.date()}_{window_id}"
        
        return session_keys
    
    def _find_active_m15_fvg_fast(self, m15_fvgs: list, m15_data: pd.DataFrame, current_price: float,
                                  current_time: pd.Timestamp, current_m1_idx: int):
        """Find active M15 FVG that current price is within (optimized)."""
        # Only look at recent FVGs (within lookback window)
        lookback_candles = self.m15_lookback_candles
        m15_len = len(m15_data)
        
        for fvg in m15_fvgs:
            # Skip FVGs that are too old (beyond lookback)
            if fvg.end_idx < m15_len - lookback_candles:
                continue
            
            # Check if price is within FVG range
            if fvg.bottom <= current_price <= fvg.top:
                # Check if FVG is still unmitigated (optimized)
                if self._is_fvg_unmitigated_fast(m15_data, fvg, fvg.end_idx, m15_len):
                    return fvg
        
        return None
    
    def _check_smc_filters_fast(self, data: pd.DataFrame, m15_data: pd.DataFrame,
                                current_idx: int, current_time: pd.Timestamp, direction: str) -> bool:
        """Check Smart Money Concepts filters (optimized version)."""
        # Skip filters if none are enabled
        if not self.use_swing_hl_filter and not self.use_bos_choch_filter and not self.use_liquidity_filter:
            return True
        
        # Use existing filter methods with minor optimizations
        if self.use_swing_hl_filter:
            if not self._check_swing_filter_proper(m15_data, current_time, direction):
                return False
        
        if self.use_bos_choch_filter:
            if not self._check_bos_choch_filter_proper(m15_data, current_time, direction):
                return False
        
        if self.use_liquidity_filter:
            if not self._check_liquidity_filter_proper(m15_data, current_time, direction):
                return False
        
        return True
    
    def _find_active_m15_fvg(self, m15_fvgs: list, m15_data: pd.DataFrame, current_price: float, 
                            current_time: pd.Timestamp, current_m1_idx: int):
        """Find active M15 FVG that current price is within."""
        # Only look at recent FVGs (within lookback window)
        lookback_candles = self.m15_lookback_candles
        
        for fvg in m15_fvgs:
            # Skip FVGs that are too old (beyond lookback)
            if fvg.end_idx < len(m15_data) - lookback_candles:
                continue
            
            # Check if price is within FVG range
            if fvg.bottom <= current_price <= fvg.top:
                # Check if FVG is still unmitigated
                if self._is_fvg_unmitigated(m15_data, fvg, fvg.end_idx, len(m15_data)):
                    return fvg
        
        return None
    
    def _is_fvg_unmitigated_fast(self, data: pd.DataFrame, fvg, fvg_end_idx: int, current_idx: int) -> bool:
        """
        Check if an FVG is unmitigated (optimized with arrays).
        """
        # Get arrays for fast access
        high_arr = data['High'].values
        low_arr = data['Low'].values
        
        # Look at all candles from after FVG formation until current candle
        for i in range(fvg_end_idx + 1, min(current_idx, len(data))):
            if fvg.fvg_type == 'bullish':
                # Bullish FVG is mitigated if price closes below the bottom of the gap
                if low_arr[i] <= fvg.bottom:
                    return False  # FVG is mitigated
            elif fvg.fvg_type == 'bearish':
                # Bearish FVG is mitigated if price closes above the top of the gap
                if high_arr[i] >= fvg.top:
                    return False  # FVG is mitigated
        
        return True  # FVG is unmitigated
    
    def _is_fvg_unmitigated(self, data: pd.DataFrame, fvg, fvg_end_idx: int, current_idx: int) -> bool:
        """
        Check if an FVG is unmitigated (hasn't been fully retraced).
        
        Args:
            data: OHLCV data (M15 or M1)
            fvg: FVG object
            fvg_end_idx: Index where FVG formation completed
            current_idx: Current candle index to check up to
            
        Returns:
            True if FVG is unmitigated (hasn't been fully closed)
        """
        # Look at all candles from after FVG formation until current candle
        for i in range(fvg_end_idx + 1, min(current_idx, len(data))):
            candle_high = data.iloc[i]['High']
            candle_low = data.iloc[i]['Low']
            
            # Check if FVG has been fully mitigated
            if fvg.fvg_type == 'bullish':
                # Bullish FVG is mitigated if price closes below the bottom of the gap
                if candle_low <= fvg.bottom:
                    return False  # FVG is mitigated
            elif fvg.fvg_type == 'bearish':
                # Bearish FVG is mitigated if price closes above the top of the gap
                if candle_high >= fvg.top:
                    return False  # FVG is mitigated
        
        return True  # FVG is unmitigated
    
    def _check_smc_filters(self, data: pd.DataFrame, m15_data: pd.DataFrame, 
                          current_idx: int, current_time: pd.Timestamp, direction: str) -> bool:
        """Check Smart Money Concepts filters using framework detectors."""
        
        # Skip filters if none are enabled
        if not self.use_swing_hl_filter and not self.use_bos_choch_filter and not self.use_liquidity_filter:
            return True
        
        # Check swing filter using SwingDetector
        if self.use_swing_hl_filter:
            if not self._check_swing_filter_proper(m15_data, current_time, direction):
                return False
        
        # Check BOS/CHoCH filter using BoSCHoCHDetector
        if self.use_bos_choch_filter:
            if not self._check_bos_choch_filter_proper(m15_data, current_time, direction):
                return False
        
        # Check liquidity filter (simplified volume-based for now)
        if self.use_liquidity_filter:
            if not self._check_liquidity_filter_proper(m15_data, current_time, direction):
                return False
        
        return True
    
    def _check_swing_filter_proper(self, m15_data: pd.DataFrame, current_time: pd.Timestamp, direction: str) -> bool:
        """Check swing filter using SwingDetector."""
        if len(m15_data) < self.swing_length * 4:
            return True  # Not enough data, allow trade
        
        # Look back specified number of candles
        lookback_candles = min(20, len(m15_data) - self.swing_length * 2)
        recent_data = m15_data.tail(lookback_candles)
        
        if direction == 'bullish' and self.require_swing_low_for_long:
            # Find pivot lows in recent data
            pivot_lows = self.swing_detector.find_all_pivot_lows(
                recent_data['Low'], self.swing_length, self.swing_length
            )
            return len(pivot_lows) > 0
        
        elif direction == 'bearish' and self.require_swing_high_for_short:
            # Find pivot highs in recent data  
            pivot_highs = self.swing_detector.find_all_pivot_highs(
                recent_data['High'], self.swing_length, self.swing_length
            )
            return len(pivot_highs) > 0
        
        return True
    
    def _check_bos_choch_filter_proper(self, m15_data: pd.DataFrame, current_time: pd.Timestamp, direction: str) -> bool:
        """Check BOS/CHoCH filter using BoSCHoCHDetector."""
        if len(m15_data) < self.bos_lookback_candles + 20:
            return True  # Not enough data, allow trade
        
        # Get recent data for BOS/CHoCH analysis
        lookback_candles = min(self.bos_lookback_candles + 20, len(m15_data))
        recent_data = m15_data.tail(lookback_candles)
        
        # Detect BOS/CHoCH patterns
        try:
            bos_choch_result = self.bos_choch_detector.detect_bos_choch(recent_data)
            
            if direction == 'bullish':
                if self.require_bos_for_entry:
                    # Look for recent bullish BOS
                    recent_bos = bos_choch_result[~pd.isna(bos_choch_result['bos']) & (bos_choch_result['bos'] == 1)]
                    if len(recent_bos) == 0:
                        return False
                
                if self.require_choch_for_entry:
                    # Look for recent bullish CHoCH
                    recent_choch = bos_choch_result[~pd.isna(bos_choch_result['choch']) & (bos_choch_result['choch'] == 1)]
                    if len(recent_choch) == 0:
                        return False
            
            elif direction == 'bearish':
                if self.require_bos_for_entry:
                    # Look for recent bearish BOS
                    recent_bos = bos_choch_result[~pd.isna(bos_choch_result['bos']) & (bos_choch_result['bos'] == -1)]
                    if len(recent_bos) == 0:
                        return False
                
                if self.require_choch_for_entry:
                    # Look for recent bearish CHoCH
                    recent_choch = bos_choch_result[~pd.isna(bos_choch_result['choch']) & (bos_choch_result['choch'] == -1)]
                    if len(recent_choch) == 0:
                        return False
            
            return True
            
        except Exception:
            return True  # Allow trade if detector fails
    
    def _check_liquidity_filter_proper(self, m15_data: pd.DataFrame, current_time: pd.Timestamp, direction: str) -> bool:
        """Check liquidity filter using volume analysis."""
        if not self.require_liquidity_sweep:
            return True
        
        if len(m15_data) < self.liquidity_lookback_candles:
            return True  # Not enough data, allow trade
        
        # Simple liquidity detection using volume spikes
        recent_data = m15_data.tail(self.liquidity_lookback_candles)
        
        # Calculate volume threshold (2x average volume indicates potential liquidity sweep)
        avg_volume = recent_data['Volume'].mean()
        volume_threshold = avg_volume * 2.0
        
        # Look for high volume candles (liquidity sweeps)
        high_volume_candles = recent_data[recent_data['Volume'] > volume_threshold]
        
        return len(high_volume_candles) > 0
    
    def get_description(self) -> str:
        """Return strategy description."""
        sessions = []
        if self.enable_london_open:
            sessions.append("London")
        if self.enable_ny_open:
            sessions.append("NY Open")  
        if self.enable_ny_afternoon:
            sessions.append("NY PM")
        
        filters = []
        if self.use_swing_hl_filter:
            filters.append("Swing")
        if self.use_bos_choch_filter:
            filters.append("BOS")
        if self.use_liquidity_filter:
            filters.append("Liquidity")
        
        return (f"Silver Bullet FVG Strategy (1M + 15M): M15 FVG setup + M1 FVG trigger, "
                f"M15 lookback {self.m15_lookback_candles} candles, R:R {self.risk_reward_ratio}:1, "
                f"Sessions: {'/'.join(sessions) if sessions else 'None'}, "
                f"Filters: {'/'.join(filters) if filters else 'None'}. "
                f"Position size: {self.position_size * 100}%.")
    
    def get_strategy_name(self) -> str:
        """Return strategy name for identification."""
        return self.name