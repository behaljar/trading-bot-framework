"""
FVG Strategy (M15 Timeframe)

Simple FVG strategy that:
- Operates only on M15 data 
- Detects M15 FVGs for entry signals
- Only trades during specific execution windows in NY timezone
- Allows only one trade per session window
- Excludes weekend trading

Entry Conditions:
1. M15 FVG detected (entry on close of 3rd candle in FVG formation)
2. Must be within execution time window
3. Only ONE trade allowed per session window  
4. NOT on weekends (Saturday/Sunday)

Exit Conditions:
- Stop Loss: Low/High of first candle in M15 FVG formation
- Take Profit: Configurable Risk/Reward ratio (default 2:1)
- Time-based exit: 2 hours maximum hold time

Execution Windows (NY Time):
- 03:00 - 04:00 (London Open)
- 10:00 - 11:00 (NY Open)  
- 14:00 - 15:00 (NY Afternoon)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import pytz
from .base_strategy import BaseStrategy
from .detectors.fvg_detector import FVGDetector, FVG


class FVGStrategy(BaseStrategy):
    """
    Simple FVG strategy using M15 FVG detection with execution windows.
    """
    
    def __init__(self, 
                 risk_reward_ratio: float = 2.0,
                 max_hold_hours: int = 2,
                 min_fvg_sensitivity: float = 0.1,
                 position_size: float = 0.1,
                 **kwargs):
        """
        Initialize FVG strategy.
        
        Args:
            risk_reward_ratio: Risk/Reward ratio for take profit (default: 2.0)
            max_hold_hours: Maximum hold time in hours (default: 2)
            min_fvg_sensitivity: Minimum FVG sensitivity ratio (default: 0.1)
            position_size: Position size as fraction of equity (default: 0.1)
        """
        super().__init__("fvg", kwargs)
        
        self.risk_reward_ratio = risk_reward_ratio
        self.max_hold_hours = max_hold_hours
        self.min_fvg_sensitivity = min_fvg_sensitivity
        self.position_size = position_size
        
        # Initialize FVG detector
        self.fvg_detector = FVGDetector(min_sensitivity=min_fvg_sensitivity)
        
        # Execution windows in NY timezone (24-hour format)
        self.execution_windows = [
            (3, 4),   # London Open: 03:00 - 04:00 NY
            (10, 11), # NY Open: 10:00 - 11:00 NY
            (14, 15)  # NY Afternoon: 14:00 - 15:00 NY
        ]
        
        # Track trades per session to enforce one trade per window
        self.session_trades = {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate FVG-based trading signals on M15 timeframe.
        
        Args:
            data: DataFrame with M15 OHLCV data and DatetimeIndex
            
        Returns:
            DataFrame with trading signals and position management columns
        """
        print(f"DEBUG: FVG Strategy generate_signals called with {len(data)} data points")
        print(f"DEBUG: Data range: {data.index[0]} to {data.index[-1]}")
        
        if not self.validate_data(data):
            raise ValueError("Invalid input data format")
        
        # Initialize signal columns
        signals_df = data.copy()
        signals_df['signal'] = 0
        signals_df['position_size'] = 0.0
        signals_df['stop_loss'] = np.nan
        signals_df['take_profit'] = np.nan
        signals_df['hold_until'] = pd.NaT
        
        # Need sufficient data for analysis
        if len(data) < 50:
            print(f"DEBUG: Insufficient data: {len(data)} < 50, returning empty signals")
            return signals_df
        
        # Detect timeframe - strategy only works with M15 data
        timeframe = self._detect_timeframe(data)
        print(f"DEBUG: Detected timeframe: {timeframe}")
        if timeframe != '15T':
            print(f"DEBUG: Strategy requires 15T (M15) timeframe, got {timeframe}, returning empty signals")
            return signals_df
        
        # Detect M15 FVGs
        m15_fvgs = self.fvg_detector.detect_fvgs(data, merge_consecutive=True)
        print(f"DEBUG: Found {len(m15_fvgs)} M15 FVGs")
        
        if len(m15_fvgs) == 0:
            print("DEBUG: No FVGs found, returning empty signals")
            return signals_df
        
        # Show first few FVGs for debugging
        for i, fvg in enumerate(m15_fvgs[:3]):
            print(f"  M15 FVG {i}: {fvg.fvg_type}, idx {fvg.start_idx}-{fvg.end_idx}, range {fvg.bottom:.6f}-{fvg.top:.6f}")
        
        # Generate signals based on FVG formations
        signals_generated = 0
        execution_window_count = 0
        weekend_count = 0
        session_limit_count = 0
        
        for fvg in m15_fvgs:
            # Entry occurs on the close of the 3rd candle (end_idx)
            entry_idx = fvg.end_idx
            
            if entry_idx >= len(data):
                continue
                
            current_time = data.index[entry_idx]
            current_price = data.iloc[entry_idx]['close']
            
            # Check if we're in an execution window
            if not self._is_execution_window(current_time):
                continue
            execution_window_count += 1
                
            # Check if it's a weekend
            if self._is_weekend(current_time):
                weekend_count += 1
                continue
                
            # Check session trade limit
            session_key = self._get_session_key(current_time)
            if session_key in self.session_trades:
                session_limit_count += 1
                continue
            
            # Generate signal based on FVG type
            if fvg.fvg_type == 'bullish':
                signal = 1  # Long
                # Stop loss at low of first candle in FVG formation
                stop_loss = data.iloc[fvg.start_idx]['low']
            elif fvg.fvg_type == 'bearish':
                signal = -1  # Short
                # Stop loss at high of first candle in FVG formation
                stop_loss = data.iloc[fvg.start_idx]['high']
            else:
                continue
            
            # Calculate take profit based on risk/reward ratio
            if signal == 1:  # Long
                risk_distance = current_price - stop_loss
                if risk_distance <= 0:
                    continue  # Invalid stop loss
                take_profit = current_price + (risk_distance * self.risk_reward_ratio)
            else:  # Short
                risk_distance = stop_loss - current_price
                if risk_distance <= 0:
                    continue  # Invalid stop loss
                take_profit = current_price - (risk_distance * self.risk_reward_ratio)
            
            # Set the signal
            signals_df.iloc[entry_idx, signals_df.columns.get_loc('signal')] = signal
            signals_df.iloc[entry_idx, signals_df.columns.get_loc('position_size')] = self.position_size
            signals_df.iloc[entry_idx, signals_df.columns.get_loc('stop_loss')] = stop_loss
            signals_df.iloc[entry_idx, signals_df.columns.get_loc('take_profit')] = take_profit
            
            # Set hold until time (max hold period)
            hold_until = current_time + timedelta(hours=self.max_hold_hours)
            signals_df.iloc[entry_idx, signals_df.columns.get_loc('hold_until')] = hold_until
            
            # Mark this session as traded
            self.session_trades[session_key] = current_time
            signals_generated += 1
            
            print(f"DEBUG: Signal generated at {current_time}: {fvg.fvg_type} FVG, signal={signal}, price={current_price:.6f}, SL={stop_loss:.6f}, TP={take_profit:.6f}")
        
        print(f"DEBUG SUMMARY:")
        print(f"  Total M15 candles processed: {len(data)}")
        print(f"  FVGs found: {len(m15_fvgs)}")
        print(f"  Execution window opportunities: {execution_window_count}")
        print(f"  Weekend blocked: {weekend_count}")
        print(f"  Session limit blocked: {session_limit_count}")
        print(f"  Signals generated: {signals_generated}")
        
        return signals_df
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """
        Detect the timeframe of the data based on index frequency.
        """
        if len(data) < 2:
            return 'unknown'
        
        # Calculate the most common time difference
        time_diffs = data.index[1:] - data.index[:-1]
        most_common_diff = pd.Series(time_diffs).mode().iloc[0]
        
        if most_common_diff == timedelta(minutes=15):
            return '15T'
        elif most_common_diff == timedelta(hours=1):
            return '1H'
        elif most_common_diff == timedelta(hours=4):
            return '4H'
        else:
            return 'unknown'
    
    def _is_execution_window(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp falls within execution windows (NY timezone).
        """
        # Convert to NY timezone
        ny_tz = pytz.timezone('America/New_York')
        
        # Convert timestamp to NY time if it's not already
        if timestamp.tz is None:
            # Assume UTC if no timezone info
            timestamp = timestamp.tz_localize('UTC')
        
        ny_time = timestamp.astimezone(ny_tz)
        current_hour = ny_time.hour
        
        # Check if current hour falls within any execution window
        for start_hour, end_hour in self.execution_windows:
            if start_hour <= current_hour < end_hour:
                return True
        
        return False
    
    def _is_weekend(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp falls on weekend (Saturday/Sunday).
        """
        # Convert to NY timezone for consistent weekend detection
        ny_tz = pytz.timezone('America/New_York')
        
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
            
        ny_time = timestamp.astimezone(ny_tz)
        
        # Saturday = 5, Sunday = 6 in Python's weekday()
        return ny_time.weekday() >= 5
    
    def _get_session_key(self, timestamp: pd.Timestamp) -> str:
        """
        Get session key for tracking trades per session.
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
        
        # Create session key: YYYY-MM-DD_window_id
        return f"{ny_time.date()}_{window_id}"
    
    def get_description(self) -> str:
        """Return strategy description."""
        return (f"FVG Strategy (M15 timeframe): M15 FVG detection, "
                f"R:R ratio {self.risk_reward_ratio}:1, max hold {self.max_hold_hours}h, "
                f"execution windows: London Open (03:00-04:00), NY Open (10:00-11:00), "
                f"NY Afternoon (14:00-15:00) NY time. Position size: {self.position_size * 100}%.")
    
    def get_strategy_name(self) -> str:
        """Return strategy name for identification."""
        return self.name