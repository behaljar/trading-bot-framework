import pandas as pd
import logging
from typing import Optional
from .base_preprocessor import BasePreprocessor


class TimeGapDetectorPreprocessor(BasePreprocessor):
    """
    Preprocessor that detects and reports significant time gaps in the data.
    This helps identify missing data periods that might affect trading strategies.
    """
    
    def __init__(self, gap_threshold_minutes: int = 60):
        """
        Initialize the TimeGapDetectorPreprocessor.
        
        Args:
            gap_threshold_minutes: Minimum gap size in minutes to report (default: 60)
        """
        super().__init__("TimeGapDetectorPreprocessor")
        self.gap_threshold_minutes = gap_threshold_minutes
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and report time gaps in the data.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Original DataFrame (no modifications, only reporting)
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for time gap detection")
            return df
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.warning("DataFrame index is not DatetimeIndex, skipping time gap detection")
            return df
        
        # Calculate time differences between consecutive rows
        time_diffs = df.index.to_series().diff()
        
        # Find gaps larger than threshold
        gap_threshold = pd.Timedelta(minutes=self.gap_threshold_minutes)
        gaps = time_diffs[time_diffs > gap_threshold]
        
        if len(gaps) > 0:
            logging.warning(f"Found {len(gaps)} time gaps larger than {self.gap_threshold_minutes} minutes")
            
            # Report details of significant gaps
            for idx, gap_duration in gaps.items():
                prev_time = df.index[df.index.get_loc(idx) - 1]
                gap_hours = gap_duration.total_seconds() / 3600
                gap_days = gap_duration.total_seconds() / 86400
                
                if gap_days >= 1:
                    logging.info(f"Gap: {prev_time} to {idx} ({gap_days:.1f} days)")
                else:
                    logging.info(f"Gap: {prev_time} to {idx} ({gap_hours:.1f} hours)")
            
            # Calculate statistics
            max_gap = gaps.max()
            avg_gap = gaps.mean()
            
            max_gap_hours = max_gap.total_seconds() / 3600
            avg_gap_hours = avg_gap.total_seconds() / 3600
            
            logging.info(f"Gap statistics - Max: {max_gap_hours:.1f} hours, Avg: {avg_gap_hours:.1f} hours")
            
            # Check for weekend gaps (common in traditional markets)
            weekend_gaps = []
            for idx, gap_duration in gaps.items():
                prev_time = df.index[df.index.get_loc(idx) - 1]
                # Check if gap spans a weekend
                if prev_time.weekday() == 4 and idx.weekday() == 0:  # Friday to Monday
                    weekend_gaps.append(idx)
            
            if weekend_gaps:
                logging.info(f"Detected {len(weekend_gaps)} probable weekend gaps")
        else:
            # Calculate average time between samples
            if len(time_diffs) > 1:
                avg_interval = time_diffs[1:].mean()
                avg_minutes = avg_interval.total_seconds() / 60
                logging.info(f"No significant time gaps found. Average interval: {avg_minutes:.1f} minutes")
            else:
                logging.info("Insufficient data to detect time gaps")
        
        # Check data frequency consistency
        if len(time_diffs) > 1:
            # Remove NaN from first diff
            valid_diffs = time_diffs.dropna()
            if len(valid_diffs) > 0:
                # Calculate the mode (most common interval)
                interval_counts = valid_diffs.value_counts()
                if len(interval_counts) > 0:
                    most_common_interval = interval_counts.index[0]
                    most_common_minutes = most_common_interval.total_seconds() / 60
                    
                    # Infer likely timeframe
                    if 0.5 <= most_common_minutes <= 1.5:
                        timeframe = "1-minute"
                    elif 4 <= most_common_minutes <= 6:
                        timeframe = "5-minute"
                    elif 14 <= most_common_minutes <= 16:
                        timeframe = "15-minute"
                    elif 29 <= most_common_minutes <= 31:
                        timeframe = "30-minute"
                    elif 59 <= most_common_minutes <= 61:
                        timeframe = "1-hour"
                    elif 239 <= most_common_minutes <= 241:
                        timeframe = "4-hour"
                    elif 1439 <= most_common_minutes <= 1441:
                        timeframe = "daily"
                    else:
                        timeframe = f"{most_common_minutes:.1f}-minute"
                    
                    logging.info(f"Detected timeframe: {timeframe}")
        
        return df