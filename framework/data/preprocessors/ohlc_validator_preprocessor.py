import pandas as pd
import logging
from .base_preprocessor import BasePreprocessor


class OhlcValidatorPreprocessor(BasePreprocessor):
    """
    Preprocessor that validates OHLC (Open, High, Low, Close) data consistency.
    Ensures that High >= Low, High >= Open/Close, and Low <= Open/Close.
    """
    
    def __init__(self):
        super().__init__("OhlcValidatorPreprocessor")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLC data consistency.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with validated OHLC data
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for OHLC validation")
            return df
        
        # Check if OHLC columns exist (case-insensitive)
        required_cols = ['Open', 'High', 'Low', 'Close']
        
        # Find actual column names
        ohlc_cols = {}
        for req_col in required_cols:
            for actual_col in df.columns:
                if actual_col.lower() == req_col.lower():
                    ohlc_cols[req_col] = actual_col
                    break
        
        # Check if all required columns are present
        missing_cols = [col for col in required_cols if col not in ohlc_cols]
        if missing_cols:
            logging.warning(f"Missing OHLC columns: {missing_cols}, skipping OHLC validation")
            return df
        
        df_clean = df.copy()
        invalid_count = 0
        
        # Validate High >= Low
        high_col = ohlc_cols['High']
        low_col = ohlc_cols['Low']
        invalid_high_low = df_clean[high_col] < df_clean[low_col]
        if invalid_high_low.any():
            count = invalid_high_low.sum()
            logging.warning(f"Found {count} rows where High < Low, fixing by swapping values")
            # Swap high and low where invalid
            df_clean.loc[invalid_high_low, [high_col, low_col]] = df_clean.loc[invalid_high_low, [low_col, high_col]].values
            invalid_count += count
        
        # Validate High >= Open and High >= Close
        open_col = ohlc_cols['Open']
        close_col = ohlc_cols['Close']
        
        invalid_high_open = df_clean[high_col] < df_clean[open_col]
        if invalid_high_open.any():
            count = invalid_high_open.sum()
            logging.warning(f"Found {count} rows where High < Open, adjusting High")
            # Use maximum to ensure high is at least as large as open
            df_clean.loc[invalid_high_open, high_col] = df_clean.loc[invalid_high_open, [high_col, open_col]].max(axis=1)
            invalid_count += count
        
        invalid_high_close = df_clean[high_col] < df_clean[close_col]
        if invalid_high_close.any():
            count = invalid_high_close.sum()
            logging.warning(f"Found {count} rows where High < Close, adjusting High")
            # Use maximum to ensure high is at least as large as close
            df_clean.loc[invalid_high_close, high_col] = df_clean.loc[invalid_high_close, [high_col, close_col]].max(axis=1)
            invalid_count += count
        
        # Validate Low <= Open and Low <= Close
        invalid_low_open = df_clean[low_col] > df_clean[open_col]
        if invalid_low_open.any():
            count = invalid_low_open.sum()
            logging.warning(f"Found {count} rows where Low > Open, adjusting Low")
            # Use minimum to ensure low is at most as large as open
            df_clean.loc[invalid_low_open, low_col] = df_clean.loc[invalid_low_open, [low_col, open_col]].min(axis=1)
            invalid_count += count
        
        invalid_low_close = df_clean[low_col] > df_clean[close_col]
        if invalid_low_close.any():
            count = invalid_low_close.sum()
            logging.warning(f"Found {count} rows where Low > Close, adjusting Low")
            # Use minimum to ensure low is at most as large as close
            df_clean.loc[invalid_low_close, low_col] = df_clean.loc[invalid_low_close, [low_col, close_col]].min(axis=1)
            invalid_count += count
        
        # Final validation check
        final_invalid_high_low = df_clean[high_col] < df_clean[low_col]
        if final_invalid_high_low.any():
            count = final_invalid_high_low.sum()
            logging.error(f"Still have {count} rows with invalid High/Low after fixes")
            # As a last resort, remove these rows
            df_clean = df_clean[~final_invalid_high_low]
            logging.warning(f"Removed {count} rows with persistent OHLC violations")
        
        if invalid_count > 0:
            logging.info(f"OHLC validation completed, fixed {invalid_count} issues")
        else:
            logging.info("OHLC validation completed, no issues found")
        
        return df_clean