import pandas as pd
import logging
from .base_preprocessor import BasePreprocessor


class NanValuePreprocessor(BasePreprocessor):
    """
    Preprocessor that handles NaN values by dropping affected rows.
    This is the safest approach for trading data to avoid look-ahead bias.
    """
    
    def __init__(self):
        super().__init__("NanValuePreprocessor")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values by dropping rows containing them.
        
        Args:
            df: DataFrame with potential NaN values
            
        Returns:
            DataFrame with rows containing NaN values removed
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for NaN value processing")
            return df
            
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        
        if nan_count > 0:
            initial_rows = len(df)
            
            # Log which columns have NaN values
            nan_columns = df.isnull().sum()
            nan_columns = nan_columns[nan_columns > 0]
            for col, count in nan_columns.items():
                logging.info(f"Column '{col}': {count} NaN values")
            
            # Drop rows with any NaN values
            df_clean = df.dropna()
            
            rows_dropped = initial_rows - len(df_clean)
            logging.warning(f"Found {nan_count} NaN values in {rows_dropped} rows, dropping them")
            
            if df_clean.empty:
                logging.warning("All rows contained NaN values and were dropped")
        else:
            logging.info("No NaN values found")
            df_clean = df
        
        return df_clean