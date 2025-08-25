import pandas as pd
import numpy as np
import logging
from .base_preprocessor import BasePreprocessor


class InfiniteValuePreprocessor(BasePreprocessor):
    """
    Preprocessor that handles infinite values (inf and -inf) in the DataFrame.
    """
    
    def __init__(self):
        super().__init__("InfiniteValuePreprocessor")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle infinite values in the DataFrame.
        
        Args:
            df: DataFrame with potential infinite values
            
        Returns:
            DataFrame with infinite values handled
        """
        if df.empty:
            logging.warning("Empty DataFrame provided for infinite value processing")
            return df
            
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        if inf_count > 0:
            logging.warning(f"Found {inf_count} infinite values")
            
            # Log which columns have infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                inf_in_col = np.isinf(df[col]).sum()
                if inf_in_col > 0:
                    logging.info(f"Column '{col}': {inf_in_col} infinite values")
            
            # Replace infinite values with NaN (to be handled by NanValuePreprocessor)
            df_clean = df.replace([np.inf, -np.inf], np.nan)
            
            # Log the result
            remaining_inf = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
            if remaining_inf == 0:
                logging.info(f"Successfully replaced {inf_count} infinite values with NaN")
            else:
                logging.error(f"Failed to replace all infinite values, {remaining_inf} remain")
        else:
            logging.info("No infinite values found")
            df_clean = df
        
        return df_clean