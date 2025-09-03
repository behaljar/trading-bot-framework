import pandas as pd
import logging
from .base_preprocessor import BasePreprocessor


class ColumnCapitalizationPreprocessor(BasePreprocessor):
    """
    Preprocessor that capitalizes the first letter of column names, except for 'timestamp'.
    
    Examples:
        - 'open' -> 'Open'
        - 'high' -> 'High'
        - 'low' -> 'Low'
        - 'close' -> 'Close'
        - 'volume' -> 'Volume'
        - 'timestamp' -> 'timestamp' (stays lowercase)
    """
    
    def __init__(self, name: str = "ColumnCapitalizationPreprocessor"):
        super().__init__(name)
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame by capitalizing the first letter of all column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with capitalized column names
        """
        if df.empty:
            logging.warning(f"{self.name}: DataFrame is empty, skipping processing")
            return df
            
        # Get original column names
        original_columns = list(df.columns)
        
        # Create mapping of old column names to capitalized names (except timestamp)
        column_mapping = {}
        for col in original_columns:
            if col.lower() == 'timestamp':
                column_mapping[col] = 'timestamp'  # Keep timestamp lowercase
            else:
                column_mapping[col] = col.capitalize()
        
        # Only log if there are actual changes to be made
        changes_to_make = [col for col in original_columns if column_mapping[col] != col]
        
        if changes_to_make:
            logging.info(f"{self.name}: Capitalizing column names: {changes_to_make}")
            # Apply the column name changes
            df_processed = df.rename(columns=column_mapping)
            logging.info(f"{self.name}: Columns renamed from {original_columns} to {list(df_processed.columns)}")
        else:
            logging.info(f"{self.name}: All column names already capitalized, no changes needed")
            df_processed = df.copy()
            
        return df_processed
        
    def can_process(self, df: pd.DataFrame) -> bool:
        """
        Check if this preprocessor can process the given DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if DataFrame is not empty
        """
        return not df.empty