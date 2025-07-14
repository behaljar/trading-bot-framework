"""
Environment variable utilities.
"""
from typing import Optional


def clean_env_value(value: Optional[str]) -> Optional[str]:
    """
    Clean environment variable value by removing inline comments.
    
    Args:
        value: Raw environment variable value
        
    Returns:
        Cleaned value with comments removed, or None if empty
    """
    if not value:
        return None
    
    # Remove inline comments (everything after #)
    cleaned = value.split('#')[0].strip()
    
    # Return None if empty or just whitespace
    if not cleaned or cleaned.isspace():
        return None
        
    return cleaned