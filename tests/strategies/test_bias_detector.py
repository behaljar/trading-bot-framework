#!/usr/bin/env python3
"""
Test script for Bias Detector

Tests the bias detector logic with sample data and real market data.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add framework to path
script_dir = Path(__file__).parent
framework_dir = script_dir.parent.parent
sys.path.append(str(framework_dir))

from framework.strategies.detectors.bias_detector import BiasDetector, BiasType


def create_test_data():
    """Create synthetic test data with known bias patterns"""
    
    # Create dates
    dates = pd.date_range('2024-01-01', periods=10, freq='1D')
    
    # Test case 1: Bullish bias (C2 close above C1)
    # C1: High=100, Low=90, Close=95
    # C2: High=105, Low=98, Close=102 (close above C1 high of 100)
    # Expected: Bullish bias, target C2 high (105)
    
    test_data = pd.DataFrame({
        'Open': [92, 98, 101, 96, 94, 89, 91, 88, 92, 95],
        'High': [100, 105, 103, 99, 97, 92, 94, 91, 96, 99],
        'Low': [90, 98, 99, 94, 92, 87, 89, 86, 90, 93],
        'Close': [95, 102, 100, 96, 94, 89, 91, 88, 94, 97],
        'Volume': [1000] * 10
    }, index=dates)
    
    return test_data


def test_basic_bias_logic():
    """Test basic bias detection logic"""
    print("=== Testing Basic Bias Detection Logic ===")
    
    detector = BiasDetector()
    test_data = create_test_data()
    
    print(f"Test data shape: {test_data.shape}")
    print("First 5 rows:")
    print(test_data.head())
    
    # Detect bias signals
    bias_df = detector.detect_bias_signals(test_data)
    
    print(f"\nDetected {len(bias_df)} bias signals:")
    print(bias_df[['bias', 'c2_close', 'c1_high', 'c1_low', 'validation']].head())
    
    # Calculate accuracy
    accuracy_stats = detector.calculate_accuracy(bias_df)
    print(f"\nAccuracy Statistics:")
    for key, value in accuracy_stats.items():
        if key != 'sample_validations':
            print(f"{key}: {value}")
    
    return bias_df


def test_specific_cases():
    """Test specific bias cases with controlled data"""
    print("\n=== Testing Specific Bias Cases ===")
    
    detector = BiasDetector()
    
    # Case 1: C2 close above C1 high → Bullish
    dates = pd.date_range('2024-01-01', periods=3, freq='1D')
    case1_data = pd.DataFrame({
        'Open': [90, 98, 104],
        'High': [100, 105, 108],  # C3 high (108) > C2 high (105) → Should validate
        'Low': [85, 95, 102],
        'Close': [95, 102, 106],  # C2 close (102) > C1 high (100) → Bullish
        'Volume': [1000, 1000, 1000]
    }, index=dates)
    
    bias_df1 = detector.detect_bias_signals(case1_data)
    print("Case 1 - Bullish bias (C2 close above C1):")
    print(bias_df1[['bias', 'validation', 'validation_reason']])
    
    # Case 2: C2 close below C1 low → Bearish
    case2_data = pd.DataFrame({
        'Open': [95, 88, 82],
        'High': [100, 92, 85],
        'Low': [90, 85, 80],  # C3 low (80) < C2 low (85) → Should validate
        'Close': [95, 87, 82],  # C2 close (87) < C1 low (90) → Bearish
        'Volume': [1000, 1000, 1000]
    }, index=dates)
    
    bias_df2 = detector.detect_bias_signals(case2_data)
    print("\nCase 2 - Bearish bias (C2 close below C1):")
    print(bias_df2[['bias', 'validation', 'validation_reason']])
    
    # Case 3: C2 inside C1, but C2 high > C1 high → Bearish
    case3_data = pd.DataFrame({
        'Open': [92, 95, 88],
        'High': [100, 105, 92],  # C2 high (105) > C1 high (100), C2 close inside C1
        'Low': [85, 93, 85],     # C3 low (85) < C2 low (93) → Should validate
        'Close': [90, 95, 89],   # C2 close (95) inside C1 (85-100)
        'Volume': [1000, 1000, 1000]
    }, index=dates)
    
    bias_df3 = detector.detect_bias_signals(case3_data)
    print("\nCase 3 - Bearish bias (C2 inside C1, but C2 high > C1 high):")
    print(bias_df3[['bias', 'validation', 'validation_reason']])


def test_with_real_data():
    """Test bias detector with real market data if available"""
    print("\n=== Testing with Real Market Data ===")
    
    # Try to load real data
    data_path = Path(framework_dir) / "data" / "cleaned" / "BTC_USDT_binance_4h_2024-09-03_2025-09-03_cleaned.csv"
    
    if data_path.exists():
        print(f"Loading real data from: {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Use a subset for testing (last 100 candles)
        test_data = data.tail(100).copy()
        
        detector = BiasDetector()
        bias_df = detector.detect_bias_signals(test_data)
        
        print(f"Processed {len(test_data)} candles, detected {len(bias_df)} bias signals")
        
        # Calculate accuracy
        accuracy_stats = detector.calculate_accuracy(bias_df)
        
        print(f"\nReal Data Accuracy Statistics:")
        print(f"Overall Accuracy: {accuracy_stats.get('overall_accuracy_pct', 0)}%")
        print(f"Bullish Accuracy: {accuracy_stats.get('bullish_accuracy_pct', 0)}% ({accuracy_stats.get('bullish_signals', 0)} signals)")
        print(f"Bearish Accuracy: {accuracy_stats.get('bearish_accuracy_pct', 0)}% ({accuracy_stats.get('bearish_signals', 0)} signals)")
        print(f"Bias Distribution: {accuracy_stats.get('bias_distribution', {})}")
        
        # Show sample results
        print(f"\nSample bias signals (last 10):")
        sample_cols = ['bias', 'c2_close', 'c3_high', 'c3_low', 'validation']
        print(bias_df[sample_cols].tail(10))
        
        # Test current bias
        current_bias = detector.get_current_bias(test_data)
        if current_bias:
            print(f"\nCurrent bias: {current_bias.bias.value}")
            if current_bias.target_high:
                print(f"Target high: {current_bias.target_high}")
            if current_bias.target_low:
                print(f"Target low: {current_bias.target_low}")
        
        return bias_df
    else:
        print(f"Real data not found at: {data_path}")
        print("Skipping real data test")
        return None


if __name__ == "__main__":
    print("Bias Detector Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        test_bias_df = test_basic_bias_logic()
        test_specific_cases()
        real_data_df = test_with_real_data()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()