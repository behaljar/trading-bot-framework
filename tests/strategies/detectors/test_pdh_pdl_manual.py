#!/usr/bin/env python
"""
Manual test script for the simplified PDH/PDL detector
This script demonstrates real-world usage with live market data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime, timedelta
from framework.strategies.detectors.pdh_pdl_detector import PDHPDLDetector
from framework.data.sources.ccxt_source import CCXTSource


def main():
    """Test PDH/PDL detector with real market data"""
    
    # Initialize detector
    detector = PDHPDLDetector()
    
    # Download fresh data from Binance
    print("Downloading BTC/USDT data from Binance...")
    source = CCXTSource(
        exchange_name="binance",
        api_key=None,
        api_secret=None,
        sandbox=False
    )
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)  # 10 days of data
    
    df = source.get_historical_data(
        symbol="BTC/USDT:USDT",
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        timeframe="15m"
    )
    
    if len(df) == 0:
        print("Unable to fetch market data")
        return
    
    # Ensure proper column names
    df.columns = [col.capitalize() for col in df.columns]
    
    print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    print()
    
    # Detect levels
    print("Detecting PDH/PDL and PWH/PWL levels...")
    levels = detector.detect_levels(df)
    
    # Display results
    print("\n" + "=" * 60)
    print("BTC/USDT Previous Day/Week High/Low Levels")
    print("=" * 60)
    
    # Daily levels
    if levels['PDH']:
        pdh = levels['PDH']
        status = "✓ BROKEN" if pdh.is_broken else "✗ UNBROKEN"
        print(f"\nPDH (Previous Day High): ${pdh.price:,.2f}")
        print(f"  Date: {pdh.date.strftime('%Y-%m-%d')}")
        print(f"  Status: {status}")
        if pdh.is_broken and pdh.break_time:
            print(f"  Break Time: {pdh.break_time}")
    
    if levels['PDL']:
        pdl = levels['PDL']
        status = "✓ BROKEN" if pdl.is_broken else "✗ UNBROKEN"
        print(f"\nPDL (Previous Day Low): ${pdl.price:,.2f}")
        print(f"  Date: {pdl.date.strftime('%Y-%m-%d')}")
        print(f"  Status: {status}")
        if pdl.is_broken and pdl.break_time:
            print(f"  Break Time: {pdl.break_time}")
    
    # Weekly levels
    if levels['PWH']:
        pwh = levels['PWH']
        status = "✓ BROKEN" if pwh.is_broken else "✗ UNBROKEN"
        print(f"\nPWH (Previous Week High): ${pwh.price:,.2f}")
        print(f"  Week Ending: {pwh.date.strftime('%Y-%m-%d')}")
        print(f"  Status: {status}")
        if pwh.is_broken and pwh.break_time:
            print(f"  Break Time: {pwh.break_time}")
    
    if levels['PWL']:
        pwl = levels['PWL']
        status = "✓ BROKEN" if pwl.is_broken else "✗ UNBROKEN"
        print(f"\nPWL (Previous Week Low): ${pwl.price:,.2f}")
        print(f"  Week Ending: {pwl.date.strftime('%Y-%m-%d')}")
        print(f"  Status: {status}")
        if pwl.is_broken and pwl.break_time:
            print(f"  Break Time: {pwl.break_time}")
    
    # Get summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    summary = detector.get_summary(levels)
    
    if 'daily_range' in summary:
        print(f"\nDaily Range: ${summary['daily_range']:,.2f} ({summary['daily_range_pct']:.2f}%)")
    
    if 'weekly_range' in summary:
        print(f"Weekly Range: ${summary['weekly_range']:,.2f} ({summary['weekly_range_pct']:.2f}%)")
    
    # Current price analysis
    current_price = df['Close'].iloc[-1]
    print(f"\nCurrent Price: ${current_price:,.2f}")
    
    # Check position relative to levels
    print("\nPosition Analysis:")
    
    if levels['PDH'] and levels['PDL']:
        if current_price > levels['PDH'].price:
            print("  ↑ Above PDH (Bullish)")
        elif current_price < levels['PDL'].price:
            print("  ↓ Below PDL (Bearish)")
        else:
            pct_from_pdl = ((current_price - levels['PDL'].price) / 
                           (levels['PDH'].price - levels['PDL'].price) * 100)
            print(f"  ↔ Within daily range ({pct_from_pdl:.1f}% from PDL)")
    
    if levels['PWH'] and levels['PWL']:
        if current_price > levels['PWH'].price:
            print("  ↑ Above PWH (Strong Bullish)")
        elif current_price < levels['PWL'].price:
            print("  ↓ Below PWL (Strong Bearish)")
        else:
            pct_from_pwl = ((current_price - levels['PWL'].price) / 
                           (levels['PWH'].price - levels['PWL'].price) * 100)
            print(f"  ↔ Within weekly range ({pct_from_pwl:.1f}% from PWL)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()