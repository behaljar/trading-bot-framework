#!/usr/bin/env python
"""
Manual test script for Liquidity Objective Detector
This script demonstrates finding potential targets using real market data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime, timedelta
from framework.strategies.detectors.liquidity_objective_detector import (
    LiquidityObjectiveDetector, 
    TradeDirection
)
from framework.data.sources.ccxt_source import CCXTSource


def main():
    """Test liquidity objective detector with real market data"""
    
    # Initialize detector
    detector = LiquidityObjectiveDetector(
        swing_sensitivity=3,
        fvg_min_sensitivity=0.1,
        max_objectives=15
    )
    
    # Download fresh data from Binance
    print("Downloading BTC/USDT data from Binance...")
    source = CCXTSource(
        exchange_name="binance",
        api_key=None,
        api_secret=None,
        sandbox=False
    )
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)  # 10 days for comprehensive analysis
    
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
    
    current_price = df['Close'].iloc[-1]
    
    print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    print(f"Current BTC/USDT Price: ${current_price:,.2f}")
    print()
    
    # Detect bullish objectives
    print("=" * 80)
    print("BULLISH LIQUIDITY OBJECTIVES (Potential Long Targets)")
    print("=" * 80)
    
    bullish_objectives = detector.detect_objectives(
        data=df,
        trade_direction=TradeDirection.BULLISH,
        current_price=current_price,
        max_distance_pct=0.10  # Within 10% above current price
    )
    
    if bullish_objectives:
        print(f"\nFound {len(bullish_objectives)} bullish targets within 10% of current price:\n")
        for i, obj in enumerate(bullish_objectives, 1):
            confidence_stars = "★" * int(obj.confidence * 5)
            print(f"{i:2d}. {obj.level_type.upper():12} | ${obj.price:8,.2f} | "
                  f"+{obj.distance_pct*100:5.2f}% | {confidence_stars:5} ({obj.confidence:.2f})")
            
            # Show additional details
            if 'timestamp' in obj.details:
                print(f"     └─ Formed: {obj.details['timestamp']}")
            elif 'level_date' in obj.details:
                print(f"     └─ Date: {obj.details['level_date']}")
    else:
        print("No bullish objectives found within 10% range.")
    
    # Get bullish summary
    bullish_summary = detector.get_summary(bullish_objectives, current_price)
    if bullish_objectives:
        print(f"\n📊 Bullish Summary:")
        print(f"   • Closest Target: {bullish_summary['closest_objective']['level_type']} "
              f"at ${bullish_summary['closest_objective']['price']:,.2f} "
              f"(+{bullish_summary['closest_objective']['distance_pct']:.2f}%)")
        print(f"   • Highest Confidence: {bullish_summary['highest_confidence']['level_type']} "
              f"at ${bullish_summary['highest_confidence']['price']:,.2f} "
              f"(confidence: {bullish_summary['highest_confidence']['confidence']:.2f})")
        print(f"   • Average Distance: {bullish_summary['avg_distance_pct']:.2f}%")
        print(f"   • Level Types: {bullish_summary['level_types']}")
    
    # Detect bearish objectives
    print("\n" + "=" * 80)
    print("BEARISH LIQUIDITY OBJECTIVES (Potential Short Targets)")
    print("=" * 80)
    
    bearish_objectives = detector.detect_objectives(
        data=df,
        trade_direction=TradeDirection.BEARISH,
        current_price=current_price,
        max_distance_pct=0.10  # Within 10% below current price
    )
    
    if bearish_objectives:
        print(f"\nFound {len(bearish_objectives)} bearish targets within 10% of current price:\n")
        for i, obj in enumerate(bearish_objectives, 1):
            confidence_stars = "★" * int(obj.confidence * 5)
            print(f"{i:2d}. {obj.level_type.upper():12} | ${obj.price:8,.2f} | "
                  f"-{obj.distance_pct*100:5.2f}% | {confidence_stars:5} ({obj.confidence:.2f})")
            
            # Show additional details
            if 'timestamp' in obj.details:
                print(f"     └─ Formed: {obj.details['timestamp']}")
            elif 'level_date' in obj.details:
                print(f"     └─ Date: {obj.details['level_date']}")
    else:
        print("No bearish objectives found within 10% range.")
    
    # Get bearish summary
    bearish_summary = detector.get_summary(bearish_objectives, current_price)
    if bearish_objectives:
        print(f"\n📊 Bearish Summary:")
        print(f"   • Closest Target: {bearish_summary['closest_objective']['level_type']} "
              f"at ${bearish_summary['closest_objective']['price']:,.2f} "
              f"(-{bearish_summary['closest_objective']['distance_pct']:.2f}%)")
        print(f"   • Highest Confidence: {bearish_summary['highest_confidence']['level_type']} "
              f"at ${bearish_summary['highest_confidence']['price']:,.2f} "
              f"(confidence: {bearish_summary['highest_confidence']['confidence']:.2f})")
        print(f"   • Average Distance: {bearish_summary['avg_distance_pct']:.2f}%")
        print(f"   • Level Types: {bearish_summary['level_types']}")
    
    # Extended range analysis
    print("\n" + "=" * 80)
    print("EXTENDED RANGE ANALYSIS (Within 20%)")
    print("=" * 80)
    
    # Get more objectives with larger range
    extended_bullish = detector.detect_objectives(
        data=df,
        trade_direction=TradeDirection.BULLISH,
        current_price=current_price,
        max_distance_pct=0.20  # Within 20% above
    )
    
    extended_bearish = detector.detect_objectives(
        data=df,
        trade_direction=TradeDirection.BEARISH,
        current_price=current_price,
        max_distance_pct=0.20  # Within 20% below
    )
    
    print(f"\nExtended Range Targets:")
    print(f"   • Bullish (10-20% away): {len(extended_bullish) - len(bullish_objectives)} additional targets")
    print(f"   • Bearish (10-20% away): {len(extended_bearish) - len(bearish_objectives)} additional targets")
    
    # Show key levels analysis
    print("\n" + "=" * 80)
    print("KEY LEVEL ANALYSIS")
    print("=" * 80)
    
    all_objectives = bullish_objectives + bearish_objectives
    level_type_analysis = {}
    
    for obj in all_objectives:
        if obj.level_type not in level_type_analysis:
            level_type_analysis[obj.level_type] = {'count': 0, 'avg_confidence': 0, 'prices': []}
        level_type_analysis[obj.level_type]['count'] += 1
        level_type_analysis[obj.level_type]['avg_confidence'] += obj.confidence
        level_type_analysis[obj.level_type]['prices'].append(obj.price)
    
    for level_type, data in level_type_analysis.items():
        data['avg_confidence'] /= data['count']
        price_range = f"${min(data['prices']):,.0f} - ${max(data['prices']):,.0f}"
        print(f"   • {level_type.upper():12}: {data['count']} levels, "
              f"confidence: {data['avg_confidence']:.2f}, range: {price_range}")
    
    # Trading recommendations
    print("\n" + "=" * 80)
    print("TRADING INSIGHTS")
    print("=" * 80)
    
    if bullish_objectives and bearish_objectives:
        closest_bull = min(bullish_objectives, key=lambda x: x.distance_pct)
        closest_bear = min(bearish_objectives, key=lambda x: x.distance_pct)
        
        print(f"\n🎯 Nearest Targets:")
        print(f"   • Bullish: ${closest_bull.price:,.2f} ({closest_bull.level_type}) - "
              f"{closest_bull.distance_pct*100:.2f}% away")
        print(f"   • Bearish: ${closest_bear.price:,.2f} ({closest_bear.level_type}) - "
              f"{closest_bear.distance_pct*100:.2f}% away")
        
        # Determine bias based on closest targets
        if closest_bull.distance_pct < closest_bear.distance_pct:
            print(f"\n📈 Short-term bias: Bullish (closer upside target)")
        else:
            print(f"\n📉 Short-term bias: Bearish (closer downside target)")
    
    print(f"\n⚠️  Risk Management:")
    print(f"   • Use identified levels for take-profit targets")
    print(f"   • Consider stop-losses beyond key support/resistance")
    print(f"   • Higher confidence levels (★★★+) are more reliable")
    print(f"   • Multiple objectives at similar prices indicate strong levels")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()