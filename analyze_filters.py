#!/usr/bin/env python3
"""
Analyze filter combinations from optimization results
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Read the heatmap data
data_file = Path("output/optimizations/optimize_breakout_BTC_USDT_20250701_223705/optimize_breakout_BTC_USDT_20250701_223705_heatmap_data.csv")
df = pd.read_csv(data_file)

print("=== FILTER COMBINATION ANALYSIS ===\n")

# Create filter combination labels
def create_filter_label(row):
    filters = []
    if row['use_volume_filter']:
        filters.append('Vol')
    if row['use_trend_filter']:
        filters.append('Trend')
    if row['use_ema_filter']:
        filters.append('EMA')
    
    if not filters:
        return 'No Filters'
    return '+'.join(filters)

df['filter_combo'] = df.apply(create_filter_label, axis=1)

# Group by filter combinations and show statistics
print("1. PERFORMANCE BY FILTER COMBINATION:")
print("="*50)
filter_stats = df.groupby('filter_combo')['Return [%]'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
filter_stats = filter_stats.sort_values('mean', ascending=False)
print(filter_stats)

print("\n2. BEST PERFORMING COMBINATIONS:")
print("="*40)
best_combos = df.nlargest(10, 'Return [%]')[['filter_combo', 'use_volume_filter', 'use_trend_filter', 'use_ema_filter', 'ema_period', 'Return [%]']]
print(best_combos.to_string(index=False))

print("\n3. WORST PERFORMING COMBINATIONS:")
print("="*40)
worst_combos = df.nsmallest(10, 'Return [%]')[['filter_combo', 'use_volume_filter', 'use_trend_filter', 'use_ema_filter', 'ema_period', 'Return [%]']]
print(worst_combos.to_string(index=False))

print("\n4. INDIVIDUAL FILTER IMPACT:")
print("="*30)

# Volume filter impact
vol_on = df[df['use_volume_filter'] == True]['Return [%]'].mean()
vol_off = df[df['use_volume_filter'] == False]['Return [%]'].mean()
print(f"Volume Filter ON:  {vol_on:.2f}%")
print(f"Volume Filter OFF: {vol_off:.2f}%")
print(f"Volume Filter Impact: {vol_off - vol_on:.2f}% (negative = filter hurts performance)")

# Trend filter impact
trend_on = df[df['use_trend_filter'] == True]['Return [%]'].mean()
trend_off = df[df['use_trend_filter'] == False]['Return [%]'].mean()
print(f"\nTrend Filter ON:  {trend_on:.2f}%")
print(f"Trend Filter OFF: {trend_off:.2f}%")
print(f"Trend Filter Impact: {trend_off - trend_on:.2f}% (negative = filter hurts performance)")

# EMA filter impact
ema_on = df[df['use_ema_filter'] == True]['Return [%]'].mean()
ema_off = df[df['use_ema_filter'] == False]['Return [%]'].mean()
print(f"\nEMA Filter ON:  {ema_on:.2f}%")
print(f"EMA Filter OFF: {ema_off:.2f}%")
print(f"EMA Filter Impact: {ema_on - ema_off:.2f}% (positive = filter helps performance)")

print("\n5. EMA PERIOD ANALYSIS (when EMA filter is ON):")
print("="*45)
ema_on_data = df[df['use_ema_filter'] == True]
ema_period_stats = ema_on_data.groupby('ema_period')['Return [%]'].agg(['mean', 'count']).round(2)
print(ema_period_stats)

# Create a comprehensive filter combination heatmap
plt.figure(figsize=(14, 10))

# Create a pivot table for filter combinations
filter_pivot = df.pivot_table(
    values='Return [%]', 
    index=['use_volume_filter', 'use_trend_filter'], 
    columns=['use_ema_filter', 'ema_period'],
    aggfunc='mean'
)

# Create labels for the axes
row_labels = []
for vol, trend in filter_pivot.index:
    label = f"Vol:{vol}, Trend:{trend}"
    row_labels.append(label)

col_labels = []
for ema, period in filter_pivot.columns:
    if ema:
        label = f"EMA:{period}"
    else:
        label = "No EMA"
    col_labels.append(label)

# Plot the heatmap
sns.heatmap(filter_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            xticklabels=col_labels, yticklabels=row_labels,
            cbar_kws={'label': 'Return [%]'})

plt.title('Filter Combination Performance Heatmap\nBTC/USDT Breakout Strategy', fontsize=16, pad=20)
plt.xlabel('EMA Filter Configuration', fontsize=12)
plt.ylabel('Volume & Trend Filter Configuration', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the plot
output_path = Path("output/optimizations/optimize_breakout_BTC_USDT_20250701_223705/filter_analysis_heatmap.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n6. COMPREHENSIVE HEATMAP SAVED:")
print(f"   {output_path}")

plt.show()

print("\n=== SUMMARY & RECOMMENDATIONS ===")
print("="*35)

best_row = df.loc[df['Return [%]'].idxmax()]
print(f"BEST CONFIGURATION:")
print(f"  Volume Filter: {best_row['use_volume_filter']}")
print(f"  Trend Filter:  {best_row['use_trend_filter']}")
print(f"  EMA Filter:    {best_row['use_ema_filter']} (period: {best_row['ema_period']})")
print(f"  Return:        {best_row['Return [%]']:.2f}%")

print(f"\nKEY INSIGHTS:")
print(f"  • Volume filter appears to HURT performance ({vol_off - vol_on:.2f}%)")
print(f"  • Trend filter appears to HURT performance ({trend_off - trend_on:.2f}%)")
print(f"  • EMA filter appears to HELP performance ({ema_on - ema_off:.2f}%)")
print(f"  • Best combination: Only EMA filter with period {best_row['ema_period']}")