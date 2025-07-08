#!/usr/bin/env python3

import sys
sys.path.append('.')
from data.ccxt_source import CCXTSource
from strategies.test_strategy import TestStrategy
from datetime import datetime, timedelta

source = CCXTSource('binance', None, None, True)
end_date = datetime.now()
start_date = end_date - timedelta(days=1)
df = source.get_historical_data('BTC/USDT:USDT', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1m')

strategy = TestStrategy({})
signals = strategy.generate_signals(df)

print(f'Total bars: {len(df)}')

# Find the most recent signal
last_signal_idx = -1
for i in range(len(df)-1, -1, -1):
    if signals.iloc[i] != 0:
        last_signal_idx = i
        break

if last_signal_idx >= 0:
    bar_time = df.index[last_signal_idx]
    close_price = df.iloc[last_signal_idx]['Close']
    signal = signals.iloc[last_signal_idx]
    print(f'Most recent signal: Bar #{last_signal_idx} | {bar_time} | Close: {close_price:.2f} | Signal: {signal}')
else:
    print('No signals found')

print(f'Last bar: #{len(df)-1} | Time: {df.index[-1]} | Signal: {signals.iloc[-1]}')

# Check if we can find a signal that's close to the end
print('\nSignals in last 20 bars:')
for i in range(max(0, len(df)-20), len(df)):
    signal = signals.iloc[i]
    if signal != 0:
        bar_time = df.index[i]
        close_price = df.iloc[i]['Close']
        print(f'Bar #{i}: {bar_time} | Close: {close_price:.2f} | Signal: {signal}')