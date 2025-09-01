#!/bin/bash
# Examples of custom parameter space optimization

echo "============================================================"
echo "CUSTOM PARAMETER SPACE OPTIMIZATION EXAMPLES"
echo "============================================================"

# Example 1: Fine-tuned SMA optimization with JSON string
echo "Example 1: Fine-tuned SMA parameters (JSON string)"
echo "------------------------------------------------------------"
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2024-02-01 \
  --param-space '{"short_window": {"type": "int", "min": 8, "max": 15, "step": 1}, "long_window": {"type": "int", "min": 20, "max": 35, "step": 3}}' \
  --output output/optimizations/example1_fine_tuned_sma.json

echo ""
echo "Example 2: Choice-based parameter optimization"
echo "------------------------------------------------------------"
python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2024-02-01 \
  --param-space '{"short_window": {"type": "choice", "choices": [10, 15, 20, 25]}, "long_window": {"type": "choice", "choices": [30, 50, 70, 100]}}' \
  --output output/optimizations/example2_choice_based.json

echo ""
echo "Example 3: Using environment variable for complex parameter space"
echo "------------------------------------------------------------"

# Set complex parameter space in environment variable
export SMA_COMPLEX_PARAMS='{
  "short_window": {"type": "int", "min": 12, "max": 18, "step": 2},
  "long_window": {"type": "int", "min": 35, "max": 55, "step": 5},
  "stop_loss_pct": {"type": "float", "min": 0.02, "max": 0.04, "step": 0.01},
  "take_profit_pct": {"type": "float", "min": 0.03, "max": 0.07, "step": 0.02}
}'

python scripts/optimize_strategy.py \
  --strategy sma \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2024-02-01 \
  --param-space SMA_COMPLEX_PARAMS \
  --output output/optimizations/example3_complex_env.json

echo ""
echo "Example 4: FVG strategy with custom parameters"
echo "------------------------------------------------------------"
python scripts/optimize_strategy.py \
  --strategy fvg \
  --data-file data/cleaned/BTC_USDT_binance_15m_2024-01-01_2025-08-20_cleaned.csv \
  --start 2024-01-01 \
  --end 2024-02-01 \
  --param-space '{"h1_lookback_candles": {"type": "choice", "choices": [18, 24, 30, 36]}, "risk_reward_ratio": {"type": "float", "min": 2.0, "max": 4.0, "step": 0.5}}' \
  --output output/optimizations/example4_fvg_custom.json

echo ""
echo "============================================================"
echo "All optimization examples completed!"
echo "Check output/optimizations/ for results and heatmaps"
echo "============================================================"