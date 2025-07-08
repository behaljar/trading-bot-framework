.PHONY: install test run clean dev test-orders sandbox live help

# Default target
.DEFAULT_GOAL := help

# === HELP ===
help:
	@echo "🤖 Trading Bot - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "📦 SETUP:"
	@echo "  make install     - Install dependencies"
	@echo "  make clean       - Clean Python cache files"
	@echo ""
	@echo "🧪 TESTING:"
	@echo "  make test        - Run unit tests"
	@echo "  make test-orders - Test order execution (safe)"
	@echo "  make test-positions - Test position synchronization"
	@echo ""
	@echo "🚀 TRADING:"
	@echo "  make sandbox     - Paper trading (safe)"
	@echo "  make live        - Live trading (real money!)"
	@echo ""
	@echo "📄 PAPER TRADING:"
	@echo "  make paper-yahoo - Paper trade with Yahoo Finance"
	@echo "  make paper-ccxt  - Paper trade with CCXT exchanges"
	@echo "  make paper-ibkr  - Paper trade with IBKR"
	@echo ""
	@echo "🏦 IBKR INTEGRATION:"
	@echo "  make ibkr-test   - Test IBKR connection"
	@echo "  make ibkr-paper  - IBKR paper trading"
	@echo "  make ibkr-live   - IBKR live trading (REAL MONEY!)"
	@echo ""
	@echo "📊 UTILITIES:"
	@echo "  make logs        - Show recent logs"
	@echo "  make status      - Show current state"
	@echo ""
	@echo "⚠️  IMPORTANT: Always test with 'make sandbox' first!"

# === SETUP ===
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

clean:
	@echo "🧹 Cleaning up Python cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .coverage htmlcov/
	@echo "✅ Cleanup complete"

# === TESTING ===
test:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/ -v --cov=trading_bot

test-orders:
	@echo "🧪 Testing order execution (SAFE - uses sandbox)..."
	@echo "⚠️  Make sure your .env has USE_SANDBOX=true"
	python tests/test_ccxt_orders.py

test-positions:
	@echo "🧪 Testing position synchronization..."
	@echo "⚠️  This will show existing positions on your account"
	python tests/test_ccxt_positions.py

# === TRADING ===
paper:
	@echo "📄 Starting local paper trading simulation..."
	@echo "💡 This runs a local simulation with virtual portfolio"
	python scripts/run_paper_trading.py

paper-yahoo:
	@echo "📄 Starting paper trading with Yahoo Finance data..."
	python scripts/run_paper_trading.py --source yahoo --symbols AAPL MSFT GOOGL

paper-ccxt:
	@echo "📄 Starting paper trading with CCXT data..."
	python scripts/run_paper_trading.py --source ccxt --symbols BTC/USDT ETH/USDT

paper-ibkr:
	@echo "📄 Starting paper trading with IBKR data..."
	@echo "⚠️  Make sure your .env is configured for IBKR paper trading"
	python scripts/run_paper_trading.py --source ibkr --symbols SPY AAPL MSFT

# === IBKR COMMANDS ===
ibkr-test:
	@echo "🧪 Testing IBKR connection and functionality..."
	@echo "⚠️  Make sure TWS or IB Gateway is running on paper trading mode"
	@echo "⚠️  Make sure your .env is configured for IBKR paper trading"
	python tests/test_ibkr_connection.py

ibkr-paper:
	@echo "📄 Starting IBKR paper trading mode..."
	@echo "⚠️  Make sure TWS/IB Gateway is running on port 7497 (paper)"
	@echo "⚠️  Make sure your .env is configured for IBKR paper trading"
	python trading_bot/main.py

ibkr-live:
	@echo "🚨 Starting IBKR LIVE trading mode..."
	@echo "⚠️  WARNING: This will place REAL trades with REAL money!"
	@echo "⚠️  Make sure TWS/IB Gateway is running on port 7496 (live)"
	@echo "⚠️  Make sure your .env is configured for IBKR live trading"
	@read -p "Are you sure you want to trade with REAL money? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	python trading_bot/main.py

sandbox:
	@echo "📄 Starting SANDBOX trading (safe with testnet)..."
	@echo "⚠️  Using testnet/sandbox - no real money at risk"
	@if [ ! -f .env ]; then \
		echo "❌ No .env file found. Copy .env.sandbox to .env first"; \
		exit 1; \
	fi
	@if ! grep -q "USE_SANDBOX=true" .env; then \
		echo "⚠️  WARNING: USE_SANDBOX not set to true in .env"; \
		echo "❌ Aborting for safety. Set USE_SANDBOX=true"; \
		exit 1; \
	fi
	python main.py

live:
	@echo "💰 Starting LIVE trading (REAL MONEY!)..."
	@echo "⚠️⚠️⚠️  THIS USES REAL MONEY! ⚠️⚠️⚠️"
	@if [ ! -f .env ]; then \
		echo "❌ No .env file found. Copy .env.live to .env first"; \
		exit 1; \
	fi
	@if grep -q "USE_SANDBOX=true" .env; then \
		echo "❌ USE_SANDBOX=true in .env - this would use sandbox mode"; \
		echo "❌ For live trading, set USE_SANDBOX=false"; \
		exit 1; \
	fi
	@echo "🔄 Starting in 5 seconds... Press Ctrl+C to abort"
	@sleep 5
	python main.py

# === MONITORING ===
logs:
	@echo "📋 Showing recent logs..."
	@if [ -d logs ]; then \
		tail -f logs/trading_bot_$(shell date +%Y%m%d).log 2>/dev/null || \
		tail -f logs/*.log 2>/dev/null || \
		echo "No log files found in logs/"; \
	else \
		echo "No logs directory found"; \
	fi

status:
	@echo "📊 Trading Bot Status"
	@echo "===================="
	@if [ -f .env ]; then \
		echo "✅ Configuration file: .env exists"; \
		if grep -q "USE_SANDBOX=true" .env; then \
			echo "🧪 Mode: SANDBOX (safe)"; \
		elif grep -q "USE_SANDBOX=false" .env; then \
			echo "💰 Mode: LIVE (real money)"; \
		else \
			echo "⚠️  Mode: Unknown (check USE_SANDBOX in .env)"; \
		fi; \
		echo "🔗 Exchange: $$(grep EXCHANGE_NAME .env | cut -d'=' -f2 || echo 'Not set')"; \
		echo "📈 Strategy: $$(grep STRATEGY_NAME .env | cut -d'=' -f2 || echo 'Not set')"; \
		echo "💎 Symbols: $$(grep SYMBOLS .env | cut -d'=' -f2 || echo 'Not set')"; \
	else \
		echo "❌ No .env file found"; \
		echo "📝 Copy .env.sandbox or .env.live to .env"; \
	fi
	@if [ -d data/state ]; then \
		echo "💾 State directory: exists"; \
		if [ -f data/state/positions.json ]; then \
			echo "📊 Saved positions: found"; \
		fi; \
	fi

# === DEVELOPMENT (Legacy - for reference) ===
dev: sandbox

# === ALIASES ===
run: live
paper: sandbox