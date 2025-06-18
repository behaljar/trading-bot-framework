.PHONY: install test run clean dev dev-dashboard dev-notebook

# === DEVELOPMENT ===
# Spustí bot v development módu
dev:
	@echo "🚀 Starting trading bot in development mode..."
	ENVIRONMENT=development DEBUG=true TRADING_MODE=paper python main.py

# Spustí development dashboard
dev-dashboard:
	@echo "📊 Starting development dashboard..."
	streamlit run monitoring/dashboard.py --server.port 8501

# Spustí Jupyter pro vývoj strategií
dev-notebook:
	@echo "📓 Starting Jupyter Lab..."
	jupyter lab notebooks/ --ip=0.0.0.0 --port=8888 --no-browser

# Spustí vše najednou (bot + dashboard)
dev-all:
	@echo "🚀 Starting full development environment..."
	@trap 'kill %1; kill %2' EXIT; \
	 make dev & \
	 make dev-dashboard & \
	 wait

# === PRODUCTION ===
# Spustí v production módu
run:
	@echo "⚡ Starting trading bot in production mode..."
	ENVIRONMENT=production python main.py

# === TESTING ===
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v --cov=strategies --cov=data --cov=risk --cov=execution --cov=monitoring --cov=utils

# === UTILITIES ===
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .coverage htmlcov/

# === TRADING OPERATIONS ===
backtest:
	@echo "📈 Running backtest..."
	python scripts/run_backtest.py --strategy sma --symbol AAPL

paper:
	@echo "📄 Starting paper trading..."
	TRADING_MODE=paper python scripts/deploy_strategy.py --mode once

live:
	@echo "💰 Starting live trading..."
	TRADING_MODE=live python scripts/deploy_strategy.py --mode scheduled

# === MONITORING ===
monitor:
	@echo "👀 Starting performance monitoring..."
	python scripts/monitor_performance.py

logs:
	@echo "📋 Showing recent logs..."
	tail -f logs/trading_bot_$(shell date +%Y%m%d).log