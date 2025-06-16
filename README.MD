# Trading Bot Framework

Educational algorithmic trading bot framework designed for the trading bot course final projects. This framework provides a complete skeleton for developing, testing, and deploying trading bots in a paper trading environment.

## ⚠️ Important Notice
This framework is designed for **educational purposes only**.
- All trading is done in paper trading mode (no real money)
- For production use, additional components would be needed
- Always test thoroughly before considering any real trading

## 🚀 Quick Start

### 1. Install Dependencies
```bash
make install
# or: pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and customize (or use environment variables):

```yaml
# Trading configuration
initial_capital: 10000.0
max_position_size: 0.1
stop_loss_pct: 0.05
commission: 0.002

# Data sources
data_source: "yahoo"  # yahoo or ccxt
symbols: ["AAPL", "MSFT", "GOOGL"]
timeframe: "1d"

# Strategy
strategy_name: "sma"
strategy_params:
  short_window: 20
  long_window: 50
```

### 3. Run the Bot

```bash
# Development mode (paper trading)
make dev

# Run backtest
make backtest

# Start development dashboard
make dev-dashboard
```

## 📁 Project Structure

```
trading-bot-framework/
├── config/              # Configuration
│   ├── settings.py      # Main configuration class
├── data/                # Data components
│   ├── __init__.py      # Data source abstractions
│   ├── yahoo_finance_source.py
│   ├── ccxt_source.py
│   └── csv_data_source.py
├── strategies/          # Trading strategies
│   ├── base_strategy.py # Abstract base class
│   ├── trend_following.py # SMA strategy
├── risk/                # Risk management
│   └── risk_manager.py  # Risk manager
├── execution/           # Execution layer
├── monitoring/          # Monitoring and reporting
│   └── alert_system.py
├── utils/               # Utilities
│   └── logger.py        # Logging setup
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
└── scripts/             # Utility scripts
    └── run_backtest.py  # Backtesting script
```

## 🔧 Available Strategies

### Built-in Strategies
- **SMA Strategy**: Simple Moving Average crossover

### Creating Your Own Strategy

Extend the `BaseStrategy` class:

```python
from strategies.base_strategy import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, params=None):
        super().__init__(params)
        # Your initialization here
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on data

        Args:
            data: DataFrame with OHLCV data and optionally additional feature columns

        Returns:
            DataFrame with columns:
            - signal: int (1=BUY, -1=SELL, 0=HOLD)
            - stop_loss: float (optional, stop loss price)
            - take_profit: float (optional, take profit price)
        """
    
    def get_strategy_name(self):
        return "My_Custom_Strategy"
```

## 🧪 Testing

```bash
# Run all tests
make test
# or: python -m pytest tests/ -v --cov=trading_bot

# Test specific strategy
python -m pytest tests/test_strategies.py::TestSMAStrategy

# Test with coverage report
python -m pytest --cov=trading_bot tests/
```

## 📈 Usage Examples

### Simple Backtest
```bash
# Run SMA strategy backtest on AAPL
python scripts/run_backtest.py --strategy sma --symbol AAPL --start 2022-01-01 --end 2023-12-31
```

### Paper Trading
```bash
# Start paper trading with your configured strategy
make dev
# or: python main.py
```

## 🔍 Development Tips

### Common Issues:
1. **API limits**: Implement rate limiting for data sources
2. **Missing data**: Add robust error handling
3. **Overfitting**: Use walk-forward analysis
4. **Data quality**: Validate data before processing

### Logging:
All components use centralized logging:
```python
from utils.logger import setup_logger
logger = setup_logger("INFO")
logger.info("Your message here")
```

## 📚 Recommended Resources

- [Backtesting.py documentation](https://kernc.github.io/backtesting.py/)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [yfinance documentation](https://pypi.org/project/yfinance/)
- [CCXT documentation](https://docs.ccxt.com/)

## 📄 License

MIT License - see LICENSE file for details.