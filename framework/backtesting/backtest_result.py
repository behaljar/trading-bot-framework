"""
Backtest result data structures
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List
import pandas as pd

from .trade import Trade


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    strategy_name: str
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    trades: List[Trade]
    portfolio_values: pd.Series
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    
    @property
    def duration_days(self) -> int:
        """Get backtest duration in days"""
        return (self.end_date - self.start_date).days
    
    @property
    def average_return_per_trade(self) -> float:
        """Get average return per trade"""
        if self.total_trades > 0:
            return self.total_return / self.total_trades
        return 0.0
    
    @property
    def annualized_return(self) -> float:
        """Get annualized return percentage"""
        if self.duration_days > 0:
            years = self.duration_days / 365.25
            return ((self.final_capital / self.initial_capital) ** (1/years) - 1) * 100
        return 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'annualized_return': self.annualized_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'duration_days': self.duration_days,
            'average_return_per_trade': self.average_return_per_trade
        }
    
    def summary(self) -> str:
        """Get formatted summary string"""
        return f"""
Backtest Results for {self.strategy_name} on {self.symbol}
{'='*60}
Period: {self.start_date.date()} to {self.end_date.date()} ({self.duration_days} days)
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${self.final_capital:,.2f}
Total Return: ${self.total_return:,.2f} ({self.total_return_pct:.2f}%)
Annualized Return: {self.annualized_return:.2f}%
Max Drawdown: {self.max_drawdown:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.2f}
Win Rate: {self.win_rate:.2f}%
Total Trades: {self.total_trades}
Avg Return per Trade: ${self.average_return_per_trade:.2f}
        """.strip()