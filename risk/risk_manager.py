"""
Risk management system
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    """Class for storing risk metrics"""
    portfolio_value: float
    total_exposure: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    sharpe_ratio: float

class RiskManager:
    """Risk manager"""

    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.equity_curve = []

    def check_position_size(self, symbol: str, size: float, price: float) -> float:
        """Checks and adjusts position size according to risk limits"""
        max_position_value = self.config.initial_capital * self.config.max_position_size
        requested_value = abs(size * price)

        if requested_value > max_position_value:
            # Adjust position size according to limit
            adjusted_size = max_position_value / price
            if size < 0:
                adjusted_size = -adjusted_size
            return adjusted_size

        return size

    def should_apply_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Checks whether stop-loss should be applied"""
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        if position['size'] == 0:
            return False

        entry_price = position['entry_price']
        size = position['size']

        if size > 0:  # Long position
            loss_pct = (entry_price - current_price) / entry_price
        else:  # Short position
            loss_pct = (current_price - entry_price) / entry_price

        return loss_pct >= self.config.stop_loss_pct

    def update_position(self, symbol: str, size: float, price: float):
        """Updates position in portfolio"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'size': 0,
                'entry_price': 0,
                'unrealized_pnl': 0
            }

        if size == 0:
            self.positions[symbol] = {'size': 0, 'entry_price': 0, 'unrealized_pnl': 0}
        else:
            self.positions[symbol]['size'] = size
            self.positions[symbol]['entry_price'] = price

    def calculate_portfolio_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculates portfolio risk metrics"""
        if len(returns) < 2:
            return RiskMetrics(0, 0, 0, 0, 0)

        # Portfolio value
        portfolio_value = self.config.initial_capital * (1 + returns).cumprod().iloc[-1]

        # Total exposure
        total_exposure = sum(abs(pos['size'] * pos['entry_price'])
                           for pos in self.positions.values()
                           if pos['size'] != 0)

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # VaR 95%
        var_95 = np.percentile(returns, 5)

        # Sharpe ratio
        excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            max_drawdown=max_drawdown,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio
        )