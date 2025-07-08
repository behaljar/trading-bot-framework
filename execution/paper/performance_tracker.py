"""
Performance metrics tracking and calculation for paper trading.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .virtual_portfolio import VirtualPortfolio, VirtualOrder


class PerformanceTracker:
    """Tracks and calculates comprehensive performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        self.equity_curve = []
        self.daily_returns = []
        self.trade_returns = []
        
    def update_equity(self, timestamp: datetime, portfolio_value: float):
        """Update equity curve with latest portfolio value."""
        self.equity_curve.append({
            'timestamp': timestamp,
            'value': portfolio_value
        })
        
    def calculate_metrics(self, portfolio: VirtualPortfolio, 
                         start_time: datetime) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Basic metrics from portfolio
        basic_metrics = portfolio.get_performance_summary()
        
        # Calculate returns
        if len(self.equity_curve) < 2:
            # Add missing metrics with default values
            basic_metrics.update({
                'total_return': basic_metrics.get('total_return_pct', 0) / 100,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration_days': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade_duration_hours': 0.0,
                'total_days': 1
            })
            return basic_metrics
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate daily returns
        daily_values = equity_df.resample('D').last()
        daily_returns = daily_values['value'].pct_change().dropna()
        
        # Calculate metrics
        total_days = (datetime.now() - start_time).days or 1
        annual_factor = 365 / total_days
        
        # Returns
        total_return = (portfolio.get_total_value() / portfolio.initial_balance - 1)
        annual_return = (1 + total_return) ** annual_factor - 1
        
        # Volatility
        daily_vol = daily_returns.std() if len(daily_returns) > 1 else 0
        annual_vol = daily_vol * np.sqrt(252)  # Annualized volatility
        
        # Sharpe ratio
        sharpe = ((annual_return - self.risk_free_rate) / annual_vol) if annual_vol > 0 else 0
        
        # Max drawdown
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity_df['value'])
        
        # Win rate and profit factor
        trades = portfolio.trade_history
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(wins) / len(trades) if trades else 0
            
            total_wins = sum(t['pnl'] for t in wins) if wins else 0
            total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            avg_win = total_wins / len(wins) if wins else 0
            avg_loss = total_losses / len(losses) if losses else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Average trade duration
            avg_duration = self._calculate_avg_trade_duration(trades)
        else:
            win_rate = profit_factor = expectancy = avg_duration = 0
            avg_win = avg_loss = 0
            
        # Compile all metrics
        metrics = {
            **basic_metrics,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_duration_days': max_dd_duration,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_duration_hours': avg_duration,
            'total_days': total_days
        }
        
        return metrics
        
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if len(equity_series) < 2:
            return 0, 0
            
        # Calculate running maximum
        running_max = equity_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_series - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Calculate duration
        dd_start = drawdown.idxmax()  # Start of worst drawdown
        dd_end = drawdown.idxmin()    # Bottom of worst drawdown
        
        # Find recovery point
        recovery_idx = equity_series[dd_end:].loc[equity_series >= running_max[dd_end]].index
        recovery_date = recovery_idx[0] if len(recovery_idx) > 0 else equity_series.index[-1]
        
        duration = (recovery_date - dd_start).days
        
        return abs(max_dd), duration
        
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours."""
        if not trades:
            return 0
            
        # This would need entry_time in trade history
        # For now, return placeholder
        return 24.0  # Default 24 hours
        
    def generate_report(self, metrics: Dict) -> str:
        """Generate formatted performance report."""
        report = """
================================================================================
                        PAPER TRADING PERFORMANCE REPORT
================================================================================

RETURNS
-------
Total Return:           {total_return:>10.2%}
Annual Return:          {annual_return:>10.2%}
Sharpe Ratio:           {sharpe_ratio:>10.2f}

RISK METRICS
------------
Annual Volatility:      {annual_volatility:>10.2%}
Max Drawdown:           {max_drawdown:>10.2%}
Max DD Duration:        {max_drawdown_duration_days:>10d} days

TRADING STATISTICS
------------------
Total Trades:           {num_trades:>10d}
Win Rate:               {win_rate:>10.2%}
Profit Factor:          {profit_factor:>10.2f}
Expectancy:             ${expectancy:>10.2f}

Average Win:            ${avg_win:>10.2f}
Average Loss:           ${avg_loss:>10.2f}

PORTFOLIO SUMMARY
-----------------
Initial Balance:        ${initial_balance:>10,.2f}
Final Balance:          ${total_value:>10,.2f}
Total P&L:              ${total_pnl:>10,.2f}
Total Commission:       ${total_commission:>10,.2f}

Trading Period:         {total_days:>10d} days
================================================================================
        """.format(**metrics)
        
        return report