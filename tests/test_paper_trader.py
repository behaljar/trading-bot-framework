"""
Tests for paper trading components.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from config.settings import Config
from execution.paper import (
    VirtualPortfolio, VirtualPosition, VirtualOrder,
    OrderSimulator, PaperTrader, PerformanceTracker
)
from strategies.base_strategy import Signal


class TestVirtualPortfolio:
    """Test virtual portfolio management."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = VirtualPortfolio(10000, 'USD')
        assert portfolio.initial_balance == 10000
        assert portfolio.balance == 10000
        assert portfolio.base_currency == 'USD'
        assert len(portfolio.positions) == 0
        assert len(portfolio.order_history) == 0
        
    def test_execute_buy_order(self):
        """Test executing a buy order."""
        portfolio = VirtualPortfolio(10000)
        
        order = VirtualOrder(
            order_id='TEST001',
            symbol='AAPL',
            side='buy',
            size=10,
            order_type='market',
            filled_price=150.0,
            status='filled',
            commission=1.5
        )
        
        result = portfolio.execute_order(order)
        assert result is True
        assert 'AAPL' in portfolio.positions
        assert portfolio.positions['AAPL'].size == 10
        assert portfolio.positions['AAPL'].entry_price == 150.0
        assert portfolio.balance == 10000 - (10 * 150.0) - 1.5
        
    def test_execute_sell_order(self):
        """Test executing a sell order to close position."""
        portfolio = VirtualPortfolio(10000)
        
        # First buy
        buy_order = VirtualOrder(
            order_id='TEST001',
            symbol='AAPL',
            side='buy',
            size=10,
            order_type='market',
            filled_price=150.0,
            status='filled',
            commission=1.5
        )
        portfolio.execute_order(buy_order)
        
        # Then sell
        sell_order = VirtualOrder(
            order_id='TEST002',
            symbol='AAPL',
            side='sell',
            size=10,
            order_type='market',
            filled_price=160.0,
            status='filled',
            commission=1.6
        )
        portfolio.execute_order(sell_order)
        
        assert 'AAPL' not in portfolio.positions
        assert len(portfolio.trade_history) == 1
        assert portfolio.trade_history[0]['pnl'] == 100.0  # (160-150) * 10
        
    def test_insufficient_balance(self):
        """Test order rejection due to insufficient balance."""
        portfolio = VirtualPortfolio(1000)
        
        order = VirtualOrder(
            order_id='TEST001',
            symbol='AAPL',
            side='buy',
            size=100,
            order_type='market',
            filled_price=150.0,
            status='filled',
            commission=15.0
        )
        
        result = portfolio.execute_order(order)
        assert result is False
        assert order.status == 'rejected'
        assert len(portfolio.positions) == 0
        
    def test_position_update_unrealized_pnl(self):
        """Test updating position prices and unrealized PnL."""
        portfolio = VirtualPortfolio(10000)
        
        # Create position
        order = VirtualOrder(
            order_id='TEST001',
            symbol='AAPL',
            side='buy',
            size=10,
            order_type='market',
            filled_price=150.0,
            status='filled',
            commission=1.5
        )
        portfolio.execute_order(order)
        
        # Update price
        portfolio.update_position_price('AAPL', 160.0)
        
        position = portfolio.positions['AAPL']
        assert position.current_price == 160.0
        assert position.unrealized_pnl == 100.0  # (160-150) * 10


class TestOrderSimulator:
    """Test order simulation."""
    
    def test_market_order_simulation(self):
        """Test market order simulation with spread and slippage."""
        config = Mock()
        config.paper_commission_rate = 0.001
        config.paper_spread_bps = 10
        config.paper_slippage_bps = 5
        
        simulator = OrderSimulator(config)
        
        order = VirtualOrder(
            order_id='TEST001',
            symbol='AAPL',
            side='buy',
            size=10,
            order_type='market'
        )
        
        # Simulate with mid price
        filled_order = simulator.simulate_market_order(order, 100.0)
        
        assert filled_order.status == 'filled'
        assert filled_order.filled_price > 100.0  # Should be higher for buy
        assert filled_order.commission > 0
        assert filled_order.slippage > 0
        
    def test_sell_order_simulation(self):
        """Test sell order simulation."""
        config = Mock()
        config.paper_commission_rate = 0.001
        config.paper_spread_bps = 10
        config.paper_slippage_bps = 5
        
        simulator = OrderSimulator(config)
        
        order = VirtualOrder(
            order_id='TEST001',
            symbol='AAPL',
            side='sell',
            size=10,
            order_type='market'
        )
        
        # Simulate with mid price
        filled_order = simulator.simulate_market_order(order, 100.0)
        
        assert filled_order.status == 'filled'
        assert filled_order.filled_price < 100.0  # Should be lower for sell
        

class TestPerformanceTracker:
    """Test performance metrics calculation."""
    
    def test_metrics_calculation(self):
        """Test basic metrics calculation."""
        tracker = PerformanceTracker()
        portfolio = VirtualPortfolio(10000)
        
        # Add some equity points
        start_time = datetime.now()
        tracker.update_equity(start_time, 10000)
        tracker.update_equity(start_time, 10100)
        tracker.update_equity(start_time, 10050)
        
        # Create a winning trade
        portfolio.trade_history.append({
            'symbol': 'AAPL',
            'pnl': 100,
            'commission': 2
        })
        
        metrics = tracker.calculate_metrics(portfolio, start_time)
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        

class TestPaperTrader:
    """Test main paper trader."""
    
    def test_paper_trader_initialization(self):
        """Test paper trader initialization."""
        config = Mock()
        config.initial_capital = 10000
        config.state_directory = '/tmp/test'
        config.paper_commission_rate = 0.001
        config.paper_spread_bps = 10
        config.paper_slippage_bps = 5
        
        data_source = Mock()
        strategy = Mock()
        risk_manager = Mock()
        
        trader = PaperTrader(config, data_source, strategy, risk_manager)
        
        assert trader.portfolio.initial_balance == 10000
        assert trader.order_simulator is not None
        assert trader.performance_tracker is not None
        
    def test_execute_buy_signal(self):
        """Test executing a buy signal."""
        config = Mock()
        config.initial_capital = 10000
        config.state_directory = '/tmp/test'
        config.paper_commission_rate = 0.001
        config.paper_spread_bps = 10
        config.paper_slippage_bps = 5
        
        data_source = Mock()
        data_source.get_current_price.return_value = {'bid': 99.9, 'ask': 100.1, 'last': 100.0}
        
        strategy = Mock()
        risk_manager = Mock()
        risk_manager.calculate_position_size.return_value = 10
        
        trader = PaperTrader(config, data_source, strategy, risk_manager)
        
        # Create mock data
        import pandas as pd
        current_data = pd.Series({'close': 100.0})
        
        # Execute buy signal
        order = trader.execute_signal('AAPL', Signal.BUY, current_data)
        
        assert order is not None
        assert order.side == 'buy'
        assert order.size == 10
        assert 'AAPL' in trader.portfolio.positions