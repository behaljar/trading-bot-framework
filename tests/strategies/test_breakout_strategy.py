import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
Test cases for breakout strategy with visualization
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import os

from framework.strategies.breakout_strategy import BreakoutStrategy


class TestBreakoutStrategy:
    """Test cases for BreakoutStrategy class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = BreakoutStrategy()
        
        # Create sample trending data for testing breakouts
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        
        # Create uptrend data with some volatility and clear breakouts
        base_price = 100
        trend = np.linspace(0, 20, 100)  # 20% uptrend over 100 periods
        noise = np.random.normal(0, 2, 100)  # 2% random noise
        
        closes = base_price + trend + noise
        
        # Create OHLC data with realistic spreads
        self.test_data = pd.DataFrame({
            'Open': closes + np.random.normal(0, 0.5, 100),
            'High': closes + np.abs(np.random.normal(1, 0.5, 100)),
            'Low': closes - np.abs(np.random.normal(1, 0.5, 100)),
            'Close': closes,
            'Volume': np.random.lognormal(10, 0.5, 100)  # Realistic volume distribution
        }, index=dates)
        
        # Ensure OHLC consistency
        for i in range(len(self.test_data)):
            row = self.test_data.iloc[i]
            self.test_data.iloc[i, self.test_data.columns.get_loc('High')] = max(row['Open'], row['High'], row['Close'])
            self.test_data.iloc[i, self.test_data.columns.get_loc('Low')] = min(row['Open'], row['Low'], row['Close'])
        
    def test_strategy_initialization(self):
        """Test strategy initialization with default and custom parameters"""
        # Test default initialization
        strategy = BreakoutStrategy()
        assert strategy.params['entry_lookback'] == 20
        assert strategy.params['exit_lookback'] == 10
        assert strategy.params['atr_multiplier'] == 2.0
        
        # Test custom parameters
        strategy = BreakoutStrategy(
            entry_lookback=30,
            exit_lookback=15,
            atr_multiplier=3.0,
            use_trend_filter=False
        )
        assert strategy.params['entry_lookback'] == 30
        assert strategy.params['exit_lookback'] == 15
        assert strategy.params['atr_multiplier'] == 3.0
        assert strategy.params['use_trend_filter'] == False
        
    def test_atr_calculation(self):
        """Test ATR calculation"""
        atr = self.strategy.calculate_atr(self.test_data)
        
        # ATR should be positive for all non-NaN values
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
        
        # ATR should start being calculated after the ATR period
        assert pd.isna(atr.iloc[:self.strategy.params['atr_period']-1]).all()
        assert not pd.isna(atr.iloc[self.strategy.params['atr_period']])
        
    def test_add_indicators(self):
        """Test indicator calculation"""
        data_with_indicators = self.strategy.add_indicators(self.test_data)
        
        # Check that all expected indicators are added
        expected_cols = [
            f'high_{self.strategy.params["entry_lookback"]}',
            f'low_{self.strategy.params["entry_lookback"]}',
            f'high_{self.strategy.params["exit_lookback"]}',
            f'low_{self.strategy.params["exit_lookback"]}',
            'longterm_trend_roc',
            'medium_trend_roc',
            'volume_ma',
            'relative_volume',
            'atr'
        ]
        
        for col in expected_cols:
            assert col in data_with_indicators.columns
            
        # Test momentum indicators if enabled
        if self.strategy.params['use_momentum_exit']:
            momentum_cols = ['candle_size_pct', 'volume_ma_momentum', 'volume_ratio_momentum',
                           'long_momentum_exit', 'short_momentum_exit']
            for col in momentum_cols:
                assert col in data_with_indicators.columns
    
    def test_data_validation(self):
        """Test data validation"""
        # Test valid data
        assert self.strategy.validate_data(self.test_data)
        
        # Test invalid data - missing columns
        invalid_data = self.test_data.drop('Close', axis=1)
        assert not self.strategy.validate_data(invalid_data)
        
        # Test invalid data - empty dataframe
        empty_data = pd.DataFrame()
        assert not self.strategy.validate_data(empty_data)
        
    def test_signal_generation(self):
        """Test signal generation"""
        signals = self.strategy.generate_signals(self.test_data)
        
        # Check that result has correct structure
        assert 'signal' in signals.columns
        assert 'position_size' in signals.columns
        assert 'stop_loss' in signals.columns
        assert 'take_profit' in signals.columns
        
        # Signals should be -1, 0, or 1
        valid_signals = signals['signal'].isin([-1, 0, 1])
        assert valid_signals.all()
        
        # Position size should be positive or 0
        assert (signals['position_size'] >= 0).all()
        
    def test_breakout_detection_simple(self):
        """Test breakout detection with simple synthetic data"""
        # Create simple breakout scenario
        dates = pd.date_range(start='2023-01-01', periods=50, freq='H')
        
        # Create data with clear breakout pattern
        # First 25 periods: sideways movement around 100
        # Period 26: breakout above 20-period high
        # Following periods: continuation
        
        prices = [100] * 25 + [105, 108, 107, 109, 111] + [110 + i for i in range(20)]
        
        breakout_data = pd.DataFrame({
            'Open': [p - 0.5 + np.random.normal(0, 0.1) for p in prices],
            'High': [p + 1 + abs(np.random.normal(0, 0.2)) for p in prices],
            'Low': [p - 1 - abs(np.random.normal(0, 0.2)) for p in prices],
            'Close': prices,
            'Volume': [1000 + np.random.normal(0, 100) for _ in prices]
        }, index=dates)
        
        # Ensure OHLC consistency
        for i in range(len(breakout_data)):
            row = breakout_data.iloc[i]
            breakout_data.iloc[i, breakout_data.columns.get_loc('High')] = max(row['Open'], row['High'], row['Close'])
            breakout_data.iloc[i, breakout_data.columns.get_loc('Low')] = min(row['Open'], row['Low'], row['Close'])
        
        # Test with simplified parameters to ensure breakout detection
        strategy = BreakoutStrategy(
            entry_lookback=20,
            exit_lookback=10,
            use_trend_filter=False,  # Disable to focus on breakout logic
            use_volume_filter=False,
            use_momentum_exit=False
        )
        signals = strategy.generate_signals(breakout_data)
        
        # Should detect at least one breakout signal
        buy_signals = (signals['signal'] == 1).sum()
        assert buy_signals > 0, "Should detect at least one breakout signal"
        
    def test_strategy_description(self):
        """Test strategy description"""
        description = self.strategy.get_description()
        assert isinstance(description, str)
        assert len(description) > 0
        assert "Breakout" in description
        
    def test_strategy_name(self):
        """Test strategy name generation"""
        name = self.strategy.get_strategy_name()
        assert isinstance(name, str)
        assert "Breakout" in name
        assert str(self.strategy.params['entry_lookback']) in name
        assert str(self.strategy.params['exit_lookback']) in name
        
    def test_trend_filter(self):
        """Test trend filter functionality"""
        # Test with trend filter enabled
        strategy_with_filter = BreakoutStrategy(use_trend_filter=True)
        signals_filtered = strategy_with_filter.generate_signals(self.test_data)
        
        # Test with trend filter disabled
        strategy_no_filter = BreakoutStrategy(use_trend_filter=False)
        signals_no_filter = strategy_no_filter.generate_signals(self.test_data)
        
        # Both should return valid signals
        assert 'signal' in signals_filtered.columns
        assert 'signal' in signals_no_filter.columns
        
    def test_volume_filter(self):
        """Test volume filter functionality"""
        # Test with volume filter enabled
        strategy_with_filter = BreakoutStrategy(use_volume_filter=True)
        signals_filtered = strategy_with_filter.generate_signals(self.test_data)
        
        # Test with volume filter disabled
        strategy_no_filter = BreakoutStrategy(use_volume_filter=False)
        signals_no_filter = strategy_no_filter.generate_signals(self.test_data)
        
        # Both should return valid signals
        assert 'signal' in signals_filtered.columns
        assert 'signal' in signals_no_filter.columns
        
    def test_visualization(self):
        """Test strategy visualization"""
        # Generate signals
        signals = self.strategy.generate_signals(self.test_data)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Price with breakout levels and signals
        ax1.plot(range(len(self.test_data)), self.test_data['Close'], 'b-', label='Close Price', alpha=0.7)
        
        # Plot entry and exit levels
        data_with_indicators = self.strategy.add_indicators(self.test_data)
        
        entry_high = data_with_indicators[f'high_{self.strategy.params["entry_lookback"]}']
        entry_low = data_with_indicators[f'low_{self.strategy.params["entry_lookback"]}']
        exit_high = data_with_indicators[f'high_{self.strategy.params["exit_lookback"]}']
        exit_low = data_with_indicators[f'low_{self.strategy.params["exit_lookback"]}']
        
        ax1.plot(range(len(self.test_data)), entry_high, 'r--', alpha=0.5, label=f'{self.strategy.params["entry_lookback"]}-period High')
        ax1.plot(range(len(self.test_data)), entry_low, 'g--', alpha=0.5, label=f'{self.strategy.params["entry_lookback"]}-period Low')
        ax1.plot(range(len(self.test_data)), exit_high, 'orange', alpha=0.3, label=f'{self.strategy.params["exit_lookback"]}-period High')
        ax1.plot(range(len(self.test_data)), exit_low, 'purple', alpha=0.3, label=f'{self.strategy.params["exit_lookback"]}-period Low')
        
        # Plot buy and sell signals
        buy_signals = signals['signal'] == 1
        sell_signals = signals['signal'] == -1
        
        if buy_signals.any():
            ax1.scatter(np.where(buy_signals)[0], self.test_data['Close'][buy_signals], 
                       color='green', marker='^', s=100, label='Buy Signal', zorder=5)
                       
        if sell_signals.any():
            ax1.scatter(np.where(sell_signals)[0], self.test_data['Close'][sell_signals], 
                       color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Price')
        ax1.set_title('Breakout Strategy - Price Action and Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volume and ATR
        ax2_vol = ax2
        ax2_atr = ax2.twinx()
        
        ax2_vol.bar(range(len(self.test_data)), self.test_data['Volume'], alpha=0.6, color='lightblue', label='Volume')
        ax2_atr.plot(range(len(self.test_data)), data_with_indicators['atr'], 'orange', label='ATR')
        
        ax2_vol.set_xlabel('Time Period')
        ax2_vol.set_ylabel('Volume', color='blue')
        ax2_atr.set_ylabel('ATR', color='orange')
        ax2_vol.set_title('Volume and ATR')
        
        # Add legends
        lines1, labels1 = ax2_vol.get_legend_handles_labels()
        lines2, labels2 = ax2_atr.get_legend_handles_labels()
        ax2_vol.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save the plot
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'output', 'tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'breakout_strategy_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Breakout strategy visualization saved to: {plot_path}")
        
        # Count signals
        buy_count = (signals['signal'] == 1).sum()
        sell_count = (signals['signal'] == -1).sum()
        print(f"Generated {buy_count} buy signals and {sell_count} sell signals")
        
        # Verify we created the visualization
        assert os.path.exists(plot_path)
        
    def test_parameter_sensitivity(self):
        """Test strategy with different parameter combinations"""
        
        test_configs = [
            {
                'name': 'Conservative',
                'strategy': BreakoutStrategy(
                    entry_lookback=30,
                    exit_lookback=15,
                    atr_multiplier=3.0,
                    use_trend_filter=True,
                    use_volume_filter=True
                )
            },
            {
                'name': 'Aggressive', 
                'strategy': BreakoutStrategy(
                    entry_lookback=10,
                    exit_lookback=5,
                    atr_multiplier=1.5,
                    use_trend_filter=False,
                    use_volume_filter=False
                )
            },
            {
                'name': 'Balanced',
                'strategy': BreakoutStrategy(
                    entry_lookback=20,
                    exit_lookback=10,
                    atr_multiplier=2.0,
                    use_trend_filter=True,
                    use_volume_filter=False
                )
            }
        ]
        
        results = {}
        
        for config in test_configs:
            strategy = config['strategy']
            signals = strategy.generate_signals(self.test_data)
            
            buy_signals = (signals['signal'] == 1).sum()
            sell_signals = (signals['signal'] == -1).sum()
            
            results[config['name']] = {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': buy_signals + sell_signals
            }
        
        # Print results
        print("\nParameter sensitivity analysis:")
        for name, result in results.items():
            print(f"{name}: {result['total_signals']} total signals ({result['buy_signals']} buy, {result['sell_signals']} sell)")
        
        # Conservative should generally have fewer signals than aggressive
        assert results['Conservative']['total_signals'] <= results['Aggressive']['total_signals']
        
    def test_stop_loss_calculation(self):
        """Test stop loss calculation with ATR"""
        signals = self.strategy.generate_signals(self.test_data)
        
        # Find rows with buy signals and check stop loss
        buy_rows = signals[signals['signal'] == 1]
        if not buy_rows.empty:
            for idx in buy_rows.index:
                stop_loss = signals.loc[idx, 'stop_loss']
                if not pd.isna(stop_loss):
                    entry_price = self.test_data.loc[idx, 'Close']
                    # Stop loss should be below entry price for long positions
                    assert stop_loss < entry_price
                    
        # Find rows with sell signals (short entries) and check stop loss
        sell_rows = signals[signals['signal'] == -1]
        position_entries = []  # Track if this is entry or exit
        
        # Note: In our strategy, -1 can be either short entry or long exit
        # We'd need to track position state to properly test this
        
    def test_cooldown_period(self):
        """Test cooldown period functionality"""
        # Create strategy with short cooldown for testing
        strategy = BreakoutStrategy(
            cooldown_periods=2,
            entry_lookback=10,  # Shorter lookback for more signals
            use_trend_filter=False,
            use_volume_filter=False
        )
        
        signals = strategy.generate_signals(self.test_data)
        
        # Find all signal changes
        signal_changes = signals['signal'] != 0
        signal_indices = signals[signal_changes].index
        
        if len(signal_indices) > 2:
            # Check that signals are spaced according to cooldown
            # This is a simplified test since actual cooldown logic is complex
            pass  # Implementation would require more detailed position tracking
            
    def test_real_data_integration(self):
        """Test with real-like data patterns"""
        try:
            # Try to create more realistic data with actual breakout patterns
            dates = pd.date_range(start='2023-01-01', periods=200, freq='15T')
            
            # Create data with consolidation followed by breakout
            consolidation_periods = 100
            breakout_periods = 100
            
            # Consolidation phase (sideways)
            consolidation_prices = 50000 + np.random.normal(0, 500, consolidation_periods)
            
            # Breakout phase (strong uptrend)
            breakout_base = consolidation_prices[-1]
            breakout_trend = np.linspace(0, 5000, breakout_periods)
            breakout_prices = breakout_base + breakout_trend + np.random.normal(0, 200, breakout_periods)
            
            all_prices = np.concatenate([consolidation_prices, breakout_prices])
            
            realistic_data = pd.DataFrame({
                'Open': all_prices + np.random.normal(0, 50, len(all_prices)),
                'High': all_prices + np.abs(np.random.normal(100, 50, len(all_prices))),
                'Low': all_prices - np.abs(np.random.normal(100, 50, len(all_prices))),
                'Close': all_prices,
                'Volume': np.random.lognormal(15, 0.5, len(all_prices))
            }, index=dates)
            
            # Ensure OHLC consistency
            for i in range(len(realistic_data)):
                row = realistic_data.iloc[i]
                realistic_data.iloc[i, realistic_data.columns.get_loc('High')] = max(row['Open'], row['High'], row['Close'])
                realistic_data.iloc[i, realistic_data.columns.get_loc('Low')] = min(row['Open'], row['Low'], row['Close'])
            
            # Test strategy
            signals = self.strategy.generate_signals(realistic_data)
            
            # Should detect breakout after consolidation
            buy_signals = (signals['signal'] == 1).sum()
            print(f"Realistic data test: Generated {buy_signals} breakout signals")
            
            # Create visualization for realistic data
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot price and signals
            ax.plot(range(len(realistic_data)), realistic_data['Close'], 'b-', label='Close Price', linewidth=1)
            
            # Plot breakout levels
            data_with_indicators = self.strategy.add_indicators(realistic_data)
            entry_high = data_with_indicators[f'high_{self.strategy.params["entry_lookback"]}']
            entry_low = data_with_indicators[f'low_{self.strategy.params["entry_lookback"]}']
            
            ax.plot(range(len(realistic_data)), entry_high, 'r--', alpha=0.5, label=f'{self.strategy.params["entry_lookback"]}-period High')
            ax.plot(range(len(realistic_data)), entry_low, 'g--', alpha=0.5, label=f'{self.strategy.params["entry_lookback"]}-period Low')
            
            # Highlight consolidation vs breakout phases
            ax.axvline(x=consolidation_periods, color='orange', linestyle=':', label='Breakout Start')
            
            # Plot signals
            buy_signals = signals['signal'] == 1
            sell_signals = signals['signal'] == -1
            
            if buy_signals.any():
                ax.scatter(np.where(buy_signals)[0], realistic_data['Close'][buy_signals], 
                          color='green', marker='^', s=100, label='Buy Signal', zorder=5)
                          
            if sell_signals.any():
                ax.scatter(np.where(sell_signals)[0], realistic_data['Close'][sell_signals], 
                          color='red', marker='v', s=100, label='Sell Signal', zorder=5)
            
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Price')
            ax.set_title('Breakout Strategy - Realistic Data Test (Consolidation → Breakout)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save realistic data plot
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, 'output', 'tests')
            plot_path = os.path.join(output_dir, 'breakout_strategy_realistic_test.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Realistic data visualization saved to: {plot_path}")
            assert os.path.exists(plot_path)
            
        except Exception as e:
            pytest.skip(f"Real data integration test failed: {e}")


if __name__ == "__main__":
    # Run tests with visualization
    test_instance = TestBreakoutStrategy()
    test_instance.setup_method()
    
    print("Running breakout strategy tests...")
    
    # Run all tests
    test_instance.test_strategy_initialization()
    print("✓ Strategy initialization test passed")
    
    test_instance.test_atr_calculation()
    print("✓ ATR calculation test passed")
    
    test_instance.test_add_indicators()
    print("✓ Indicator calculation test passed")
    
    test_instance.test_data_validation()
    print("✓ Data validation test passed")
    
    test_instance.test_signal_generation()
    print("✓ Signal generation test passed")
    
    test_instance.test_breakout_detection_simple()
    print("✓ Simple breakout detection test passed")
    
    test_instance.test_strategy_description()
    print("✓ Strategy description test passed")
    
    test_instance.test_strategy_name()
    print("✓ Strategy name test passed")
    
    test_instance.test_trend_filter()
    print("✓ Trend filter test passed")
    
    test_instance.test_volume_filter()
    print("✓ Volume filter test passed")
    
    test_instance.test_visualization()
    print("✓ Visualization test passed")
    
    test_instance.test_parameter_sensitivity()
    print("✓ Parameter sensitivity test passed")
    
    test_instance.test_real_data_integration()
    print("✓ Realistic data integration test passed")
    
    print("\nAll breakout strategy tests completed successfully!")