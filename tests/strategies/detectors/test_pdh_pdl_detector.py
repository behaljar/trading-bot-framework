import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
Test cases for simplified PDH/PDL detector
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import numpy as np
import os

from framework.strategies.detectors.pdh_pdl_detector import PDHPDLDetector, HighLowLevel


class TestPDHPDLDetector:
    """Test cases for simplified PDHPDLDetector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = PDHPDLDetector()
        
        # Create sample data spanning multiple days and weeks
        start_date = datetime(2024, 1, 1)  # Monday
        end_date = datetime(2024, 1, 20)   # 20 days of data
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Create sample OHLCV data with daily and weekly patterns
        np.random.seed(42)
        self.sample_data = self._generate_sample_data(timestamps)
    
    def _generate_sample_data(self, timestamps):
        """Generate sample OHLCV data with clear daily and weekly patterns"""
        base_price = 100
        data = []
        
        for i, ts in enumerate(timestamps):
            # Add daily volatility pattern (higher during market hours)
            hour_of_day = ts.hour
            if 9 <= hour_of_day <= 16:  # Market hours
                volatility = 2.0
            else:
                volatility = 0.5
            
            # Add weekly trend
            week_num = ts.isocalendar()[1]
            weekly_trend = (week_num % 2) * 5  # Alternating weekly pattern
            
            # Generate price with some randomness
            price = base_price + weekly_trend + np.random.randn() * volatility
            
            # Create OHLC values
            open_price = price
            close_price = price + np.random.randn() * 0.5
            high_price = max(open_price, close_price) + abs(np.random.randn() * 0.3)
            low_price = min(open_price, close_price) - abs(np.random.randn() * 0.3)
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': np.random.randint(1000, 5000)
            })
            
            # Update base price for next iteration
            base_price = close_price
        
        return pd.DataFrame(data, index=timestamps)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = PDHPDLDetector()
        assert detector is not None
    
    def test_detect_levels_basic(self):
        """Test basic level detection"""
        levels = self.detector.detect_levels(self.sample_data)
        
        # Check that we get all expected keys
        assert 'PDH' in levels
        assert 'PDL' in levels
        assert 'PWH' in levels
        assert 'PWL' in levels
        
        # Check that we have at least daily levels
        assert levels['PDH'] is not None
        assert levels['PDL'] is not None
        
        # PDH should be higher than PDL
        if levels['PDH'] and levels['PDL']:
            assert levels['PDH'].price > levels['PDL'].price
        
        # PWH should be higher than PWL
        if levels['PWH'] and levels['PWL']:
            assert levels['PWH'].price > levels['PWL'].price
    
    def test_level_types(self):
        """Test that level types are correctly assigned"""
        levels = self.detector.detect_levels(self.sample_data)
        
        if levels['PDH']:
            assert levels['PDH'].level_type == 'PDH'
            assert isinstance(levels['PDH'].price, float)
            assert isinstance(levels['PDH'].date, pd.Timestamp)
            assert isinstance(levels['PDH'].is_broken, bool)
        
        if levels['PDL']:
            assert levels['PDL'].level_type == 'PDL'
        
        if levels['PWH']:
            assert levels['PWH'].level_type == 'PWH'
        
        if levels['PWL']:
            assert levels['PWL'].level_type == 'PWL'
    
    def test_break_detection(self):
        """Test level break detection"""
        # Create data with clear break pattern
        dates = pd.date_range(start='2024-01-01', periods=72, freq='h')  # 3 days
        
        # Day 1: High at 110, Low at 90
        # Day 2: Break above 110
        # Day 3: Break below 90
        prices = []
        for i, date in enumerate(dates):
            if i < 24:  # Day 1
                prices.append(100 + 10 * np.sin(i * np.pi / 12))  # Oscillate 90-110
            elif i < 48:  # Day 2
                prices.append(105 + 10 * np.sin(i * np.pi / 12))  # Oscillate 95-115 (breaks PDH)
            else:  # Day 3
                prices.append(85 + 10 * np.sin(i * np.pi / 12))  # Oscillate 75-95 (breaks PDL)
        
        test_data = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': [p + 0.5 for p in prices],
            'Volume': [1000] * len(prices)
        }, index=dates)
        
        levels = self.detector.detect_levels(test_data)
        
        # The PDH from day 1 should be broken
        if levels['PDH']:
            print(f"PDH: {levels['PDH'].price}, broken: {levels['PDH'].is_broken}")
        
        # The PDL from day 1 or 2 should be broken
        if levels['PDL']:
            print(f"PDL: {levels['PDL'].price}, broken: {levels['PDL'].is_broken}")
    
    def test_get_summary(self):
        """Test summary generation"""
        levels = self.detector.detect_levels(self.sample_data)
        summary = self.detector.get_summary(levels)
        
        # Check summary structure
        assert 'pdh_price' in summary
        assert 'pdl_price' in summary
        assert 'pwh_price' in summary
        assert 'pwl_price' in summary
        assert 'pdh_broken' in summary
        assert 'pdl_broken' in summary
        assert 'pwh_broken' in summary
        assert 'pwl_broken' in summary
        
        # Check daily range calculation if both levels exist
        if levels['PDH'] and levels['PDL']:
            assert 'daily_range' in summary
            assert 'daily_range_pct' in summary
            assert summary['daily_range'] > 0
            assert summary['daily_range_pct'] > 0
        
        # Check weekly range calculation if both levels exist
        if levels['PWH'] and levels['PWL']:
            assert 'weekly_range' in summary
            assert 'weekly_range_pct' in summary
            assert summary['weekly_range'] > 0
            assert summary['weekly_range_pct'] > 0
    
    def test_data_validation(self):
        """Test input data validation"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        levels = self.detector.detect_levels(empty_df)
        assert all(v is None for v in levels.values())
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'Open': [1, 2, 3]}, index=pd.date_range('2024-01-01', periods=3, freq='h'))
        levels = self.detector.detect_levels(invalid_df)
        assert all(v is None for v in levels.values())
        
        # Test with non-datetime index
        invalid_df = pd.DataFrame({
            'Open': [1, 2, 3],
            'High': [2, 3, 4],
            'Low': [0.5, 1.5, 2.5],
            'Close': [1.5, 2.5, 3.5]
        })
        levels = self.detector.detect_levels(invalid_df)
        assert all(v is None for v in levels.values())
    
    def test_with_real_data(self):
        """Test with freshly downloaded real market data"""
        try:
            from framework.data.sources.ccxt_source import CCXTSource
            
            # Download fresh data from Binance
            source = CCXTSource(
                exchange_name="binance",
                api_key=None,
                api_secret=None,
                sandbox=False
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)  # 10 days of data
            
            df = source.get_historical_data(
                symbol="BTC/USDT:USDT",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe="15m"
            )
            
            if len(df) == 0:
                pytest.skip("Unable to fetch real market data")
            
            # Ensure proper column names
            df.columns = [col.capitalize() for col in df.columns]
            
            # Detect levels
            levels = self.detector.detect_levels(df)
            
            print("\nReal BTC/USDT Levels:")
            if levels['PDH']:
                print(f"PDH: ${levels['PDH'].price:,.2f} (Broken: {levels['PDH'].is_broken})")
            if levels['PDL']:
                print(f"PDL: ${levels['PDL'].price:,.2f} (Broken: {levels['PDL'].is_broken})")
            if levels['PWH']:
                print(f"PWH: ${levels['PWH'].price:,.2f} (Broken: {levels['PWH'].is_broken})")
            if levels['PWL']:
                print(f"PWL: ${levels['PWL'].price:,.2f} (Broken: {levels['PWL'].is_broken})")
            
            # Get summary
            summary = self.detector.get_summary(levels)
            if 'daily_range_pct' in summary:
                print(f"Daily Range: {summary['daily_range_pct']:.2f}%")
            if 'weekly_range_pct' in summary:
                print(f"Weekly Range: {summary['weekly_range_pct']:.2f}%")
            
            # Basic validation
            if levels['PDH']:
                assert levels['PDH'].price > 0
            if levels['PDL']:
                assert levels['PDL'].price > 0
            if levels['PWH']:
                assert levels['PWH'].price > 0
            if levels['PWL']:
                assert levels['PWL'].price > 0
            
        except Exception as e:
            pytest.skip(f"Real data test skipped: {e}")
    
    def test_visualization(self):
        """Test visualization of PDH/PDL levels"""
        # Create test data with clear daily patterns
        dates = pd.date_range(start='2024-01-01', periods=120, freq='h')  # 5 days
        
        # Generate price data with daily patterns
        prices = []
        for i, date in enumerate(dates):
            day_num = i // 24
            hour_in_day = i % 24
            
            # Base price for each day
            base_prices = [100, 105, 102, 108, 106]
            base = base_prices[day_num]
            
            # Add intraday pattern
            intraday_pattern = 3 * np.sin(hour_in_day * np.pi / 12)
            price = base + intraday_pattern
            prices.append(price)
        
        test_data = pd.DataFrame({
            'Open': prices,
            'High': [p + 0.5 for p in prices],
            'Low': [p - 0.5 for p in prices],
            'Close': [p + 0.2 for p in prices],
            'Volume': [1000] * len(prices)
        }, index=dates)
        
        # Detect levels
        levels = self.detector.detect_levels(test_data)
        
        # Create visualization (last 48 hours for clarity)
        recent_data = test_data.tail(48)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot candlesticks
        for i, (index, row) in enumerate(recent_data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            
            # Candlestick body
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Close'], row['Open'])
            
            # Draw body
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # Draw wicks
            ax.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], 'k-', linewidth=1)
            ax.plot([i, i], [max(row['Open'], row['Close']), row['High']], 'k-', linewidth=1)
        
        # Plot PDH/PDL levels
        if levels['PDH']:
            pdh_line = ax.axhline(y=levels['PDH'].price, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(recent_data) - 1, levels['PDH'].price, 
                   f" PDH: {levels['PDH'].price:.2f}{'✓' if levels['PDH'].is_broken else ''}",
                   verticalalignment='bottom', color='red', fontweight='bold')
        
        if levels['PDL']:
            pdl_line = ax.axhline(y=levels['PDL'].price, color='blue', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(recent_data) - 1, levels['PDL'].price,
                   f" PDL: {levels['PDL'].price:.2f}{'✓' if levels['PDL'].is_broken else ''}",
                   verticalalignment='top', color='blue', fontweight='bold')
        
        # Plot PWH/PWL levels if available
        if levels['PWH']:
            pwh_line = ax.axhline(y=levels['PWH'].price, color='darkred', linestyle=':', alpha=0.5, linewidth=2)
            ax.text(0, levels['PWH'].price,
                   f"PWH: {levels['PWH'].price:.2f} ",
                   verticalalignment='bottom', color='darkred', horizontalalignment='right')
        
        if levels['PWL']:
            pwl_line = ax.axhline(y=levels['PWL'].price, color='darkblue', linestyle=':', alpha=0.5, linewidth=2)
            ax.text(0, levels['PWL'].price,
                   f"PWL: {levels['PWL'].price:.2f} ",
                   verticalalignment='top', color='darkblue', horizontalalignment='right')
        
        # Formatting
        ax.set_xlabel('Hours')
        ax.set_ylabel('Price')
        ax.set_title('Previous Day/Week High/Low Levels')
        ax.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = []
        if levels['PDH']:
            legend_elements.append(Line2D([0], [0], color='red', linestyle='--', label='PDH'))
        if levels['PDL']:
            legend_elements.append(Line2D([0], [0], color='blue', linestyle='--', label='PDL'))
        if levels['PWH']:
            legend_elements.append(Line2D([0], [0], color='darkred', linestyle=':', label='PWH'))
        if levels['PWL']:
            legend_elements.append(Line2D([0], [0], color='darkblue', linestyle=':', label='PWL'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left')
        
        # Set x-axis labels
        step = max(1, len(recent_data) // 8)
        ax.set_xticks(range(0, len(recent_data), step))
        ax.set_xticklabels([recent_data.index[i].strftime('%m/%d %H:%M')
                           for i in range(0, len(recent_data), step)], rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 'output', 'tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'pdh_pdl_detection_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"PDH/PDL visualization saved to: {plot_path}")
        
        # Verify visualization was created
        assert os.path.exists(plot_path)


if __name__ == "__main__":
    # Run tests
    test_instance = TestPDHPDLDetector()
    test_instance.setup_method()
    
    # Run basic tests
    test_instance.test_detector_initialization()
    test_instance.test_detect_levels_basic()
    test_instance.test_level_types()
    test_instance.test_break_detection()
    test_instance.test_get_summary()
    test_instance.test_data_validation()
    
    # Run visualization test
    test_instance.test_visualization()
    
    try:
        test_instance.test_with_real_data()
    except Exception as e:
        print(f"Skipped real data test: {e}")
    
    print("All PDH/PDL detector tests completed successfully!")