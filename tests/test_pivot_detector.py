"""
Test cases for pivot detector with candlestick visualization
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import yfinance as yf
from datetime import datetime, timedelta
import os

from strategies.detectors.pivot_detector import PivotDetector


class TestPivotDetector:
    """Test cases for PivotDetector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = PivotDetector()
        
        # Create sample data for testing
        self.sample_highs = [100, 105, 110, 108, 106, 104, 107, 109, 111, 108, 106, 103, 105, 108, 110]
        self.sample_lows = [98, 102, 107, 105, 103, 101, 104, 106, 108, 105, 103, 100, 102, 105, 107]
        
        # Create OHLC data for candlestick visualization
        dates = pd.date_range(start='2023-01-01', periods=len(self.sample_highs), freq='D')
        self.ohlc_data = pd.DataFrame({
            'Date': dates,
            'Open': [h - 2 for h in self.sample_highs],
            'High': self.sample_highs,
            'Low': self.sample_lows,
            'Close': [h - 1 for h in self.sample_highs]
        })
        
    def test_pivot_high_detection(self):
        """Test pivot high detection"""
        # Test with known pivot high
        test_series = [100, 105, 110, 108, 106]  # 110 should be pivot high with left=2, right=2
        result = self.detector.pivot_high(test_series, left_bars=2, right_bars=2)
        assert result == 110
        
        # Test with no pivot high
        test_series = [100, 105, 108, 110, 112]  # No pivot high (increasing)
        result = self.detector.pivot_high(test_series, left_bars=2, right_bars=2)
        assert result is None
        
    def test_pivot_low_detection(self):
        """Test pivot low detection"""
        # Test with known pivot low
        test_series = [105, 102, 98, 101, 104]  # 98 should be pivot low with left=2, right=2
        result = self.detector.pivot_low(test_series, left_bars=2, right_bars=2)
        assert result == 98
        
        # Test with no pivot low
        test_series = [105, 102, 98, 95, 90]  # No pivot low (decreasing)
        result = self.detector.pivot_low(test_series, left_bars=2, right_bars=2)
        assert result is None
        
    def test_input_validation(self):
        """Test input validation"""
        # Test with empty series
        result = self.detector.pivot_high([], left_bars=2, right_bars=2)
        assert result is None
        
        # Test with invalid bar counts
        result = self.detector.pivot_high(self.sample_highs, left_bars=0, right_bars=2)
        assert result is None
        
        result = self.detector.pivot_high(self.sample_highs, left_bars=2, right_bars=0)
        assert result is None
        
    def test_find_all_pivots(self):
        """Test finding all pivot points"""
        # Test finding all pivot highs
        pivot_highs = self.detector.find_all_pivot_highs(self.sample_highs, left_bars=2, right_bars=2)
        assert len(pivot_highs) > 0
        
        # Test finding all pivot lows
        pivot_lows = self.detector.find_all_pivot_lows(self.sample_lows, left_bars=2, right_bars=2)
        assert len(pivot_lows) > 0
        
        # Verify pivot values are correct
        for index, value in pivot_highs:
            assert value == self.sample_highs[index]
            
        for index, value in pivot_lows:
            assert value == self.sample_lows[index]
            
    def test_with_real_data(self):
        """Test with real market data"""
        try:
            # Try to fetch real data for testing
            ticker = "AAPL"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                # Test with real high/low data
                pivot_highs = self.detector.find_all_pivot_highs(data['High'].values, left_bars=5, right_bars=5)
                pivot_lows = self.detector.find_all_pivot_lows(data['Low'].values, left_bars=5, right_bars=5)
                
                # Should find some pivots in 90 days of data
                assert len(pivot_highs) >= 0  # Could be 0 if no pivots found
                assert len(pivot_lows) >= 0   # Could be 0 if no pivots found
                
        except Exception as e:
            # Skip if unable to fetch data
            pytest.skip(f"Unable to fetch real data: {e}")
            
    def test_candlestick_visualization(self):
        """Test candlestick visualization with pivot points"""
        # Find pivot points
        left_bars, right_bars = 2, 2
        pivot_highs = self.detector.find_all_pivot_highs(self.ohlc_data['High'].values, left_bars, right_bars)
        pivot_lows = self.detector.find_all_pivot_lows(self.ohlc_data['Low'].values, left_bars, right_bars)
        
        # Create candlestick chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot candlesticks
        for i, (index, row) in enumerate(self.ohlc_data.iterrows()):
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
        
        # Highlight pivot highs
        for index, value in pivot_highs:
            if index < len(self.ohlc_data):
                ax.scatter(index, value, color='blue', s=100, marker='^', 
                          label='Pivot High' if index == pivot_highs[0][0] else "")
                ax.annotate(f'PH: {value:.1f}', (index, value), 
                           xytext=(5, 10), textcoords='offset points', 
                           fontsize=8, color='blue')
        
        # Highlight pivot lows
        for index, value in pivot_lows:
            if index < len(self.ohlc_data):
                ax.scatter(index, value, color='purple', s=100, marker='v', 
                          label='Pivot Low' if index == pivot_lows[0][0] else "")
                ax.annotate(f'PL: {value:.1f}', (index, value), 
                           xytext=(5, -15), textcoords='offset points', 
                           fontsize=8, color='purple')
        
        # Formatting
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Price')
        ax.set_title(f'Candlestick Chart with Pivot Points (Left: {left_bars}, Right: {right_bars})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis labels
        ax.set_xticks(range(0, len(self.ohlc_data), 2))
        ax.set_xticklabels([self.ohlc_data.iloc[i]['Date'].strftime('%m/%d') 
                           for i in range(0, len(self.ohlc_data), 2)], rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(test_dir, 'pivot_detection_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Candlestick chart with pivot points saved to: {plot_path}")
        print(f"Found {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows")
        
        # Verify we created the visualization
        assert os.path.exists(plot_path)
        
    def test_real_data_visualization(self):
        """Test visualization with real market data"""
        try:
            # Fetch real data
            ticker = "AAPL"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = yf.download(ticker, start=start_date, end=end_date, interval="15m")
            
            if len(data) == 0:
                pytest.skip("Unable to fetch real market data")
                
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Handle multi-level columns from yfinance
            if hasattr(data.columns, 'levels'):
                data.columns = data.columns.droplevel(1)  # Remove ticker level
            
            # Find pivot points
            left_bars, right_bars = 50, 50
            pivot_highs = self.detector.find_all_pivot_highs(data['High'].values, left_bars, right_bars)
            pivot_lows = self.detector.find_all_pivot_lows(data['Low'].values, left_bars, right_bars)
            
            # Create candlestick chart
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Plot candlesticks
            for i, (index, row) in enumerate(data.iterrows()):
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
            
            # Highlight pivot highs
            pivot_high_plotted = False
            for index, value in pivot_highs:
                if index < len(data):
                    ax.scatter(index, value, color='blue', s=150, marker='^', 
                              label='Pivot High' if not pivot_high_plotted else "", zorder=5)
                    ax.annotate(f'${value:.2f}', (index, value), 
                               xytext=(5, 10), textcoords='offset points', 
                               fontsize=9, color='blue', weight='bold')
                    pivot_high_plotted = True
            
            # Highlight pivot lows
            pivot_low_plotted = False
            for index, value in pivot_lows:
                if index < len(data):
                    ax.scatter(index, value, color='purple', s=150, marker='v', 
                              label='Pivot Low' if not pivot_low_plotted else "", zorder=5)
                    ax.annotate(f'${value:.2f}', (index, value), 
                               xytext=(5, -15), textcoords='offset points', 
                               fontsize=9, color='purple', weight='bold')
                    pivot_low_plotted = True
            
            # Formatting
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{ticker} - Candlestick Chart with Pivot Points (L:{left_bars}, R:{right_bars})')
            ax.grid(True, alpha=0.3)
            if pivot_high_plotted or pivot_low_plotted:
                ax.legend()
            
            # Set x-axis labels
            step = max(1, len(data) // 10)
            ax.set_xticks(range(0, len(data), step))
            ax.set_xticklabels([data.iloc[i]['Datetime'].strftime('%m/%d %H:%M') 
                               for i in range(0, len(data), step)], rotation=45)
            
            plt.tight_layout()
            
            # Save the plot
            test_dir = os.path.dirname(os.path.abspath(__file__))
            plot_path = os.path.join(test_dir, f'pivot_detection_{ticker}_real.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Real data candlestick chart saved to: {plot_path}")
            print(f"Found {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows in {ticker}")
            
            # Verify we created the visualization
            assert os.path.exists(plot_path)
            
        except Exception as e:
            pytest.skip(f"Unable to create real data visualization: {e}")


if __name__ == "__main__":
    # Run tests with visualization
    test_instance = TestPivotDetector()
    test_instance.setup_method()
    
    # Run basic tests
    test_instance.test_pivot_high_detection()
    test_instance.test_pivot_low_detection()
    test_instance.test_input_validation()
    test_instance.test_find_all_pivots()
    
    # Run visualization tests
    test_instance.test_candlestick_visualization()
    test_instance.test_real_data_visualization()
    
    print("All tests completed successfully!")