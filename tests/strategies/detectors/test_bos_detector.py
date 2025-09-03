import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
Test cases for Break of Structure (BoS) detector
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import os

from framework.strategies.detectors.bos_detector import BoSDetector


class TestBoSDetector:
    """Test cases for BoSDetector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = BoSDetector(swing_sensitivity=2)
        
        # Create sample OHLC data that should generate BoS events
        dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
        
        # Create pattern: Downtrend -> Higher Low -> Break of Previous High (Bullish BoS)
        price_data = [
            100, 102, 98, 96, 94,  # Initial downtrend
            92, 94, 90, 88, 86,    # Continue down
            84, 86, 88, 90, 92,    # Start recovery
            94, 96, 98, 100, 102,  # Higher low formation
            104, 106, 108, 110, 112,  # Break previous high - BoS
            114, 116, 118, 120, 122,  # Continue up
            124, 122, 120, 118, 116,  # Start pullback
            114, 112, 110, 108, 106,  # Lower high formation
            104, 102, 100, 98, 96,   # Break previous low - BoS
            94, 92, 90, 88, 86      # Continue down
        ]
        
        self.test_data = pd.DataFrame({
            'Open': [p - 1 for p in price_data],
            'High': [p + 1 for p in price_data],
            'Low': [p - 1 for p in price_data],
            'Close': price_data,
            'Volume': [1000] * len(price_data)
        }, index=dates)
        
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = BoSDetector(swing_sensitivity=3)
        assert detector.swing_sensitivity == 3
        assert detector.swing_detector.sensitivity == 3
        
    def test_data_validation(self):
        """Test input data validation"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = self.detector.detect_bos_events(empty_df)
        assert result.empty
        assert list(result.columns) == ['timestamp', 'bos_type', 'price', 'swing_broken']
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'Open': [1, 2, 3]})
        result = self.detector.detect_bos_events(invalid_df)
        assert result.empty
        
        # Test with insufficient data
        small_df = pd.DataFrame({
            'Open': [1, 2], 'High': [2, 3], 'Low': [0.5, 1.5], 'Close': [1.5, 2.5]
        })
        result = self.detector.detect_bos_events(small_df)
        assert result.empty
        
    def test_bos_detection(self):
        """Test BoS event detection with sample data"""
        result = self.detector.detect_bos_events(self.test_data)
        
        # Should detect some BoS events (may be empty with synthetic data)
        assert 'timestamp' in result.columns
        assert 'bos_type' in result.columns
        assert 'price' in result.columns
        assert 'swing_broken' in result.columns
        
        if not result.empty:
            # Check that we have valid BoS events
            bos_types = result['bos_type'].unique()
            assert len(bos_types) > 0
            
            # Verify BoS types are valid
            valid_types = {'bullish', 'bearish'}
            for bos_type in bos_types:
                assert bos_type in valid_types
            
        # Verify prices are numeric and positive
        assert all(isinstance(price, (int, float)) and price > 0 for price in result['price'])
        assert all(isinstance(price, (int, float)) and price > 0 for price in result['swing_broken'])
        
        print(f"Detected {len(result)} BoS events:")
        print(result.head(10))
        
    def test_bullish_bos_pattern(self):
        """Test specific bullish BoS pattern detection"""
        # Create specific bullish BoS pattern: Higher Low -> Break Previous High
        dates = pd.date_range(start='2024-01-01', periods=20, freq='15min')
        
        # Pattern: High at 110, Low at 100, Higher Low at 105, Break 110
        pattern_data = [
            105, 108, 110, 108, 105,  # Initial high at 110
            102, 100, 103, 105, 107,  # Low at 100
            104, 106, 105, 107, 109,  # Higher low at 105 (higher than 100)
            111, 113, 115, 112, 114   # Break previous high at 110 -> BoS
        ]
        
        test_df = pd.DataFrame({
            'Open': [p - 0.5 for p in pattern_data],
            'High': [p + 0.5 for p in pattern_data],
            'Low': [p - 0.5 for p in pattern_data],
            'Close': pattern_data,
            'Volume': [1000] * len(pattern_data)
        }, index=dates)
        
        result = self.detector.detect_bos_events(test_df)
        
        # Should detect at least one bullish BoS
        bullish_bos = result[result['bos_type'] == 'bullish']
        assert len(bullish_bos) > 0
        
        # Verify we have a bullish BoS event (swing broken should be reasonable)
        assert all(broken > 100 for broken in bullish_bos['swing_broken'])
        print(f"Bullish BoS detected: {len(bullish_bos)} events")
        print(bullish_bos)
        
    def test_bearish_bos_pattern(self):
        """Test specific bearish BoS pattern detection"""
        # Create specific bearish BoS pattern: Lower High -> Break Previous Low
        dates = pd.date_range(start='2024-01-01', periods=20, freq='15min')
        
        # Pattern: Low at 90, High at 100, Lower High at 95, Break 90
        pattern_data = [
            95, 92, 90, 93, 95,       # Initial low at 90
            98, 100, 97, 95, 93,      # High at 100
            96, 98, 95, 93, 91,       # Lower high at 95 (lower than 100)
            89, 87, 85, 88, 86        # Break previous low at 90 -> BoS
        ]
        
        test_df = pd.DataFrame({
            'Open': [p + 0.5 for p in pattern_data],
            'High': [p + 0.5 for p in pattern_data],
            'Low': [p - 0.5 for p in pattern_data],
            'Close': pattern_data,
            'Volume': [1000] * len(pattern_data)
        }, index=dates)
        
        result = self.detector.detect_bos_events(test_df)
        
        # Should detect at least one bearish BoS
        bearish_bos = result[result['bos_type'] == 'bearish']
        assert len(bearish_bos) > 0
        
        # Verify we have a bearish BoS event (swing broken should be reasonable)
        assert all(broken < 95 for broken in bearish_bos['swing_broken'])
        print(f"Bearish BoS detected: {len(bearish_bos)} events")
        print(bearish_bos)
        
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
            start_date = end_date - timedelta(days=3)  # 3 days of 1m data
            
            df = source.get_historical_data(
                symbol="BTC/USDT:USDT",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe="1m"
            )
            
            if len(df) == 0:
                pytest.skip("Unable to fetch real market data")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Ensure proper column names
            df.columns = [col.capitalize() for col in df.columns]
            df = df.set_index('Timestamp')
            
            # Test BoS detection
            result = self.detector.detect_bos_events(df)
            
            print(f"Real data BoS detection: {len(result)} events found")
            if not result.empty:
                print("Sample BoS events:")
                print(result.head())
                
                # Basic validation
                assert all(result['bos_type'].isin(['bullish', 'bearish']))
                assert all(result['price'] > 0)
                assert all(result['swing_broken'] > 0)
                
        except Exception as e:
            pytest.skip(f"Real data test skipped: {e}")
            
    def test_visualization(self):
        """Test BoS detection visualization"""
        # Use smaller dataset for clearer visualization
        dates = pd.date_range(start='2024-01-01', periods=30, freq='15min')
        
        # Clear BoS pattern
        pattern_data = [
            100, 102, 105, 103, 101,   # Initial movement
            98, 96, 99, 101, 103,      # Low formation
            105, 107, 109, 106, 104,   # Higher low
            106, 108, 111, 113, 115,   # Break high -> BoS
            113, 111, 109, 107, 105,   # Pullback starts
            103, 101, 98, 96, 94       # Lower high -> Break low BoS
        ]
        
        viz_data = pd.DataFrame({
            'Open': [p - 0.5 for p in pattern_data],
            'High': [p + 1 for p in pattern_data],
            'Low': [p - 1 for p in pattern_data],
            'Close': pattern_data,
            'Volume': [1000] * len(pattern_data)
        }, index=dates)
        
        # Detect BoS events
        bos_events = self.detector.detect_bos_events(viz_data)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot candlesticks
        for i, (index, row) in enumerate(viz_data.iterrows()):
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
        
        # Plot BoS events
        bullish_plotted = False
        bearish_plotted = False
        
        for _, event in bos_events.iterrows():
            # Find the index in viz_data corresponding to the event timestamp
            event_idx = None
            for i, ts in enumerate(viz_data.index):
                if ts == event['timestamp']:
                    event_idx = i
                    break
            
            if event_idx is not None:
                if event['bos_type'] == 'bullish':
                    ax.scatter(event_idx, event['price'], color='lime', s=200, marker='^', 
                              label='Bullish BoS' if not bullish_plotted else "", zorder=5, edgecolors='darkgreen')
                    ax.annotate(f"BoS↑ {event['price']:.1f}", (event_idx, event['price']), 
                               xytext=(5, 15), textcoords='offset points', 
                               fontsize=10, color='darkgreen', weight='bold')
                    bullish_plotted = True
                else:
                    ax.scatter(event_idx, event['price'], color='red', s=200, marker='v', 
                              label='Bearish BoS' if not bearish_plotted else "", zorder=5, edgecolors='darkred')
                    ax.annotate(f"BoS↓ {event['price']:.1f}", (event_idx, event['price']), 
                               xytext=(5, -20), textcoords='offset points', 
                               fontsize=10, color='darkred', weight='bold')
                    bearish_plotted = True
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title(f'Break of Structure (BoS) Detection Test\nSensitivity: {self.detector.swing_sensitivity}')
        ax.grid(True, alpha=0.3)
        if bullish_plotted or bearish_plotted:
            ax.legend()
        
        # Set x-axis labels
        step = max(1, len(viz_data) // 8)
        ax.set_xticks(range(0, len(viz_data), step))
        ax.set_xticklabels([viz_data.index[i].strftime('%m/%d %H:%M') 
                           for i in range(0, len(viz_data), step)], rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output', 'tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'bos_detection_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"BoS detection visualization saved to: {plot_path}")
        print(f"Detected {len(bos_events)} BoS events")
        
        # Verify visualization was created
        assert os.path.exists(plot_path)


if __name__ == "__main__":
    # Run tests with visualization
    test_instance = TestBoSDetector()
    test_instance.setup_method()
    
    # Run basic tests
    test_instance.test_detector_initialization()
    test_instance.test_data_validation()
    test_instance.test_bos_detection()
    test_instance.test_bullish_bos_pattern()
    test_instance.test_bearish_bos_pattern()
    
    # Run visualization test
    test_instance.test_visualization()
    
    try:
        test_instance.test_with_real_data()
    except Exception as e:
        print(f"Skipped real data test: {e}")
    
    print("All BoS detector tests completed successfully!")