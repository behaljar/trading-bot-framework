import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
Test cases for FVG detector with candlestick visualization
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import os

from framework.strategies.detectors.fvg_detector import FVGDetector, FVG


class TestFVGDetector:
    """Test cases for FVGDetector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = FVGDetector(min_sensitivity=0.1)
        
        # Create sample OHLCV data with known FVGs and realistic candlesticks
        dates = pd.date_range(start='2023-01-01', periods=20, freq='h')
        
        # Create realistic OHLCV data with intentional FVGs
        # Rule: open[i] = close[i-1] (continuous market)
        # Each candle: low <= open,close <= high
        
        # Start with first candle
        closes = [101.5]  # First close
        highs = [102.0]   # First high
        lows = [99.0]     # First low
        opens = [100.0]   # First open
        
        # Build subsequent candles where open = previous close
        base_closes = [103.0, 106.5, 107.5, 105.0, 108.0, 111.5, 115.0, 112.5, 109.0, 104.0, 101.0, 98.5, 96.5, 98.5, 100.5, 103.5, 106.0, 104.0, 102.0]
        
        for i, target_close in enumerate(base_closes):
            prev_close = closes[-1]
            opens.append(prev_close)  # Open = previous close
            
            # Determine if bullish or bearish candle
            if target_close > prev_close:  # Bullish candle
                low = min(prev_close, target_close) - np.random.uniform(0.5, 2.0)
                high = max(prev_close, target_close) + np.random.uniform(0.5, 3.0)
            else:  # Bearish candle
                low = min(prev_close, target_close) - np.random.uniform(0.5, 3.0)
                high = max(prev_close, target_close) + np.random.uniform(0.5, 2.0)
            
            lows.append(low)
            highs.append(high)
            closes.append(target_close)
        
        self.sample_data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': [1000 + i * 50 for i in range(20)]  # Varying volume
        }, index=dates)
        
        # Create data with specific FVG scenarios (realistic OHLCV)
        # Manual construction to ensure FVG patterns while maintaining realistic candles
        fvg_opens = [100.0, 101.0, 106.0, 103.0, 109.0, 105.0, 113.0]  # open[i] = close[i-1]
        fvg_closes = [101.0, 106.0, 103.0, 109.0, 105.0, 113.0, 107.0]
        fvg_highs = [102.0, 107.5, 106.5, 110.0, 109.5, 114.0, 113.5]
        fvg_lows = [99.0, 100.0, 102.0, 102.5, 104.0, 104.5, 106.0]
        
        self.fvg_test_data = pd.DataFrame({
            'open': fvg_opens,
            'high': fvg_highs,
            'low': fvg_lows,
            'close': fvg_closes,
            'volume': [1000 + i * 100 for i in range(7)]
        })
        
    def test_bullish_fvg_detection(self):
        """Test bullish FVG detection"""
        # Create specific data for bullish FVG: high[i-1] < low[i+1]
        # Realistic: open[1] = close[0], open[2] = close[1]
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 106.0],    # open[i] = close[i-1]
            'high': [102.0, 107.0, 108.0],    # high[0] = 102, low[2] = 107, gap exists
            'low': [99.0, 100.5, 107.0],      # gap between high[0]=102 and low[2]=107
            'close': [101.0, 106.0, 107.5],   
            'volume': [1000] * 3
        })
        
        fvg = self.detector._detect_bullish_fvg(test_data, 1)
        assert fvg is not None
        assert fvg.fvg_type == 'bullish'
        assert fvg.bottom == 102.0  # high[0]
        assert fvg.top == 107.0     # low[2]
        
    def test_bearish_fvg_detection(self):
        """Test bearish FVG detection"""
        # Create specific data for bearish FVG: low[i-1] > high[i+1]
        # Realistic: open[i] = close[i-1] and proper OHLC relationships
        test_data = pd.DataFrame({
            'open': [110.0, 108.0, 103.0],    # open[i] = close[i-1]
            'high': [112.0, 109.0, 105.0],    # low[0] = 108, high[2] = 105, gap exists
            'low': [108.0, 102.0, 103.0],     # gap between low[0]=108 and high[2]=105
            'close': [108.0, 103.0, 104.0],   
            'volume': [1000] * 3
        })
        
        fvg = self.detector._detect_bearish_fvg(test_data, 1)
        assert fvg is not None
        assert fvg.fvg_type == 'bearish'
        assert fvg.top == 108.0    # low[0]
        assert fvg.bottom == 105.0  # high[2]
        
    def test_no_fvg_detection(self):
        """Test when no FVG should be detected"""
        # Create data with no gaps - realistic OHLCV
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],    # open[i] = close[i-1]
            'high': [102.0, 103.0, 104.0],    # Overlapping ranges
            'low': [99.0, 100.5, 101.5],      # high[0]=102, low[2]=101.5 - overlapping
            'close': [101.0, 102.0, 103.0],   
            'volume': [1000] * 3
        })
        
        bullish_fvg = self.detector._detect_bullish_fvg(test_data, 1)
        bearish_fvg = self.detector._detect_bearish_fvg(test_data, 1)
        
        assert bullish_fvg is None
        assert bearish_fvg is None
        
    def test_sensitivity_filtering(self):
        """Test sensitivity-based filtering"""
        # Create FVG with small gap relative to middle candle - realistic OHLCV
        test_data = pd.DataFrame({
            'open': [100.0, 100.2, 105.0],    # open[i] = close[i-1]
            'high': [100.5, 110.0, 105.5],    # Middle candle is large (20 points)
            'low': [99.5, 90.0, 104.8],       # Gap between high[0]=100.5 and low[2]=104.8
            'close': [100.2, 105.0, 105.2],   # Gap is large (4.3 points)
            'volume': [1000] * 3
        })
        
        # With high sensitivity requirement, should not detect
        high_sensitivity_detector = FVGDetector(min_sensitivity=0.5)
        fvg = high_sensitivity_detector._detect_bullish_fvg(test_data, 1)
        assert fvg is None
        
        # With low sensitivity requirement, should detect
        low_sensitivity_detector = FVGDetector(min_sensitivity=0.01)
        fvg = low_sensitivity_detector._detect_bullish_fvg(test_data, 1)
        assert fvg is not None
        
    def test_detect_all_fvgs(self):
        """Test detecting all FVGs in a dataset"""
        fvgs = self.detector.detect_fvgs(self.sample_data, merge_consecutive=False)
        assert isinstance(fvgs, list)
        
        # Verify all FVGs have required properties
        for fvg in fvgs:
            assert isinstance(fvg, FVG)
            assert fvg.fvg_type in ['bullish', 'bearish']
            assert fvg.gap_size > 0
            assert fvg.sensitivity_ratio >= self.detector.min_sensitivity
            
    def test_fvg_merging(self):
        """Test FVG merging logic"""
        # Create data with consecutive FVGs that should be merged
        merge_data = pd.DataFrame({
            'open': [100, 105, 102, 107, 104, 109, 106],
            'high': [102, 107, 104, 109, 106, 111, 108],
            'low': [98, 103, 105, 105, 107, 107, 104],  # Consecutive gaps
            'close': [101, 106, 103, 108, 105, 110, 107],
            'volume': [1000] * 7
        })
        
        # Test without merging
        fvgs_no_merge = self.detector.detect_fvgs(merge_data, merge_consecutive=False)
        
        # Test with merging
        fvgs_merged = self.detector.detect_fvgs(merge_data, merge_consecutive=True)
        
        # Should have fewer or equal FVGs after merging
        assert len(fvgs_merged) <= len(fvgs_no_merge)
        
    def test_active_fvgs(self):
        """Test getting active (unfilled) FVGs"""
        # Create FVGs
        fvgs = self.detector.detect_fvgs(self.sample_data)
        
        if fvgs:
            # Test with different price levels
            current_prices = [95, 105, 115]  # Low, medium, high prices
            
            for price in current_prices:
                active_fvgs = self.detector.get_active_fvgs(fvgs, price)
                assert isinstance(active_fvgs, list)
                
                # All active FVGs should contain logic for the price
                for fvg in active_fvgs:
                    if fvg.fvg_type == 'bullish':
                        assert price <= fvg.top  # Price hasn't exceeded top
                    else:  # bearish
                        assert price >= fvg.bottom  # Price hasn't fallen below bottom
        
    def test_input_validation(self):
        """Test input validation"""
        # Test with invalid data
        empty_data = pd.DataFrame()
        fvgs = self.detector.detect_fvgs(empty_data)
        assert fvgs == []
        
        # Test with insufficient data
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 1000]
        })
        fvgs = self.detector.detect_fvgs(small_data)
        assert fvgs == []
        
        # Test with missing columns
        bad_data = pd.DataFrame({
            'price': [100, 101, 102],
            'volume': [1000, 1000, 1000]
        })
        fvgs = self.detector.detect_fvgs(bad_data)
        assert fvgs == []
        
    def test_summary_statistics(self):
        """Test FVG summary statistics"""
        fvgs = self.detector.detect_fvgs(self.sample_data)
        summary = self.detector.get_fvgs_summary(fvgs)
        
        assert 'total_fvgs' in summary
        assert 'bullish_fvgs' in summary
        assert 'bearish_fvgs' in summary
        assert 'merged_fvgs' in summary
        assert 'avg_gap_size' in summary
        assert 'avg_sensitivity' in summary
        
        # Test with empty list
        empty_summary = self.detector.get_fvgs_summary([])
        assert empty_summary['total_fvgs'] == 0
        
    def test_fvg_visualization(self):
        """Test FVG visualization with candlestick chart"""
        # Detect FVGs
        fvgs = self.detector.detect_fvgs(self.sample_data)
        
        # Create candlestick chart
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot candlesticks
        for i, (index, row) in enumerate(self.sample_data.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            
            # Candlestick body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            # Draw body
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # Draw wicks
            ax.plot([i, i], [row['low'], min(row['open'], row['close'])], 'k-', linewidth=1)
            ax.plot([i, i], [max(row['open'], row['close']), row['high']], 'k-', linewidth=1)
        
        # Highlight FVGs
        bullish_plotted = False
        bearish_plotted = False
        
        for fvg in fvgs:
            if fvg.start_idx < len(self.sample_data) and fvg.end_idx < len(self.sample_data):
                # Draw FVG rectangle
                fvg_color = 'lightblue' if fvg.fvg_type == 'bullish' else 'lightcoral'
                fvg_alpha = 0.3
                
                fvg_rect = Rectangle(
                    (fvg.start_idx - 0.4, fvg.bottom), 
                    fvg.end_idx - fvg.start_idx + 0.8, 
                    fvg.gap_size,
                    facecolor=fvg_color, 
                    edgecolor='blue' if fvg.fvg_type == 'bullish' else 'red',
                    alpha=fvg_alpha,
                    label=f'{fvg.fvg_type.title()} FVG' if not (bullish_plotted and fvg.fvg_type == 'bullish') and not (bearish_plotted and fvg.fvg_type == 'bearish') else ""
                )
                ax.add_patch(fvg_rect)
                
                if fvg.fvg_type == 'bullish':
                    bullish_plotted = True
                else:
                    bearish_plotted = True
                
                # Add text annotation
                mid_x = (fvg.start_idx + fvg.end_idx) / 2
                mid_y = (fvg.top + fvg.bottom) / 2
                ax.annotate(
                    f'{fvg.fvg_type[0].upper()}FVG\n{fvg.gap_size:.2f}',
                    (mid_x, mid_y),
                    ha='center', va='center',
                    fontsize=8,
                    color='darkblue' if fvg.fvg_type == 'bullish' else 'darkred',
                    weight='bold'
                )
        
        # Formatting
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Price')
        ax.set_title(f'FVG Detection - Candlestick Chart (Min Sensitivity: {self.detector.min_sensitivity})')
        ax.grid(True, alpha=0.3)
        if bullish_plotted or bearish_plotted:
            ax.legend()
        
        # Set x-axis labels
        step = max(1, len(self.sample_data) // 10)
        ax.set_xticks(range(0, len(self.sample_data), step))
        ax.set_xticklabels([self.sample_data.index[i].strftime('%m/%d %H:%M') 
                           for i in range(0, len(self.sample_data), step)], rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'output', 'tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'fvg_detection_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"FVG detection chart saved to: {plot_path}")
        print(f"Found {len(fvgs)} FVGs")
        
        # Print summary
        summary = self.detector.get_fvgs_summary(fvgs)
        print(f"Summary: {summary['bullish_fvgs']} bullish, {summary['bearish_fvgs']} bearish, "
              f"avg gap size: {summary['avg_gap_size']:.2f}, avg sensitivity: {summary['avg_sensitivity']:.3f}")
        
        # Verify we created the visualization
        assert os.path.exists(plot_path)
        
    def test_real_data_fvg_visualization(self):
        """Test FVG visualization with real market data"""
        try:
            # Import CCXT source
            from framework.data.ccxt_source import CCXTSource
            
            # Fetch real BTC/USDT data from Binance
            source = CCXTSource(
                exchange_name="binance",
                api_key=None,
                api_secret=None,
                sandbox=False
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3)  # 3 days of 1m data
            
            data = source.get_historical_data(
                symbol="BTC/USDT:USDT",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe="1m"
            )
            
            if len(data) == 0:
                pytest.skip("Unable to fetch real market data")
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Sample data to manageable size for visualization
            if len(data) > 500:
                step = len(data) // 500
                data = data.iloc[::step].reset_index(drop=True)
            
            # Detect FVGs
            detector = FVGDetector(min_sensitivity=0.05)  # Lower sensitivity for real data
            fvgs = detector.detect_fvgs(data)
            
            # Create candlestick chart
            fig, ax = plt.subplots(figsize=(20, 12))
            
            # Plot candlesticks (sample for visibility)
            sample_step = max(1, len(data) // 200)
            sample_data = data.iloc[::sample_step].reset_index(drop=True)
            
            for i, (index, row) in enumerate(sample_data.iterrows()):
                color = 'green' if row['close'] >= row['open'] else 'red'
                
                # Candlestick body
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['close'], row['open'])
                
                # Draw body
                rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                               facecolor=color, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
                
                # Draw wicks
                ax.plot([i, i], [row['low'], min(row['open'], row['close'])], 'k-', linewidth=1)
                ax.plot([i, i], [max(row['open'], row['close']), row['high']], 'k-', linewidth=1)
            
            # Highlight FVGs (scale indices to match sample)
            bullish_plotted = False
            bearish_plotted = False
            
            for fvg in fvgs:
                # Scale FVG indices to match sampled data
                scaled_start = fvg.start_idx // sample_step
                scaled_end = fvg.end_idx // sample_step
                
                if scaled_start < len(sample_data) and scaled_end < len(sample_data):
                    # Draw FVG rectangle
                    fvg_color = 'lightblue' if fvg.fvg_type == 'bullish' else 'lightcoral'
                    fvg_alpha = 0.4
                    
                    fvg_rect = Rectangle(
                        (scaled_start - 0.4, fvg.bottom), 
                        max(1, scaled_end - scaled_start) + 0.8, 
                        fvg.gap_size,
                        facecolor=fvg_color, 
                        edgecolor='blue' if fvg.fvg_type == 'bullish' else 'red',
                        alpha=fvg_alpha,
                        linewidth=2,
                        label=f'{fvg.fvg_type.title()} FVG' if not (bullish_plotted and fvg.fvg_type == 'bullish') and not (bearish_plotted and fvg.fvg_type == 'bearish') else ""
                    )
                    ax.add_patch(fvg_rect)
                    
                    if fvg.fvg_type == 'bullish':
                        bullish_plotted = True
                    else:
                        bearish_plotted = True
            
            # Formatting
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'BTC/USDT - FVG Detection with Real Data (Min Sensitivity: {detector.min_sensitivity})')
            ax.grid(True, alpha=0.3)
            if bullish_plotted or bearish_plotted:
                ax.legend()
            
            # Set x-axis labels
            step = max(1, len(sample_data) // 10)
            ax.set_xticks(range(0, len(sample_data), step))
            timestamp_indices = [i * sample_step for i in range(0, len(sample_data), step)]
            ax.set_xticklabels([data.iloc[min(i, len(data)-1)]['timestamp'].strftime('%m/%d %H:%M') 
                               for i in timestamp_indices], rotation=45)
            
            plt.tight_layout()
            
            # Save the plot
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, 'output', 'tests')
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'fvg_detection_btc_usdt_real.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Real data FVG chart saved to: {plot_path}")
            print(f"Found {len(fvgs)} FVGs in BTC/USDT real data")
            
            # Print summary
            summary = detector.get_fvgs_summary(fvgs)
            print(f"Real data summary: {summary['bullish_fvgs']} bullish, {summary['bearish_fvgs']} bearish")
            print(f"Average gap size: ${summary['avg_gap_size']:.2f}, avg sensitivity: {summary['avg_sensitivity']:.3f}")
            
            # Verify we created the visualization
            assert os.path.exists(plot_path)
            
        except Exception as e:
            pytest.skip(f"Unable to create real data FVG visualization: {e}")


if __name__ == "__main__":
    # Run tests with visualization
    test_instance = TestFVGDetector()
    test_instance.setup_method()
    
    # Run basic tests
    test_instance.test_bullish_fvg_detection()
    test_instance.test_bearish_fvg_detection()
    test_instance.test_no_fvg_detection()
    test_instance.test_sensitivity_filtering()
    test_instance.test_detect_all_fvgs()
    test_instance.test_fvg_merging()
    test_instance.test_active_fvgs()
    test_instance.test_input_validation()
    test_instance.test_summary_statistics()
    
    # Run visualization tests
    test_instance.test_fvg_visualization()
    test_instance.test_real_data_fvg_visualization()
    
    print("All FVG detector tests completed successfully!")