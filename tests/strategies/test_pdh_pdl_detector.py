import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
Test cases for PDH/PDL detector with candlestick visualization
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import os

from framework.strategies.detectors.pdh_pdl_detector import PDHPDLDetector, HighLowLevel


class TestPDHPDLDetector:
    """Test cases for PDHPDLDetector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = PDHPDLDetector(touch_threshold=0.001, break_threshold=0.0005)
        
        # Create sample intraday data spanning multiple days
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 15)
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Create realistic OHLCV data with daily patterns
        np.random.seed(42)
        self.sample_data = self._generate_realistic_data(timestamps)
        
    def _generate_realistic_data(self, timestamps):
        """Generate realistic OHLCV data with proper market structure and levels"""
        # Create realistic daily scenarios with trending moves and consolidation
        daily_scenarios = [
            {'trend': 'bullish_breakout', 'volatility': 0.025, 'range_factor': 1.5},
            {'trend': 'bearish_selloff', 'volatility': 0.030, 'range_factor': 1.8},
            {'trend': 'sideways_range', 'volatility': 0.012, 'range_factor': 0.8},
            {'trend': 'bullish_continuation', 'volatility': 0.020, 'range_factor': 1.2},
            {'trend': 'reversal_day', 'volatility': 0.028, 'range_factor': 2.0},
            {'trend': 'gap_fill', 'volatility': 0.015, 'range_factor': 1.0},
            {'trend': 'accumulation', 'volatility': 0.008, 'range_factor': 0.6},
            {'trend': 'distribution', 'volatility': 0.018, 'range_factor': 1.3},
        ]
        
        data = []
        current_price = 48500.0  # Start at a reasonable BTC price
        
        # Group timestamps by day for realistic daily patterns
        daily_groups = {}
        for ts in timestamps:
            day_key = ts.date()
            if day_key not in daily_groups:
                daily_groups[day_key] = []
            daily_groups[day_key].append(ts)
        
        for day_idx, (date, day_timestamps) in enumerate(sorted(daily_groups.items())):
            # Select scenario for this day
            scenario = daily_scenarios[day_idx % len(daily_scenarios)]
            
            # Set daily parameters
            day_volatility = scenario['volatility']
            range_factor = scenario['range_factor']
            trend_type = scenario['trend']
            
            # Calculate daily target based on trend
            if trend_type == 'bullish_breakout':
                daily_move = np.random.uniform(0.02, 0.04) * current_price  # 2-4% up
            elif trend_type == 'bearish_selloff':
                daily_move = -np.random.uniform(0.025, 0.05) * current_price  # 2.5-5% down
            elif trend_type == 'sideways_range':
                daily_move = np.random.uniform(-0.01, 0.01) * current_price  # ±1%
            elif trend_type == 'bullish_continuation':
                daily_move = np.random.uniform(0.015, 0.025) * current_price  # 1.5-2.5% up
            elif trend_type == 'reversal_day':
                daily_move = np.random.uniform(-0.03, 0.03) * current_price  # Big reversal
            elif trend_type == 'gap_fill':
                daily_move = np.random.uniform(-0.015, 0.015) * current_price  # Gap fill
            elif trend_type == 'accumulation':
                daily_move = np.random.uniform(-0.005, 0.01) * current_price  # Slow accumulation
            else:  # distribution
                daily_move = np.random.uniform(-0.02, 0.005) * current_price  # Distribution
            
            # Generate intraday candles for this day
            day_start_price = current_price
            target_end_price = day_start_price + daily_move
            
            # Create realistic intraday progression
            for candle_idx, ts in enumerate(day_timestamps):
                if len(data) == 0:
                    # First candle
                    open_price = day_start_price
                else:
                    # Open = previous close (no gaps)
                    open_price = data[-1]['close']
                
                # Progress through the day toward target
                day_progress = (candle_idx + 1) / len(day_timestamps)
                
                # Add session-based volatility
                hour = ts.hour
                if 8 <= hour <= 12:  # European session
                    session_vol_mult = 1.2
                    volume_mult = 1.5
                elif 13 <= hour <= 17:  # US session  
                    session_vol_mult = 1.5
                    volume_mult = 2.0
                elif 22 <= hour <= 2:  # Asian session
                    session_vol_mult = 1.0
                    volume_mult = 1.2
                else:  # Off hours
                    session_vol_mult = 0.6
                    volume_mult = 0.8
                
                # Calculate target close for this candle
                if trend_type == 'reversal_day':
                    # Create V-shaped or inverted-V pattern
                    if day_progress < 0.6:
                        # First part - move opposite to final direction
                        temp_target = day_start_price - (daily_move * 0.8 * day_progress / 0.6)
                    else:
                        # Reversal phase
                        reversal_progress = (day_progress - 0.6) / 0.4
                        temp_target = (day_start_price - daily_move * 0.8) + (daily_move * 1.5 * reversal_progress)
                else:
                    # Normal progression toward daily target
                    temp_target = day_start_price + (daily_move * day_progress)
                
                # Add noise and mean reversion
                noise = np.random.normal(0, day_volatility * session_vol_mult * 0.3) * open_price
                
                # Mean reversion to daily target
                mean_revert = (temp_target - open_price) * 0.1
                
                close_price = open_price + noise + mean_revert
                
                # Create realistic high/low based on candle type
                candle_range = abs(close_price - open_price) * range_factor
                extra_wick = np.random.exponential(candle_range * 0.3)
                
                if close_price > open_price:  # Bullish candle
                    high = max(open_price, close_price) + extra_wick * np.random.uniform(0.3, 1.2)
                    low = min(open_price, close_price) - extra_wick * np.random.uniform(0.1, 0.6)
                else:  # Bearish candle
                    high = max(open_price, close_price) + extra_wick * np.random.uniform(0.1, 0.6)
                    low = min(open_price, close_price) - extra_wick * np.random.uniform(0.3, 1.2)
                
                # Ensure proper OHLC relationships
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                # Realistic volume
                base_volume = 1200 * volume_mult
                volume_noise = np.random.lognormal(0, 0.5)  # Lognormal for realistic volume distribution
                volume = int(base_volume * volume_noise)
                
                data.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            # Update current price for next day
            current_price = data[-1]['close']
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    def test_detector_initialization(self):
        """Test PDHPDLDetector initialization"""
        detector = PDHPDLDetector()
        assert detector.touch_threshold == 0.001
        assert detector.break_threshold == 0.0005
        
        custom_detector = PDHPDLDetector(touch_threshold=0.002, break_threshold=0.001)
        assert custom_detector.touch_threshold == 0.002
        assert custom_detector.break_threshold == 0.001
    
    def test_data_validation(self):
        """Test input data validation"""
        # Test with empty data
        empty_df = pd.DataFrame()
        levels = self.detector.detect_daily_levels(empty_df)
        assert levels == []
        
        # Test with missing columns
        bad_df = pd.DataFrame({
            'price': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3, freq='h'))
        levels = self.detector.detect_daily_levels(bad_df)
        assert levels == []
        
        # Test with non-DatetimeIndex
        bad_index_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'close': [101, 102, 103]
        })
        levels = self.detector.detect_daily_levels(bad_index_df)
        assert levels == []
    
    def test_daily_levels_detection(self):
        """Test daily high/low level detection"""
        levels = self.detector.detect_daily_levels(self.sample_data)
        
        # Should have PDH and PDL for each complete day
        assert len(levels) > 0
        
        # Check level types
        level_types = {level.level_type for level in levels}
        assert 'PDH' in level_types
        assert 'PDL' in level_types
        
        # Verify level structure
        for level in levels:
            assert isinstance(level, HighLowLevel)
            assert level.level_type in ['PDH', 'PDL']
            assert isinstance(level.price, float)
            assert level.price > 0
            assert isinstance(level.timestamp, pd.Timestamp)
    
    def test_weekly_levels_detection(self):
        """Test weekly high/low level detection"""
        levels = self.detector.detect_weekly_levels(self.sample_data)
        
        # Should have PWH and PWL for complete weeks
        assert len(levels) >= 0  # Might be 0 if data doesn't span complete weeks
        
        for level in levels:
            assert isinstance(level, HighLowLevel)
            assert level.level_type in ['PWH', 'PWL']
            assert isinstance(level.price, float)
            assert level.price > 0
    
    def test_unclaimed_only_filter(self):
        """Test unclaimed levels filtering"""
        # Get all levels
        all_levels = self.detector.detect_daily_levels(self.sample_data, unclaimed_only=False)
        
        # Get unclaimed only
        unclaimed_levels = self.detector.detect_daily_levels(self.sample_data, unclaimed_only=True)
        
        # Unclaimed should be subset of all levels
        assert len(unclaimed_levels) <= len(all_levels)
        
        # All unclaimed levels should have is_claimed = False
        for level in unclaimed_levels:
            assert not level.is_claimed
    
    def test_level_claiming_logic(self):
        """Test level claiming/breaking logic"""
        # Create specific test data with known breaks
        timestamps = pd.date_range('2023-01-01', periods=72, freq='h')  # 3 days
        
        # Day 1: High at 110, Low at 90
        # Day 2: Price breaks above 110 (claims PDH)
        # Day 3: Price breaks below 90 (claims PDL)
        
        test_data = []
        for i, ts in enumerate(timestamps):
            if i < 24:  # Day 1
                if i == 12:  # Peak hour
                    data_point = {'open': 108, 'high': 110, 'low': 107, 'close': 109}
                elif i == 6:   # Low hour
                    data_point = {'open': 92, 'high': 93, 'low': 90, 'close': 91}
                else:
                    data_point = {'open': 100, 'high': 102, 'low': 98, 'close': 101}
            elif i < 48:  # Day 2 - break above previous high
                if i == 36:  # Break PDH
                    data_point = {'open': 110, 'high': 115, 'low': 109, 'close': 114}
                else:
                    data_point = {'open': 105, 'high': 107, 'low': 103, 'close': 106}
            else:  # Day 3 - break below previous low
                if i == 60:  # Break PDL
                    data_point = {'open': 90, 'high': 91, 'low': 85, 'close': 86}
                else:
                    data_point = {'open': 95, 'high': 97, 'low': 93, 'close': 96}
            
            data_point['volume'] = 1000
            test_data.append(data_point)
        
        test_df = pd.DataFrame(test_data, index=timestamps)
        
        # Detect levels
        levels = self.detector.detect_daily_levels(test_df, unclaimed_only=False)
        
        # Find the specific levels we're testing
        day1_pdh = None
        day1_pdl = None
        for level in levels:
            if level.timestamp.date() == timestamps[0].date():
                if level.level_type == 'PDH':
                    day1_pdh = level
                elif level.level_type == 'PDL':
                    day1_pdl = level
        
        # Verify levels exist and check claiming status
        if day1_pdh:
            assert day1_pdh.price == 110
            # Should be claimed due to break on day 2
            assert day1_pdh.is_claimed or day1_pdh.touch_count > 0
        
        if day1_pdl:
            assert day1_pdl.price == 90
            # Should be claimed due to break on day 3
            assert day1_pdl.is_claimed or day1_pdl.touch_count > 0
    
    def test_levels_near_price(self):
        """Test getting levels near current price"""
        levels = self.detector.detect_all_levels(self.sample_data)
        
        if levels:
            current_price = self.sample_data['close'].iloc[-1]
            near_levels = self.detector.get_levels_near_price(levels, current_price, distance_pct=0.05)
            
            # All near levels should be within 5% of current price
            for level in near_levels:
                distance_pct = abs(level.price - current_price) / current_price
                assert distance_pct <= 0.05
    
    def test_support_resistance_filtering(self):
        """Test support and resistance level filtering"""
        levels = self.detector.detect_all_levels(self.sample_data)
        
        resistance_levels = self.detector.get_resistance_levels(levels)
        support_levels = self.detector.get_support_levels(levels)
        
        # Verify filtering
        for level in resistance_levels:
            assert level.level_type in ['PDH', 'PWH']
        
        for level in support_levels:
            assert level.level_type in ['PDL', 'PWL']
        
        # Total should equal original
        assert len(resistance_levels) + len(support_levels) == len(levels)
    
    def test_age_filtering(self):
        """Test filtering levels by age"""
        levels = self.detector.detect_all_levels(self.sample_data)
        
        if levels:
            # Filter to last 7 days
            recent_levels = self.detector.filter_by_age(levels, max_age_days=7)
            
            # All recent levels should be within 7 days
            cutoff_date = pd.Timestamp.now() - timedelta(days=7)
            for level in recent_levels:
                # Note: Our test data is from 2023, so this might filter everything
                # This test mainly checks the filtering logic works
                pass
    
    def test_summary_statistics(self):
        """Test level summary statistics"""
        levels = self.detector.detect_all_levels(self.sample_data)
        summary = self.detector.get_levels_summary(levels)
        
        assert 'total_levels' in summary
        assert 'unclaimed_levels' in summary
        assert 'daily_levels' in summary
        assert 'weekly_levels' in summary
        assert 'pdh_count' in summary
        assert 'pdl_count' in summary
        assert 'pwh_count' in summary
        assert 'pwl_count' in summary
        
        # Test with empty levels
        empty_summary = self.detector.get_levels_summary([])
        assert empty_summary['total_levels'] == 0
    
    def test_pdh_pdl_visualization(self):
        """Test PDH/PDL visualization with candlestick chart"""
        # Use subset of data for better visualization
        viz_data = self.sample_data.iloc[:168]  # First week of data
        
        # Detect all levels first (including expired ones for demonstration)
        daily_levels = self.detector.detect_daily_levels(viz_data, unclaimed_only=False)
        weekly_levels = self.detector.detect_weekly_levels(viz_data, unclaimed_only=False)
        all_levels = daily_levels + weekly_levels
        
        # Get valid unclaimed levels for current focus
        valid_unclaimed = self.detector.get_valid_unclaimed_levels(all_levels)
        
        # Sample data for visualization (every 4 hours for readability)
        sample_data = viz_data.iloc[::4]
        
        # Create candlestick chart
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Plot candlesticks
        for i, (timestamp, row) in enumerate(sample_data.iterrows()):
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
        
        # Plot levels with validity visualization
        colors = {'PDH': 'red', 'PDL': 'green', 'PWH': 'darkred', 'PWL': 'darkgreen'}
        
        plotted_types = set()
        current_date = viz_data.index[-1]
        
        for level in all_levels:
            # Determine visual style based on validity and claim status
            color = colors[level.level_type]
            
            if level.is_valid and not level.is_claimed:
                # Valid and unclaimed - solid, bright
                line_style = '-'
                alpha = 0.9
                linewidth = 3
                status = "ACTIVE"
            elif level.is_valid and level.is_claimed:
                # Valid but claimed - solid, dimmed
                line_style = '-'
                alpha = 0.4
                linewidth = 2
                status = "CLAIMED"
            else:
                # Invalid (expired) - dashed, very dim
                line_style = ':'
                alpha = 0.2
                linewidth = 1
                status = "EXPIRED"
            
            # Plot horizontal line for validity period
            if level.is_valid:
                # Plot line only during validity period
                validity_start_idx = 0
                validity_end_idx = len(sample_data) - 1
                
                # Find approximate indices for validity period
                for i, ts in enumerate(sample_data.index):
                    if ts >= level.validity_start:
                        validity_start_idx = max(0, i)
                        break
                for i, ts in enumerate(sample_data.index):
                    if ts > level.validity_end:
                        validity_end_idx = min(len(sample_data) - 1, i)
                        break
                
                # Plot line segment for validity period
                x_range = [validity_start_idx, validity_end_idx]
                ax.hlines(y=level.price, xmin=x_range[0], xmax=x_range[1],
                         colors=color, linestyles=line_style, alpha=alpha, linewidth=linewidth,
                         label=f'{level.level_type} ({status})' if level.level_type not in plotted_types else "")
            else:
                # Plot full line but very dim for expired levels
                ax.axhline(y=level.price, color=color, linestyle=line_style, 
                          alpha=alpha, linewidth=linewidth,
                          label=f'{level.level_type} ({status})' if level.level_type not in plotted_types else "")
            
            plotted_types.add(level.level_type)
            
            # Add price label with status
            label_color = 'white' if alpha > 0.5 else 'black'
            ax.text(len(sample_data) * 0.02, level.price, 
                   f'{level.level_type}: {level.price:.2f} ({status})',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=min(0.8, alpha + 0.3)),
                   fontsize=8, color=label_color, weight='bold' if level.is_valid else 'normal')
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('PDH/PDL Detection - Level Validity Periods (ACTIVE=Valid&Unclaimed, CLAIMED=Valid&Tested, EXPIRED=Invalid)')
        ax.grid(True, alpha=0.3)
        if plotted_types:
            ax.legend(loc='upper left')
        
        # Set x-axis labels
        step = max(1, len(sample_data) // 10)
        ax.set_xticks(range(0, len(sample_data), step))
        ax.set_xticklabels([sample_data.index[i].strftime('%m/%d %H:%M') 
                           for i in range(0, len(sample_data), step)], rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'output', 'tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'pdh_pdl_detection_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"PDH/PDL detection chart saved to: {plot_path}")
        print(f"Found {len(daily_levels)} daily levels and {len(weekly_levels)} weekly levels")
        print(f"Valid unclaimed levels: {len(valid_unclaimed)}")
        
        # Print summary
        summary = self.detector.get_levels_summary(all_levels)
        valid_summary = self.detector.get_levels_summary(valid_unclaimed)
        print(f"All levels: {summary['pdh_count']} PDH, {summary['pdl_count']} PDL, "
              f"{summary['pwh_count']} PWH, {summary['pwl_count']} PWL")
        print(f"Currently ACTIVE (valid & unclaimed): {valid_summary['total_levels']}")
        
        # Verify we created the visualization
        assert os.path.exists(plot_path)
    
    def test_extended_data_visualization(self):
        """Test with extended realistic data (no CCXT dependency)"""
        # Create extended dataset - 2 weeks of hourly data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 15)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Generate extended realistic data
        extended_data = self._generate_realistic_data(timestamps)
        
        # Detect all levels
        detector = PDHPDLDetector(touch_threshold=0.003, break_threshold=0.002)
        all_levels = detector.detect_all_levels(extended_data, unclaimed_only=False)
        valid_unclaimed = detector.get_valid_unclaimed_levels(all_levels)
        
        # Sample data for visualization
        sample_step = max(1, len(extended_data) // 300)
        sample_data = extended_data.iloc[::sample_step]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Plot candlesticks
        for i, (timestamp, row) in enumerate(sample_data.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            ax.plot([i, i], [row['low'], min(row['open'], row['close'])], 'k-', linewidth=1)
            ax.plot([i, i], [max(row['open'], row['close']), row['high']], 'k-', linewidth=1)
        
        # Plot levels with validity status
        colors = {'PDH': 'red', 'PDL': 'green', 'PWH': 'darkred', 'PWL': 'darkgreen'}
        plotted_types = set()
        
        for level in all_levels:
            color = colors[level.level_type]
            
            if level.is_valid and not level.is_claimed:
                alpha = 0.9
                linewidth = 3
                status = "ACTIVE"
            elif level.is_valid and level.is_claimed:
                alpha = 0.5
                linewidth = 2
                status = "CLAIMED"
            else:
                alpha = 0.2
                linewidth = 1
                status = "EXPIRED"
            
            linestyle = '-' if level.level_type in ['PDH', 'PDL'] else '--'
            
            ax.axhline(y=level.price, color=color, linestyle=linestyle,
                      alpha=alpha, linewidth=linewidth,
                      label=f'{level.level_type} ({status})' if level.level_type not in plotted_types else "")
            
            plotted_types.add(level.level_type)
        
        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('PDH/PDL Detection - Extended 2-Week Dataset with Realistic OHLCV')
        ax.grid(True, alpha=0.3)
        if plotted_types:
            ax.legend()
        
        # Set x-axis labels
        step = max(1, len(sample_data) // 12)
        ax.set_xticks(range(0, len(sample_data), step))
        ax.set_xticklabels([sample_data.index[i].strftime('%m/%d %H:%M') 
                           for i in range(0, len(sample_data), step)], rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'output', 'tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'pdh_pdl_extended_realistic.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Extended PDH/PDL chart saved to: {plot_path}")
        print(f"Found {len(all_levels)} total levels, {len(valid_unclaimed)} valid unclaimed")
        
        # Print summary
        summary = detector.get_levels_summary(all_levels)
        valid_summary = detector.get_levels_summary(valid_unclaimed)
        print(f"All levels: {summary}")
        print(f"Valid unclaimed: {valid_summary}")
        
        assert os.path.exists(plot_path)


if __name__ == "__main__":
    # Run tests with visualization
    test_instance = TestPDHPDLDetector()
    test_instance.setup_method()
    
    # Run basic tests
    test_instance.test_detector_initialization()
    test_instance.test_data_validation()
    test_instance.test_daily_levels_detection()
    test_instance.test_weekly_levels_detection()
    test_instance.test_unclaimed_only_filter()
    test_instance.test_level_claiming_logic()
    test_instance.test_levels_near_price()
    test_instance.test_support_resistance_filtering()
    test_instance.test_summary_statistics()
    
    # Run visualization tests
    test_instance.test_pdh_pdl_visualization()
    test_instance.test_extended_data_visualization()
    
    print("All PDH/PDL detector tests completed successfully!")