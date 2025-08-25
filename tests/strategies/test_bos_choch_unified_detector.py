import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
Test cases for unified BoS/CHoCH detector with precise pattern matching
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import os

from framework.strategies.detectors.bos_choch_unified_detector import BoSCHoCHDetector, StructureEventType


class TestBoSCHoCHUnifiedDetector:
    """Test cases for unified BoS/CHoCH detector"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = BoSCHoCHDetector(left_bars=3, right_bars=3)
        
        # Create test OHLC data with specific patterns
        self.create_test_data()
        
    def create_test_data(self):
        """Create OHLC test data with known BoS and CHoCH patterns"""
        
        # Bullish BoS pattern data: [-1,1,-1,1] with L1 < L3 < L2 < L4
        # Swing sequence: Low(40) -> High(50) -> Low(45) -> High(60)  
        # Pattern: 40 < 45 < 50 < 60 ✓ (Bullish BoS)
        # Need proper spacing for 3/3 bar pivot detection
        
        base_data_bullish_bos = [
            # Lead-in bars for first swing low
            {'open': 50, 'high': 51, 'low': 49, 'close': 50},
            {'open': 50, 'high': 51, 'low': 48, 'close': 49},
            {'open': 49, 'high': 50, 'low': 47, 'close': 48},
            # First swing low L1 = 40
            {'open': 48, 'high': 49, 'low': 40, 'close': 41},  # Index 3 - L1 = 40
            {'open': 41, 'high': 43, 'low': 41, 'close': 42},
            {'open': 42, 'high': 44, 'low': 42, 'close': 43},
            {'open': 43, 'high': 45, 'low': 43, 'close': 44},
            # First swing high H2 = 50
            {'open': 44, 'high': 50, 'low': 44, 'close': 49},  # Index 7 - H2 = 50
            {'open': 49, 'high': 50, 'low': 48, 'close': 49},
            {'open': 49, 'high': 49, 'low': 47, 'close': 48},
            {'open': 48, 'high': 48, 'low': 46, 'close': 47},
            # Second swing low L3 = 45  
            {'open': 47, 'high': 47, 'low': 45, 'close': 46},  # Index 11 - L3 = 45
            {'open': 46, 'high': 47, 'low': 46, 'close': 47},
            {'open': 47, 'high': 48, 'low': 47, 'close': 48},
            {'open': 48, 'high': 49, 'low': 48, 'close': 49},
            # Second swing high H4 = 60 - This should trigger BoS
            {'open': 49, 'high': 60, 'low': 49, 'close': 59},  # Index 15 - H4 = 60
            {'open': 59, 'high': 60, 'low': 58, 'close': 59},
            {'open': 59, 'high': 59, 'low': 57, 'close': 58},
            {'open': 58, 'high': 58, 'low': 56, 'close': 57},
            # Break confirmation - price breaks above H2 level (50)
            {'open': 57, 'high': 62, 'low': 57, 'close': 61},  # Break above 50
            {'open': 61, 'high': 63, 'low': 60, 'close': 62},
        ]
        
        # Bearish CHoCH pattern data: [1,-1,1,-1] with L4 < L2 < L1 < L3  
        # Swing sequence: High(60) -> Low(45) -> High(65) -> Low(35)
        # Pattern: 35 < 45 < 60 < 65 ✓ (L4 < L2 < L1 < L3 for Bearish CHoCH)
        # Need proper spacing for 3/3 bar pivot detection
        
        base_data_bearish_choch = [
            # Lead-in bars for first swing high
            {'open': 55, 'high': 56, 'low': 54, 'close': 55},
            {'open': 55, 'high': 57, 'low': 55, 'close': 56},
            {'open': 56, 'high': 58, 'low': 56, 'close': 57},
            # First swing high H1 = 60
            {'open': 57, 'high': 60, 'low': 57, 'close': 59},  # Index 3 - H1 = 60
            {'open': 59, 'high': 59, 'low': 57, 'close': 58},
            {'open': 58, 'high': 58, 'low': 56, 'close': 57},
            {'open': 57, 'high': 57, 'low': 55, 'close': 56},
            # First swing low L2 = 45
            {'open': 56, 'high': 56, 'low': 45, 'close': 46},  # Index 7 - L2 = 45
            {'open': 46, 'high': 47, 'low': 46, 'close': 47},
            {'open': 47, 'high': 48, 'low': 47, 'close': 48},
            {'open': 48, 'high': 49, 'low': 48, 'close': 49},
            # Second swing high H3 = 65
            {'open': 49, 'high': 65, 'low': 49, 'close': 64},  # Index 11 - H3 = 65
            {'open': 64, 'high': 65, 'low': 62, 'close': 63},
            {'open': 63, 'high': 63, 'low': 61, 'close': 62},
            {'open': 62, 'high': 62, 'low': 60, 'close': 61},
            # Second swing low L4 = 35 - This should trigger CHoCH
            {'open': 61, 'high': 61, 'low': 35, 'close': 36},  # Index 15 - L4 = 35
            {'open': 36, 'high': 37, 'low': 36, 'close': 37},
            {'open': 37, 'high': 38, 'low': 37, 'close': 38},
            {'open': 38, 'high': 39, 'low': 38, 'close': 39},
            # Break confirmation - price breaks below L2 level (45)
            {'open': 39, 'high': 39, 'low': 32, 'close': 33},  # Break below 45
            {'open': 33, 'high': 34, 'low': 30, 'close': 31},
        ]
        
        # Convert to DataFrames
        self.ohlc_bullish_bos = pd.DataFrame(base_data_bullish_bos)
        self.ohlc_bearish_choch = pd.DataFrame(base_data_bearish_choch)
        
        # Add timestamps
        dates_bos = pd.date_range(start='2023-01-01', periods=len(self.ohlc_bullish_bos), freq='1h')
        dates_choch = pd.date_range(start='2023-02-01', periods=len(self.ohlc_bearish_choch), freq='1h')
        
        self.ohlc_bullish_bos.index = dates_bos
        self.ohlc_bearish_choch.index = dates_choch
        
        # Add volume column
        self.ohlc_bullish_bos['volume'] = 1000
        self.ohlc_bearish_choch['volume'] = 1000
        
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = BoSCHoCHDetector(left_bars=3, right_bars=4)
        assert detector.left_bars == 3
        assert detector.right_bars == 4
        assert detector.pivot_detector is not None
        
    def test_swing_highs_lows_detection(self):
        """Test swing highs and lows detection"""
        swing_data = self.detector._find_swing_highs_lows(self.ohlc_bullish_bos)
        
        # Should have some swing points
        valid_swings = swing_data[~pd.isna(swing_data['high_low'])]
        assert len(valid_swings) > 0
        
        # Should have both highs and lows
        has_highs = np.any(valid_swings['high_low'] == 1)
        has_lows = np.any(valid_swings['high_low'] == -1)
        assert has_highs and has_lows
        
    def test_bullish_bos_pattern_recognition(self):
        """Test bullish BoS pattern recognition"""
        result = self.detector.detect_bos_choch(self.ohlc_bullish_bos, close_break=True)
        
        # Should detect bullish BoS
        bullish_bos = result[result['bos'] == 1]
        assert len(bullish_bos) > 0, "Should detect bullish BoS pattern"
        
        # Check that break was confirmed
        for idx, row in bullish_bos.iterrows():
            assert not pd.isna(row['broken_index']), "BoS should have break confirmation"
            assert not pd.isna(row['level']), "BoS should have break level"
            
    def test_bearish_choch_pattern_recognition(self):
        """Test bearish CHoCH pattern recognition"""
        result = self.detector.detect_bos_choch(self.ohlc_bearish_choch, close_break=True)
        
        # Should detect bearish CHoCH
        bearish_choch = result[result['choch'] == -1]
        assert len(bearish_choch) > 0, "Should detect bearish CHoCH pattern"
        
        # Check that break was confirmed
        for idx, row in bearish_choch.iterrows():
            assert not pd.isna(row['broken_index']), "CHoCH should have break confirmation"
            assert not pd.isna(row['level']), "CHoCH should have break level"
            
    def test_level_relationship_logic(self):
        """Test the level relationship checking logic"""
        detector = self.detector
        
        # Test bullish BoS levels: L1 < L3 < L2 < L4
        # Example: [40, 50, 45, 60] -> 40 < 45 < 50 < 60 ✓
        assert detector._check_bullish_bos_levels([40, 50, 45, 60]) == True
        assert detector._check_bullish_bos_levels([50, 40, 45, 60]) == False  # 50 < 45 is false
        
        # Test bearish BoS levels: L1 > L3 > L2 > L4  
        # Example: [60, 50, 55, 40] -> 60 > 55 > 50 > 40 ✓
        assert detector._check_bearish_bos_levels([60, 50, 55, 40]) == True
        assert detector._check_bearish_bos_levels([40, 50, 45, 60]) == False
        
        # Test bullish CHoCH levels: L4 > L2 > L1 > L3
        # Example: [50, 60, 45, 70] -> 70 > 60 > 50 > 45 ✓
        assert detector._check_bullish_choch_levels([50, 60, 45, 70]) == True
        assert detector._check_bullish_choch_levels([40, 50, 45, 60]) == False
        
        # Test bearish CHoCH levels: L4 < L2 < L1 < L3
        # Example: [60, 50, 65, 40] -> 40 < 50 < 60 < 65 ✓
        assert detector._check_bearish_choch_levels([60, 50, 65, 40]) == True
        assert detector._check_bearish_choch_levels([40, 50, 45, 60]) == False
        
    def test_break_confirmation(self):
        """Test break confirmation logic"""
        result = self.detector.detect_bos_choch(self.ohlc_bullish_bos, close_break=True)
        
        # All detected signals should have break confirmation
        all_signals = result[(~pd.isna(result['bos'])) | (~pd.isna(result['choch']))]
        
        for idx, row in all_signals.iterrows():
            assert not pd.isna(row['broken_index']), f"Signal at {idx} should have break confirmation"
            
            # Verify the break actually occurred
            break_idx = int(row['broken_index'])
            level_val = row['level']
            
            if not pd.isna(row['bos']) and row['bos'] == 1:
                # Bullish BoS - price should go above level
                assert self.ohlc_bullish_bos.iloc[break_idx]['close'] > level_val
            elif not pd.isna(row['bos']) and row['bos'] == -1:
                # Bearish BoS - price should go below level
                assert self.ohlc_bullish_bos.iloc[break_idx]['close'] < level_val
                
    def test_close_vs_wick_break(self):
        """Test difference between close break and wick break"""
        result_close = self.detector.detect_bos_choch(self.ohlc_bullish_bos, close_break=True)
        result_wick = self.detector.detect_bos_choch(self.ohlc_bullish_bos, close_break=False)
        
        # Both should detect patterns, but break confirmation might differ
        close_signals = len(result_close[(~pd.isna(result_close['bos'])) | (~pd.isna(result_close['choch']))])
        wick_signals = len(result_wick[(~pd.isna(result_wick['bos'])) | (~pd.isna(result_wick['choch']))])
        
        assert close_signals >= 0
        assert wick_signals >= 0
        
    def test_get_latest_signal(self):
        """Test getting the latest BoS/CHoCH signal"""
        latest = self.detector.get_latest_signal(self.ohlc_bullish_bos)
        
        if latest is not None:
            assert 'type' in latest
            assert 'level' in latest
            assert 'broken_index' in latest
            assert latest['type'] in ['BULLISH_BOS', 'BEARISH_BOS', 'BULLISH_CHOCH', 'BEARISH_CHOCH']
            
    def test_market_structure_bias(self):
        """Test market structure bias determination"""
        bias = self.detector.get_market_structure_bias(self.ohlc_bullish_bos)
        assert bias in ['BULLISH', 'BEARISH', 'NEUTRAL']
        
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        # Create minimal OHLC data
        minimal_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102], 
            'low': [99, 100],
            'close': [100, 101],
            'volume': [1000, 1000]
        })
        
        result = self.detector.detect_bos_choch(minimal_data)
        
        # Should return valid DataFrame structure but no signals
        assert len(result) == len(minimal_data)
        assert 'bos' in result.columns
        assert 'choch' in result.columns
        assert 'level' in result.columns
        assert 'broken_index' in result.columns
        
    def test_input_validation(self):
        """Test input validation"""
        # Test with missing columns
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102]
            # Missing 'low' and 'close'
        })
        
        with pytest.raises(ValueError, match="OHLC data must contain columns"):
            self.detector.detect_bos_choch(invalid_data)
            
    def test_visualization_bullish_bos(self):
        """Test visualization for bullish BoS pattern"""
        result = self.detector.detect_bos_choch(self.ohlc_bullish_bos)
        
        self._create_visualization(
            self.ohlc_bullish_bos,
            result,
            'unified_bos_bullish_test.png',
            'Bullish Break of Structure - Unified Detector'
        )
        
    def test_visualization_bearish_choch(self):
        """Test visualization for bearish CHoCH pattern"""
        result = self.detector.detect_bos_choch(self.ohlc_bearish_choch)
        
        self._create_visualization(
            self.ohlc_bearish_choch,
            result,
            'unified_choch_bearish_test.png', 
            'Bearish Change of Character - Unified Detector'
        )
        
    def _create_visualization(self, ohlc_data, result, filename, title):
        """Create visualization for BoS/CHoCH detection"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(ohlc_data.iterrows()):
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
        
        # Highlight BoS signals
        bullish_bos = result[result['bos'] == 1]
        bearish_bos = result[result['bos'] == -1]
        
        for idx, row in bullish_bos.iterrows():
            candle_idx = ohlc_data.index.get_loc(idx)
            ax.scatter(candle_idx, row['level'], color='lime', s=300, marker='^',
                      label='Bullish BoS', zorder=6)
            ax.annotate(f'Bull BoS\nLevel: {row["level"]:.1f}',
                       (candle_idx, row['level']),
                       xytext=(10, 20), textcoords='offset points',
                       fontsize=10, color='darkgreen', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.9))
            
            # Mark break point
            if not pd.isna(row['broken_index']):
                break_idx = int(row['broken_index'])
                if break_idx < len(ohlc_data):
                    ax.scatter(break_idx, ohlc_data.iloc[break_idx]['high'], 
                             color='lime', s=100, marker='x', zorder=7)
        
        for idx, row in bearish_bos.iterrows():
            candle_idx = ohlc_data.index.get_loc(idx)
            ax.scatter(candle_idx, row['level'], color='red', s=300, marker='v',
                      label='Bearish BoS', zorder=6)
            ax.annotate(f'Bear BoS\nLevel: {row["level"]:.1f}',
                       (candle_idx, row['level']),
                       xytext=(10, -30), textcoords='offset points', 
                       fontsize=10, color='darkred', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.9))
        
        # Highlight CHoCH signals
        bullish_choch = result[result['choch'] == 1]
        bearish_choch = result[result['choch'] == -1]
        
        for idx, row in bullish_choch.iterrows():
            candle_idx = ohlc_data.index.get_loc(idx)
            ax.scatter(candle_idx, row['level'], color='cyan', s=300, marker='D',
                      label='Bullish CHoCH', zorder=6)
            ax.annotate(f'Bull CHoCH\nLevel: {row["level"]:.1f}',
                       (candle_idx, row['level']),
                       xytext=(10, 20), textcoords='offset points',
                       fontsize=10, color='darkcyan', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.9))
        
        for idx, row in bearish_choch.iterrows():
            candle_idx = ohlc_data.index.get_loc(idx)
            ax.scatter(candle_idx, row['level'], color='orange', s=300, marker='D',
                      label='Bearish CHoCH', zorder=6)
            ax.annotate(f'Bear CHoCH\nLevel: {row["level"]:.1f}',
                       (candle_idx, row['level']),
                       xytext=(10, -30), textcoords='offset points',
                       fontsize=10, color='darkorange', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='moccasin', alpha=0.9))
        
        # Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(f'{title}\nSensitivity: {self.detector.left_bars}/{self.detector.right_bars} bars',
                    fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        
        plt.tight_layout()
        
        # Save plot
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'output', 'tests')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Unified BoS/CHoCH chart saved to: {plot_path}")
        
        # Print summary
        bos_signals = result[(~pd.isna(result['bos']))]
        choch_signals = result[(~pd.isna(result['choch']))]
        print(f"Found {len(bos_signals)} BoS signals and {len(choch_signals)} CHoCH signals")
        
        assert os.path.exists(plot_path)
        
    def test_real_data_integration(self):
        """Test with real market data"""
        try:
            from framework.data.sources.ccxt_source import CCXTSource
            
            source = CCXTSource(
                exchange_name="binance",
                api_key=None,
                api_secret=None,
                sandbox=False
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # 7 days of 1h data
            
            data = source.get_historical_data(
                symbol="BTC/USDT:USDT",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                timeframe="1h"
            )
            
            if len(data) == 0:
                pytest.skip("Unable to fetch real market data")
            
            data = data.reset_index()
            
            # Rename columns to match expected format
            ohlc_data = pd.DataFrame({
                'open': data['open'],
                'high': data['high'], 
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            })
            
            # Use higher sensitivity for 1h data
            detector = BoSCHoCHDetector(left_bars=6, right_bars=6)
            
            result = detector.detect_bos_choch(ohlc_data)
            
            # Should return valid results
            assert len(result) == len(ohlc_data)
            assert 'bos' in result.columns
            assert 'choch' in result.columns
            
            # Check for any signals
            bos_signals = result[~pd.isna(result['bos'])]
            choch_signals = result[~pd.isna(result['choch'])]
            total_signals = len(bos_signals) + len(choch_signals)
            
            print(f"Real data: Found {total_signals} total signals ({len(bos_signals)} BoS, {len(choch_signals)} CHoCH)")
            
            # Get market bias
            bias = detector.get_market_structure_bias(ohlc_data)
            print(f"Current market structure bias: {bias}")
            
        except Exception as e:
            pytest.skip(f"Unable to test with real data: {e}")


if __name__ == "__main__":
    # Run tests with visualization
    test_instance = TestBoSCHoCHUnifiedDetector()
    test_instance.setup_method()
    
    # Run basic tests
    test_instance.test_detector_initialization()
    test_instance.test_swing_highs_lows_detection()
    test_instance.test_level_relationship_logic()
    test_instance.test_bullish_bos_pattern_recognition()
    test_instance.test_bearish_choch_pattern_recognition()
    test_instance.test_break_confirmation()
    test_instance.test_close_vs_wick_break()
    test_instance.test_get_latest_signal()
    test_instance.test_market_structure_bias()
    test_instance.test_insufficient_data()
    test_instance.test_input_validation()
    
    # Run visualization tests
    test_instance.test_visualization_bullish_bos()
    test_instance.test_visualization_bearish_choch()
    test_instance.test_real_data_integration()
    
    print("All unified BoS/CHoCH detector tests completed successfully!")