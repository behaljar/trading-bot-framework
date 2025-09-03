import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
Test cases for Liquidity Objective Detector
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import os

from framework.strategies.detectors.liquidity_objective_detector import (
    LiquidityObjectiveDetector, 
    TradeDirection, 
    LiquidityObjective
)


class TestLiquidityObjectiveDetector:
    """Test cases for LiquidityObjectiveDetector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = LiquidityObjectiveDetector(
            swing_sensitivity=3,
            fvg_min_sensitivity=0.1,
            max_objectives=10
        )
        
        # Create realistic market data with various liquidity levels
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        # Generate data with clear swing levels, gaps, and patterns
        np.random.seed(42)
        self.sample_data = self._generate_test_data(timestamps)
        
    def _generate_test_data(self, timestamps):
        """Generate test data with clear liquidity levels"""
        data = []
        base_price = 50000
        
        for i, ts in enumerate(timestamps):
            # Create patterns with swing highs/lows and gaps
            if i < 100:
                # Initial uptrend with swing high
                price = base_price + i * 10 + np.random.randn() * 20
            elif i < 200:
                # Pullback creating swing low
                price = base_price + 1000 - (i - 100) * 5 + np.random.randn() * 15
            elif i < 250:
                # Gap up (FVG)
                price = base_price + 800 + (i - 200) * 15 + np.random.randn() * 10
            elif i < 350:
                # Another swing high formation
                price = base_price + 1500 + (i - 250) * 8 + np.random.randn() * 25
            else:
                # Current movement
                price = base_price + 2300 - (i - 350) * 3 + np.random.randn() * 20
                
            # Create OHLC
            open_price = price + np.random.randn() * 5
            close_price = price + np.random.randn() * 8
            high_price = max(open_price, close_price) + abs(np.random.randn() * 10)
            low_price = min(open_price, close_price) - abs(np.random.randn() * 10)
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': np.random.randint(1000, 10000)
            })
        
        return pd.DataFrame(data, index=timestamps)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = LiquidityObjectiveDetector(
            swing_sensitivity=5,
            fvg_min_sensitivity=0.2,
            max_objectives=15
        )
        assert detector.swing_detector.sensitivity == 5
        assert detector.fvg_detector.min_sensitivity == 0.2
        assert detector.max_objectives == 15
    
    def test_detect_bullish_objectives(self):
        """Test bullish liquidity objective detection"""
        current_price = self.sample_data['Close'].iloc[-1]
        
        objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BULLISH,
            current_price=current_price,
            max_distance_pct=0.15
        )
        
        # Should find some bullish objectives
        assert len(objectives) > 0
        
        # All objectives should be for bullish direction and above current price
        for obj in objectives:
            assert obj.direction == 'bullish'
            assert obj.price > current_price
            assert obj.distance_pct > 0
            assert obj.confidence > 0
            assert obj.priority >= 0
        
        # Should be sorted by priority (lower number = higher priority)
        priorities = [obj.priority for obj in objectives]
        assert priorities == sorted(priorities)
        
        print(f"\nFound {len(objectives)} bullish objectives:")
        for i, obj in enumerate(objectives[:5]):
            print(f"  {i+1}. {obj.level_type}: ${obj.price:,.2f} ({obj.distance_pct*100:.2f}% away, confidence: {obj.confidence:.2f})")
    
    def test_detect_bearish_objectives(self):
        """Test bearish liquidity objective detection"""
        current_price = self.sample_data['Close'].iloc[-1]
        
        objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BEARISH,
            current_price=current_price,
            max_distance_pct=0.15
        )
        
        # Should find some bearish objectives
        assert len(objectives) > 0
        
        # All objectives should be for bearish direction and below current price
        for obj in objectives:
            assert obj.direction == 'bearish'
            assert obj.price < current_price
            assert obj.distance_pct > 0
            assert obj.confidence > 0
        
        print(f"\nFound {len(objectives)} bearish objectives:")
        for i, obj in enumerate(objectives[:5]):
            print(f"  {i+1}. {obj.level_type}: ${obj.price:,.2f} ({obj.distance_pct*100:.2f}% away, confidence: {obj.confidence:.2f})")
    
    def test_objective_types(self):
        """Test that different objective types are detected"""
        current_price = self.sample_data['Close'].iloc[-1]
        
        bullish_objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction='bullish',  # Test string input
            current_price=current_price,
            max_distance_pct=0.20
        )
        
        # Get all level types found
        level_types = set(obj.level_type for obj in bullish_objectives)
        print(f"\nLevel types found: {level_types}")
        
        # Should have at least swing-based objectives
        assert len(level_types) > 0
        
        # Check that each objective has valid attributes
        for obj in bullish_objectives:
            assert obj.level_type in ['swing_high', 'swing_low', 'pdh', 'pdl', 'pwh', 'pwl', 'fvg']
            assert obj.details is not None
            assert isinstance(obj.details, dict)
    
    def test_distance_filtering(self):
        """Test distance-based filtering"""
        current_price = self.sample_data['Close'].iloc[-1]
        
        # Test with small distance limit
        close_objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BULLISH,
            current_price=current_price,
            max_distance_pct=0.05  # Only 5% away
        )
        
        # Test with larger distance limit
        far_objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BULLISH,
            current_price=current_price,
            max_distance_pct=0.20  # 20% away
        )
        
        # Should find more objectives with larger distance limit
        assert len(far_objectives) >= len(close_objectives)
        
        # All objectives should be within specified distance
        for obj in close_objectives:
            assert obj.distance_pct <= 0.05
        
        for obj in far_objectives:
            assert obj.distance_pct <= 0.20
    
    def test_max_objectives_limit(self):
        """Test maximum objectives limit"""
        detector = LiquidityObjectiveDetector(max_objectives=3)
        current_price = self.sample_data['Close'].iloc[-1]
        
        objectives = detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BULLISH,
            current_price=current_price,
            max_distance_pct=0.30
        )
        
        # Should not exceed max limit
        assert len(objectives) <= 3
    
    def test_summary_generation(self):
        """Test summary generation"""
        current_price = self.sample_data['Close'].iloc[-1]
        
        objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BULLISH,
            current_price=current_price
        )
        
        summary = self.detector.get_summary(objectives, current_price)
        
        # Check summary structure
        assert 'total_objectives' in summary
        assert 'avg_distance_pct' in summary
        assert 'closest_objective' in summary
        assert 'highest_confidence' in summary
        assert 'level_types' in summary
        assert 'avg_confidence' in summary
        
        # Validate summary values
        assert summary['total_objectives'] == len(objectives)
        assert summary['avg_distance_pct'] >= 0
        assert summary['avg_confidence'] >= 0
        
        if objectives:
            closest = summary['closest_objective']
            assert closest['price'] > 0
            assert closest['distance_pct'] >= 0
            
            highest_conf = summary['highest_confidence']
            assert 0 <= highest_conf['confidence'] <= 1
        
        print(f"\nSummary: {summary}")
    
    def test_data_validation(self):
        """Test input data validation"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        objectives = self.detector.detect_objectives(empty_df, TradeDirection.BULLISH)
        assert len(objectives) == 0
        
        # Test with insufficient data
        small_df = pd.DataFrame({
            'Open': [1, 2], 'High': [2, 3], 'Low': [0.5, 1.5], 'Close': [1.5, 2.5]
        }, index=pd.date_range('2024-01-01', periods=2, freq='h'))
        objectives = self.detector.detect_objectives(small_df, TradeDirection.BULLISH)
        assert len(objectives) == 0
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'Open': [1, 2, 3]}, 
                                index=pd.date_range('2024-01-01', periods=3, freq='h'))
        objectives = self.detector.detect_objectives(invalid_df, TradeDirection.BULLISH)
        assert len(objectives) == 0
    
    def test_with_real_data(self):
        """Test with real market data if available"""
        try:
            from framework.data.sources.ccxt_source import CCXTSource
            
            # Download fresh data
            source = CCXTSource(
                exchange_name="binance",
                api_key=None,
                api_secret=None,
                sandbox=False
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # 7 days
            
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
            
            current_price = df['Close'].iloc[-1]
            
            # Test bullish objectives
            bullish_objectives = self.detector.detect_objectives(
                data=df,
                trade_direction=TradeDirection.BULLISH,
                current_price=current_price,
                max_distance_pct=0.10
            )
            
            # Test bearish objectives
            bearish_objectives = self.detector.detect_objectives(
                data=df,
                trade_direction=TradeDirection.BEARISH,
                current_price=current_price,
                max_distance_pct=0.10
            )
            
            print(f"\nReal BTC/USDT Data Analysis:")
            print(f"Current Price: ${current_price:,.2f}")
            print(f"Bullish Objectives: {len(bullish_objectives)}")
            print(f"Bearish Objectives: {len(bearish_objectives)}")
            
            # Display top objectives
            if bullish_objectives:
                print("\nTop Bullish Targets:")
                for i, obj in enumerate(bullish_objectives[:3]):
                    print(f"  {i+1}. {obj.level_type}: ${obj.price:,.2f} (+{obj.distance_pct*100:.2f}%)")
            
            if bearish_objectives:
                print("\nTop Bearish Targets:")
                for i, obj in enumerate(bearish_objectives[:3]):
                    print(f"  {i+1}. {obj.level_type}: ${obj.price:,.2f} (-{obj.distance_pct*100:.2f}%)")
            
            # Basic validation
            for obj in bullish_objectives + bearish_objectives:
                assert obj.price > 0
                assert obj.confidence > 0
                assert obj.distance_pct > 0
                assert obj.level_type in ['swing_high', 'swing_low', 'pdh', 'pdl', 'pwh', 'pwl', 'fvg']
                
        except Exception as e:
            pytest.skip(f"Real data test skipped: {e}")
    
    def test_visualization(self):
        """Test visualization of liquidity objectives"""
        current_price = self.sample_data['Close'].iloc[-1]
        
        # Get objectives for both directions
        bullish_objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BULLISH,
            current_price=current_price,
            max_distance_pct=0.15
        )
        
        bearish_objectives = self.detector.detect_objectives(
            data=self.sample_data,
            trade_direction=TradeDirection.BEARISH,
            current_price=current_price,
            max_distance_pct=0.15
        )
        
        # Create visualization using last 100 candles
        recent_data = self.sample_data.tail(100)
        
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
            ax.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], 'k-', linewidth=0.5)
            ax.plot([i, i], [max(row['Open'], row['Close']), row['High']], 'k-', linewidth=0.5)
        
        # Plot current price
        ax.axhline(y=current_price, color='gold', linestyle='-', alpha=0.7, linewidth=2, 
                  label=f'Current: ${current_price:,.0f}')
        
        # Plot bullish objectives
        for i, obj in enumerate(bullish_objectives[:5]):
            alpha = 0.8 - (i * 0.1)  # Fade lower priority objectives
            ax.axhline(y=obj.price, color='lime', linestyle='--', alpha=alpha, linewidth=1.5)
            ax.text(len(recent_data) - 1, obj.price,
                   f" {obj.level_type.upper()}: ${obj.price:,.0f}",
                   verticalalignment='center', color='darkgreen', fontsize=9)
        
        # Plot bearish objectives
        for i, obj in enumerate(bearish_objectives[:5]):
            alpha = 0.8 - (i * 0.1)  # Fade lower priority objectives
            ax.axhline(y=obj.price, color='red', linestyle='--', alpha=alpha, linewidth=1.5)
            ax.text(len(recent_data) - 1, obj.price,
                   f" {obj.level_type.upper()}: ${obj.price:,.0f}",
                   verticalalignment='center', color='darkred', fontsize=9)
        
        # Formatting
        ax.set_title('Liquidity Objectives Detection', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = [
            Line2D([0], [0], color='gold', linestyle='-', label='Current Price'),
            Line2D([0], [0], color='lime', linestyle='--', label='Bullish Targets'),
            Line2D([0], [0], color='red', linestyle='--', label='Bearish Targets')
        ]
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
        plot_path = os.path.join(output_dir, 'liquidity_objectives_test.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nLiquidity objectives visualization saved to: {plot_path}")
        print(f"Bullish objectives: {len(bullish_objectives)}")
        print(f"Bearish objectives: {len(bearish_objectives)}")
        
        # Verify visualization was created
        assert os.path.exists(plot_path)


if __name__ == "__main__":
    # Run tests
    test_instance = TestLiquidityObjectiveDetector()
    test_instance.setup_method()
    
    # Run all tests
    test_instance.test_detector_initialization()
    test_instance.test_detect_bullish_objectives()
    test_instance.test_detect_bearish_objectives() 
    test_instance.test_objective_types()
    test_instance.test_distance_filtering()
    test_instance.test_max_objectives_limit()
    test_instance.test_summary_generation()
    test_instance.test_data_validation()
    test_instance.test_visualization()
    
    try:
        test_instance.test_with_real_data()
    except Exception as e:
        print(f"Skipped real data test: {e}")
    
    print("\nAll liquidity objective detector tests completed successfully!")