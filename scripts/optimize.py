"""
Enhanced optimization runner supporting multiple data sources and strategies
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from config.settings import load_config
from data import YahooFinanceSource, CCXTSource, CSVDataSource
from strategies.trend_following import SMAStrategy
from strategies.mean_reversion import RSIStrategy
from utils.logger import setup_logger

# Global variables for optimization wrapper
_strategy_class = None
_data = None
_optimization_params = None

class OptimizationWrapper(Strategy):
    def init(self):
        super().init()
        # Get strategy parameters from optimization
        strategy_params = {}
        for param_name in _optimization_params.keys():
            if hasattr(self, param_name):
                strategy_params[param_name] = getattr(self, param_name)
        
        # Create strategy instance with optimized parameters
        self.strategy_instance = _strategy_class(strategy_params)
        
        # Convert backtesting data object to DataFrame for our strategy
        df_data = pd.DataFrame({
            'Open': self.data.Open,
            'High': self.data.High, 
            'Low': self.data.Low,
            'Close': self.data.Close,
            'Volume': self.data.Volume
        })
        df_data.index = self.data.index
        
        # Add any additional columns from the original data
        for col in _data.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_data[col] = _data[col]
        
        self.signals = self.strategy_instance.generate_signals(df_data)

    def next(self):
        super().next()
        current_idx = len(self.data.Close) - 1
        if current_idx >= len(self.signals):
            return

        # Handle both Series and DataFrame formats
        if isinstance(self.signals, pd.Series):
            signal = self.signals.iloc[current_idx]
            stop_loss = None
            take_profit = None
        else:
            signal = self.signals.iloc[current_idx]['signal']
            stop_loss = self.signals.iloc[current_idx].get('stop_loss', None)
            take_profit = self.signals.iloc[current_idx].get('take_profit', None)
        
        current_price = self.data.Close[-1]
        
        if current_idx < len(self.signals):
            # Close any existing position first if signal changes direction
            if self.position and (
                (signal == 1 and self.position.is_short) or 
                (signal == -1 and self.position.is_long)
            ):
                self.position.close()
            
            if signal == 1:  # Buy signal (long)
                if not self.position:
                    current_account_value = self.equity
                    risk_amount = current_account_value * 0.01  # 1% risk
                    
                    if stop_loss and stop_loss > 0:
                        risk_per_share = current_price - stop_loss
                        if risk_per_share > 0:
                            units = max(1, int(risk_amount / risk_per_share))
                            self.buy(size=units, sl=stop_loss, tp=take_profit)
                    else:
                        # No stop loss - use fixed sizing
                        target_value = current_account_value * 0.01
                        units = max(1, int(target_value / current_price))
                        self.buy(size=units)

            elif signal == -1:  # Sell signal (short)
                if not self.position:
                    current_account_value = self.equity
                    risk_amount = current_account_value * 0.01  # 1% risk
                    
                    if stop_loss and stop_loss > 0:
                        risk_per_share = stop_loss - current_price
                        if risk_per_share > 0:
                            units = max(1, int(risk_amount / risk_per_share))
                            self.sell(size=units, sl=stop_loss, tp=take_profit)
                    else:
                        # No stop loss - use fixed sizing
                        target_value = current_account_value * 0.01
                        units = max(1, int(target_value / current_price))
                        self.sell(size=units)

def optimize_strategy(strategy_name: str, symbol: str, start_date: str, end_date: str, 
                     optimization_params: dict, data_source: str = None, timeframe: str = None,
                     metric: str = 'Return [%]', maximize: bool = True, max_tries: int = None):
    """
    Optimize strategy parameters using Backtesting.py optimize method
    
    Args:
        strategy_name: Name of strategy to optimize
        symbol: Symbol to optimize on  
        start_date: Start date for optimization
        end_date: End date for optimization
        optimization_params: Dict of parameters to optimize with ranges
        data_source: Data source override
        timeframe: Timeframe override
        metric: Metric to optimize for
        maximize: Whether to maximize or minimize the metric
        max_tries: Maximum optimization attempts
    """
    
    config = load_config()
    logger = setup_logger()

    # Use specified data source or config default
    if data_source:
        config.data_source = data_source
    
    # Use specified timeframe or config default
    if timeframe:
        config.timeframe = timeframe

    logger.info(f"Optimizing {strategy_name} on {symbol} using {config.data_source} data ({config.timeframe})")
    logger.info(f"Optimization metric: {metric} ({'maximize' if maximize else 'minimize'})")
    logger.info(f"Parameters to optimize: {optimization_params}")

    # Initialize data source
    if config.data_source == "yahoo":
        source = YahooFinanceSource()
    elif config.data_source == "ccxt":
        source = CCXTSource(
            exchange_name=config.exchange_name,
            api_key=config.api_key,
            api_secret=config.api_secret,
            sandbox=config.use_sandbox
        )
    elif config.data_source == "csv":
        source = CSVDataSource(data_directory=config.csv_data_directory)
    elif config.data_source == "csv_processed":
        source = CSVDataSource(data_directory=config.csv_data_directory, use_processed=True)
    else:
        logger.error(f"Unknown data source: {config.data_source}")
        return None

    # Load data
    data = source.get_historical_data(symbol, start_date, end_date, config.timeframe)
        
    if data.empty:
        logger.error(f"Failed to load data for {symbol} from {config.data_source}")
        return None
    
    # Convert to microBTC for fractional trading if dealing with BTC data
    if 'BTC' in symbol.upper():
        logger.info("Converting prices to microBTC for fractional trading")
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col] / 1e6
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'] * 1e6

    # Log available feature columns
    feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    if feature_columns:
        logger.info(f"Found additional feature columns: {', '.join(feature_columns)}")

    # Select strategy and set global variables for optimization wrapper
    global _strategy_class, _data, _optimization_params
    
    if strategy_name.lower() in ["sma", "sma_crossover", "trend_following"]: 
        _strategy_class = SMAStrategy
    elif strategy_name.lower() in ["rsi", "rsi_mean_reversion"]:
        _strategy_class = RSIStrategy
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None

    _data = data
    _optimization_params = optimization_params

    # Add optimization parameters as class attributes
    for param_name in optimization_params.keys():
        setattr(OptimizationWrapper, param_name, None)

    # Run backtest with optimization
    bt = Backtest(
        data,
        OptimizationWrapper,
        cash=config.initial_capital,
        commission=config.commission,
        trade_on_close=True,
        hedging=True,
        exclusive_orders=True,
        spread=config.slippage,
        margin=0.01
    )

    logger.info("Starting optimization...")
    
    # Convert optimization parameters to the format expected by backtesting library
    opt_kwargs = {}
    for param_name, param_range in optimization_params.items():
        if isinstance(param_range, (list, tuple)) and len(param_range) >= 2:
            if len(param_range) == 2:
                # Range format: (start, end) - create range with reasonable step
                start, end = param_range
                if isinstance(start, int) and isinstance(end, int):
                    opt_kwargs[param_name] = list(range(start, end + 1, max(1, (end - start) // 10)))
                else:
                    # For float ranges, create reasonable steps
                    step = (end - start) / 10
                    opt_kwargs[param_name] = [round(x, 2) for x in np.arange(start, end + step/2, step)]
            else:
                # List of specific values
                opt_kwargs[param_name] = list(param_range)
        else:
            logger.warning(f"Invalid range for parameter {param_name}: {param_range}")
            continue
    
    try:
        optimization_result = bt.optimize(
            **opt_kwargs,
            maximize=metric if maximize else f'-{metric}',
            max_tries=max_tries,
            random_state=42,
            return_heatmap=True
        )
        
        logger.info("Optimization completed!")
        
        # Extract heatmap from optimization result
        if isinstance(optimization_result, tuple) and len(optimization_result) == 2:
            best_result, heatmap_series = optimization_result
            optimization_result = best_result  # Use best result for other operations
        else:
            heatmap_series = None
        
        # Create output filename with strategy, symbol, and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "output"
        output_dir.mkdir(exist_ok=True)
        
        output_filename = f"optimize_{strategy_name}_{symbol}_{timestamp}"
        
        # Save optimization results
        best_params = optimization_result._strategy.__dict__
        best_params = {k: v for k, v in best_params.items() if k in optimization_params.keys()}
        
        results_dict = {
            'best_parameters': best_params,
            'best_result': dict(optimization_result),
            'optimization_metric': metric,
            'maximize': maximize,
            'metadata': {
                'strategy': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'data_source': config.data_source,
                'timeframe': config.timeframe,
                'data_points': len(data),
                'date_range_start': data.index[0].isoformat(),
                'date_range_end': data.index[-1].isoformat(),
                'timestamp': datetime.now().isoformat(),
                'optimization_params': optimization_params
            }
        }
        
        # Clean results_dict for JSON serialization
        for key, value in results_dict['best_result'].items():
            if hasattr(value, 'isoformat'):  # datetime objects
                results_dict['best_result'][key] = value.isoformat()
            elif not isinstance(value, (int, float, str, bool, type(None))):
                results_dict['best_result'][key] = str(value)
        
        json_file = output_dir / f"{output_filename}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to: {json_file}")
        
        # Generate and save heatmaps using backtesting.lib
        if heatmap_series is not None:
            try:
                logger.info("Found heatmap data, generating heatmaps...")
                generate_heatmaps(heatmap_series, metric, output_dir, output_filename)
            except Exception as e:
                logger.warning(f"Error generating heatmaps: {e}")
        else:
            logger.info("No heatmap data available - try running with more parameter combinations or return_heatmap=True")
        
        # Print best results
        logger.info("=== OPTIMIZATION RESULTS ===")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {metric}: {optimization_result[metric]:.2f}")
        
        for key, value in optimization_result.items():
            if isinstance(value, (int, float)):
                if 'Return' in key or 'Drawdown' in key or 'Rate' in key:
                    logger.info(f"{key}: {value:.2f}%" if '%' not in key else f"{key}: {value:.2f}")
                elif 'Ratio' in key or 'Factor' in key:
                    logger.info(f"{key}: {value:.3f}")
                else:
                    logger.info(f"{key}: {value}")
        
        logger.info(f"\nAll optimization outputs saved to: {output_dir}")
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None

def _move_stray_html_files(current_dir, output_dir, expected_filename, logger):
    """Move any stray HTML files created by bokeh to the proper output directory"""
    import shutil
    from pathlib import Path
    
    # Common stray filenames that bokeh might create
    stray_patterns = ['optimize.html', 'heatmap.html', 'plot.html', 'bokeh_plot.html']
    
    for pattern in stray_patterns:
        stray_file = current_dir / pattern
        if stray_file.exists():
            try:
                # Move to output directory with proper naming
                target_file = output_dir / f"{expected_filename}.html"
                shutil.move(str(stray_file), str(target_file))
                logger.info(f"Moved stray file {stray_file} to {target_file}")
            except Exception as e:
                logger.warning(f"Could not move stray file {stray_file}: {e}")

def generate_heatmaps(heatmap_series, metric, output_dir, output_filename):
    """Generate and save optimization heatmaps using backtesting.lib"""
    import os
    import shutil
    from pathlib import Path
    
    logger = setup_logger()
    
    try:
        # Save heatmap data to CSV for reference
        heatmap_csv = output_dir / f"{output_filename}_heatmap_data.csv"
        heatmap_series.to_csv(heatmap_csv)
        logger.info(f"Heatmap data saved to: {heatmap_csv}")
        
        # Log heatmap structure for debugging
        logger.info(f"Heatmap series shape: {heatmap_series.shape}")
        logger.info(f"Heatmap index levels: {heatmap_series.index.nlevels}")
        logger.info(f"Heatmap index names: {heatmap_series.index.names}")
        
        # Generate heatmaps using backtesting.lib plot_heatmaps
        # plot_heatmaps() creates interactive bokeh plots by default
        
        # Get current working directory to check for stray files
        current_dir = Path.cwd()
        
        # Default heatmap (max aggregation)
        try:
            from bokeh.plotting import output_file, save, reset_output
            
            # Reset any previous bokeh output settings
            reset_output()
            
            heatmap_html_file = output_dir / f"{output_filename}_heatmap_interactive.html"
            output_file(str(heatmap_html_file))
            
            fig = plot_heatmaps(heatmap_series)
            if fig:
                save(fig)
                logger.info(f"Interactive heatmap saved to: {heatmap_html_file}")
                
                # Check for and move any stray HTML files created by bokeh
                _move_stray_html_files(current_dir, output_dir, f"{output_filename}_heatmap_interactive", logger)
                
        except Exception as e:
            logger.warning(f"Could not generate interactive heatmap: {e}")
        
        # Mean aggregation heatmap
        try:
            from bokeh.plotting import output_file, save, reset_output
            
            # Reset any previous bokeh output settings
            reset_output()
            
            heatmap_mean_file = output_dir / f"{output_filename}_heatmap_mean.html"
            output_file(str(heatmap_mean_file))
            
            fig_mean = plot_heatmaps(heatmap_series, agg='mean')
            if fig_mean:
                save(fig_mean)
                logger.info(f"Mean heatmap saved to: {heatmap_mean_file}")
                
                # Check for and move any stray HTML files created by bokeh
                _move_stray_html_files(current_dir, output_dir, f"{output_filename}_heatmap_mean", logger)
                
        except Exception as e:
            logger.warning(f"Could not generate mean heatmap: {e}")
        
        # Generate static PNG heatmaps as fallback using matplotlib/seaborn
        try:
            generate_static_heatmaps(heatmap_series, metric, output_dir, output_filename)
        except Exception as e:
            logger.warning(f"Could not generate static heatmaps: {e}")
        
        # Final cleanup: check for any remaining stray HTML files
        _move_stray_html_files(current_dir, output_dir, output_filename, logger)
        
    except Exception as e:
        logger.warning(f"Could not generate heatmaps with plot_heatmaps: {e}")
        # Fallback to basic heatmap generation
        try:
            generate_static_heatmaps(heatmap_series, metric, output_dir, output_filename)
        except Exception as e2:
            logger.warning(f"Could not generate basic heatmap either: {e2}")
        
        # Final cleanup even in fallback case
        current_dir = Path.cwd()
        _move_stray_html_files(current_dir, output_dir, output_filename, logger)

def generate_static_heatmaps(heatmap_series, metric, output_dir, output_filename):
    """Generate static PNG heatmaps using matplotlib/seaborn"""
    logger = setup_logger()
    
    try:
        # Get parameter names from the MultiIndex
        param_names = heatmap_series.index.names
        
        if len(param_names) >= 2:
            # Create 2D projections for all parameter pairs
            for i in range(len(param_names)):
                for j in range(i + 1, len(param_names)):
                    param1, param2 = param_names[i], param_names[j]
                    
                    # Group by the two parameters and aggregate
                    try:
                        # Get all combinations for these two parameters
                        hm_data = heatmap_series.groupby([param1, param2]).agg(['mean', 'max', 'min'])
                        
                        for agg_method in ['mean', 'max']:
                            try:
                                # Create pivot table for heatmap
                                pivot_data = hm_data[agg_method].unstack()
                                
                                plt.figure(figsize=(10, 8))
                                sns.heatmap(pivot_data, annot=True, fmt='.2f', 
                                          cmap='RdYlGn', center=pivot_data.median().median())
                                plt.title(f'{metric} Heatmap ({agg_method.title()}) - {param1} vs {param2}')
                                plt.xlabel(param2)
                                plt.ylabel(param1)
                                plt.tight_layout()
                                
                                # Save heatmap
                                heatmap_file = output_dir / f"{output_filename}_heatmap_{param1}_{param2}_{agg_method}.png"
                                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                                plt.close()
                                
                                logger.info(f"Static {agg_method} heatmap ({param1} vs {param2}) saved to: {heatmap_file}")
                                
                            except Exception as e:
                                logger.warning(f"Could not create {agg_method} heatmap for {param1} vs {param2}: {e}")
                                
                    except Exception as e:
                        logger.warning(f"Could not process parameter pair {param1}, {param2}: {e}")
        
        # Also create a summary heatmap if we have many parameters
        if len(param_names) > 2:
            try:
                # Create a correlation-style plot showing best performance regions
                # Take the top 10% of results
                top_results = heatmap_series.nlargest(max(1, len(heatmap_series) // 10))
                
                # Convert to DataFrame for easier manipulation
                df = top_results.reset_index()
                df['metric'] = top_results.values
                
                # Create pairwise correlation plot
                plt.figure(figsize=(12, 10))
                # Create a correlation matrix of the top performing parameter combinations
                corr_data = df[param_names].corr()
                sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0)
                plt.title(f'Parameter Correlation in Top 10% Results')
                plt.tight_layout()
                
                summary_file = output_dir / f"{output_filename}_heatmap_summary.png"
                plt.savefig(summary_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Summary correlation heatmap saved to: {summary_file}")
                
            except Exception as e:
                logger.warning(f"Could not create summary heatmap: {e}")
                
    except Exception as e:
        logger.warning(f"Could not generate static heatmaps: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    parser.add_argument('--strategy', required=True, help='Strategy name (sma, rsi)')
    parser.add_argument('--symbol', help='Symbol to optimize')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-source', help='Data source (yahoo, ccxt, csv, csv_processed) - overrides config')
    parser.add_argument('--timeframe', help='Timeframe (1m, 5m, 15m, 30m, 1h, 1d) - overrides config')
    parser.add_argument('--metric', default='Return [%]', help='Optimization metric')
    parser.add_argument('--minimize', action='store_true', help='Minimize metric instead of maximize')
    parser.add_argument('--max-tries', type=int, help='Maximum optimization attempts')
    
    # Strategy-specific parameters
    parser.add_argument('--short-window', help='SMA short window range (e.g., "5,50" or "10,20,30")')
    parser.add_argument('--long-window', help='SMA long window range (e.g., "20,100" or "50,60,70")')
    parser.add_argument('--stop-loss', help='Stop loss percentage range (e.g., "1,5" or "2,3,4")')
    parser.add_argument('--take-profit', help='Take profit percentage range (e.g., "2,10" or "4,6,8")')

    args = parser.parse_args()

    # Load config to get defaults if arguments not provided
    config = load_config()
    
    # Use config defaults if arguments not provided
    symbol = args.symbol or (config.symbols[0] if config.symbols else "AAPL")
    
    # Build optimization parameters based on strategy
    optimization_params = {}
    
    if args.strategy.lower() in ["sma", "sma_crossover", "trend_following"]:
        if args.short_window:
            values = args.short_window.split(',')
            if len(values) == 2:
                optimization_params['short_window'] = (int(values[0]), int(values[1]))
            else:
                optimization_params['short_window'] = [int(v) for v in values]
        else:
            optimization_params['short_window'] = (5, 50)
            
        if args.long_window:
            values = args.long_window.split(',')
            if len(values) == 2:
                optimization_params['long_window'] = (int(values[0]), int(values[1]))
            else:
                optimization_params['long_window'] = [int(v) for v in values]
        else:
            optimization_params['long_window'] = (20, 100)
            
        if args.stop_loss:
            values = args.stop_loss.split(',')
            if len(values) == 2:
                optimization_params['stop_loss_pct'] = (float(values[0]), float(values[1]))
            else:
                optimization_params['stop_loss_pct'] = [float(v) for v in values]
            
        if args.take_profit:
            values = args.take_profit.split(',')
            if len(values) == 2:
                optimization_params['take_profit_pct'] = (float(values[0]), float(values[1]))
            else:
                optimization_params['take_profit_pct'] = [float(v) for v in values]
    
    # Add more strategy parameter definitions here as needed
    
    if not optimization_params:
        print(f"No optimization parameters defined for strategy: {args.strategy}")
        sys.exit(1)
    
    optimize_strategy(
        args.strategy, 
        symbol, 
        args.start, 
        args.end,
        optimization_params,
        args.data_source, 
        args.timeframe,
        args.metric,
        not args.minimize,
        args.max_tries
    )