#!/usr/bin/env python3
"""
Data Download Script

Downloads financial data from various sources and saves to CSV format.
Supports Yahoo Finance for stocks and CCXT for cryptocurrencies.
Handles large date ranges by chunking requests and combining data.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.yahoo_finance import YahooFinanceSource
from data.ccxt_source import CCXTSource


def validate_date(date_string):
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return date_string
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")


def get_default_dates():
    """Get default start and end dates (1 year back to today)"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    return start_date, end_date


def chunk_date_range(start_date, end_date, chunk_months=6):
    """Split date range into chunks to avoid API limits"""
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    chunks = []
    current_start = start_dt
    
    while current_start < end_dt:
        # Calculate chunk end (add months)
        year = current_start.year
        month = current_start.month + chunk_months
        
        if month > 12:
            month = month - 12
            year += 1
            
        chunk_end = datetime(year, month, current_start.day)
        
        # Don't exceed the actual end date
        if chunk_end > end_dt:
            chunk_end = end_dt
            
        chunks.append((
            current_start.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        ))
        
        # Move to next chunk
        current_start = chunk_end + timedelta(days=1)
    
    return chunks


def download_yahoo_data(symbol, start_date, end_date, timeframe, output_path):
    """Download data from Yahoo Finance"""
    print(f"Downloading {symbol} from Yahoo Finance...")
    
    source = YahooFinanceSource()
    
    # For Yahoo Finance, we can usually get all data in one request
    # But let's chunk it for very large ranges to be safe
    date_diff = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
    
    if date_diff.days > 1000:  # More than ~3 years, chunk it
        print(f"Large date range detected ({date_diff.days} days). Chunking requests...")
        chunks = chunk_date_range(start_date, end_date, chunk_months=12)
        all_data = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            print(f"  Downloading chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
            
            chunk_data = source.get_historical_data(symbol, chunk_start, chunk_end, timeframe)
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
            
            # Small delay between requests
            time.sleep(0.5)
        
        if all_data:
            data = pd.concat(all_data, ignore_index=False)
            data = data.sort_index()
            # Remove duplicates that might occur at chunk boundaries
            data = data[~data.index.duplicated(keep='first')]
        else:
            data = pd.DataFrame()
    else:
        data = source.get_historical_data(symbol, start_date, end_date, timeframe)
    
    if data.empty:
        print(f"No data retrieved for {symbol}")
        return False
    
    # Save to CSV
    data.to_csv(output_path)
    print(f"Saved {len(data)} rows to {output_path}")
    return True


def download_ccxt_data(symbol, start_date, end_date, timeframe, output_path, exchange="binance", sandbox=True):
    """Download data from CCXT exchange with chunking for large ranges"""
    print(f"Downloading {symbol} from {exchange} via CCXT...")
    
    try:
        source = CCXTSource(exchange_name=exchange, sandbox=sandbox)
        
        # Always chunk CCXT data since exchanges have strict limits
        chunks = chunk_date_range(start_date, end_date, chunk_months=3)  # Smaller chunks for CCXT
        all_data = []
        
        print(f"Downloading {len(chunks)} chunks...")
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            print(f"  Downloading chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
            
            chunk_data = source.get_historical_data(symbol, chunk_start, chunk_end, timeframe)
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
                print(f"    Retrieved {len(chunk_data)} rows")
            else:
                print(f"    No data for this chunk")
            
            # Rate limiting - important for CCXT
            time.sleep(2)
        
        if all_data:
            print("Combining all chunks...")
            data = pd.concat(all_data, ignore_index=False)
            data = data.sort_index()
            # Remove duplicates that might occur at chunk boundaries
            data = data[~data.index.duplicated(keep='first')]
        else:
            data = pd.DataFrame()
        
        if data.empty:
            print(f"No data retrieved for {symbol}")
            return False
        
        # Save to CSV
        data.to_csv(output_path)
        print(f"Saved {len(data)} total rows to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error with CCXT download: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download financial data and save to CSV with chunking support for large date ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download AAPL stock data (Yahoo Finance)
  python scripts/download_data.py --source yahoo --symbol AAPL

  # Download Bitcoin data from 2017 to now (CCXT with chunking)
  python scripts/download_data.py --source ccxt --symbol BTC/USDT --start 2017-01-01 --exchange binance

  # Download with custom date range and timeframe
  python scripts/download_data.py --source yahoo --symbol MSFT --start 2020-01-01 --end 2023-12-31 --timeframe 1h

  # Download from sandbox/testnet (live exchange is default)
  python scripts/download_data.py --source ccxt --symbol ETH/USDT --sandbox --start 2018-01-01

Supported timeframes:
  Yahoo Finance: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
  CCXT: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M (depends on exchange)

Note: Large date ranges are automatically chunked to avoid API limits.
      CCXT data is downloaded in 3-month chunks with rate limiting.
        """
    )
    
    parser.add_argument("--source", required=True, choices=["yahoo", "ccxt"],
                       help="Data source to use")
    parser.add_argument("--symbol", required=True,
                       help="Symbol to download (e.g., AAPL, BTC/USDT)")
    parser.add_argument("--start", type=validate_date,
                       help="Start date (YYYY-MM-DD). Default: 1 year ago")
    parser.add_argument("--end", type=validate_date,
                       help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument("--timeframe", default="1d",
                       help="Timeframe (default: 1d)")
    parser.add_argument("--exchange", default="binance",
                       help="Exchange name for CCXT (default: binance)")
    parser.add_argument("--sandbox", action="store_true",
                       help="Use sandbox/testnet instead of live exchange (CCXT only)")
    parser.add_argument("--output-dir", default="data/csv",
                       help="Output directory (default: data/csv)")
    parser.add_argument("--filename",
                       help="Custom filename (default: SYMBOL.csv)")
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.start or not args.end:
        default_start, default_end = get_default_dates()
        start_date = args.start or default_start
        end_date = args.end or default_end
    else:
        start_date = args.start
        end_date = args.end
    
    # Validate date range
    if datetime.strptime(start_date, '%Y-%m-%d') >= datetime.strptime(end_date, '%Y-%m-%d'):
        print("Error: Start date must be before end date")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if args.filename:
        filename = args.filename
    else:
        # Clean symbol for filename (replace / with _)
        clean_symbol = args.symbol.replace('/', '_')
        filename = f"{clean_symbol}.csv"
    
    output_path = output_dir / filename
    
    # Calculate total days
    total_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    
    print(f"Configuration:")
    print(f"  Source: {args.source}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Date range: {start_date} to {end_date} ({total_days} days)")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Output: {output_path}")
    if args.source == "ccxt":
        print(f"  Exchange: {args.exchange}")
        print(f"  Sandbox: {args.sandbox}")
    print()
    
    # Download data based on source
    success = False
    
    if args.source == "yahoo":
        success = download_yahoo_data(
            args.symbol, start_date, end_date, args.timeframe, output_path
        )
    elif args.source == "ccxt":
        success = download_ccxt_data(
            args.symbol, start_date, end_date, args.timeframe, output_path,
            exchange=args.exchange, sandbox=args.sandbox
        )
    
    if success:
        print(f"\n✓ Successfully downloaded data for {args.symbol}")
        print(f"  File saved to: {output_path}")
    else:
        print(f"\n✗ Failed to download data for {args.symbol}")
        sys.exit(1)


if __name__ == "__main__":
    main()