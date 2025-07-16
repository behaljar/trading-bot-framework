#!/usr/bin/env python3
"""
Database monitoring script for trading bot
"""
import psycopg2
import time
import os
from datetime import datetime
from psycopg2.extras import RealDictCursor

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        database=os.getenv('DB_NAME', 'trading_bot'),
        user=os.getenv('DB_USER', 'trading_user'),
        password=os.getenv('DB_PASSWORD', 'trading_password')
    )

def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def monitor_trades():
    """Monitor recent trades"""
    print_section("RECENT TRADES")
    
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT 
                trade_id,
                symbol,
                side,
                quantity,
                price,
                status,
                exchange,
                created_at,
                commission
            FROM trading.trades 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        trades = cursor.fetchall()
        
        if trades:
            print(f"{'Trade ID':<12} {'Symbol':<12} {'Side':<6} {'Qty':<10} {'Price':<10} {'Status':<8} {'Time':<20}")
            print("-" * 80)
            for trade in trades:
                print(f"{trade['trade_id']:<12} {trade['symbol']:<12} {trade['side']:<6} {trade['quantity']:<10.4f} {trade['price']:<10.4f} {trade['status']:<8} {trade['created_at'].strftime('%H:%M:%S'):<20}")
        else:
            print("No trades found")
    
    conn.close()

def monitor_positions():
    """Monitor current positions"""
    print_section("CURRENT POSITIONS")
    
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT 
                symbol,
                quantity,
                average_price,
                current_price,
                unrealized_pnl,
                realized_pnl,
                exchange,
                is_active,
                updated_at
            FROM trading.positions 
            WHERE is_active = true
            ORDER BY updated_at DESC
        """)
        
        positions = cursor.fetchall()
        
        if positions:
            print(f"{'Symbol':<12} {'Quantity':<10} {'Avg Price':<10} {'Curr Price':<10} {'Unrealized':<10} {'Realized':<10}")
            print("-" * 80)
            for pos in positions:
                print(f"{pos['symbol']:<12} {pos['quantity']:<10.4f} {pos['average_price']:<10.4f} {pos['current_price'] or 0:<10.4f} {pos['unrealized_pnl'] or 0:<10.2f} {pos['realized_pnl'] or 0:<10.2f}")
        else:
            print("No active positions")
    
    conn.close()

def monitor_balances():
    """Monitor account balances"""
    print_section("ACCOUNT BALANCES")
    
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT 
                account_id,
                asset,
                free_balance,
                locked_balance,
                exchange,
                updated_at
            FROM trading.account_balances 
            ORDER BY updated_at DESC
            LIMIT 10
        """)
        
        balances = cursor.fetchall()
        
        if balances:
            print(f"{'Account':<15} {'Asset':<8} {'Free':<12} {'Locked':<12} {'Exchange':<10} {'Updated':<20}")
            print("-" * 80)
            for balance in balances:
                print(f"{balance['account_id']:<15} {balance['asset']:<8} {balance['free_balance']:<12.4f} {balance['locked_balance']:<12.4f} {balance['exchange']:<10} {balance['updated_at'].strftime('%H:%M:%S'):<20}")
        else:
            print("No balances found")
    
    conn.close()

def monitor_state():
    """Monitor trading state"""
    print_section("TRADING STATE")
    
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT 
                instance_id,
                exchange,
                state_type,
                state_key,
                updated_at
            FROM trading.trading_state 
            ORDER BY updated_at DESC
            LIMIT 10
        """)
        
        states = cursor.fetchall()
        
        if states:
            print(f"{'Instance':<20} {'Exchange':<10} {'Type':<12} {'Key':<20} {'Updated':<20}")
            print("-" * 80)
            for state in states:
                print(f"{state['instance_id']:<20} {state['exchange']:<10} {state['state_type']:<12} {state['state_key'] or 'N/A':<20} {state['updated_at'].strftime('%H:%M:%S'):<20}")
        else:
            print("No trading state found")
    
    conn.close()

def monitor_performance():
    """Monitor daily performance"""
    print_section("DAILY PERFORMANCE")
    
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT 
                trading_date,
                strategy_name,
                symbol,
                trades_count,
                daily_pnl,
                avg_price
            FROM trading.v_daily_performance
            WHERE trading_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY trading_date DESC, symbol
        """)
        
        performance = cursor.fetchall()
        
        if performance:
            print(f"{'Date':<12} {'Strategy':<15} {'Symbol':<10} {'Trades':<8} {'P&L':<10} {'Avg Price':<10}")
            print("-" * 80)
            for perf in performance:
                print(f"{perf['trading_date']:<12} {perf['strategy_name']:<15} {perf['symbol']:<10} {perf['trades_count']:<8} {perf['daily_pnl']:<10.2f} {perf['avg_price']:<10.4f}")
        else:
            print("No performance data found")
    
    conn.close()

def main():
    """Main monitoring loop"""
    print("🔍 Trading Bot Database Monitor")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"📊 Trading Bot Database Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            monitor_trades()
            monitor_positions()
            monitor_balances()
            monitor_state()
            monitor_performance()
            
            print(f"\n{'='*60}")
            print("🔄 Refreshing in 10 seconds... (Ctrl+C to exit)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()