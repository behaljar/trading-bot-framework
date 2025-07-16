# 📊 Database Usage Guide

## Quick Start Monitoring

### 1. **Real-time Database Monitor**
```bash
# Run the database monitor script
python scripts/monitor_db.py
```

### 2. **Manual Database Queries**
```bash
# Connect to database
docker compose exec postgres psql -U trading_user -d trading_bot

# Or from outside container
psql -h localhost -p 5432 -U trading_user -d trading_bot
```

## 📈 Key Database Tables

### **trades** - All executed trades
```sql
SELECT * FROM trading.trades 
ORDER BY created_at DESC 
LIMIT 10;
```

### **positions** - Current and historical positions
```sql
SELECT * FROM trading.positions 
WHERE is_active = true;
```

### **account_balances** - Account balance snapshots
```sql
SELECT * FROM trading.account_balances 
ORDER BY updated_at DESC;
```

### **trading_state** - Bot state persistence
```sql
SELECT * FROM trading.trading_state 
ORDER BY updated_at DESC;
```

## 🔄 When Database Updates

### **Balance Updates Happen:**
1. **Bot Startup** - Initial balance recorded
2. **After Each Trade** - Balance updated when order fills
3. **Position Changes** - Balance updated when positions open/close

### **Trade Logging Happens:**
1. **Order Placed** - Status: 'placed'
2. **Order Filled** - Status: 'filled' with execution details
3. **Position Updated** - Position table updated with new quantities

### **State Persistence Happens:**
1. **Continuous** - Every trading cycle
2. **On Trade** - After each order execution
3. **On Shutdown** - Final state save

## 📊 Useful Queries

### **Today's Trading Performance**
```sql
SELECT 
    symbol,
    COUNT(*) as trades,
    SUM(CASE WHEN side = 'buy' THEN -quantity * price ELSE quantity * price END) as pnl
FROM trading.trades 
WHERE created_at >= CURRENT_DATE 
    AND status = 'filled'
GROUP BY symbol;
```

### **Current Position Status**
```sql
SELECT 
    symbol,
    quantity,
    average_price,
    current_price,
    (current_price - average_price) * quantity as unrealized_pnl
FROM trading.positions 
WHERE is_active = true;
```

### **Balance History**
```sql
SELECT 
    asset,
    free_balance,
    locked_balance,
    free_balance + locked_balance as total_balance,
    updated_at
FROM trading.account_balances 
WHERE asset = 'USDT'
ORDER BY updated_at DESC;
```

### **Trade Execution Timeline**
```sql
SELECT 
    trade_id,
    symbol,
    side,
    quantity,
    price,
    quantity * price as volume,
    commission,
    created_at
FROM trading.trades 
WHERE status = 'filled' 
ORDER BY created_at DESC;
```

## 🎯 Monitoring Best Practices

### **1. Set up Grafana Dashboard**
- Connect Grafana to PostgreSQL
- Create charts for P&L, trades, balances
- Set up alerts for significant losses

### **2. Regular Database Maintenance**
```sql
-- Clean up old trading state (older than 30 days)
DELETE FROM trading.trading_state 
WHERE updated_at < NOW() - INTERVAL '30 days';

-- Clean up expired locks
DELETE FROM trading.trading_locks 
WHERE expires_at < NOW();
```

### **3. Performance Monitoring**
```sql
-- Check database performance
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables 
WHERE schemaname = 'trading';
```

## 🚨 Important Notes

1. **Balance Updates**: Balances are updated **after** successful trades, not before
2. **State Persistence**: Uses PostgreSQL by default, falls back to files if DB unavailable
3. **Performance**: Database logging is non-blocking - failures don't stop trading
4. **Backups**: Set up regular database backups for production use

## 🔧 Configuration

All database settings are configured via environment variables:
- `DB_HOST` - PostgreSQL host (default: postgres)
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name (default: trading_bot)
- `DB_USER` - Database user (default: trading_user)
- `DB_PASSWORD` - Database password

## 📝 Testing

The test strategy generates signals every 2 bars now, so you should see:
1. **Trades** appearing in the database
2. **Positions** being opened and closed
3. **Balances** being updated after each trade
4. **State** being persisted continuously

Monitor with: `python scripts/monitor_db.py`