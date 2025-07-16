-- Create schema for trading bot
CREATE SCHEMA IF NOT EXISTS trading;

-- Set default search path
SET search_path TO trading, public;

-- Create enum types
CREATE TYPE trade_side AS ENUM ('buy', 'sell');
CREATE TYPE trade_status AS ENUM ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected', 'expired');
CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit');

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(255) UNIQUE NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side trade_side NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) DEFAULT 0,
    commission_asset VARCHAR(50),
    status trade_status NOT NULL DEFAULT 'pending',
    order_type order_type NOT NULL DEFAULT 'market',
    strategy_name VARCHAR(100),
    signal_reason TEXT,
    exchange VARCHAR(50),
    account_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
    average_price DECIMAL(20, 8) NOT NULL DEFAULT 0,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    exchange VARCHAR(50),
    account_id VARCHAR(100),
    strategy_name VARCHAR(100),
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(symbol, exchange, account_id, strategy_name)
);

-- Create strategy_signals table
CREATE TABLE IF NOT EXISTS strategy_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    signal_type VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    signal_strength DECIMAL(5, 4), -- 0.0 to 1.0
    price DECIMAL(20, 8) NOT NULL,
    indicators JSONB DEFAULT '{}'::jsonb, -- Store indicator values
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(50),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    win_rate DECIMAL(5, 4) DEFAULT 0,
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create account_balances table
CREATE TABLE IF NOT EXISTS account_balances (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(100) NOT NULL,
    exchange VARCHAR(50),
    asset VARCHAR(50) NOT NULL,
    free_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
    locked_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
    total_balance DECIMAL(20, 8) GENERATED ALWAYS AS (free_balance + locked_balance) STORED,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(account_id, exchange, asset)
);

-- Create market_data table for caching
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '1h', '1d'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, exchange, timestamp, timeframe)
);

-- State persistence tables
CREATE TABLE IF NOT EXISTS trading_state (
    id SERIAL PRIMARY KEY,
    instance_id VARCHAR(255) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    state_type VARCHAR(50) NOT NULL,  -- 'positions', 'orders', 'daily_pnl', 'checkpoint'
    state_key VARCHAR(255),  -- For keyed state like positions[symbol], orders[id]
    state_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(instance_id, exchange, state_type, state_key)
);

-- Trading locks table
CREATE TABLE IF NOT EXISTS trading_locks (
    id SERIAL PRIMARY KEY,
    lock_id VARCHAR(255) UNIQUE NOT NULL,
    instance_id VARCHAR(255) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    acquired_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '1 hour'
);


-- Create indexes
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_created_at ON trades(created_at);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_strategy ON trades(strategy_name);

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_active ON positions(is_active);
CREATE INDEX idx_positions_strategy ON positions(strategy_name);

CREATE INDEX idx_signals_symbol ON strategy_signals(symbol);
CREATE INDEX idx_signals_created_at ON strategy_signals(created_at);
CREATE INDEX idx_signals_strategy ON strategy_signals(strategy_name);

CREATE INDEX idx_market_data_lookup ON market_data(symbol, exchange, timeframe, timestamp DESC);

-- State persistence indexes
CREATE INDEX idx_trading_state_instance ON trading_state(instance_id, exchange, state_type);
CREATE INDEX idx_trading_state_updated ON trading_state(updated_at);

CREATE INDEX idx_trading_locks_instance ON trading_locks(instance_id, exchange);
CREATE INDEX idx_trading_locks_expires ON trading_locks(expires_at);

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_balances_updated_at BEFORE UPDATE ON account_balances
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_state_updated_at BEFORE UPDATE ON trading_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for reporting
CREATE OR REPLACE VIEW v_active_positions AS
SELECT 
    p.*,
    t.total_trades,
    t.total_volume
FROM positions p
LEFT JOIN (
    SELECT 
        symbol,
        exchange,
        account_id,
        COUNT(*) as total_trades,
        SUM(quantity * price) as total_volume
    FROM trades
    WHERE status = 'filled'
    GROUP BY symbol, exchange, account_id
) t ON p.symbol = t.symbol 
    AND p.exchange = t.exchange 
    AND p.account_id = t.account_id
WHERE p.is_active = TRUE;

CREATE OR REPLACE VIEW v_daily_performance AS
SELECT 
    DATE(created_at) as trading_date,
    strategy_name,
    symbol,
    COUNT(*) as trades_count,
    SUM(CASE WHEN side = 'buy' THEN -quantity * price ELSE quantity * price END) as daily_pnl,
    AVG(price) as avg_price
FROM trades
WHERE status = 'filled'
GROUP BY DATE(created_at), strategy_name, symbol
ORDER BY trading_date DESC;

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trading_user;