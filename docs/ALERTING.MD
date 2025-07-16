# Trading Bot Alert System

Comprehensive notification system for the trading bot that supports multiple communication channels for various types of alerts.

## Overview

The alert system automatically sends notifications about:
- **Executed trades** - position opening/closing
- **P&L alerts** - profits and losses
- **Warnings and errors** - technical issues, API errors
- **Emergency stop** - automatic halt when loss limits are reached

## Supported Channels

### 📧 Email
Classic email notifications via SMTP server.

**Configuration:**
```env
ENABLE_EMAIL_ALERTS=true
ALERT_EMAIL=your_email@example.com
ALERT_EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

**For Gmail:**
1. Enable 2FA in your Google account
2. Generate app password: https://support.google.com/accounts/answer/185833
3. Use app password instead of regular password

### 💬 Slack
Modern Slack notifications using Block Kit for better presentation.

**Setup:**
1. Create Slack app: https://api.slack.com/apps
2. Enable "Incoming Webhooks"
3. Create webhook for channel
4. Copy webhook URL

**Configuration:**
```env
ENABLE_SLACK_ALERTS=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 🎮 Discord
Discord webhook with rich embed formatting and color coding based on alert level.

**Setup:**
1. Go to Discord server → Server Settings → Integrations
2. Click "Create Webhook"
3. Select channel and copy webhook URL

**Configuration:**
```env
ENABLE_DISCORD_ALERTS=true
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK/URL
```

### 🤖 Telegram
Telegram bot with Markdown formatting and emoji indicators.

**Setup:**
1. Create bot via @BotFather
2. Get bot token
3. Find your chat ID via @userinfobot

**Configuration:**
```env
ENABLE_TELEGRAM_ALERTS=true
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

## Alert Types

### ℹ️ Info (Blue)
- Successful trade execution
- Position opening
- System information

### ⚠️ Warning (Orange)
- Position closing with loss
- Insufficient data warnings
- Less critical errors

### 🚨 Error (Red)
- Emergency stop activation
- Critical API errors
- Connection failures

## Sample Messages

### Trade Executed
```
🎯 Trade Executed: LONG BTC/USDT
Size: 0.001000
Price: $43,250.00
Exchange: BINANCE
```

### Position Closed
```
💰 Position Closed: BTC/USDT
Size: 0.001000
Close Price: $43,450.00
P&L: 0.20 USDT
Daily P&L: 1.25 USDT
```

### Emergency Stop
```
🚨 EMERGENCY STOP: Daily loss limit reached!
Daily P&L: -200.50 USDT
Limit: 200.00 USDT
Trading halted.
```

## Integration into Strategy

The alert system is automatically integrated into all traders (CCXT, IBKR). For custom usage:

```python
from monitoring.alert_system import AlertSystem

# In initialization
self.alert_system = AlertSystem(config)

# Sending alerts
self.alert_system.send_alert("Custom message", "info")
```

## Security

⚠️ **Important security notes:**

1. **Never share API keys** in alerts
2. **Protect webhook URLs** - they function as passwords
3. **Use app passwords** for email instead of main password
4. **Test notifications** before live trading
5. **Set rate limits** for webhooks

## Testing

To test the alert system:

```bash
# Test all channels
python -c "
from config.settings import load_config
from monitoring.alert_system import AlertSystem

config = load_config()
alerts = AlertSystem(config)
alerts.send_alert('Test notification - trading bot alert system', 'info')
"
```

## Troubleshooting

### Email not working
- Check SMTP settings
- Verify app password (not regular password)
- Try different SMTP server

### Slack/Discord/Telegram not working
- Verify webhook URL
- Check bot permissions
- Test webhook via curl

### Too many notifications
- Set higher thresholds
- Use only critical alerts
- Implement rate limiting

## Environment Configuration

### Development
```env
ENABLE_EMAIL_ALERTS=false
ENABLE_SLACK_ALERTS=true
ENABLE_DISCORD_ALERTS=false
ENABLE_TELEGRAM_ALERTS=false
```

### Production
```env
ENABLE_EMAIL_ALERTS=true
ENABLE_SLACK_ALERTS=true
ENABLE_DISCORD_ALERTS=true
ENABLE_TELEGRAM_ALERTS=true
```

### Error monitoring only
```env
ENABLE_EMAIL_ALERTS=true     # Only for critical errors
ENABLE_SLACK_ALERTS=false
ENABLE_DISCORD_ALERTS=false
ENABLE_TELEGRAM_ALERTS=false
```

## Rate Limiting

To prevent spam, it's recommended to implement custom rate limiting:
- Maximum 1 alert per minute for the same symbol
- Merge similar messages
- Prioritize critical alerts

The alert system is designed for high reliability and provides redundant communication channels for critical trading notifications.