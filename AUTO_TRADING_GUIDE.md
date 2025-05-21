# Automatic Trading Guide

This guide will help you set up and use the automatic trading functionality with Alpaca Markets.

## Setup Instructions

1. **Install dependencies**:
   ```
   pip install alpaca-trade-api
   ```

2. **Configure Alpaca API credentials**:
   - Create a file named `.env` in the project root
   - Add your Alpaca API credentials:
   ```
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_SECRET_KEY=your_alpaca_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
   - For paper trading (recommended for testing), use `https://paper-api.alpaca.markets`
   - For live trading, use `https://api.alpaca.markets`

3. **Verify your Alpaca account**:
   - Make sure your Alpaca account is funded (for paper trading, this is automatic)
   - Ensure your account has the appropriate permissions for trading

## Using the Auto Trading Features

### Command Line Auto Trader

The command line auto trader (`auto_trader.py`) can be run with various options:

```
python auto_trader.py [options]
```

Options:
- `--symbols`: List of stock symbols to trade (default: AAPL, MSFT, GOOGL, AMZN, META)
- `--interval`: Data interval (1d, 1h, 5m)
- `--update-interval`: Seconds between updates
- `--no-trade`: Run in signal-only mode without executing trades
- `--market-regime`: Strategy bias (trending, ranging, auto)
- `--risk-percent`: Portfolio percentage to risk per trade
- `--max-positions`: Maximum number of active positions

Example:
```
python auto_trader.py --symbols AAPL TSLA NVDA --interval 1h --risk-percent 0.01 --max-positions 3
```

### GUI Interface

The GUI interface (`auto_interface.py`) provides an interactive way to control the trading bot:

```
python auto_interface.py
```

Features:
1. **Dashboard Tab**: View real-time trading signals
2. **Auto Trading Tab**: Configure and monitor automatic trading
   - Enable/disable auto trading
   - Set market regime (trending, ranging, or auto-detect)
   - Configure risk percentage
   - Set maximum positions
   - View current positions and performance
3. **Charts Tab**: View generated charts for each symbol and strategy
4. **Settings Tab**: Configure general settings
5. **Logs Tab**: Monitor detailed logs of bot activity

## How the Auto Trader Works

1. The bot fetches market data for the specified symbols
2. All available strategies generate signals
3. The best strategy for each symbol is selected based on:
   - Historical performance
   - Current market regime
   - Signal strength
4. If auto trading is enabled, trades are executed through Alpaca:
   - Buy signals open long positions (if no position exists)
   - Sell signals close existing positions
   - Position sizing is based on the risk percentage

## Risk Management

- **Position Sizing**: Each position is sized based on the specified risk percentage of your portfolio value
- **Maximum Positions**: Limits the number of concurrent open positions
- **Paper Trading**: Use paper trading to test strategies without risking real money

## Best Practices

1. **Start with Paper Trading**: Always test strategies with paper trading before using real money
2. **Start Small**: Use small risk percentages (0.5-2%) when beginning
3. **Monitor Performance**: Regularly review strategy performance and positions
4. **Adjust as Needed**: Some strategies perform better in trending markets, others in ranging markets

## Troubleshooting

If you encounter issues:
1. Check logs for detailed error messages
2. Verify your Alpaca API credentials in the `.env` file
3. Ensure your Internet connection is stable
4. Check if the market is open (US stock market hours) 