# Stock Trading Bot

A stock trading bot with signal generation and automated trading capabilities through Alpaca Markets(paper only/fake money, you can make it for real money just plug in your live trading API key and secret key from alpaca).

## New Unified Interface


- **Main Interface**: `auto_interface.py` - The primary GUI application with both signal generation and automated trading abilities
- **Command Line Auto Trader**: `auto_trader.py` - Command line version of the automated trading system
- **API Integration**: `trader.py` - Connects to Alpaca Markets API for order execution


## Features

- **Data Fetching**: Retrieves historical stock data using yfinance
- **Advanced Trading Strategies**: 
  - Enhanced Moving Average Crossover
  - RSI Strategy with dynamic thresholds
  - Momentum Strategy with volume confirmation
  - Breakout Strategy (requires 30+ data bars)
  - Mean Reversion Strategy (requires 30+ data bars)
  - Dual Strategy System
- **Visualizations**: Generates charts with trading signals
- **Automated Trading**: Executes trades through Alpaca Markets API
- **Strategy Selection**: Automatically picks the best strategy based on market conditions
- **Risk Management**: Position sizing, maximum positions limit, stop losses

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StockTradingBot.git
cd StockTradingBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install alpaca-trade-api
```

3. Set up your Alpaca API credentials in a `.env` file:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Usage

### GUI Interface

Launch the GUI interface with all features:
```bash
python auto_interface.py
```

This provides:
- Dashboard tab for viewing signals
- Auto-Trading tab for configuring and monitoring automated trading
- Charts tab for viewing visualizations
- Settings tab for configuration
- Logs tab for monitoring activity

### Command Line

Run the automated trader from the command line:
```bash
python auto_trader.py --symbols AAPL MSFT --interval 1h --period 3mo --market-regime auto
```

Options:
- `--symbols`: List of stock symbols to trade
- `--interval`: Data interval (1d, 1h, 5m)
- `--period`: Data period (1mo, 3mo, 6mo, 1y)
- `--update-interval`: Time between updates
- `--no-trade`: Run in signal-only mode
- `--market-regime`: Strategy bias (trending, ranging, auto)
- `--risk-percent`: Portfolio percentage to risk per trade
- `--max-positions`: Maximum number of active positions

## Recommendations for Best Results

1. **Data Requirements**:
   - Breakout and Mean Reversion strategies require at least 30 bars of data
   - Use longer periods (3mo or 6mo) with shorter intervals (1h) for sufficient data points
   - Initially disable complex strategies in Settings → Strategy Selection

2. **Starting Settings**:
   - Interval: 1h
   - Period: 3mo
   - Risk %: 0.5-2%
   - Enabled Strategies: MA Crossover, RSI, Momentum (initially)

## For Detailed Information

See `AUTO_TRADING_GUIDE.md` for comprehensive instructions on setting up automated trading.

## Disclaimer

This bot is for educational purposes only. Don't use it for real trading without proper testing and validation. The author is not responsible for any financial losses.
