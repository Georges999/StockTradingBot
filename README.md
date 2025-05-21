# Stock Trading Bot

A simple, clean stock trading bot with minimal dependencies. This bot fetches stock data, applies trading strategies, and visualizes trading signals.

## Features

- **Data Fetching**: Retrieves historical stock data using yfinance
- **Trading Strategies**: 
  - Moving Average Crossover
  - RSI (Relative Strength Index)
  - Momentum
- **Visualizations**: Generates charts with trading signals
- **Modular Design**: Easy to extend with new strategies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StockTradingBot.git
cd StockTradingBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the bot with default settings:
```bash
python main.py
```

### Command Line Arguments

- `--symbols`: List of stock symbols to trade (default: AAPL, MSFT, GOOGL, AMZN, META)
- `--strategy`: Trading strategy to use (default: MA Crossover)
- `--interval`: Data interval (default: 1d)
- `--update-interval`: Time in seconds between updates (default: 60)
- `--iterations`: Maximum number of iterations to run (default: None, runs indefinitely)

Example:
```bash
python main.py --symbols AAPL TSLA --strategy "RSI Strategy" --interval 1h --update-interval 300 --iterations 5
```

## Project Structure

- `main.py`: Main entry point
- `data_fetcher.py`: Fetches stock data
- `strategy.py`: Implements trading strategies
- `visualizer.py`: Creates charts and visualizations
- `figures/`: Directory for saved charts
- `logs/`: Directory for log files

## Extending the Bot

### Adding a New Strategy

1. Create a new strategy class in `strategy.py` that inherits from the `Strategy` base class
2. Implement the required methods
3. Add the strategy to the `StrategyManager` in `main.py`

Example:
```python
class MyNewStrategy(Strategy):
    def __init__(self):
        super().__init__("My New Strategy")
        
    def generate_signals(self, data):
        # Implement your strategy logic here
        # Return DataFrame with signals
        pass
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This bot is for educational purposes only. Don't use it for real trading without proper testing and validation. The author is not responsible for any financial losses.