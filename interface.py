#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI interface for the Stock Trading Bot.
"""

import os
import sys
import glob
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinter.font import Font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pandas as pd
from PIL import Image, ImageTk

from data_fetcher import DataFetcher
from strategy import MovingAverageCrossover, RSIStrategy, MomentumStrategy, StrategyManager
from visualizer import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
    logger.info("Created logs directory")

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')
    logger.info("Created figures directory")

class LogHandler(logging.Handler):
    """Custom logging handler that writes to a tkinter Text widget."""
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
    def emit(self, record):
        msg = self.formatter.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        self.text_widget.after(0, append)

class StockTradingBotGUI(tk.Tk):
    """GUI interface for the Stock Trading Bot."""
    
    def __init__(self):
        """Initialize the GUI."""
        super().__init__()
        
        # Configure the window
        self.title("Stock Trading Bot")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Set up variables
        self.bot_running = False
        self.bot_thread = None
        self.symbols = tk.StringVar(value="AAPL,MSFT,GOOGL,AMZN,META")
        self.strategy = tk.StringVar(value="MA Crossover")
        self.interval = tk.StringVar(value="1d")
        self.update_interval = tk.IntVar(value=60)
        
        # Initialize bot components 
        self.data_fetcher = DataFetcher()
        self.strategy_manager = StrategyManager()
        self.visualizer = Visualizer()
        
        # Set up strategies
        self.setup_strategies()
        
        # Create the main UI
        self.create_widgets()
        
        # Add custom logging to the text widget
        self.log_handler = LogHandler(self.log_text)
        logger.addHandler(self.log_handler)
        
        # Create signal dictionary to store the latest signals
        self.signals = {}
        
        # Load initial available charts
        self.update_available_charts()
        
        logger.info("Stock Trading Bot Interface started")
    
    def setup_strategies(self):
        """Set up trading strategies."""
        self.strategy_manager.add_strategy(MovingAverageCrossover(short_window=20, long_window=50))
        self.strategy_manager.add_strategy(RSIStrategy(period=14, overbought=70, oversold=30))
        self.strategy_manager.add_strategy(MomentumStrategy(period=10, threshold=0.05))
    
    def create_widgets(self):
        """Create the interface widgets."""
        # Create a main frame that fills the window
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a notebook for tabbed interface
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create the tabs
        dashboard_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)
        charts_tab = ttk.Frame(notebook)
        logs_tab = ttk.Frame(notebook)
        
        notebook.add(dashboard_tab, text="Dashboard")
        notebook.add(charts_tab, text="Charts")
        notebook.add(settings_tab, text="Settings")
        notebook.add(logs_tab, text="Logs")
        
        # Set up the dashboard tab
        self.setup_dashboard(dashboard_tab)
        
        # Set up the settings tab
        self.setup_settings(settings_tab)
        
        # Set up the charts tab
        self.setup_charts(charts_tab)
        
        # Set up the logs tab
        self.setup_logs(logs_tab)
        
        # Create the status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_dashboard(self, parent):
        """Setup the dashboard tab with controls and signal display."""
        # Create a frame for controls
        control_frame = ttk.LabelFrame(parent, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create buttons for starting and stopping the bot
        self.start_button = ttk.Button(control_frame, text="Start Bot", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Bot", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Add a refresh button for signals
        refresh_button = ttk.Button(control_frame, text="Refresh Data", command=self.refresh_data)
        refresh_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create a frame for the signal display
        signal_frame = ttk.LabelFrame(parent, text="Latest Trading Signals")
        signal_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview to display signals
        columns = ("Symbol", "Price", "Signal", "Strategy", "Time")
        self.signal_tree = ttk.Treeview(signal_frame, columns=columns, show="headings")
        
        # Define column headings
        for col in columns:
            self.signal_tree.heading(col, text=col)
            if col == "Time":
                self.signal_tree.column(col, width=150)
            else:
                self.signal_tree.column(col, width=100)
        
        self.signal_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a scrollbar for the treeview
        scrollbar = ttk.Scrollbar(signal_frame, orient=tk.VERTICAL, command=self.signal_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.signal_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind double-click event to show chart
        self.signal_tree.bind("<Double-1>", self.show_chart_for_symbol)
    
    def setup_settings(self, parent):
        """Setup the settings tab with configuration options."""
        # Create a frame for general settings
        general_frame = ttk.LabelFrame(parent, text="General Settings")
        general_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Symbols
        ttk.Label(general_frame, text="Symbols (comma-separated):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(general_frame, textvariable=self.symbols, width=40).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Strategy
        ttk.Label(general_frame, text="Strategy:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        strategy_combo = ttk.Combobox(general_frame, textvariable=self.strategy, state="readonly", width=37)
        strategy_combo['values'] = ("MA Crossover", "RSI Strategy", "Momentum Strategy")
        strategy_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Interval
        ttk.Label(general_frame, text="Data Interval:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        interval_combo = ttk.Combobox(general_frame, textvariable=self.interval, state="readonly", width=37)
        interval_combo['values'] = ("1d", "1h", "5m")
        interval_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Update Interval
        ttk.Label(general_frame, text="Update Interval (seconds):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(general_frame, textvariable=self.update_interval, width=10).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create a frame for strategy settings
        strategy_frame = ttk.LabelFrame(parent, text="Strategy Settings")
        strategy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Moving Average Strategy Settings
        ma_frame = ttk.LabelFrame(strategy_frame, text="Moving Average Crossover")
        ma_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.ma_short = tk.IntVar(value=20)
        self.ma_long = tk.IntVar(value=50)
        
        ttk.Label(ma_frame, text="Short Window:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(ma_frame, textvariable=self.ma_short, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(ma_frame, text="Long Window:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(ma_frame, textvariable=self.ma_long, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # RSI Strategy Settings
        rsi_frame = ttk.LabelFrame(strategy_frame, text="RSI Strategy")
        rsi_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.rsi_period = tk.IntVar(value=14)
        self.rsi_overbought = tk.IntVar(value=70)
        self.rsi_oversold = tk.IntVar(value=30)
        
        ttk.Label(rsi_frame, text="Period:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(rsi_frame, textvariable=self.rsi_period, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(rsi_frame, text="Overbought:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(rsi_frame, textvariable=self.rsi_overbought, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(rsi_frame, text="Oversold:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(rsi_frame, textvariable=self.rsi_oversold, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Momentum Strategy Settings
        momentum_frame = ttk.LabelFrame(strategy_frame, text="Momentum Strategy")
        momentum_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.momentum_period = tk.IntVar(value=10)
        self.momentum_threshold = tk.DoubleVar(value=0.05)
        
        ttk.Label(momentum_frame, text="Period:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(momentum_frame, textvariable=self.momentum_period, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(momentum_frame, text="Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(momentum_frame, textvariable=self.momentum_threshold, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Buttons to save and load settings
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Apply Settings", command=self.apply_settings).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Reset to Defaults", command=self.reset_settings).pack(side=tk.LEFT, padx=5, pady=5)
    
    def setup_charts(self, parent):
        """Setup the charts tab with chart selection and display."""
        # Create a frame for the list of available charts
        list_frame = ttk.LabelFrame(parent, text="Available Charts")
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create a listbox for the charts
        self.chart_listbox = tk.Listbox(list_frame, width=40)
        self.chart_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a scrollbar for the listbox
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.chart_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chart_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection event to display chart
        self.chart_listbox.bind("<<ListboxSelect>>", self.display_selected_chart)
        
        # Refresh button for chart list
        refresh_button = ttk.Button(list_frame, text="Refresh Charts", command=self.update_available_charts)
        refresh_button.pack(side=tk.BOTTOM, padx=5, pady=5)
        
        # Create a frame for the chart display
        chart_frame = ttk.LabelFrame(parent, text="Chart View")
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a label to display the chart image
        self.chart_label = ttk.Label(chart_frame)
        self.chart_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_logs(self, parent):
        """Setup the logs tab with log display."""
        # Create a frame for the log display
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a text widget for the logs
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add a button to clear logs
        clear_button = ttk.Button(log_frame, text="Clear Logs", command=self.clear_logs)
        clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add a button to save logs
        save_button = ttk.Button(log_frame, text="Save Logs", command=self.save_logs)
        save_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    def start_bot(self):
        """Start the trading bot in a separate thread."""
        if self.bot_running:
            return
        
        # Update UI
        self.bot_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Bot running...")
        
        # Get settings
        symbol_list = [s.strip() for s in self.symbols.get().split(',')]
        strategy_name = self.strategy.get()
        interval = self.interval.get()
        update_interval = self.update_interval.get()
        
        # Start bot in a separate thread
        self.bot_thread = threading.Thread(
            target=self.run_bot, 
            args=(symbol_list, strategy_name, interval, update_interval)
        )
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        logger.info(f"Bot started with symbols: {symbol_list}, strategy: {strategy_name}")
    
    def stop_bot(self):
        """Stop the trading bot."""
        if not self.bot_running:
            return
        
        # Update UI
        self.bot_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Bot stopped")
        
        logger.info("Bot stopped")
    
    def run_bot(self, symbols, strategy_name, interval, update_interval):
        """Run the trading bot main loop."""
        try:
            while self.bot_running:
                # Process each symbol
                for symbol in symbols:
                    try:
                        if not self.bot_running:
                            break
                            
                        # Fetch stock data
                        logger.info(f"Fetching data for {symbol}")
                        stock_data = self.data_fetcher.get_stock_data(symbol, period='1mo', interval=interval)
                        
                        if stock_data.empty:
                            logger.warning(f"No data retrieved for {symbol}")
                            continue
                            
                        # Generate signals
                        logger.info(f"Generating signals for {symbol}")
                        signals = self.strategy_manager.generate_signals(stock_data, strategy_name)
                        
                        if not signals:
                            logger.warning(f"No signals generated for {symbol}")
                            continue
                            
                        # Get the signals DataFrame for the selected strategy
                        signal_data = signals.get(strategy_name)
                        
                        if signal_data is None:
                            logger.warning(f"Strategy '{strategy_name}' not found")
                            continue
                        
                        # Get the last signal
                        last_signal = signal_data['signal'].iloc[-1]
                        last_close = signal_data['close'].iloc[-1]
                        
                        # Convert signal to text
                        signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"
                        logger.info(f"{symbol} signal: {signal_text} at ${last_close:.2f}")
                        
                        # Update signals dictionary
                        signal_info = {
                            'symbol': symbol,
                            'price': f"${last_close:.2f}",
                            'signal': signal_text,
                            'strategy': strategy_name,
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Store in signals dictionary
                        self.signals[symbol] = signal_info
                        
                        # Update the signal display
                        self.after(0, self.update_signal_display)
                        
                        # Visualize the data
                        if 'rsi' in signal_data.columns:
                            self.visualizer.plot_rsi(signal_data, symbol)
                        else:
                            self.visualizer.plot_signals(signal_data, symbol)
                            
                        # Update available charts
                        self.after(0, self.update_available_charts)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                
                # Clean up old figures
                self.visualizer.clean_old_figures(max_figures=30)
                
                # Wait for the next update if still running
                if self.bot_running:
                    logger.info(f"Waiting {update_interval} seconds for next update")
                    for _ in range(update_interval):
                        if not self.bot_running:
                            break
                        time.sleep(1)
        
        except Exception as e:
            logger.error(f"Bot error: {str(e)}")
        finally:
            logger.info("Bot thread stopped")
            # Ensure the UI is updated
            self.after(0, self.update_ui_on_bot_stop)
    
    def update_ui_on_bot_stop(self):
        """Update the UI when the bot stops."""
        self.bot_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Bot stopped")
    
    def update_signal_display(self):
        """Update the signal display treeview."""
        # Clear the treeview
        for i in self.signal_tree.get_children():
            self.signal_tree.delete(i)
        
        # Add signals to the treeview
        for symbol, info in self.signals.items():
            # Set row color based on signal
            tag = info['signal'].lower()
            self.signal_tree.insert('', 'end', values=(
                info['symbol'],
                info['price'],
                info['signal'],
                info['strategy'],
                info['time']
            ), tags=(tag,))
        
        # Configure tag colors
        self.signal_tree.tag_configure('buy', background='#d0f0d0')  # Light green
        self.signal_tree.tag_configure('sell', background='#f0d0d0')  # Light red
        self.signal_tree.tag_configure('hold', background='#f0f0d0')  # Light yellow
    
    def update_available_charts(self):
        """Update the list of available charts."""
        # Clear the listbox
        self.chart_listbox.delete(0, tk.END)
        
        # Get all chart files
        chart_files = glob.glob(os.path.join('figures', '*.png'))
        chart_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first
        
        # Add chart files to the listbox
        for file in chart_files:
            filename = os.path.basename(file)
            self.chart_listbox.insert(tk.END, filename)
    
    def display_selected_chart(self, event):
        """Display the selected chart."""
        # Get the selected chart file
        selected_indices = self.chart_listbox.curselection()
        if not selected_indices:
            return
        
        selected_file = self.chart_listbox.get(selected_indices[0])
        file_path = os.path.join('figures', selected_file)
        
        # Display the chart
        self.display_chart(file_path)
    
    def display_chart(self, file_path):
        """Display a chart image."""
        try:
            # Load the image
            image = Image.open(file_path)
            
            # Resize image to fit the label
            label_width = self.chart_label.winfo_width()
            label_height = self.chart_label.winfo_height()
            
            if label_width == 1:  # Not yet realized
                label_width = 600
                label_height = 400
            
            # Calculate new size while preserving aspect ratio
            img_width, img_height = image.size
            ratio = min(label_width/img_width, label_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update the label
            self.chart_label.configure(image=photo)
            self.chart_label.image = photo  # Keep a reference to prevent garbage collection
            
            # Update status
            self.status_var.set(f"Displaying chart: {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error displaying chart: {str(e)}")
            self.status_var.set(f"Error displaying chart: {str(e)}")
    
    def show_chart_for_symbol(self, event):
        """Show chart for the selected symbol in the treeview."""
        # Get the selected item
        selected_item = self.signal_tree.focus()
        if not selected_item:
            return
        
        # Get the symbol from the selected item
        symbol = self.signal_tree.item(selected_item)['values'][0]
        
        # Find the latest chart for the symbol
        chart_files = glob.glob(os.path.join('figures', f"{symbol}_*.png"))
        if chart_files:
            # Sort by modification time, newest first
            chart_files.sort(key=os.path.getmtime, reverse=True)
            
            # Display the latest chart
            self.display_chart(chart_files[0])
    
    def refresh_data(self):
        """Refresh data for the currently configured symbols without starting the bot."""
        symbol_list = [s.strip() for s in self.symbols.get().split(',')]
        strategy_name = self.strategy.get()
        interval = self.interval.get()
        
        # Update the status
        self.status_var.set("Refreshing data...")
        
        # Start in a separate thread to avoid freezing the UI
        threading.Thread(target=self.refresh_data_thread, args=(symbol_list, strategy_name, interval), daemon=True).start()
    
    def refresh_data_thread(self, symbols, strategy_name, interval):
        """Thread function to refresh data."""
        try:
            for symbol in symbols:
                try:
                    # Fetch stock data
                    logger.info(f"Fetching data for {symbol}")
                    stock_data = self.data_fetcher.get_stock_data(symbol, period='1mo', interval=interval)
                    
                    if stock_data.empty:
                        logger.warning(f"No data retrieved for {symbol}")
                        continue
                        
                    # Generate signals
                    logger.info(f"Generating signals for {symbol}")
                    signals = self.strategy_manager.generate_signals(stock_data, strategy_name)
                    
                    if not signals:
                        logger.warning(f"No signals generated for {symbol}")
                        continue
                        
                    # Get the signals DataFrame for the selected strategy
                    signal_data = signals.get(strategy_name)
                    
                    if signal_data is None:
                        logger.warning(f"Strategy '{strategy_name}' not found")
                        continue
                    
                    # Get the last signal
                    last_signal = signal_data['signal'].iloc[-1]
                    last_close = signal_data['close'].iloc[-1]
                    
                    # Convert signal to text
                    signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"
                    logger.info(f"{symbol} signal: {signal_text} at ${last_close:.2f}")
                    
                    # Update signals dictionary
                    signal_info = {
                        'symbol': symbol,
                        'price': f"${last_close:.2f}",
                        'signal': signal_text,
                        'strategy': strategy_name,
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Store in signals dictionary
                    self.signals[symbol] = signal_info
                    
                    # Update the signal display
                    self.after(0, self.update_signal_display)
                    
                    # Visualize the data
                    if 'rsi' in signal_data.columns:
                        self.visualizer.plot_rsi(signal_data, symbol)
                    else:
                        self.visualizer.plot_signals(signal_data, symbol)
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
            
            # Update available charts
            self.after(0, self.update_available_charts)
            
            # Update status
            self.after(0, lambda: self.status_var.set("Data refresh complete"))
            
        except Exception as e:
            logger.error(f"Refresh error: {str(e)}")
            self.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
    
    def apply_settings(self):
        """Apply the settings to the strategies."""
        try:
            # Update the MA Crossover strategy
            ma_strategy = self.strategy_manager.get_strategy("MA Crossover")
            if ma_strategy:
                ma_strategy.short_window = self.ma_short.get()
                ma_strategy.long_window = self.ma_long.get()
            
            # Update the RSI strategy
            rsi_strategy = self.strategy_manager.get_strategy("RSI Strategy")
            if rsi_strategy:
                rsi_strategy.period = self.rsi_period.get()
                rsi_strategy.overbought = self.rsi_overbought.get()
                rsi_strategy.oversold = self.rsi_oversold.get()
            
            # Update the Momentum strategy
            momentum_strategy = self.strategy_manager.get_strategy("Momentum Strategy")
            if momentum_strategy:
                momentum_strategy.period = self.momentum_period.get()
                momentum_strategy.threshold = self.momentum_threshold.get()
            
            messagebox.showinfo("Settings", "Settings applied successfully.")
            logger.info("Settings applied")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying settings: {str(e)}")
            logger.error(f"Error applying settings: {str(e)}")
    
    def reset_settings(self):
        """Reset settings to defaults."""
        # Reset strategy parameters
        self.ma_short.set(20)
        self.ma_long.set(50)
        self.rsi_period.set(14)
        self.rsi_overbought.set(70)
        self.rsi_oversold.set(30)
        self.momentum_period.set(10)
        self.momentum_threshold.set(0.05)
        
        # Reset general settings
        self.symbols.set("AAPL,MSFT,GOOGL,AMZN,META")
        self.strategy.set("MA Crossover")
        self.interval.set("1d")
        self.update_interval.set(60)
        
        messagebox.showinfo("Settings", "Settings reset to defaults.")
        logger.info("Settings reset to defaults")
    
    def clear_logs(self):
        """Clear the log display."""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        logger.info("Logs cleared")
    
    def save_logs(self):
        """Save the logs to a file."""
        filename = filedialog.asksaveasfilename(
            initialdir="logs",
            title="Save Logs",
            filetypes=(("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")),
            defaultextension=".log"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                
                messagebox.showinfo("Save Logs", f"Logs saved to {filename}")
                logger.info(f"Logs saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving logs: {str(e)}")
                logger.error(f"Error saving logs: {str(e)}")

if __name__ == "__main__":
    app = StockTradingBotGUI()
    app.mainloop() 