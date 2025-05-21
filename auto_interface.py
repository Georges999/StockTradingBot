#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAIN INTERFACE FILE - This is the unified GUI interface for the Stock Trading Bot with Alpaca integration.
All other interface files (interface.py) can be safely deleted.
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
from strategy import (
    EnhancedMovingAverageCrossover,
    EnhancedRSIStrategy,
    EnhancedMomentumStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    DualStrategySystem,
    StrategyManager
)
from visualizer import Visualizer
from auto_trader import AutoTrader

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

class AutoTradingInterface(tk.Tk):
    """GUI interface for the Stock Trading Bot with auto-trading."""
    
    def __init__(self):
        """Initialize the GUI."""
        super().__init__()
        
        # Configure the window
        self.title("Stock Trading Bot with Auto-Trading")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Set up variables
        self.symbols = tk.StringVar(value="AAPL,MSFT,GOOGL,AMZN,META")
        self.strategy = tk.StringVar(value="MA Crossover")
        self.interval = tk.StringVar(value="1d")
        self.period = tk.StringVar(value="1mo")
        self.update_interval = tk.IntVar(value=60)
        self.market_regime = tk.StringVar(value="auto")
        self.risk_percent = tk.DoubleVar(value=0.02)
        self.max_positions = tk.IntVar(value=5)
        self.auto_trade_enabled = tk.BooleanVar(value=False)
        
        # Strategy enable/disable variables
        self.ma_crossover_enabled = tk.BooleanVar(value=True)
        self.rsi_strategy_enabled = tk.BooleanVar(value=True)
        self.momentum_strategy_enabled = tk.BooleanVar(value=True)
        self.breakout_strategy_enabled = tk.BooleanVar(value=False)  # Disabled by default as it needs more data
        self.mean_reversion_enabled = tk.BooleanVar(value=False)     # Disabled by default as it needs more data
        self.dual_strategy_enabled = tk.BooleanVar(value=True)
        
        # Advanced settings
        self.notify_on_signals = tk.BooleanVar(value=False)
        self.save_charts = tk.BooleanVar(value=True)
        self.max_charts = tk.IntVar(value=30)
        self.stop_loss_pct = tk.DoubleVar(value=5.0)
        self.take_profit_pct = tk.DoubleVar(value=10.0)
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.strategy_manager = StrategyManager()
        self.visualizer = Visualizer()
        
        # Initialize auto trader (but don't start it yet)
        self.auto_trader = None
        
        # Create the main UI
        self.create_widgets()
        
        # Add custom logging to the text widget
        self.log_handler = LogHandler(self.log_text)
        logger.addHandler(self.log_handler)
        
        # Create signal dictionary to store the latest signals
        self.signals = {}
        
        # Load initial available charts
        self.update_available_charts()
        
        logger.info("Auto Trading Interface started")
    
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
        auto_trading_tab = ttk.Frame(notebook)
        charts_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)
        logs_tab = ttk.Frame(notebook)
        
        notebook.add(dashboard_tab, text="Dashboard")
        notebook.add(auto_trading_tab, text="Auto Trading")
        notebook.add(charts_tab, text="Charts")
        notebook.add(settings_tab, text="Settings")
        notebook.add(logs_tab, text="Logs")
        
        # Set up the dashboard tab
        self.setup_dashboard(dashboard_tab)
        
        # Set up the auto trading tab
        self.setup_auto_trading(auto_trading_tab)
        
        # Set up the charts tab
        self.setup_charts(charts_tab)
        
        # Set up the settings tab
        self.setup_settings(settings_tab)
        
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
    
    def setup_auto_trading(self, parent):
        """Setup the auto trading tab with controls and position display."""
        # Create a frame for auto trading controls
        control_frame = ttk.LabelFrame(parent, text="Auto Trading Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Auto trading enable/disable
        auto_trade_check = ttk.Checkbutton(control_frame, text="Enable Auto Trading", 
                                           variable=self.auto_trade_enabled)
        auto_trade_check.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Market regime selection
        ttk.Label(control_frame, text="Market Regime:").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        regime_combo = ttk.Combobox(control_frame, textvariable=self.market_regime, state="readonly", width=15)
        regime_combo['values'] = ("auto", "trending", "ranging")
        regime_combo.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Risk percentage
        ttk.Label(control_frame, text="Risk %:").grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.risk_percent, width=6).grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        
        # Max positions
        ttk.Label(control_frame, text="Max Positions:").grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.max_positions, width=6).grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        
        # Apply settings button
        ttk.Button(control_frame, text="Apply Settings", command=self.apply_auto_trade_settings).grid(
            row=0, column=7, padx=5, pady=5, sticky=tk.W)
        
        # Create a frame for the positions display
        positions_frame = ttk.LabelFrame(parent, text="Current Positions")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview for positions
        columns = ("Symbol", "Quantity", "Entry Price", "Current Price", "P/L", "P/L %")
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings")
        
        # Define column headings
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=100)
        
        self.positions_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a scrollbar for the treeview
        scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.positions_tree.configure(yscrollcommand=scrollbar.set)
        
        # Create a frame for performance metrics
        performance_frame = ttk.LabelFrame(parent, text="Strategy Performance")
        performance_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview for performance
        columns = ("Strategy", "Trades", "Wins", "Losses", "Win Rate", "Total P/L")
        self.performance_tree = ttk.Treeview(performance_frame, columns=columns, show="headings")
        
        # Define column headings
        for col in columns:
            self.performance_tree.heading(col, text=col)
            self.performance_tree.column(col, width=100)
        
        self.performance_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a scrollbar for the treeview
        scrollbar = ttk.Scrollbar(performance_frame, orient=tk.VERTICAL, command=self.performance_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.performance_tree.configure(yscrollcommand=scrollbar.set)
        
        # Add refresh button for positions and performance
        refresh_button = ttk.Button(parent, text="Refresh Positions & Performance", 
                                   command=self.refresh_positions_and_performance)
        refresh_button.pack(side=tk.BOTTOM, padx=5, pady=5)
    
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
    
    def setup_settings(self, parent):
        """Setup the settings tab with configuration options."""
        # Create a frame for general settings
        general_frame = ttk.LabelFrame(parent, text="General Settings")
        general_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Symbols
        ttk.Label(general_frame, text="Symbols (comma-separated):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(general_frame, textvariable=self.symbols, width=40).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Strategy
        ttk.Label(general_frame, text="Default Strategy:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        strategy_combo = ttk.Combobox(general_frame, textvariable=self.strategy, state="readonly", width=37)
        strategy_combo['values'] = (
            "MA Crossover", 
            "RSI Strategy", 
            "Momentum Strategy",
            "Breakout Strategy",
            "Mean Reversion Strategy",
            "Dual Strategy System"
        )
        strategy_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Interval
        ttk.Label(general_frame, text="Data Interval:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        interval_combo = ttk.Combobox(general_frame, textvariable=self.interval, state="readonly", width=37)
        interval_combo['values'] = ("1d", "1h", "5m")
        interval_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Period
        ttk.Label(general_frame, text="Data Period:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        period_combo = ttk.Combobox(general_frame, textvariable=self.period, state="readonly", width=37)
        period_combo['values'] = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        period_combo.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Update Interval
        ttk.Label(general_frame, text="Update Interval (seconds):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(general_frame, textvariable=self.update_interval, width=10).grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create a frame for strategy selection
        strategy_frame = ttk.LabelFrame(parent, text="Strategy Selection")
        strategy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Strategy checkboxes
        ttk.Checkbutton(strategy_frame, text="Moving Average Crossover", variable=self.ma_crossover_enabled).grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Checkbutton(strategy_frame, text="RSI Strategy", variable=self.rsi_strategy_enabled).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Checkbutton(strategy_frame, text="Momentum Strategy", variable=self.momentum_strategy_enabled).grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Checkbutton(strategy_frame, text="Breakout Strategy (30+ bars)", variable=self.breakout_strategy_enabled).grid(
            row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Checkbutton(strategy_frame, text="Mean Reversion Strategy (30+ bars)", variable=self.mean_reversion_enabled).grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Checkbutton(strategy_frame, text="Dual Strategy System", variable=self.dual_strategy_enabled).grid(
            row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create a frame for advanced settings
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Settings")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Risk and Position Management
        ttk.Label(advanced_frame, text="Risk % per Trade:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.risk_percent, width=8).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(advanced_frame, text="Max Active Positions:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.max_positions, width=8).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(advanced_frame, text="Stop Loss %:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.stop_loss_pct, width=8).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(advanced_frame, text="Take Profit %:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.take_profit_pct, width=8).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Visualization settings
        ttk.Checkbutton(advanced_frame, text="Save Charts", variable=self.save_charts).grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(advanced_frame, text="Max Charts to Keep:").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.max_charts, width=8).grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Checkbutton(advanced_frame, text="Notifications", variable=self.notify_on_signals).grid(
            row=3, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Buttons to save and load settings
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Apply Settings", command=self.apply_settings).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Reset to Defaults", command=self.reset_settings).pack(side=tk.LEFT, padx=5, pady=5)
    
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
        """Start the trading bot."""
        if self.auto_trader and self.auto_trader.running:
            messagebox.showinfo("Already Running", "The bot is already running!")
            return
        
        # Get settings
        symbol_list = [s.strip() for s in self.symbols.get().split(',')]
        
        # Warn if too many symbols
        if len(symbol_list) > 10:
            if not messagebox.askyesno("Warning", 
                "You've selected more than 10 symbols, which may cause performance issues.\n\nProcessing many symbols at once can lead to high resource usage and potential freezing.\n\nDo you want to continue anyway?"):
                return
        
        try:
            # Initialize strategies based on checkboxes
            self._setup_strategies()
            
            # Initialize auto trader
            self.auto_trader = AutoTrader(
                symbols=symbol_list,
                interval=self.interval.get(),
                period=self.period.get(),
                update_interval=self.update_interval.get(),
                auto_trade=self.auto_trade_enabled.get(),
                market_regime=self.market_regime.get(),
                risk_pct=self.risk_percent.get(),
                max_active_positions=self.max_positions.get()
            )
            
            # Additional settings for risk management
            if hasattr(self.auto_trader, 'set_risk_parameters'):
                self.auto_trader.set_risk_parameters(
                    stop_loss_pct=self.stop_loss_pct.get(),
                    take_profit_pct=self.take_profit_pct.get()
                )
            
            # Start the auto trader
            self.auto_trader.start()
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Bot running...")
            
            # Start a thread to periodically update the UI with new signals
            self.ui_update_thread = threading.Thread(target=self.update_ui_from_bot)
            self.ui_update_thread.daemon = True
            self.ui_update_thread.start()
            
            logger.info(f"Bot started with symbols: {symbol_list}")
            
        except Exception as e:
            messagebox.showerror("Error Starting Bot", f"Error: {str(e)}")
            logger.error(f"Error starting bot: {str(e)}")
    
    def stop_bot(self):
        """Stop the trading bot."""
        if not self.auto_trader or not self.auto_trader.running:
            return
        
        try:
            # Stop the auto trader
            self.auto_trader.stop()
            
            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Bot stopped")
            
            logger.info("Bot stopped")
            
        except Exception as e:
            messagebox.showerror("Error Stopping Bot", f"Error: {str(e)}")
            logger.error(f"Error stopping bot: {str(e)}")
    
    def update_ui_from_bot(self):
        """Update the UI with data from the auto trader."""
        while self.auto_trader and self.auto_trader.running:
            try:
                # Get signals from auto trader
                signals = self.auto_trader.get_signals()
                
                # Update UI with signals
                if signals:
                    self.after(0, lambda: self.update_signal_display(signals))
                
                # Update positions and performance if auto trading is enabled
                if self.auto_trader and self.auto_trader.auto_trade:
                    self.after(0, self.refresh_positions_and_performance)
                
                # Update charts
                self.after(0, self.update_available_charts)
                
                # Wait a bit before next update
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error updating UI from bot: {str(e)}")
    
    def update_signal_display(self, signals=None):
        """Update the signal display treeview."""
        # Clear the treeview
        for i in self.signal_tree.get_children():
            self.signal_tree.delete(i)
        
        # Get signals to display
        if signals is None and self.auto_trader:
            signals = self.auto_trader.get_signals()
        
        # If we have signals, display them
        if not signals:
            return
            
        # Add signals to the treeview
        for symbol, strategies in signals.items():
            for strategy_name, info in strategies.items():
                # Set row color based on signal
                signal_text = info.get('signal_text', 'HOLD')
                tag = signal_text.lower()
                
                self.signal_tree.insert('', 'end', values=(
                    symbol,
                    f"${info.get('price', 0):.2f}",
                    signal_text,
                    strategy_name,
                    info.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                ), tags=(tag,))
        
        # Configure tag colors
        self.signal_tree.tag_configure('buy', background='#d0f0d0')  # Light green
        self.signal_tree.tag_configure('sell', background='#f0d0d0')  # Light red
        self.signal_tree.tag_configure('hold', background='#f0f0d0')  # Light yellow
    
    def refresh_positions_and_performance(self):
        """Refresh the positions and performance displays."""
        if not self.auto_trader or not self.auto_trader.trader:
            self.status_var.set("Auto-trading not enabled")
            return
            
        try:
            # Clear the positions treeview
            for i in self.positions_tree.get_children():
                self.positions_tree.delete(i)
                
            # Get current positions
            positions = self.auto_trader.get_positions()
            
            # Add positions to the treeview
            for symbol, info in positions.items():
                pl_pct = info.get('unrealized_plpc', 0) * 100
                tag = 'profit' if pl_pct > 0 else 'loss'
                
                self.positions_tree.insert('', 'end', values=(
                    symbol,
                    info.get('qty', 0),
                    f"${info.get('avg_entry_price', 0):.2f}",
                    f"${info.get('current_price', 0):.2f}",
                    f"${info.get('unrealized_pl', 0):.2f}",
                    f"{pl_pct:.2f}%"
                ), tags=(tag,))
            
            # Configure tag colors
            self.positions_tree.tag_configure('profit', background='#d0f0d0')  # Light green for profit
            self.positions_tree.tag_configure('loss', background='#f0d0d0')    # Light red for loss
            
            # Clear the performance treeview
            for i in self.performance_tree.get_children():
                self.performance_tree.delete(i)
                
            # Get performance metrics
            performance = self.auto_trader.get_performance()
            
            if performance:
                # Add performance to the treeview
                for strategy, metrics in performance.items():
                    win_rate = metrics.get('win_rate', 0)
                    tag = 'good' if win_rate > 50 else 'bad'
                    
                    self.performance_tree.insert('', 'end', values=(
                        strategy,
                        metrics.get('trades', 0),
                        metrics.get('wins', 0),
                        metrics.get('losses', 0),
                        f"{win_rate:.1f}%",
                        f"${metrics.get('profit_loss', 0):.2f}"
                    ), tags=(tag,))
                
                # Configure tag colors
                self.performance_tree.tag_configure('good', background='#d0f0d0')  # Light green for good
                self.performance_tree.tag_configure('bad', background='#f0d0d0')    # Light red for bad
            
            self.status_var.set("Positions and performance updated")
            
        except Exception as e:
            logger.error(f"Error refreshing positions and performance: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def refresh_data(self):
        """Refresh data for the currently configured symbols."""
        # If the bot is running, it will update automatically
        if self.auto_trader and self.auto_trader.running:
            messagebox.showinfo("Bot Running", "The bot is already running and updating data automatically.")
            return
            
        symbol_list = [s.strip() for s in self.symbols.get().split(',')]
        
        # Warn if too many symbols
        if len(symbol_list) > 10:
            if not messagebox.askyesno("Warning", 
                "You've selected more than 10 symbols, which may cause performance issues.\n\nDo you want to continue anyway?"):
                return
        
        # Update the status
        self.status_var.set("Refreshing data...")
        
        try:
            # Initialize strategies based on checkboxes
            self._setup_strategies()
            
            # Create a temporary AutoTrader just for data refresh
            temp_trader = AutoTrader(
                symbols=symbol_list,
                interval=self.interval.get(),
                period=self.period.get(),
                auto_trade=False
            )
            
            # Process symbols once
            temp_trader._process_symbols()
            
            # Get the signals
            signals = temp_trader.get_signals()
            
            # Update the UI
            self.update_signal_display(signals)
            
            # Update available charts
            self.update_available_charts()
            
            self.status_var.set("Data refresh complete")
            
        except Exception as e:
            messagebox.showerror("Error Refreshing Data", f"Error: {str(e)}")
            logger.error(f"Error refreshing data: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def apply_settings(self):
        """Apply the general settings."""
        if self.auto_trader and self.auto_trader.running:
            messagebox.showinfo("Bot Running", "Please stop the bot before changing settings.")
            return
        
        # Validate settings
        try:
            # Validate numeric settings
            risk_pct = float(self.risk_percent.get())
            if risk_pct <= 0 or risk_pct > 100:
                raise ValueError("Risk percentage must be between 0 and 100")
                
            max_pos = int(self.max_positions.get())
            if max_pos <= 0:
                raise ValueError("Maximum positions must be greater than 0")
                
            stop_loss = float(self.stop_loss_pct.get())
            if stop_loss <= 0 or stop_loss > 100:
                raise ValueError("Stop loss percentage must be between 0 and 100")
                
            take_profit = float(self.take_profit_pct.get())
            if take_profit <= 0 or take_profit > 100:
                raise ValueError("Take profit percentage must be between 0 and 100")
                
            max_charts = int(self.max_charts.get())
            if max_charts < 1:
                raise ValueError("Maximum charts must be at least 1")
            
            # Validate symbol list
            symbol_list = [s.strip() for s in self.symbols.get().split(',')]
            if not symbol_list:
                raise ValueError("At least one symbol must be specified")
            
            # Setup strategies based on enabled checkboxes
            self._setup_strategies()
            
            # Update visualization settings
            self.visualizer.max_figures = max_charts
            
            messagebox.showinfo("Settings", "Settings applied successfully. They will take effect when you restart the bot.")
            logger.info("Settings applied successfully")
            
        except ValueError as e:
            messagebox.showerror("Invalid Settings", str(e))
            logger.error(f"Invalid settings: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying settings: {str(e)}")
            logger.error(f"Error applying settings: {str(e)}")
    
    def apply_auto_trade_settings(self):
        """Apply the auto trading settings."""
        if self.auto_trader and self.auto_trader.running:
            messagebox.showinfo("Bot Running", "Please stop the bot before changing settings.")
            return
            
        messagebox.showinfo("Auto Trading Settings", 
                           f"Auto trading {'enabled' if self.auto_trade_enabled.get() else 'disabled'}, " +
                           f"using {self.market_regime.get()} market regime, " +
                           f"{self.risk_percent.get()}% risk, " +
                           f"max {self.max_positions.get()} positions.")
        logger.info(f"Auto trading settings updated: enabled={self.auto_trade_enabled.get()}, " +
                   f"regime={self.market_regime.get()}, risk={self.risk_percent.get()}%, " +
                   f"max_positions={self.max_positions.get()}")
    
    def reset_settings(self):
        """Reset settings to defaults."""
        self.symbols.set("AAPL,MSFT,GOOGL,AMZN,META")
        self.strategy.set("MA Crossover")
        self.interval.set("1d")
        self.period.set("1mo")
        self.update_interval.set(60)
        self.market_regime.set("auto")
        self.risk_percent.set(0.02)
        self.max_positions.set(5)
        self.auto_trade_enabled.set(False)
        
        # Reset strategy enable/disable variables
        self.ma_crossover_enabled.set(True)
        self.rsi_strategy_enabled.set(True)
        self.momentum_strategy_enabled.set(True)
        self.breakout_strategy_enabled.set(False)
        self.mean_reversion_enabled.set(False)
        self.dual_strategy_enabled.set(True)
        
        # Reset advanced settings
        self.notify_on_signals.set(False)
        self.save_charts.set(True)
        self.max_charts.set(30)
        self.stop_loss_pct.set(5.0)
        self.take_profit_pct.set(10.0)
        
        messagebox.showinfo("Settings", "Settings reset to defaults.")
        logger.info("Settings reset to defaults")
    
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
            
            if label_width <= 1:  # Not yet realized
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

    def _setup_strategies(self):
        """Set up trading strategies based on enabled settings."""
        # Clear any existing strategies
        self.strategy_manager = StrategyManager()
        
        # Add enabled strategies
        if self.ma_crossover_enabled.get():
            self.strategy_manager.add_strategy(EnhancedMovingAverageCrossover(short_window=20, long_window=50))
            
        if self.rsi_strategy_enabled.get():
            self.strategy_manager.add_strategy(EnhancedRSIStrategy(period=14, overbought=70, oversold=30))
            
        if self.momentum_strategy_enabled.get():
            self.strategy_manager.add_strategy(EnhancedMomentumStrategy(period=10, threshold=5.0))
            
        if self.breakout_strategy_enabled.get():
            self.strategy_manager.add_strategy(BreakoutStrategy())
            
        if self.mean_reversion_enabled.get():
            self.strategy_manager.add_strategy(MeanReversionStrategy())
            
        if self.dual_strategy_enabled.get():
            self.strategy_manager.add_strategy(DualStrategySystem())
        
        if not self.strategy_manager.strategies:
            # Ensure at least one strategy is enabled
            logger.warning("No strategies enabled, enabling MA Crossover as default")
            self.ma_crossover_enabled.set(True)
            self.strategy_manager.add_strategy(EnhancedMovingAverageCrossover(short_window=20, long_window=50))
        
        logger.info(f"Initialized {len(self.strategy_manager.strategies)} trading strategies")

if __name__ == "__main__":
    app = AutoTradingInterface()
    app.mainloop() 