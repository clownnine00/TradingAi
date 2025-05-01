import pandas as pd
import numpy as np
import time
import threading
from performance_analyzer import PerformanceAnalyzer

class TradeExecutor:
    """Executes and manages trades based on strategy signals"""

    def __init__(self, connector):
        self.connector = connector
        self.strategy = None
        self.performance_analyzer = PerformanceAnalyzer()  # Automatically initialize
        self.trading_thread = None
        self.monitor_thread = None
        self.trading_interval = 5  # seconds
        self.max_positions = 3
        self.risk_per_trade = 0.01  # 1% of balance
        self.symbol = None
        self.timeframe = None
        self.trades = []
        self.last_signal_time = 0
        self.stop_flag = threading.Event()

    def set_strategy(self, strategy):
        """Sets the trading strategy"""
        self.strategy = strategy

    def set_performance_analyzer(self, analyzer):
        """Sets the performance analyzer for tracking trades"""
        self.performance_analyzer = analyzer

    def start_trading(self, symbol, timeframe, risk_per_trade=None, max_positions=None):
        """Starts the trading loop and reversal monitoring"""
        self.symbol = symbol
        self.timeframe = timeframe
        self.stop_flag.clear()

        if risk_per_trade:
            self.risk_per_trade = risk_per_trade
        if max_positions:
            self.max_positions = max_positions

        self.trading_thread = threading.Thread(target=self._trade_loop, daemon=True)
        self.trading_thread.start()

    def stop_trading(self):
        """Stops the trading loop and monitoring"""
        self.stop_flag.set()
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        print("✅ Trading loop stopped.")

    def _trade_loop(self):
        """The main trading loop where trades are executed based on strategy signals"""
        print(f"🚀 Trading started on {self.symbol} | Timeframe: {self.timeframe}")
        while not self.stop_flag.is_set():
            try:
                if self._should_trade():
                    if not self.connector.is_market_open(self.symbol):
                        print(f"📛 Market closed for {self.symbol}. Waiting...")
                        time.sleep(self.trading_interval)
                        continue

                    signal = self.strategy.generate_signal(self.symbol, self.timeframe)
                    if signal:
                        self._execute_trade(signal)

                time.sleep(self.trading_interval)
            except Exception as e:
                print(f"⚠️ Trade loop error: {e}")
        print("🛑 Trading loop exited.")

    def _should_trade(self):
        """Checks if a new trade should be executed"""
        return (
            len(self.trades) < self.max_positions and
            (time.time() - self.last_signal_time) >= self.trading_interval
        )

    def _execute_trade(self, signal):
        """Executes the trade based on the generated signal"""
        size = self._calculate_position_size()

        if signal['type'] == 'buy':
            trade = self.connector.buy(self.symbol, size)
        elif signal['type'] == 'sell':
            trade = self.connector.sell(self.symbol, size)
        else:
            print("⚠️ Invalid trade signal type.")
            return

        if trade:
            self.trades.append(trade)
            self.last_signal_time = time.time()

            if self.performance_analyzer:
                self.performance_analyzer.add_trade(trade)

    def _calculate_position_size(self):
        """Calculates the position size based on risk per trade and account balance"""
        balance = self.connector.get_account_balance()
        price = self.connector.get_current_price(self.symbol)
        risk_amount = balance * self.risk_per_trade
        size = risk_amount / price if price > 0 else 0
        return round(size, 2)

    def get_trading_status(self):
        return {
            'active': not self.stop_flag.is_set(),
            'total_trades': len(self.trades),
            'open_positions': len([t for t in self.trades if t['status'] == 'open']),
            'closed_positions': len([t for t in self.trades if t['status'] == 'closed'])
        }
