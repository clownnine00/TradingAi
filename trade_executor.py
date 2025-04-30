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
        self.monitor_thread = threading.Thread(target=self._monitor_reversals, daemon=True)

        self.trading_thread.start()
        self.monitor_thread.start()

    def stop_trading(self):
        """Stops the trading loop and monitoring"""
        self.stop_flag.set()
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        print("✅ Trading loop and monitoring stopped.")

    def _trade_loop(self):
        """The main trading loop where trades are executed based on strategy signals"""
        print(f"🚀 Trading started on {self.symbol} | Timeframe: {self.timeframe}")
        while not self.stop_flag.is_set():
            if self._should_trade():
                try:
                    signal = self.strategy.generate_signal(self.symbol, self.timeframe)
                    if signal:
                        self._execute_trade(signal)
                except Exception as e:
                    print(f"⚠️ Error in strategy or execution: {e}")
            time.sleep(self.trading_interval)
        print("🛑 Trading loop exited.")

    def _should_trade(self):
        """Checks if a new trade should be executed"""
        if len([t for t in self.trades if t['status'] == 'open']) >= self.max_positions:
            return False
        if time.time() - self.last_signal_time < self.trading_interval:
            return False
        return True

    def _execute_trade(self, signal):
        """Executes the trade based on the generated signal"""
        size = self._calculate_position_size()

        if signal['type'] == 'buy':
            trade = self.connector.buy(self.symbol, size)
        elif signal['type'] == 'sell':
            trade = self.connector.sell(self.symbol, size)
        else:
            return

        if trade:
            trade['status'] = 'open'
            trade['type'] = signal['type']
            trade['entry_time'] = time.time()
            trade['entry_price'] = self.connector.get_current_price(self.symbol)
            self.trades.append(trade)

            self.performance_analyzer.add_trade(trade)

            self.last_signal_time = time.time()

    def _calculate_position_size(self):
        """Calculates the position size based on risk per trade and account balance"""
        balance = self.connector.get_account_balance()
        price = self.connector.get_current_price(self.symbol)
        risk_amount = balance * self.risk_per_trade
        return risk_amount / price if price > 0 else 0

    def _monitor_reversals(self):
        """Monitors market reversals and closes trades accordingly"""
        print("🧠 Reversal monitoring started.")
        while not self.stop_flag.is_set():
            try:
                signal = self.strategy.generate_signal(self.symbol, self.timeframe)
                current_type = signal['type'] if signal else None

                for trade in self.trades:
                    if trade['status'] == 'open':
                        if current_type and trade['type'] != current_type:
                            print(f"🔄 Reversal detected. Closing trade: {trade['type']} → {current_type}")
                            self._close_trade(trade)
            except Exception as e:
                print(f"⚠️ Error in reversal monitoring: {e}")
            time.sleep(self.trading_interval)

    def _close_trade(self, trade):
        """Closes an open trade due to market reversal"""
        result = self.connector.close_trade(self.symbol, trade)
        if result:
            trade['status'] = 'closed'
            trade['exit_time'] = time.time()
            trade['exit_price'] = self.connector.get_current_price(self.symbol)

            # Update performance stats
            self.performance_analyzer.update_trade(trade)

            print(f"✅ Trade closed due to reversal on {self.symbol}")
        else:
            print("❌ Failed to close trade")

    def get_trading_status(self):
        """Returns the current status of the trading activity"""
        return {
            'active': not self.stop_flag.is_set(),
            'total_trades': len(self.trades),
            'active_positions': len([t for t in self.trades if t['status'] == 'open']),
            'completed_trades': len([t for t in self.trades if t['status'] == 'closed']),
            'performance': self.performance_analyzer.get_summary()
        }

