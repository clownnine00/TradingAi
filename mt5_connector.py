import MetaTrader5 as mt5
import time
import threading
from datetime import datetime


# ----- MT5 Connector -----
class MT5Connector:
    """Upgraded MT5 Connector with auto-reconnect, better logging & safe checks"""

    def __init__(self):
        self.connected = False

    def connect(self, retries=3, delay=2):
        for attempt in range(retries):
            try:
                if not mt5.initialize():
                    print(f"❌ MT5 init failed [{attempt+1}/{retries}]: {mt5.last_error()}")
                    time.sleep(delay)
                    continue

                account_info = self.get_account_info()
                if account_info is None:
                    print(f"❌ Could not get account info [{attempt+1}/{retries}]: {mt5.last_error()}")
                    time.sleep(delay)
                    continue

                self.connected = True
                print(f"✅ Connected to MT5 successfully at {datetime.now().strftime('%H:%M:%S')}")
                return True

            except Exception as e:
                print(f"⚠️ Exception during MT5 connection [{attempt+1}/{retries}]: {e}")
                time.sleep(delay)

        print("❌ All connection attempts failed.")
        return False

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("🔌 Disconnected from MT5")

    def is_market_open(self, symbol):
        # Updated to use symbol_info.trade_time_flags or an alternative check for market status
        info = mt5.symbol_info(symbol)
        if info and info.trade_time_flags is not None:
            return info.trade_time_flags != 0
        return False

    def get_account_info(self):
        account_info = mt5.account_info()
        if account_info is None:
            raise ValueError("Failed to retrieve account information")
        return account_info

    def get_account_balance(self):
        info = self.get_account_info()
        return info.balance if info else 0.0

    def get_current_price(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        return tick.ask if tick else 0.0

    def buy(self, symbol, volume):
        return self._send_order(symbol, volume, action="BUY")

    def sell(self, symbol, volume):
        return self._send_order(symbol, volume, action="SELL")

    def _send_order(self, symbol, volume, action="BUY"):
        if not self.connected:
            print("❌ Not connected to MT5.")
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Symbol {symbol} not found.")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Failed to select symbol: {symbol}")
                return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"⚠️ Could not retrieve price for {symbol}")
            return None

        price = tick.ask if action == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "price": price,
            "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
            "deviation": 10,
            "magic": 123456,
            "comment": "TradingAi",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ {action} order placed for {symbol} at {price:.5f} [{datetime.now().strftime('%H:%M:%S')}]")
            return {
                'symbol': symbol,
                'type': action.lower(),
                'entry_price': price,
                'size': volume,
                'status': 'open',
                'time': datetime.now()
            }
        else:
            print(f"❌ Order failed [{result.retcode}]: {result.comment}")
            return None


# ----- Performance Analyzer -----
class PerformanceAnalyzer:
    """Tracks trade performance for analysis"""

    def __init__(self):
        self.trades = []

    def add_trade(self, trade):
        self.trades.append(trade)

    def analyze_performance(self):
        if not self.trades:
            return "No trades executed yet."

        total_profit = sum([trade['size'] * (trade['entry_price'] - trade['exit_price']) for trade in self.trades if trade['status'] == 'closed'])
        total_trades = len(self.trades)

        return f"Total Profit: {total_profit:.2f} | Total Trades: {total_trades}"


# ----- Trade Executor -----
class TradeExecutor:
    """Executes trades based on strategy with upgrades & risk management"""

    def __init__(self, connector):
        self.connector = connector
        self.strategy = None
        self.performance_analyzer = None
        self.trading_thread = None
        self.trading_interval = 5  # seconds
        self.max_positions = 3
        self.risk_per_trade = 0.01  # 1% of balance
        self.symbol = None
        self.timeframe = None
        self.trades = []
        self.last_signal_time = 0
        self.stop_flag = threading.Event()

    def set_strategy(self, strategy):
        self.strategy = strategy

    def set_performance_analyzer(self, analyzer):
        self.performance_analyzer = analyzer

    def start_trading(self, symbol, timeframe, risk_per_trade=None, max_positions=None):
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
        self.stop_flag.set()
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        print("✅ Trading loop stopped.")

    def _trade_loop(self):
        print(f"🚀 Trading started on {self.symbol} | TF: {self.timeframe} | Risk: {self.risk_per_trade*100}%")
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
        return (
            len(self.trades) < self.max_positions and
            (time.time() - self.last_signal_time) >= self.trading_interval
        )

    def _execute_trade(self, signal):
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


# Example of how to use it
connector = MT5Connector()

# Connect to MT5
if connector.connect():
    analyzer = PerformanceAnalyzer()
    trade_executor = TradeExecutor(connector)
    trade_executor.set_performance_analyzer(analyzer)

    # Define a strategy here or use any existing one
    # Example: trade_executor.set_strategy(some_strategy)

    # Start trading
    trade_executor.start_trading("EURUSD", "M1", risk_per_trade=0.01, max_positions=3)

    # Stop trading after a while
    time.sleep(60)
    trade_executor.stop_trading()

    # Performance analysis
    print(analyzer.analyze_performance())
else:
    print("❌ Failed to connect to MT5.")


