import MetaTrader5 as mt5
import time
import threading
from datetime import datetime

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
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"⚠️ Symbol {symbol} not found.")
            return False
        return info.session_deals > 0 or info.session_buy_orders > 0

    def get_account_info(self):
        account_info = mt5.account_info()
        if account_info is None:
            raise ValueError("Failed to retrieve account information")
        return {
            "login": account_info.login,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "margin_free": account_info.margin_free,
            "margin_level": account_info.margin_level,
            "profit": account_info.profit
        }

    def get_account_balance(self):
        info = self.get_account_info()
        return info["balance"] if info else 0.0

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
