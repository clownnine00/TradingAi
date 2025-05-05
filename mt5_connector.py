import MetaTrader5 as mt5
import time
from datetime import datetime

class MT5Connector:
    """Upgraded MT5 Connector with auto-reconnect, better logging & safe checks"""

    def __init__(self):
        self.connected = False

    def connect(self, retries=3, delay=2):
        """Connect to MetaTrader 5 with retry logic."""
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
        """Disconnect from MetaTrader 5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("🔌 Disconnected from MT5")

    def get_mt5_server_time(self):
        """
        Get the current server time from MetaTrader 5.
        Since the MT5 Python API does not provide a direct method for server time,
        we assume the local system time is synchronized with the server time.
        """
        print("⚠️ Using local time as a proxy for MT5 server time.")
        return datetime.now()

    def is_market_open(self, symbol):
        """Check if the market for a specific symbol is open."""
        if not self.connected:
            print("❌ Not connected to MT5. Cannot check market status.")
            return False

        # Ensure the symbol is visible in Market Watch
        if not mt5.symbol_select(symbol, True):
            print(f"⚠️ Failed to select symbol: {symbol}. Ensure it is available in Market Watch.")
            return "not_subscribed"

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"⚠️ Invalid symbol: {symbol}. Please check the symbol and try again.")
            return "invalid"

        # Debugging: Log the full symbol info
        print(f"🔍 Symbol Info for {symbol}: {symbol_info}")

        # Use MT5 server time for market checks
        now = self.get_mt5_server_time()

        # Debugging Tip
        print("🕒 Server Time (MT5):", now)

        # Check if market is closed based on time (e.g., weekends)
        if now.weekday() >= 5:  # Saturday or Sunday
            print(f"📛 Market time window closed for {symbol}.")
            return "market_closed"

        # Check if the symbol is allowed to trade
        if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            print(f"📛 Trading is disabled for {symbol}. Trade mode: {symbol_info.trade_mode}")
            return "market_closed"

        # Check trading session for the symbol
        if symbol_info.session_deals == 0 and symbol_info.session_buy_orders == 0:
            print(f"📛 Market session might not be fully active for {symbol}.")
            return "market_closed"

        # Market is open
        print(f"✅ Market is open for {symbol}.")
        return True

    def get_account_info(self):
        """Retrieve account information from MetaTrader 5."""
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
        """Get the account balance."""
        info = self.get_account_info()
        return info["balance"] if info else 0.0

    def get_current_price(self, symbol):
        """Get the current price for a specific symbol."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"⚠️ Could not retrieve price for {symbol}")
            return 0.0
        return tick.ask

    def buy(self, symbol, volume):
        """Place a buy order for a specific symbol and volume."""
        return self._send_order(symbol, volume, action="BUY")

    def sell(self, symbol, volume):
        """Place a sell order for a specific symbol and volume."""
        return self._send_order(symbol, volume, action="SELL")

    def _send_order(self, symbol, volume, action="BUY"):
        """Send an order to MetaTrader 5."""
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

        # Validate price
        if price == 0:
            print(f"⚠️ Retrieved price is 0 for {symbol}. Cannot proceed with trade.")
            return None

        # Example debug log for volume
        print(f"🧮 Calculated lot size: {volume} for symbol {symbol}")

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

    def list_available_symbols(self):
        """Fetch and return a list of all available symbols in the MT5 terminal."""
        if not self.connected:
            print("⚠️ Not connected to MT5. Cannot fetch symbols.")
            return []

        symbols = mt5.symbols_get()
        if not symbols:
            print("⚠️ No symbols found in the MT5 terminal.")
            return []

        print(f"✅ {len(symbols)} symbols available in the MT5 terminal.")
        return [symbol.name for symbol in symbols]
