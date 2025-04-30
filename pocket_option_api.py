import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PocketOptionAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.session = None
        self.connected = False
        self.symbols = []
        self.base_url = "https://api.pocketoption.com/v1"
        self.token = None

    def connect(self):
        if not self.api_key:
            logger.error("API key is required for API connection")
            return False

        try:
            self.session = requests.Session()
            retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

            auth_url = f"{self.base_url}/auth"
            response = self.session.post(auth_url, json={"api_key": self.api_key}, timeout=60)

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.token = data.get("token")
                    self.connected = True
                    logger.info("Authenticated with Pocket Option API")
                    self.fetch_symbols()
                    return True
            logger.error("Authentication failed")
            return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def disconnect(self):
        if self.session:
            try:
                if self.token:
                    self.session.post(f"{self.base_url}/logout", headers={"Authorization": f"Bearer {self.token}"}, timeout=30)
                self.session.close()
                self.connected = False
                logger.info("Disconnected from Pocket Option API")
                return True
            except Exception as e:
                logger.warning(f"Disconnection error: {e}")
        return False

    def fetch_symbols(self):
        if not self.connected:
            return []

        try:
            response = self.session.get(f"{self.base_url}/symbols", headers={"Authorization": f"Bearer {self.token}"}, timeout=30)
            data = response.json()
            if data.get("success"):
                self.symbols = data.get("symbols", [])
                logger.info(f"Fetched {len(self.symbols)} symbols")
                return self.symbols
        except Exception as e:
            logger.error(f"Symbol fetch failed: {e}")

        self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD"]
        return self.symbols

    def get_historical_data(self, symbol, timeframe, start_date, end_date=None):
        if not self.connected:
            return None

        if end_date is None:
            end_date = datetime.now()

        try:
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "from": int(start_date.timestamp()),
                "to": int(end_date.timestamp())
            }
            response = self.session.get(f"{self.base_url}/history", headers={"Authorization": f"Bearer {self.token}"}, params=params, timeout=45)
            data = response.json()
            if data.get("success") and data.get("candles"):
                candles = data["candles"]
                df = pd.DataFrame([{
                    'time': datetime.fromtimestamp(c['timestamp']),
                    'open': float(c['open']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'close': float(c['close']),
                    'volume': int(c.get('volume', 0))
                } for c in candles])
                df.set_index("time", inplace=True)
                return df
        except Exception as e:
            logger.error(f"Failed to get history: {e}")

        return None

    def place_order(self, symbol, order_type, amount, expiration=60):
        if not self.connected:
            return None

        try:
            payload = {
                "symbol": symbol,
                "direction": order_type,
                "amount": amount,
                "expiration": expiration
            }
            response = self.session.post(f"{self.base_url}/trade", headers={"Authorization": f"Bearer {self.token}"}, json=payload, timeout=30)
            data = response.json()
            if data.get("success") and data.get("order"):
                order = data["order"]
                logger.info(f"{order_type} order placed on {symbol}")
                return {
                    'order_id': order['id'],
                    'symbol': order['symbol'],
                    'direction': order['direction'],
                    'amount': order['amount'],
                    'entry_price': order['entry_price'],
                    'expiration': order['expiration']
                }
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
        return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            response = self.session.get(f"{self.base_url}/account", headers={"Authorization": f"Bearer {self.token}"}, timeout=30)
            data = response.json()
            if data.get("success") and data.get("account"):
                return data["account"]
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
        return None
