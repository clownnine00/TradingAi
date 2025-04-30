import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PocketOption")

class PocketOptionBrowser:
    def __init__(self, email=None, password=None):
        self.email = email
        self.password = password
        self.driver = None
        self.connected = False
        self.symbols = []
        self.login_url = "https://pocketoption.com/en/login/"
        self.trading_url = "https://pocketoption.com/en/cabinet/turbo-options/"

    def connect(self):
        if not self.email or not self.password:
            logger.error("Email and password are required.")
            return False

        logger.info("Launching browser...")
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')

            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            wait = WebDriverWait(self.driver, 20)
            self.driver.get(self.login_url)

            wait.until(EC.presence_of_element_located((By.ID, "email")))
            self.driver.find_element(By.ID, "email").send_keys(self.email)
            self.driver.find_element(By.ID, "password").send_keys(self.password)
            self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

            wait.until(EC.url_contains("/cabinet"))
            self.driver.get(self.trading_url)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "asset-select")))

            self.fetch_symbols()
            self.connected = True
            logger.info("Connected and logged in.")
            return True

        except TimeoutException:
            logger.error("Login timeout. Check credentials or site status.")
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")

        self.disconnect()
        return False

    def disconnect(self):
        try:
            if self.driver:
                self.driver.get("https://pocketoption.com/en/logout/")
                self.driver.quit()
                logger.info("Browser disconnected.")
        except Exception as e:
            logger.warning(f"Disconnection error: {str(e)}")
        finally:
            self.driver = None
            self.connected = False

    def fetch_symbols(self):
        if not self.connected:
            return []

        try:
            self.driver.find_element(By.CLASS_NAME, "asset-select").click()
            time.sleep(1)
            asset_items = self.driver.find_elements(By.CSS_SELECTOR, ".asset-list .asset-item")
            self.symbols = [item.text.strip() for item in asset_items if item.text.strip()]
            self.driver.find_element(By.CLASS_NAME, "trading-container").click()

            if not self.symbols:
                self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD"]
                logger.warning("No symbols found. Using default list.")
            else:
                logger.info(f"{len(self.symbols)} symbols loaded.")
        except Exception as e:
            logger.error(f"Symbol fetching failed: {str(e)}")
            self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD", "ETH/USD"]

        return self.symbols

    def get_symbols(self):
        return self.symbols if self.symbols else self.fetch_symbols()

    def get_timeframes(self):
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    def _convert_timeframe_to_minutes(self, tf):
        return {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080
        }.get(tf, 1)

    def get_historical_data(self, symbol, timeframe, start_date, end_date=None):
        logger.warning("Historical data via browser not implemented. Use API instead.")
        return None

    def place_order(self, symbol, order_type, amount, expiration=60):
        if not self.connected:
            return None

        try:
            wait = WebDriverWait(self.driver, 10)
            if self.trading_url not in self.driver.current_url:
                self.driver.get(self.trading_url)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "asset-select")))

            # Select symbol
            self.driver.find_element(By.CLASS_NAME, "asset-select").click()
            time.sleep(1)
            for el in self.driver.find_elements(By.CSS_SELECTOR, ".asset-list .asset-item"):
                if el.text.strip() == symbol:
                    el.click()
                    break
            self.driver.find_element(By.CLASS_NAME, "trading-container").click()

            # Set amount
            amount_input = self.driver.find_element(By.CSS_SELECTOR, "input[name='amount']")
            amount_input.clear()
            amount_input.send_keys(str(amount))

            # Choose CALL or PUT
            btn_class = "btn-up" if order_type.upper() == "CALL" else "btn-down"
            self.driver.find_element(By.CLASS_NAME, btn_class).click()

            logger.info(f"Placed {order_type.upper()} order for {symbol} with ${amount}")
            return {"symbol": symbol, "order": order_type, "amount": amount}

        except Exception as e:
            logger.error(f"Order placement failed: {str(e)}")
            return None
