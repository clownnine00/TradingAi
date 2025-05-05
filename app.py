import os
import pickle
import streamlit as st
import pandas as pd
from random_forest_strategy import RandomForestStrategy
from mt5_connector import MT5Connector
from trade_executor import TradeExecutor
from performance_analyzer import PerformanceAnalyzer
from sklearn.ensemble import RandomForestClassifier

# Initialize Streamlit session state variables
if "connector" not in st.session_state:
    st.session_state.connector = None
if "executor" not in st.session_state:
    st.session_state.executor = None
if "analyzer" not in st.session_state:
    st.session_state.analyzer = PerformanceAnalyzer()
if "connected" not in st.session_state:
    st.session_state.connected = False
if "trading" not in st.session_state:
    st.session_state.trading = False
if "symbols" not in st.session_state:
    st.session_state.symbols = []

# Auto-refresh interval (in seconds)
REFRESH_INTERVAL = 5

# Load or train the model
def load_or_train_model():
    model_filename = 'random_forest_model.pkl'
    strategy = RandomForestStrategy()

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            strategy.model = pickle.load(f)
    else:
        st.warning("Model not found. Training a new model from default or generated data.")
        try:
            df = pd.read_csv("training_data.csv")
        except FileNotFoundError:
            import numpy as np
            df = pd.DataFrame({
                'scaled_close': np.random.rand(1000),
                'scaled_sma20': np.random.rand(1000),
                'scaled_macd': np.random.rand(1000),
                'target': np.random.choice([0, 1], size=1000)
            })
            st.info("Dummy training data used for model training.")
        strategy.train(df)
        st.success("Model trained successfully.")
        with open(model_filename, 'wb') as f:
            pickle.dump(strategy.model, f)
    return strategy

# Lazy load the strategy
strategy = load_or_train_model()

# Streamlit page configuration
st.set_page_config(page_title="TradingAi Dashboard", layout="wide")
st.title("📈 TradingAi Dashboard")

# Sidebar for MT5 connection and trading controls
with st.sidebar:
    st.header("🔌 Connector")

    if not st.session_state.connected:
        if st.button("Connect to MT5"):
            connector = MT5Connector()
            if connector.connect():
                st.session_state.connector = connector
                st.session_state.executor = TradeExecutor(connector)
                st.session_state.executor.set_strategy(strategy)
                st.session_state.executor.set_performance_analyzer(st.session_state.analyzer)
                st.session_state.symbols = connector.list_available_symbols()
                account_info = connector.get_account_info()
                st.success(f"Connected to MetaTrader 5\nBalance: ${account_info['balance']:.2f}")
                st.session_state.connected = True
            else:
                st.error("Failed to connect to MT5")
    else:
        if st.button("Disconnect"):
            try:
                if st.session_state.executor:
                    st.session_state.executor.stop_trading()
                if st.session_state.connector:
                    st.session_state.connector.disconnect()
                # Reset session state variables on disconnect
                st.session_state.connector = None
                st.session_state.executor = None
                st.session_state.connected = False
                st.session_state.symbols = []
                st.success("Disconnected from MT5 successfully.")
            except Exception as e:
                st.error(f"An error occurred while disconnecting: {str(e)}")

    # Trading controls
    if st.session_state.connected:
        st.header("⚙ Trade Setup")

        # Dropdown for available symbols with typing support
        symbol = st.selectbox("Search or Select Symbol", st.session_state.symbols)

        # Other trading parameters
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
        max_pos = st.slider("Max Open Positions", 1, 10, 3)

        if not st.session_state.trading:
            if st.button("Start Trading"):
                market_status = st.session_state.connector.is_market_open(symbol)
                if market_status == "invalid":
                    st.error(f"The symbol '{symbol}' is invalid. Please select a valid trading pair.")
                elif market_status == "not_subscribed":
                    st.error(f"The symbol '{symbol}' is not subscribed. Ensure it is visible in the Market Watch.")
                elif not market_status:
                    st.error(f"The market is closed for '{symbol}'. Please try again later.")
                else:
                    st.session_state.executor.start_trading(symbol, timeframe, risk / 100, max_pos)
                    st.success(f"Trading started for {symbol}.")
                    st.session_state.trading = True
        else:
            if st.button("Stop Trading"):
                st.session_state.executor.stop_trading()
                st.warning("Trading stopped.")
                st.session_state.trading = False

# Auto-refresh mechanism using st_autorefresh
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=REFRESH_INTERVAL * 1000, limit=None)

# Main dashboard for live updates
st.subheader("💰 Wallet and Open Trades")

if st.session_state.connected and st.session_state.connector:
    # Fetch wallet balance
    wallet_balance = st.session_state.connector.get_account_balance()
    if wallet_balance == 0.0:
        st.warning("Unable to retrieve wallet balance. Please check the MT5 connection.")
    else:
        st.metric("💰 Wallet Balance", f"${wallet_balance:.2f}")

    # Fetch open trades
    if st.session_state.executor:
        open_trades = [
            trade for trade in st.session_state.executor.trades if trade["status"] == "open"
        ]
        st.write("### 📂 Open Trades")
        open_trades_df = pd.DataFrame(open_trades)
        if not open_trades_df.empty:
            st.dataframe(open_trades_df)
        else:
            st.info("No open trades at the moment.")
else:
    st.warning("MT5 is disconnected. Please connect to view wallet balance and trades.")
