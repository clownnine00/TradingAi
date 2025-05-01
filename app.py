import os
import pickle
import streamlit as st
import pandas as pd
from random_forest_strategy import RandomForestStrategy
from data_processor import DataProcessor
from performance_analyzer import PerformanceAnalyzer
from trade_executor import TradeExecutor
from mt5_connector import MT5Connector
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

    # Connect to MT5
    if not st.session_state.connected:
        if st.button("Connect to MT5"):
            connector = MT5Connector()
            if connector.connect():
                st.session_state.connector = connector
                st.session_state.executor = TradeExecutor(connector)
                st.session_state.executor.set_strategy(strategy)
                st.session_state.executor.set_performance_analyzer(st.session_state.analyzer)
                account_info = connector.get_account_info()
                st.success(f"Connected to MetaTrader 5\nBalance: ${account_info['balance']:.2f}")
                st.session_state.connected = True
            else:
                st.error("Failed to connect to MT5")
    else:
        if st.button("Disconnect"):
            if st.session_state.executor:
                st.session_state.executor.stop_trading()
            if st.session_state.connector:
                st.session_state.connector.disconnect()
            st.session_state.connector = None
            st.session_state.executor = None
            st.session_state.connected = False
            st.warning("Disconnected")

    # Trading controls
    if st.session_state.connected:
        st.header("⚙ Trade Setup")
        symbol = st.text_input("Symbol", "EURUSD")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])
        risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
        max_pos = st.slider("Max Open Positions", 1, 10, 3)

        if not st.session_state.trading:
            if st.button("Start Trading"):
                if not st.session_state.connector.is_market_open(symbol):
                    st.error("Market is closed or symbol is invalid. Trading not started.")
                else:
                    st.session_state.executor.start_trading(symbol, timeframe, risk / 100, max_pos)
                    st.success("Trading started.")
                    st.session_state.trading = True
        else:
            if st.button("Stop Trading"):
                st.session_state.executor.stop_trading()
                st.warning("Trading stopped.")
                st.session_state.trading = False

# Main dashboard for performance metrics
st.subheader("📊 Performance Metrics")

if st.session_state.analyzer:
    metrics = st.session_state.analyzer.get_metrics()
    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
        col2.metric("📈 Total Trades", str(metrics.get("total_trades", 0)))
        col3.metric("💰 Net Profit", f"${metrics.get('net_profit', 0):.2f}")

        st.write("### 📈 Equity Curve")
        equity_curve_df = metrics.get("equity_curve", pd.DataFrame())
        if not equity_curve_df.empty:
            st.line_chart(equity_curve_df)

        st.write("### 📅 Trade History")
        trade_df = pd.DataFrame(metrics.get("trade_history", []))
        if not trade_df.empty:
            st.dataframe(trade_df)
    else:
        st.info("No performance metrics available yet. Start trading to generate data.")
