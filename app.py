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

# Initialize core components
connector = MT5Connector()
processor = DataProcessor()
strategy = RandomForestStrategy()
analyzer = PerformanceAnalyzer()
executor = TradeExecutor(connector)

# Automatically load or train the model
model_filename = 'random_forest_model.pkl'
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

# Assign strategy and analyzer to executor
executor.set_strategy(strategy)

# Streamlit UI
st.set_page_config(page_title="TradingAi Dashboard", layout="wide")
st.title("📈 TradingAi Dashboard")

with st.sidebar:
    st.header("🔌 Connector")
    if st.button("Connect to MT5"):
        if connector.connect():
            account_info = connector.get_account_info()
            st.success(f"Connected to MetaTrader 5\nBalance: ${account_info['balance']:.2f}")
        else:
            st.error("Connection failed")

    if st.button("Disconnect"):
        connector.disconnect()
        st.warning("Disconnected")

    st.header("⚙ Trade Setup")
    symbol = st.text_input("Symbol", "EURUSD")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])
    risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
    max_pos = st.slider("Max Open Positions", 1, 10, 3)

    if st.button("Start Trading"):
        if not connector.is_market_open(symbol):
            st.error("Market is closed or symbol is invalid.")
        else:
            executor.start_trading(symbol, timeframe, risk / 100, max_pos)
            st.success("Trading started.")

    if st.button("Stop Trading"):
        executor.stop_trading()
        st.warning("Trading stopped.")

# 📊 Performance Metrics Section
st.subheader("📊 Performance Metrics")

metrics = analyzer.get_metrics()
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

