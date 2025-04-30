import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from base_strategy import BaseStrategy
import pickle
from datetime import datetime, timedelta
import time
import MetaTrader5 as mt5
from data_processor import DataProcessor

class RandomForestStrategy(BaseStrategy):
    """Random Forest Strategy for market analysis with periodic retraining"""

    def __init__(self):
        super().__init__("Random Forest Strategy", "Random Forest trading strategy using indicators")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_trained = None
        self.model_save_interval = timedelta(minutes=10)  # Save model every 10 minutes
        self.data_save_path = 'market_data.csv'  # Path to save data
        self.log_file_path = 'model_training_log.txt'  # Path to save logs
        self.feature_columns = ['scaled_close', 'scaled_sma20', 'scaled_macd']  # Feature names used for training

    def log_message(self, message):
        """Logs messages to a log file."""
        try:
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f"{datetime.now()} - {message}\n")
        except Exception as e:
            print(f"❌ Error logging message: {e}")

    def train(self, data):
        try:
            print("📊 Training model with provided data...")
            self.log_message("Starting model training")

            # Check for missing columns
            required_columns = self.feature_columns + ['target']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"❌ Missing required column: {col}")

            # Drop NaNs
            data = data.dropna()
            if len(data) < 50:
                raise ValueError("❌ Not enough data to train the model (minimum 50 rows required).")

            features = data[self.feature_columns]
            target = data['target']  # 1 = buy, 0 = sell

            scaled_features = self.scaler.fit_transform(features)
            scaled_df = pd.DataFrame(scaled_features, columns=self.feature_columns)

            X_train, X_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.2, random_state=42)

            self.model.fit(X_train, y_train)
            self.is_trained = True
            print("✅ Random Forest model successfully trained.")
            self.log_message("Random Forest model successfully trained.")

            self.last_trained = datetime.now()

            self.save_model("random_forest_model.pkl")
            self.save_data(data)  # Save the data after training
        except Exception as e:
            print(f"❌ Error during training: {e}")
            self.log_message(f"Error during training: {e}")

    def analyze(self, data):
        if not self.is_trained:
            raise ValueError("❌ Model not trained.")

        try:
            features = data[self.feature_columns]
            scaled_features = self.scaler.transform(features)
            scaled_df = pd.DataFrame(scaled_features, columns=self.feature_columns)

            predictions = self.model.predict(scaled_df)

            return {'signal': predictions}
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            self.log_message(f"Error during analysis: {e}")
            return None

    def generate_signal(self, symbol, timeframe):
        processor = DataProcessor()

        try:
            timeframe_map = {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "1h": mt5.TIMEFRAME_H1,
                "1d": mt5.TIMEFRAME_D1
            }

            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)
            bars = mt5.copy_rates_from(symbol, tf, datetime.now() - timedelta(minutes=300), 300)

            if bars is None or len(bars) < 100:
                print("⚠️ Not enough data to generate signal.")
                self.log_message("Not enough data to generate signal.")
                return None

            df = pd.DataFrame(bars)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "tick_volume": "volume"}, inplace=True)

            processed = processor.preprocess_data(df)

            if processed is None or len(processed) < 50:
                print("⚠️ Processed data invalid or too short.")
                self.log_message("Processed data invalid or too short.")
                return None

            # Auto train if not yet trained
            if not self.is_trained:
                print("🔁 Auto-training model...")
                self.log_message("Auto-training model started.")
                self.train(processed)

            # Retrain model every 10 minutes
            if self.last_trained and datetime.now() - self.last_trained >= self.model_save_interval:
                print("⏱️ Retraining model after 10 minutes...")
                self.log_message("Retraining model after 10 minutes.")
                self.train(processed)

            if not self.is_trained:
                print("❌ Model still not trained after auto-training attempt.")
                self.log_message("Model still not trained after auto-training attempt.")
                return None

            latest_features = processed[self.feature_columns].tail(1)
            scaled_latest = self.scaler.transform(latest_features)
            scaled_latest_df = pd.DataFrame(scaled_latest, columns=self.feature_columns)

            prediction = self.model.predict(scaled_latest_df)[0]

            print(f"✅ Signal generated: {'BUY' if prediction == 1 else 'SELL'}")
            self.log_message(f"Signal generated: {'BUY' if prediction == 1 else 'SELL'}")

            return {'type': 'buy' if prediction == 1 else 'sell'}

        except Exception as e:
            print(f"❌ Error during signal generation: {e}")
            self.log_message(f"Error during signal generation: {e}")
            return None

    def save_model(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler
                }, f)
            print(f"💾 Model saved to {path}")
            self.log_message(f"Model saved to {path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            self.log_message(f"Error saving model: {e}")

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
                self.model = obj['model']
                self.scaler = obj['scaler']
                self.is_trained = True
            print("✅ Random Forest model loaded successfully.")
            self.log_message("Random Forest model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.log_message(f"Error loading model: {e}")

    def save_data(self, data):
        """Save data to a CSV file"""
        try:
            data.to_csv(self.data_save_path, mode='a', header=not pd.io.common.file_exists(self.data_save_path))
            print(f"💾 Data saved to {self.data_save_path}")
            self.log_message(f"Data saved to {self.data_save_path}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            self.log_message(f"Error saving data: {e}")
