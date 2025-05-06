import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from base_strategy import BaseStrategy
import pickle
from datetime import datetime, timedelta
import os
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
        self.model_save_interval = timedelta(minutes=10)
        self.data_save_path = 'market_data.csv'
        self.log_file_path = 'model_training_log.txt'
        self.model_path = "random_forest_model.pkl"
        self.feature_columns = ['scaled_close', 'scaled_sma20', 'scaled_macd']

        # Auto-load model if it exists
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def log_message(self, message):
        """Log messages to the log file."""
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{datetime.now()} - {message}\n")
        except Exception as e:
            print(f"❌ Error logging message: {e}")

    def validate_model(self):
        """Validate that the model is an instance of RandomForestClassifier."""
        if not isinstance(self.model, RandomForestClassifier):
            print("⚠️ Invalid model detected. Reinitializing RandomForestClassifier.")
            self.log_message("⚠️ Invalid model detected. Reinitializing RandomForestClassifier.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.is_trained = False

    def train(self, data):
        """Train the Random Forest model on the provided data."""
        try:
            print("📊 Training model with provided data...")
            self.log_message("Starting model training")
            self.validate_model()

            required_columns = self.feature_columns + ['target']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"❌ Missing required column: {col}")

            data = data.dropna()
            if len(data) < 50:
                raise ValueError("❌ Not enough data to train the model (minimum 50 rows required).")

            features = data[self.feature_columns]
            target = data['target']

            scaled_features = self.scaler.fit_transform(features)
            scaled_df = pd.DataFrame(scaled_features, columns=self.feature_columns)

            X_train, X_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.2, random_state=42)

            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.last_trained = datetime.now()

            print("✅ Random Forest model successfully trained.")
            self.log_message("Random Forest model successfully trained.")
            self.save_model(self.model_path)
            self.save_data(data)

        except Exception as e:
            print(f"❌ Error during training: {e}")
            self.log_message(f"Error during training: {e}")

    def analyze(self, data):
        """Analyze the data and generate predictions and confidence levels."""
        if not self.is_trained or not isinstance(self.model, RandomForestClassifier):
            raise ValueError("❌ Model not trained or invalid.")
        try:
            features = data[self.feature_columns]
            scaled_features = self.scaler.transform(features)
            predictions = self.model.predict(scaled_features)
            probabilities = self.model.predict_proba(scaled_features)
            return {
                'signal': predictions,
                'confidence': probabilities.max(axis=1)
            }
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            self.log_message(f"Error during analysis: {e}")
            return None

    def generate_signal(self, symbol, timeframe):
        """Generate a trading signal for the given symbol and timeframe."""
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
                self.log_message("⚠️ Not enough data to generate signal.")
                return None

            df = pd.DataFrame(bars)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "tick_volume": "volume"}, inplace=True)

            processed = processor.preprocess_data(df)

            if processed is None or len(processed) < 50:
                self.log_message("⚠️ Processed data invalid or too short.")
                return None

            # Retrain periodically if needed
            if not self.is_trained:
                self.log_message("🔁 Auto-training model started.")
                self.train(processed)

            if self.last_trained and datetime.now() - self.last_trained >= self.model_save_interval:
                self.log_message("⏱️ Retraining model after interval.")
                self.train(processed)

            if not self.is_trained:
                self.log_message("❌ Model still not trained after auto-training.")
                return None

            latest_features = processed[self.feature_columns].tail(1)
            scaled_latest = self.scaler.transform(latest_features)
            prediction = self.model.predict(scaled_latest)[0]
            confidence = self.model.predict_proba(scaled_latest)[0].max()

            level = 'high' if confidence >= 0.75 else 'medium' if confidence >= 0.55 else 'low'
            signal = 'buy' if prediction == 1 else 'sell'

            print(f"✅ Signal: {signal.upper()} | Confidence: {confidence:.2f} ({level})")
            self.log_message(f"Signal: {signal.upper()} | Confidence: {confidence:.2f} ({level})")

            return {
                'type': signal,
                'confidence': round(confidence, 2),
                'confidence_level': level
            }

        except Exception as e:
            print(f"❌ Error during signal generation: {e}")
            self.log_message(f"Error during signal generation: {e}")
            return {'type': 'neutral', 'confidence': 0.0, 'confidence_level': 'low'}

    def save_model(self, path):
        """Save the model and scaler into a file."""
        try:
            if not isinstance(self.model, RandomForestClassifier):
                raise TypeError("Model is not a RandomForestClassifier.")
            if not isinstance(self.scaler, StandardScaler):
                raise TypeError("Scaler is not a StandardScaler.")
            with open(path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            print("✅ Model saved successfully.")
            self.log_message("Model saved successfully.")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            self.log_message(f"Error saving model: {e}")

    def load_model(self, path):
        """Load the model and scaler from a file."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

                if isinstance(model_data, dict) and 'model' in model_data and 'scaler' in model_data:
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']

                    if not isinstance(self.model, RandomForestClassifier):
                        raise TypeError("❌ The loaded model is not a RandomForestClassifier.")
                    if not isinstance(self.scaler, StandardScaler):
                        raise TypeError("❌ The loaded scaler is not a StandardScaler.")

                    self.is_trained = True
                    print("✅ Model loaded successfully.")
                    self.log_message("Model loaded successfully.")
                else:
                    print("❌ Invalid model data format.")
                    self.log_message("Invalid model data format.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.log_message(f"Error loading model: {e}")
            # Reinitialize model and scaler for safety
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False

    def save_data(self, data):
        """Save processed data to a CSV file."""
        try:
            data.to_csv(self.data_save_path, mode='a', header=not os.path.exists(self.data_save_path))
            print("💾 Data saved to CSV file.")
            self.log_message("Data saved to CSV file.")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            self.log_message(f"Error saving data: {e}")
