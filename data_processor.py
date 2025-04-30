import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import talib
except ImportError:
    from ta_utils import talib_mock as talib  # Fallback if TA-Lib is unavailable

from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Class to process and prepare market data for analysis and trading"""

    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        # ✅ 1. Verify data quality
        if data is None or data.empty:
            print("❌ Data is empty or None.")
            return pd.DataFrame()

        df = data.copy()

        # ✅ 2. Verify required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Missing required column: {col}")
                return pd.DataFrame()

        # ✅ 3. Check data length (minimum 50 rows)
        if len(df) < 50:
            print(f"⚠️ Not enough data to calculate indicators. Found: {len(df)} rows.")
            return pd.DataFrame()

        try:
            # Calculate necessary technical indicators
            df['sma20'] = self._calculate_sma(df['close'], 20)
            
            macd_result = self._calculate_macd(df['close'], df.index)
            df['macd'] = macd_result['macd']
            df['macd_signal'] = macd_result['signal']
            df['macd_histogram'] = macd_result['histogram']

            df['rsi'] = self.calculate_rsi(df['close'], window=14)  # RSI (14-period)

            # Drop rows with NaNs from indicator calculation
            df.dropna(inplace=True)

            # ✅ Extra safety check after dropping NaNs
            if len(df) < 10:
                print(f"⚠️ Not enough valid rows after indicators. Remaining: {len(df)} rows.")
                return pd.DataFrame()

            # Features to scale
            features = ['close', 'sma20', 'macd', 'rsi']

            # ✅ 4. Adjust scaling with column checks
            for feature in features:
                if feature not in df.columns:
                    print(f"❌ Missing feature column: {feature}")
                    return pd.DataFrame()

            scaled = self.scaler.fit_transform(df[features])
            scaled_df = pd.DataFrame(scaled, columns=[f"scaled_{col}" for col in features], index=df.index)
            df = pd.concat([df, scaled_df], axis=1)

            # ✅ Target column based on condition (Buy = 1, Sell = 0)
            df['target'] = (df['close'] > df['sma20']).astype(int)

            print("✅ Data preprocessing completed successfully.")
            return df

        except Exception as e:
            print(f"❌ Error while processing data: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data, window=14):
        """Calculate the Relative Strength Index (RSI)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_sma(self, data, period):
        """Simple Moving Average (SMA)"""
        return data.rolling(window=period).mean()

    def _calculate_macd(self, data, index):
        """MACD Indicator with proper indexing"""
        macd, signal, _ = talib.MACD(data, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_series = pd.Series(macd, index=index)
        signal_series = pd.Series(signal, index=index)
        histogram_series = macd_series - signal_series
        return {
            'macd': macd_series,
            'signal': signal_series,
            'histogram': histogram_series
        }
