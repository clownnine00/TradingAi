import pandas as pd
import numpy as np
from reinforcement_agent import QLearningAgent
from data_processor import DataProcessor
from performance_analyzer import PerformanceAnalyzer
import os

class RLTrainingSimulator:
    """Simulates a trading environment to train the reinforcement learning agent"""

    def __init__(self, data_file, symbol="EURUSD"):
        self.data_file = data_file
        self.symbol = symbol
        self.agent = QLearningAgent()
        self.processor = DataProcessor()
        self.performance = PerformanceAnalyzer()

    def load_and_prepare_data(self):
        """Loads and preprocesses the historical data"""
        df = pd.read_csv(self.data_file)
        df = self.processor.preprocess_data(df)
        return df

    def simulate(self, episodes=10):
        """Runs the training loop"""
        df = self.load_and_prepare_data()

        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")
            balance = 10000
            position = None
            entry_price = 0

            for i in range(1, len(df)):
                state = df.iloc[i - 1][['scaled_close', 'scaled_sma20', 'scaled_macd']].values
                next_state = df.iloc[i][['scaled_close', 'scaled_sma20', 'scaled_macd']].values

                action = self.agent.choose_action(state)

                reward = 0
                if action == 1 and position is None:
                    position = "buy"
                    entry_price = df.iloc[i]['close']
                elif action == 2 and position is None:
                    position = "sell"
                    entry_price = df.iloc[i]['close']
                elif action == 0 and position is not None:
                    if position == "buy":
                        reward = df.iloc[i]['close'] - entry_price
                    elif position == "sell":
                        reward = entry_price - df.iloc[i]['close']
                    trade = {
                        'symbol': self.symbol,
                        'type': position,
                        'entry_price': entry_price,
                        'exit_price': df.iloc[i]['close'],
                        'size': 1.0,
                        'time': df.index[i - 1],
                        'exit_time': df.index[i]
                    }
                    self.performance.add_trade(trade)
                    position = None

                self.agent.learn(state, action, reward, next_state)

            metrics = self.performance.get_metrics()
            print(f"Net Profit after Episode {episode + 1}: {metrics['net_profit']:.2f}")

        # Save model after training
        self.agent.save_model("trained_q_table.pkl")
