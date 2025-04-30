import numpy as np
import pickle
import os

class QLearningAgent:
    """Reinforcement Learning agent using Q-learning"""

    def __init__(self, state_size=3, action_size=3, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0):
        self.state_size = state_size  # e.g., 3 indicators: scaled_close, scaled_sma20, scaled_macd
        self.action_size = action_size  # 0: hold, 1: buy, 2: sell
        self.q_table = {}  # Dictionary for storing state-action values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration = 0.01

    def get_state_key(self, state):
        """Converts state array to a hashable key"""
        return tuple(np.round(state, 2))  # rounded for state-space compression

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)

        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[key])  # Exploit

    def learn(self, state, action, reward, next_state):
        """Update Q-values based on observed experience"""
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        best_next_action = np.max(self.q_table[next_key])
        target = reward + self.discount_factor * best_next_action
        self.q_table[key][action] += self.learning_rate * (target - self.q_table[key][action])

        # Decay exploration rate
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay

    def save_model(self, file_path):
        """Saves the Q-table to a file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, file_path):
        """Loads the Q-table from a file"""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print("Q-table loaded successfully.")
        else:
            print("Q-table file not found.")
