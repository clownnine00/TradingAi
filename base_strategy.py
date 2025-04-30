
class BaseStrategy:
    """Base interface for all trading strategies"""

    def __init__(self, name="Base Strategy", description=""):
        self.name = name
        self.description = description
        self.is_trained = False

    def train(self, data):
        """Train the strategy using historical data"""
        raise NotImplementedError("Train method must be implemented by subclass")

    def analyze(self, data):
        """Analyze current market data"""
        raise NotImplementedError("Analyze method must be implemented by subclass")

    def generate_signal(self, symbol, timeframe):
        """Generate a buy/sell signal"""
        raise NotImplementedError("generate_signal method must be implemented by subclass")

    def save_model(self, path):
        """Save model to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, path):
        """Load model from disk"""
        import pickle
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
