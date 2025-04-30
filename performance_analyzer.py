import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    """Class to analyze and report trading performance."""

    def __init__(self):
        self.trades = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'net_profit': 0.0,
            'return_pct': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'equity_curve': pd.DataFrame(),
            'trade_history': []
        }
        self.initial_balance = 10000.0
        self.current_equity = self.initial_balance
        self.equity_history = [self.initial_balance]

    def add_trade(self, trade):
        if not isinstance(trade, dict):
            return
        required_fields = ['symbol', 'type', 'entry_price', 'size', 'time']
        if not all(field in trade for field in required_fields):
            return

        self.trades.append(trade)
        if 'exit_price' in trade and 'exit_time' in trade:
            self.update_equity(trade)
        self.update_metrics()

    def update_trade(self, trade_id, exit_price, exit_time=None):
        trade = self.trades[trade_id]
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time or pd.Timestamp.now()
        self.update_equity(trade)
        self.update_metrics()

    def update_equity(self, trade):
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        size = trade['size']
        trade_type = trade['type']

        if trade_type == 'buy':
            pnl = (exit_price - entry_price) * size
        elif trade_type == 'sell':
            pnl = (entry_price - exit_price) * size
        else:
            pnl = 0.0

        trade['pnl'] = pnl
        self.current_equity += pnl
        self.equity_history.append(self.current_equity)

    def update_metrics(self):
        wins = [t for t in self.trades if 'exit_price' in t and (
            (t['type'] == 'buy' and t['exit_price'] > t['entry_price']) or
            (t['type'] == 'sell' and t['exit_price'] < t['entry_price'])
        )]
        losses = [t for t in self.trades if 'exit_price' in t and t not in wins]

        win_profits = [(t['exit_price'] - t['entry_price']) * t['size'] if t['type'] == 'buy' else
                       (t['entry_price'] - t['exit_price']) * t['size'] for t in wins]
        loss_profits = [(t['entry_price'] - t['exit_price']) * t['size'] if t['type'] == 'buy' else
                        (t['exit_price'] - t['entry_price']) * t['size'] for t in losses]

        net_profit = self.current_equity - self.initial_balance
        return_pct = (net_profit / self.initial_balance) * 100
        average_win = np.mean(win_profits) if win_profits else 0
        average_loss = np.mean(loss_profits) if loss_profits else 0
        largest_win = max(win_profits) if win_profits else 0
        largest_loss = min(loss_profits) if loss_profits else 0
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        profit_factor = sum(win_profits) / abs(sum(loss_profits)) if loss_profits else float('inf')

        equity_arr = np.array(self.equity_history)
        drawdowns = equity_arr - np.maximum.accumulate(equity_arr)
        max_drawdown = drawdowns.min() if drawdowns.size else 0

        returns = np.diff(equity_arr) / equity_arr[:-1] if equity_arr.size > 1 else []
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 0 and np.std(returns) > 0 else 0

        self.metrics.update({
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'net_profit': net_profit,
            'return_pct': return_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': pd.DataFrame({'equity': self.equity_history}),
            'trade_history': self.trades
        })

    def get_metrics(self):
        return self.metrics
