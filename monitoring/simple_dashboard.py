import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleMonitor:
    """In-memory monitor — feeds data to web dashboard"""

    def __init__(self):
        self.signals = {}
        self.signal_history = []
        self.trade_history = []
        self.account_history = []
        self.last_update = None

    def log_signal(self, symbol, signal, confidence):
        signal_name = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")
        self.signals[symbol] = {
            'symbol': symbol,
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'time': datetime.now().strftime('%H:%M:%S')
        }
        self.signal_history.append({
            'symbol': symbol,
            'signal_name': signal_name,
            'confidence': confidence,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        # Keep last 500
        if len(self.signal_history) > 500:
            self.signal_history = self.signal_history[-500:]

        self.last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log_account(self, balance, equity):
        self.account_history.append({
            'balance': balance,
            'equity': equity,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        if len(self.account_history) > 1000:
            self.account_history = self.account_history[-1000:]

    def get_dashboard_data(self):
        return {
            'signals': self.signals,
            'signal_history': self.signal_history[-50:],
            'account_history': self.account_history[-100:],
            'last_update': self.last_update
        }

    def print_status(self):
        """Console print"""
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  Bot Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

        for sym, data in self.signals.items():
            sig = data['signal_name']
            conf = data['confidence']
            t = data['time']
            logger.info(f"  {sym:12} {sig:6} Conf: {conf:.2%}  @ {t}")

        logger.info("=" * 70)