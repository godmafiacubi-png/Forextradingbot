import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LiveMonitor:
    """Monitor live trading in real-time"""
    
    def __init__(self):
        self.trades = []
        self.signals = {}
        self.start_time = datetime.now()
    
    def add_signal(self, symbol, signal, confidence, price):
        """Add signal"""
        self.signals[symbol] = {
            'signal': signal,
            'confidence': confidence,
            'price': price,
            'time': datetime.now()
        }
    
    def add_trade(self, symbol, side, entry, sl, tp, size):
        """Add opened trade"""
        trade = {
            'symbol': symbol,
            'side': side,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'size': size,
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'pnl': 0
        }
        self.trades.append(trade)
    
    def update_trade_pnl(self, symbol, current_price):
        """Update trade PnL"""
        for trade in self.trades:
            if trade['symbol'] == symbol and trade['status'] == 'OPEN':
                if trade['side'] == 'BUY':
                    pnl = (current_price - trade['entry']) * trade['size'] * 100000
                else:
                    pnl = (trade['entry'] - current_price) * trade['size'] * 100000
                
                trade['pnl'] = pnl
    
    def print_dashboard(self):
        """Print live dashboard"""
        print("\n" + "="*100)
        print(f"LIVE TRADING DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # Signals
        print("\n[SIGNALS]")
        if self.signals:
            for symbol, data in self.signals.items():
                signal_name = "BUY" if data['signal'] == 1 else "SELL" if data['signal'] == -1 else "HOLD"
                print(f"  {symbol:12} {signal_name:10} Conf: {data['confidence']:.2%} Price: {data['price']:.5f}")
        else:
            print("  No signals yet")
        
        # Open Trades
        print("\n[OPEN TRADES]")
        open_trades = [t for t in self.trades if t['status'] == 'OPEN']
        
        if open_trades:
            for trade in open_trades:
                print(f"  {trade['symbol']:12} {trade['side']:10} Entry: {trade['entry']:.5f} PnL: ${trade['pnl']:.2f}")
        else:
            print("  No open trades")
        
        # Statistics
        print("\n[STATISTICS]")
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        
        if closed_trades:
            total_pnl = sum(t['pnl'] for t in closed_trades)
            wins = len([t for t in closed_trades if t['pnl'] > 0])
            losses = len([t for t in closed_trades if t['pnl'] < 0])
            
            print(f"  Total Trades: {len(closed_trades)}")
            print(f"  Wins/Losses: {wins}/{losses}")
            if len(closed_trades) > 0:
                print(f"  Win Rate: {wins/len(closed_trades):.1%}")
            print(f"  Total PnL: ${total_pnl:.2f}")
        
        print("="*100 + "\n")