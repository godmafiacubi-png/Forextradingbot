import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedRiskManager:
    """Advanced risk management"""
    
    def __init__(self, account_balance, daily_loss_limit_percent=5, max_positions=3):
        self.initial_balance = account_balance
        self.current_balance = account_balance
        self.daily_loss_limit = account_balance * (daily_loss_limit_percent / 100)
        self.max_positions = max_positions
        self.trades = []
        self.daily_pnl = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
    
    def reset_daily_pnl(self):
        """Reset daily PnL at midnight"""
        now = datetime.now()
        if now.date() != self.daily_reset_time.date():
            self.daily_pnl = 0
            self.daily_reset_time = now
            logger.info(f"Daily PnL reset. New limit: -${self.daily_loss_limit:.2f}")
    
    def can_take_trade(self, symbol, side, entry_price, stop_loss, account_equity):
        """Check if trade should be taken"""
        self.reset_daily_pnl()
        
        # Check 1: Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False, "Daily loss limit exceeded"
        
        # Check 2: Position limit
        open_positions = len([t for t in self.trades if t['status'] == 'open'])
        if open_positions >= self.max_positions:
            logger.warning(f"Max positions reached: {open_positions}")
            return False, "Max positions reached"
        
        # Check 3: Risk/Reward ratio (min 1:2)
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return False, "Invalid stop loss"
        
        min_reward = risk * 2
        logger.debug(f"Risk: {risk:.5f}, Min Reward: {min_reward:.5f}")
        
        # Check 4: Position size
        account_risk_percent = 2.0
        max_loss_per_trade = account_equity * (account_risk_percent / 100)
        position_size = max_loss_per_trade / risk
        
        logger.info(f"Trade allowed - Position size: {position_size:.2f} lots, Risk: ${max_loss_per_trade:.2f}")
        
        return True, {
            'symbol': symbol,
            'side': side,
            'entry': entry_price,
            'sl': stop_loss,
            'size': position_size,
            'max_loss': max_loss_per_trade,
            'risk': risk,
            'reward': min_reward
        }
    
    def calculate_trailing_stop(self, current_price, entry_price, side, atr_value):
        """Calculate trailing stop"""
        if side == 'BUY':
            trailing_stop = current_price - (atr_value * 1.5)
        else:
            trailing_stop = current_price + (atr_value * 1.5)
        
        return trailing_stop
    
    def calculate_take_profit_levels(self, entry_price, stop_loss, side, num_levels=3):
        """Calculate multiple take profit levels"""
        risk = abs(entry_price - stop_loss)
        tp_levels = []
        
        if side == 'BUY':
            for i in range(1, num_levels + 1):
                tp = entry_price + (risk * i)
                tp_levels.append(tp)
        else:
            for i in range(1, num_levels + 1):
                tp = entry_price - (risk * i)
                tp_levels.append(tp)
        
        logger.debug(f"TP Levels: {tp_levels}")
        return tp_levels
    
    def log_trade(self, symbol, side, entry, sl, tp, size, status='open'):
        """Log trade"""
        trade = {
            'symbol': symbol,
            'side': side,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'size': size,
            'status': status,
            'entry_time': datetime.now(),
            'exit_time': None,
            'exit_price': None,
            'pnl': 0
        }
        self.trades.append(trade)
        logger.info(f"Trade logged: {symbol} {side} @ {entry:.5f} SL:{sl:.5f} TP:{tp:.5f}")
    
    def close_trade(self, symbol, exit_price):
        """Close trade and calculate PnL"""
        for trade in self.trades:
            if trade['symbol'] == symbol and trade['status'] == 'open':
                trade['status'] = 'closed'
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.now()
                
                if trade['side'] == 'BUY':
                    trade['pnl'] = (exit_price - trade['entry']) * trade['size'] * 100000
                else:
                    trade['pnl'] = (trade['entry'] - exit_price) * trade['size'] * 100000
                
                self.daily_pnl += trade['pnl']
                logger.info(f"Trade closed: {symbol} PnL: ${trade['pnl']:.2f}")
                return trade['pnl']
        
        return 0
    
    def get_statistics(self):
        """Get trading statistics"""
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        
        if not closed_trades:
            return None
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        wins = len([t for t in closed_trades if t['pnl'] > 0])
        losses = len([t for t in closed_trades if t['pnl'] < 0])
        
        win_rate = wins / len(closed_trades) if closed_trades else 0
        
        avg_win = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0) / wins if wins > 0 else 0
        avg_loss = sum(t['pnl'] for t in closed_trades if t['pnl'] < 0) / losses if losses > 0 else 0
        
        return {
            'total_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': abs(avg_loss),
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'daily_pnl': self.daily_pnl,
            'daily_limit_remaining': self.daily_loss_limit - abs(self.daily_pnl)
        }