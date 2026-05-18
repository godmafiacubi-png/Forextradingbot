from datetime import datetime
import logging

from execution.trade_logger import TradeJournal

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track bot performance — ใช้ PnL จริงจาก MT5"""

    def __init__(self, journal=None):
        self.trades = []
        self.open_trades = {}
        self.balance_history = []
        self.journal = journal if journal is not None else TradeJournal()

    def log_trade(self, ticket, symbol, side, price, size):
        """Log new trade opened"""
        self.open_trades[ticket] = {
            'ticket': ticket,
            'symbol': symbol,
            'side': side,
            'entry_price': price,
            'size': size,
            'open_time': datetime.now()
        }
        logger.info(f"[TRACKER] Tracked #{ticket} {side} {symbol} {size}lots @{price:.5f}; OPEN journal owned by OrderManager")

    def close_trade(self, ticket, exit_price, actual_pnl=None):
        """
        Log trade closed
        actual_pnl: PnL จริงจาก MT5 (prev.profit)
        """
        if ticket not in self.open_trades:
            if actual_pnl is not None:
                result = "WIN" if actual_pnl > 0 else "LOSS"
                logger.info(f"[TRACKER] Closed #{ticket} (untracked) {result} PnL=${actual_pnl:.2f} — excluded from WR stats")
            else:
                logger.debug(f"[TRACKER] Ticket #{ticket} not found, no PnL")
            return

        trade = self.open_trades.pop(ticket)

        if actual_pnl is not None:
            pnl = actual_pnl
        else:
            pnl = self._calculate_pnl(
                trade['side'], trade['entry_price'], exit_price,
                trade['size'], trade['symbol']
            )

        self.trades.append({
            'ticket': ticket,
            'symbol': trade['symbol'],
            'side': trade['side'],
            'entry': trade['entry_price'],
            'exit': exit_price if exit_price else 0,
            'size': trade['size'],
            'pnl': pnl,
            'open_time': trade['open_time'],
            'close_time': datetime.now()
        })

        self.journal.log_close(
            ticket, trade['symbol'], trade['side'], trade['size'], exit_price, pnl,
            comment="performance_tracker", source="performance_tracker",
        )
        result = "WIN" if pnl > 0 else "LOSS"
        logger.info(f"[TRACKER] Closed #{ticket} {trade['symbol']} {trade['side']} {result} PnL=${pnl:.2f}")

    def _calculate_pnl(self, side, entry, exit_price, size, symbol=''):
        """Fallback PnL calculation"""
        symbol_upper = symbol.upper()
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            contract_size = 100
        elif 'BTC' in symbol_upper:
            contract_size = 1
        elif 'JPY' in symbol_upper:
            contract_size = 100000
        else:
            contract_size = 100000

        if side == 'BUY':
            pnl = (exit_price - entry) * size * contract_size
        else:
            pnl = (entry - exit_price) * size * contract_size
        return pnl

    def add_trade(self, symbol, entry_price, exit_price, size, side, actual_pnl=None):
        """Add closed trade directly"""
        if actual_pnl is not None:
            pnl = actual_pnl
        else:
            pnl = self._calculate_pnl(side, entry_price, exit_price, size, symbol)

        self.trades.append({
            'symbol': symbol,
            'entry': entry_price,
            'exit': exit_price,
            'size': size,
            'side': side,
            'pnl': pnl,
            'open_time': datetime.now(),
            'close_time': datetime.now()
        })

    def get_stats(self):
        """Get performance statistics."""
        if not self.trades:
            return None

        pnls = [float(trade.get('pnl') or 0.0) for trade in self.trades]
        total_pnl = sum(pnls)
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        total = len(self.trades)
        win_rate = len(wins) / total if total > 0 else 0

        total_win = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = total_win / total_loss if total_loss > 0 else (999.0 if total_win > 0 else 0)

        max_consec_win = 0
        max_consec_loss = 0
        consec_win = 0
        consec_loss = 0
        for pnl in pnls:
            if pnl > 0:
                consec_win += 1
                consec_loss = 0
                max_consec_win = max(max_consec_win, consec_win)
            else:
                consec_loss += 1
                consec_win = 0
                max_consec_loss = max(max_consec_loss, consec_loss)

        symbol_stats = {}
        for trade in self.trades:
            sym = trade.get('symbol') or 'UNKNOWN'
            bucket = symbol_stats.setdefault(sym, {'trades': 0, 'pnl': 0.0, 'wins': 0})
            pnl = float(trade.get('pnl') or 0.0)
            bucket['trades'] += 1
            bucket['pnl'] += pnl
            if pnl > 0:
                bucket['wins'] += 1

        symbol_stats = {
            sym: {
                'trades': values['trades'],
                'pnl': round(values['pnl'], 2),
                'win_rate': round(values['wins'] / values['trades'], 3) if values['trades'] else 0,
            }
            for sym, values in symbol_stats.items()
        }

        return {
            'total_trades': total,
            'open_trades': len(self.open_trades),
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 4),
            'win_trades': len(wins),
            'lose_trades': len(losses),
            'avg_win': round(total_win / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(losses) / len(losses), 2) if losses else 0,
            'best_trade': round(max(pnls), 2) if pnls else 0,
            'worst_trade': round(min(pnls), 2) if pnls else 0,
            'max_consec_win': max_consec_win,
            'max_consec_loss': max_consec_loss,
            'profit_factor': round(profit_factor, 2),
            'symbol_stats': symbol_stats,
        }

    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()

        if not stats:
            logger.info("No closed trades yet")
            if self.open_trades:
                logger.info(f"Open trades: {len(self.open_trades)}")
                for ticket, t in self.open_trades.items():
                    logger.info(f"  #{ticket} {t['side']} {t['symbol']} {t['size']}lots @{t['entry_price']:.5f}")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("PERFORMANCE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Trades:      {stats['total_trades']}")
        logger.info(f"Open Trades:       {stats['open_trades']}")
        logger.info(f"Total PnL:         ${stats['total_pnl']:.2f}")
        logger.info(f"Win Rate:          {stats['win_rate']:.2%}")
        logger.info(f"Wins/Losses:       {stats['win_trades']}/{stats['lose_trades']}")
        logger.info(f"Avg Win:           ${stats['avg_win']:.2f}")
        logger.info(f"Avg Loss:          ${stats['avg_loss']:.2f}")
        logger.info(f"Best Trade:        ${stats['best_trade']:.2f}")
        logger.info(f"Worst Trade:       ${stats['worst_trade']:.2f}")
        logger.info(f"Profit Factor:     {stats['profit_factor']:.2f}")

        if stats.get('symbol_stats'):
            logger.info("-" * 40)
            for sym, ss in stats['symbol_stats'].items():
                logger.info(f"  {sym}: {ss['trades']}T ${ss['pnl']:.2f} WR={ss['win_rate']:.0%}")
        logger.info("=" * 60)