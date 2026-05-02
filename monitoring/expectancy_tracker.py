"""
Expectancy Tracker — tracks AvgWin vs AvgLoss per symbol
Logs warning if actual R:R drops below 1.5x
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExpectancyTracker:
    def __init__(self):
        self.trades = []  # list of {"pnl": float, "symbol": str}
        self.symbol_trades = defaultdict(list)

    def record(self, symbol: str, pnl: float):
        self.trades.append({"symbol": symbol, "pnl": pnl})
        self.symbol_trades[symbol].append(pnl)

    def report(self) -> float:
        if not self.trades:
            return 0.0

        wins   = [t["pnl"] for t in self.trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in self.trades if t["pnl"] < 0]

        wr       = len(wins) / len(self.trades)
        avg_win  = sum(wins)   / len(wins)   if wins   else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        expectancy = (wr * avg_win) + ((1 - wr) * avg_loss)

        logger.info(f"[EXPECTANCY] Total:{len(self.trades)} WR={wr:.1%} "
                    f"AvgWin=${avg_win:.2f} AvgLoss=${avg_loss:.2f} "
                    f"Expectancy=${expectancy:.2f}")

        if avg_loss != 0:
            rr_actual = abs(avg_win / avg_loss)
            flag = "✅" if rr_actual >= 1.5 else "❌ RR ต่ำกว่าเกณฑ์!"
            logger.info(f"[EXPECTANCY] Actual R:R = {rr_actual:.2f}x {flag}")

        for sym, sym_pnls in self.symbol_trades.items():
            sym_wr  = sum(1 for p in sym_pnls if p > 0) / len(sym_pnls)
            sym_pnl = sum(sym_pnls)
            logger.info(f"  {sym}: {len(sym_pnls)}T WR={sym_wr:.0%} PnL=${sym_pnl:.2f}")

        return expectancy
