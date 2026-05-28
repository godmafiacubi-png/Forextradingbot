"""Sign-consistent shaped reward calculation for Deep RL.

The reward should primarily teach the agent whether a closed trade was good or
bad.  Risk, regime, drawdown, holding time, and expensive symbols may shape the
size of the reward, but they should not invert a profitable trade into a large
negative reward or a losing trade into a positive reward.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SignConsistentShapedRewardCalculator:
    """Reward shaping that preserves the sign of realized PnL.

    Previous reward shaping could let global drawdown, hold-time penalties, or
    quiet-regime penalties overwhelm a profitable trade and return values such
    as -10 for a winning position.  That teaches the RL agent the wrong action.

    This implementation keeps the realized trade outcome as the anchor:
    - profitable trades receive a positive reward floor;
    - losing trades receive a negative reward ceiling;
    - shaping terms only adjust magnitude.
    """

    def __init__(self, risk_free_rate: float = 0.0001):
        self.risk_free = risk_free_rate
        self.returns_history = deque(maxlen=100)
        self.peak_equity = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.high_spread_symbols = {"XAUUSDm", "BTCUSDm", "XAUUSD", "BTCUSD"}
        self._current_symbol = None

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return default
            return value
        except (TypeError, ValueError):
            return default

    def calculate_trade_reward(
        self,
        pnl,
        pnl_pct,
        equity,
        hold_bars,
        regime,
        rr_ratio: float = 1.5,
        symbol=None,
    ):
        pnl = self._safe_float(pnl)
        pnl_pct = self._safe_float(pnl_pct)
        equity = self._safe_float(equity)
        hold_bars = max(0, int(self._safe_float(hold_bars)))
        rr_ratio = self._safe_float(rr_ratio, 1.5)
        regime = str(regime or "QUIET").upper()
        effective_symbol = symbol if symbol is not None else self._current_symbol

        abs_return_score = min(abs(pnl_pct) * 100.0, 3.0)

        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            reward = 0.80 + abs_return_score
            if rr_ratio >= 2.0:
                reward += 0.40
            elif rr_ratio >= 1.5:
                reward += 0.20
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            reward = -0.80 - abs_return_score
        else:
            reward = -0.05

        self.returns_history.append(pnl_pct)

        if len(self.returns_history) >= 10:
            returns = np.array(self.returns_history, dtype=float)
            sharpe = (returns.mean() - self.risk_free) / (returns.std() + 1e-10)
            # Sharpe is useful, but it should not dominate single-trade outcome.
            reward += float(np.clip(sharpe * 0.15, -0.50, 0.50))

        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0 and equity > 0:
            drawdown = max((self.peak_equity - equity) / self.peak_equity, 0.0)
            if pnl > 0:
                reward -= min(drawdown * 2.0, 0.50)
            elif pnl < 0:
                reward -= min(drawdown * 4.0, 1.00)

        hold_penalty = 0.0
        if hold_bars > 24:
            hold_penalty += min((hold_bars - 24) * 0.01, 0.30)
        if hold_bars > 48:
            hold_penalty += min((hold_bars - 48) * 0.02, 0.40)
        reward -= min(hold_penalty, 0.60)

        if self.consecutive_wins >= 3:
            reward += 0.15 * min(self.consecutive_wins - 2, 5)
        if self.consecutive_losses >= 2:
            reward -= 0.25 * min(self.consecutive_losses - 1, 5)

        if regime == "TRENDING" and pnl > 0:
            reward += 0.25
        elif regime == "RANGING" and pnl > 0:
            reward += 0.15
        elif regime == "VOLATILE" and pnl < 0:
            reward -= 0.25
        elif regime == "QUIET":
            reward -= 0.10 if pnl > 0 else 0.20

        if effective_symbol in self.high_spread_symbols:
            if 0 < pnl < 5.0:
                reward -= 0.20
            elif pnl < 0:
                reward -= 0.30

        # Preserve learning direction.  Shaping can shrink magnitude, not invert
        # the realized outcome.
        if pnl > 0:
            reward = max(reward, 0.10)
        elif pnl < 0:
            reward = min(reward, -0.10)

        return float(np.clip(reward, -10.0, 10.0))

    def calculate_hold_reward(self, had_signal, regime):
        regime = str(regime or "QUIET").upper()
        if regime == "QUIET":
            return 0.1
        if regime == "VOLATILE":
            return 0.15
        if regime == "TRENDING" and had_signal:
            return -0.2
        if regime == "RANGING":
            return 0.05
        return 0.0

    def get_stats(self):
        win_rate = 0.0
        if self.returns_history:
            wins = sum(1 for item in self.returns_history if item > 0)
            win_rate = wins / len(self.returns_history)
        return {
            "recent_returns": len(self.returns_history),
            "win_rate": round(win_rate, 3),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "peak_equity": round(self.peak_equity, 2),
        }
