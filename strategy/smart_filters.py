"""
Smart Filters v7.1
- SignalQualityScorer
- AdaptiveThreshold
- LossStreakManager
- TimeFilter
- SmartTrailingV2 (BE + Trail)
- EnhancedPartialClose
- PerformanceAutoAdjust
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class SignalQualityScorer:
    """คะแนน 0-100 ประเมินคุณภาพ signal"""

    def calculate(self, signal, confidence, ict_score, adx, rsi, htf,
                  m30_confirmed=True, structure=0, vol_spike=0):
        score = 0
        score += min(confidence * 20, 20)      # confidence: max 20 (was 25)
        score += min(ict_score * 7, 30)        # ICT: max 30 (was ict_score*5, max 20)
        if adx > 40:    score += 15
        elif adx > 30:  score += 12
        elif adx > 25:  score += 8
        elif adx > 20:  score += 5
        if signal == 1 and htf == 1:    score += 15
        elif signal == -1 and htf == -1: score += 15
        elif htf == 0:  score += 5
        if m30_confirmed: score += 10
        else:             score -= 5
        if signal == 1 and structure == 1:   score += 5
        elif signal == -1 and structure == -1: score += 5
        if vol_spike: score += 5
        if signal == 1 and 30 < rsi < 55:    score += 5
        elif signal == -1 and 45 < rsi < 70: score += 5
        return max(0, min(100, int(score))), {}

    def get_grade(self, score):
        if score >= 85:
            return 'A+', '🟢'
        elif score >= 70:
            return 'A', '🟢'
        elif score >= 55:
            return 'B', '🟡'
        elif score >= 40:
            return 'C', '🟠'
        else:
            return 'D', '🔴'


class AdaptiveThreshold:
    """ปรับ confidence threshold อัตโนมัติตาม win rate"""

    def __init__(self, base_threshold=0.40, window=20, min_thresh=0.25, max_thresh=0.60):
        self.base = base_threshold
        self.current = base_threshold
        self.window = window
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.results = deque(maxlen=window)

    def record_result(self, is_win):
        self.results.append(1 if is_win else 0)
        self._recalculate()

    def _recalculate(self):
        if len(self.results) < 5:
            self.current = self.base
            return
        wr = sum(self.results) / len(self.results)
        if wr >= 0.65:
            self.current = max(self.base - 0.05, self.min_thresh)
        elif wr >= 0.50:
            self.current = self.base
        elif wr >= 0.40:
            self.current = min(self.base + 0.05, self.max_thresh)
        else:
            self.current = min(self.base + 0.10, self.max_thresh)

    def get_threshold(self):
        return self.current

    def get_stats(self):
        wr = sum(self.results) / len(self.results) if self.results else 0
        return {
            'base': self.base,
            'current': round(self.current, 3),
            'recent_wr': round(wr, 3),
            'samples': len(self.results),
        }


class LossStreakManager:
    """จัดการ consecutive losses"""

    def __init__(self, max_streak=2, cooldown_minutes=120):
        self.max_streak = max_streak
        self.cooldown_min = cooldown_minutes
        self.current_streak = 0
        self.max_streak_hit = 0
        self.cooldown_until = None
        self.total_streaks = 0

    def record_result(self, is_win):
        if is_win:
            self.current_streak = 0
        else:
            self.current_streak += 1
            self.max_streak_hit = max(self.max_streak_hit, self.current_streak)
            if self.current_streak >= self.max_streak:
                self.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_min)
                self.total_streaks += 1
                logger.warning(f"[STREAK] {self.current_streak} losses -> cooldown {self.cooldown_min}min")

    def can_trade(self):
        if self.cooldown_until:
            if datetime.now() < self.cooldown_until:
                remaining = (self.cooldown_until - datetime.now()).seconds // 60
                return False, f"cooldown {remaining}min left"
            else:
                self.cooldown_until = None
                self.current_streak = 0
        return True, "ok"

    def get_stats(self):
        return {
            'current_streak': self.current_streak,
            'max_streak': self.max_streak_hit,
            'in_cooldown': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'total_streaks': self.total_streaks,
        }


class TimeFilter:
    """กรองเวลาที่ไม่ควรเทรด"""

    def __init__(self):
        self.last_trade_time = {}
        self.min_gap_minutes = 5

    def can_trade(self, symbol=None):
        now = datetime.now()
        if now.weekday() >= 5:
            return False, "weekend"
        if (now.hour == 23 and now.minute >= 50) or (now.hour == 0 and now.minute <= 10):
            return False, "rollover"
        if symbol and symbol in self.last_trade_time:
            gap = (now - self.last_trade_time[symbol]).seconds / 60
            if gap < self.min_gap_minutes:
                return False, f"gap {gap:.0f}m < {self.min_gap_minutes}m"
        return True, "ok"

    def record_trade(self, symbol):
        self.last_trade_time[symbol] = datetime.now()


class SmartTrailingV2:
    """
    Smart Trailing Stop + Breakeven

    Phase 1: BE     -- price moves >= BE_ATR -> move SL to entry + buffer
    Phase 2: Trail  -- price moves further -> trail SL behind price
    """

    def __init__(self):
        self.be_activated = {}

    def calculate_new_sl(self, side, entry, current_price, current_sl, atr,
                         be_atr=None, trail_atr=None, ticket=None):
        if atr <= 0:
            return current_sl

        try:
            from config.settings import BREAKEVEN_ATR, TRAILING_STOP_ATR
            if be_atr is None:
                be_atr = BREAKEVEN_ATR
            if trail_atr is None:
                trail_atr = TRAILING_STOP_ATR
        except ImportError:
            if be_atr is None:
                be_atr = 1.0
            if trail_atr is None:
                trail_atr = 1.0

        ticket_key = str(ticket) if ticket else f"{side}_{entry}"
        buffer = atr * 0.05

        if side == 'BUY':
            profit_distance = current_price - entry
            profit_atr = profit_distance / atr if atr > 0 else 0

            if profit_atr >= be_atr and ticket_key not in self.be_activated:
                new_sl = entry + buffer
                if new_sl > current_sl:
                    self.be_activated[ticket_key] = True
                    logger.info(f"[BE] #{ticket_key} BUY: SL {current_sl:.5f} -> {new_sl:.5f} (BE +{buffer:.5f})")
                    return new_sl

            if profit_atr >= be_atr + 0.5:
                trail_sl = current_price - (trail_atr * atr)
                trail_sl = max(trail_sl, entry + buffer)
                if trail_sl > current_sl:
                    logger.info(f"[TRAIL] #{ticket_key} BUY: SL {current_sl:.5f} -> {trail_sl:.5f} (+{profit_atr:.1f}ATR)")
                    return trail_sl

        elif side == 'SELL':
            profit_distance = entry - current_price
            profit_atr = profit_distance / atr if atr > 0 else 0

            if profit_atr >= be_atr and ticket_key not in self.be_activated:
                new_sl = entry - buffer
                if new_sl < current_sl or current_sl == 0:
                    self.be_activated[ticket_key] = True
                    logger.info(f"[BE] #{ticket_key} SELL: SL {current_sl:.5f} -> {new_sl:.5f} (BE -{buffer:.5f})")
                    return new_sl

            if profit_atr >= be_atr + 0.5:
                trail_sl = current_price + (trail_atr * atr)
                trail_sl = min(trail_sl, entry - buffer)
                if trail_sl < current_sl or current_sl == 0:
                    logger.info(f"[TRAIL] #{ticket_key} SELL: SL {current_sl:.5f} -> {trail_sl:.5f} (+{profit_atr:.1f}ATR)")
                    return trail_sl

        return current_sl

    def cleanup(self, active_tickets):
        active_keys = {str(t) for t in active_tickets}
        to_remove = [k for k in self.be_activated if k not in active_keys]
        for k in to_remove:
            del self.be_activated[k]


class EnhancedPartialClose:
    """Partial close logic"""

    def __init__(self):
        self.closed_levels = {}

    def check_partial_close(self, ticket, side, entry, current_price, atr, volume,
                            levels=None):
        if atr <= 0 or volume <= 0:
            return []

        if levels is None:
            try:
                from config.settings import (PARTIAL_CLOSE_1_ATR, PARTIAL_CLOSE_1_PCT,
                                             PARTIAL_CLOSE_2_ATR, PARTIAL_CLOSE_2_PCT)
                levels = [
                    (PARTIAL_CLOSE_1_ATR, PARTIAL_CLOSE_1_PCT),
                    (PARTIAL_CLOSE_2_ATR, PARTIAL_CLOSE_2_PCT),
                ]
            except ImportError:
                levels = [(0.8, 0.5), (1.5, 0.5)]

        ticket_key = str(ticket)
        if ticket_key not in self.closed_levels:
            self.closed_levels[ticket_key] = set()

        if side == 'BUY':
            profit_atr = (current_price - entry) / atr
        else:
            profit_atr = (entry - current_price) / atr

        actions = []
        for i, (atr_mult, close_pct) in enumerate(levels):
            level_key = f"L{i}_{atr_mult}"
            if level_key not in self.closed_levels[ticket_key] and profit_atr >= atr_mult:
                self.closed_levels[ticket_key].add(level_key)
                actions.append((close_pct, f"partial_{i+1}@{atr_mult}ATR"))

        return actions

    def cleanup(self, active_tickets):
        active_keys = {str(t) for t in active_tickets}
        to_remove = [k for k in self.closed_levels if k not in active_keys]
        for k in to_remove:
            del self.closed_levels[k]


class PerformanceAutoAdjust:
    """ปรับ parameters อัตโนมัติตามผลงาน"""

    def __init__(self, window=30):
        self.window = window
        self.trades = deque(maxlen=window)
        self.pnl_history = deque(maxlen=window)

    def record_trade(self, pnl):
        self.trades.append(pnl)
        self.pnl_history.append(pnl)

    def get_adjustments(self):
        if len(self.trades) < 5:
            return {}

        total = len(self.trades)
        wins = sum(1 for p in self.trades if p > 0)
        wr = wins / total
        total_pnl = sum(self.trades)

        adj = {
            'conf_adj': 0,
            'adx_adj': 0,
            'risk_mult': 1.0,
            'reason': '',
        }

        if wr >= 0.65 and total_pnl > 0:
            adj['conf_adj'] = -0.03
            adj['risk_mult'] = 1.1
            adj['reason'] = f'HOT streak WR={wr:.0%}'
        elif wr >= 0.50:
            adj['reason'] = f'Normal WR={wr:.0%}'
        elif wr >= 0.40:
            adj['conf_adj'] = 0.03
            adj['adx_adj'] = 2
            adj['risk_mult'] = 0.9
            adj['reason'] = f'Cold WR={wr:.0%}'
        else:
            adj['conf_adj'] = 0.05
            adj['adx_adj'] = 5
            adj['risk_mult'] = 0.7
            adj['reason'] = f'BAD streak WR={wr:.0%}'

        return adj

    def get_stats(self):
        total = len(self.trades)
        if total == 0:
            return {'trades': 0, 'adjustments': {}}

        wins = sum(1 for p in self.trades if p > 0)
        wr = wins / total
        total_pnl = sum(self.trades)

        return {
            'trades': total,
            'win_rate': round(wr, 3),
            'total_pnl': round(total_pnl, 2),
            'adjustments': self.get_adjustments(),
        }