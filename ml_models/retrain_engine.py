"""
Retrain Engine v1.0
- Walk-Forward sliding window retraining
- Triggered retrain when performance drifts
- Periodic retrain every N trades
- Thread-safe, non-blocking (runs in background thread)
"""

import os
import logging
import threading
import time
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict

logger = logging.getLogger(__name__)


# ============================================================
# Performance Tracker — detects drift / degradation
# ============================================================
@dataclass
class PerformanceWindow:
    window_size: int = 50
    min_samples: int = 20

    wins: deque = field(default_factory=lambda: deque(maxlen=50))
    pnls: deque = field(default_factory=lambda: deque(maxlen=50))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))

    # Baselines set after first successful train
    baseline_win_rate: float = 0.0
    baseline_sharpe: float = 0.0
    baseline_set: bool = False

    def record(self, pnl: float, pnl_pct: float):
        self.wins.append(1 if pnl > 0 else 0)
        self.pnls.append(pnl_pct)
        self.timestamps.append(datetime.now())

    @property
    def win_rate(self) -> float:
        if len(self.wins) < self.min_samples:
            return 0.5
        return sum(self.wins) / len(self.wins)

    @property
    def sharpe(self) -> float:
        if len(self.pnls) < self.min_samples:
            return 0.0
        arr = np.array(self.pnls)
        std = arr.std()
        return arr.mean() / (std + 1e-10)

    @property
    def recent_win_rate(self) -> float:
        """Win rate of last 10 trades"""
        recent = list(self.wins)[-10:]
        return sum(recent) / max(len(recent), 1)

    def set_baseline(self):
        if len(self.wins) >= self.min_samples:
            self.baseline_win_rate = self.win_rate
            self.baseline_sharpe = self.sharpe
            self.baseline_set = True
            logger.info(f"[Retrain] Baseline set: WR={self.baseline_win_rate:.2%} "
                        f"Sharpe={self.baseline_sharpe:.3f}")

    def is_degraded(self, wr_drop_threshold=0.10, sharpe_drop_threshold=0.3) -> tuple:
        """Returns (is_degraded: bool, reason: str)"""
        if not self.baseline_set or len(self.wins) < self.min_samples:
            return False, ""

        wr_drop = self.baseline_win_rate - self.win_rate
        sharpe_drop = self.baseline_sharpe - self.sharpe

        if wr_drop >= wr_drop_threshold:
            return True, f"WR dropped {wr_drop:.1%} (baseline={self.baseline_win_rate:.1%} current={self.win_rate:.1%})"
        if sharpe_drop >= sharpe_drop_threshold:
            return True, f"Sharpe dropped {sharpe_drop:.3f} (baseline={self.baseline_sharpe:.3f} current={self.sharpe:.3f})"
        if self.recent_win_rate < 0.30:
            return True, f"Recent WR critical: {self.recent_win_rate:.1%} (last 10 trades)"

        return False, ""

    def get_stats(self) -> Dict:
        return {
            "samples": len(self.wins),
            "win_rate": round(self.win_rate, 4),
            "recent_win_rate": round(self.recent_win_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "baseline_wr": round(self.baseline_win_rate, 4),
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "baseline_set": self.baseline_set,
        }


# ============================================================
# Walk-Forward Data Manager
# ============================================================
class WalkForwardManager:
    """
    Maintains a sliding window of historical data for retraining.

    train_window: number of recent bars to train on
    val_window:   number of bars to use for validation
    min_bars:     minimum bars required before training
    """

    def __init__(self, train_window=2000, val_window=200, min_bars=500):
        self.train_window = train_window
        self.val_window = val_window
        self.min_bars = min_bars

        self._data_lock = threading.Lock()
        self._buffer: deque = deque(maxlen=train_window + val_window + 100)
        self._last_train_idx = 0
        self.total_bars_seen = 0

    def add_bar(self, bar: Dict):
        """Add a new OHLCV + indicator bar"""
        with self._data_lock:
            self._buffer.append(bar)
            self.total_bars_seen += 1

    def add_bars(self, bars: List[Dict]):
        """Bulk add bars"""
        with self._data_lock:
            self._buffer.extend(bars)
            self.total_bars_seen += len(bars)

    def get_train_data(self) -> Optional[List[Dict]]:
        """Get sliding window training data"""
        with self._data_lock:
            if len(self._buffer) < self.min_bars:
                logger.info(
                    f"[WalkForward] Buffer {len(self._buffer)}/{self.min_bars} bars "
                    f"— waiting for market open"
                )
                return None
            data = list(self._buffer)
            return data[-self.train_window:]

    def get_val_data(self) -> Optional[List[Dict]]:
        """Get validation slice (most recent bars)"""
        with self._data_lock:
            if len(self._buffer) < self.val_window:
                return None
            data = list(self._buffer)
            return data[-self.val_window:]

    def has_new_data(self, min_new_bars=100) -> bool:
        """True if enough new bars have arrived since last train"""
        new_bars = self.total_bars_seen - self._last_train_idx
        return new_bars >= min_new_bars

    def mark_trained(self):
        self._last_train_idx = self.total_bars_seen

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def get_stats(self) -> Dict:
        return {
            "buffer_size": self.buffer_size,
            "total_bars_seen": self.total_bars_seen,
            "bars_since_last_train": self.total_bars_seen - self._last_train_idx,
            "train_window": self.train_window,
            "val_window": self.val_window,
        }


# ============================================================
# Retrain Engine (main class)
# ============================================================
class RetrainEngine:
    """
    Orchestrates model retraining with three triggers:

    1. Periodic:   every `retrain_every_n_trades` trades
    2. Triggered:  when performance degrades (win rate drop / Sharpe drop)
    3. Walk-forward: when enough new bars arrive

    Runs retraining in a background thread — non-blocking.
    """

    def __init__(
        self,
        train_fn: Callable,            # fn(train_data) -> bool
        retrain_every_n_trades: int = 100,
        min_new_bars_for_wf: int = 200,
        wr_drop_threshold: float = 0.10,
        sharpe_drop_threshold: float = 0.30,
        cooldown_minutes: int = 30,
        train_window: int = 2000,
        val_window: int = 200,
        min_bars: int = 500,
    ):
        self.train_fn = train_fn
        self.retrain_every_n_trades = retrain_every_n_trades
        self.min_new_bars_for_wf = min_new_bars_for_wf
        self.cooldown_minutes = cooldown_minutes

        self.performance = PerformanceWindow()
        self.walk_forward = WalkForwardManager(
            train_window=train_window,
            val_window=val_window,
            min_bars=min_bars,
        )

        # Counters
        self._trade_count = 0
        self._last_retrain_time: Optional[datetime] = None
        self._last_retrain_trigger = ""
        self._retrain_count = 0
        self._is_training = False
        self._train_lock = threading.Lock()

        self._wr_drop_threshold = wr_drop_threshold
        self._sharpe_drop_threshold = sharpe_drop_threshold

        logger.info(
            f"[RetrainEngine] Init: periodic={retrain_every_n_trades} trades, "
            f"cooldown={cooldown_minutes}min, WF window={train_window} bars"
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def add_bar(self, bar: Dict):
        """Feed new market bar"""
        self.walk_forward.add_bar(bar)
        self._check_walkforward_trigger()

    def add_bars(self, bars: List[Dict]):
        """Bulk feed bars"""
        self.walk_forward.add_bars(bars)
        self._check_walkforward_trigger()

    def record_trade(self, pnl: float, pnl_pct: float):
        """Record completed trade result"""
        self.performance.record(pnl, pnl_pct)
        self._trade_count += 1

        # Set baseline after first window fills up
        if not self.performance.baseline_set and len(self.performance.wins) >= self.performance.min_samples:
            self.performance.set_baseline()

        # Check triggers
        self._check_periodic_trigger()
        self._check_degradation_trigger()

    def force_retrain(self, reason="manual"):
        """Manually trigger retrain"""
        self._trigger_retrain(reason)

    # ----------------------------------------------------------
    # Trigger checks
    # ----------------------------------------------------------

    def _in_cooldown(self) -> bool:
        if self._last_retrain_time is None:
            return False
        elapsed = (datetime.now() - self._last_retrain_time).total_seconds() / 60
        return elapsed < self.cooldown_minutes

    def _check_periodic_trigger(self):
        if self._in_cooldown() or self._is_training:
            return
        if self._trade_count > 0 and self._trade_count % self.retrain_every_n_trades == 0:
            self._trigger_retrain(f"periodic ({self._trade_count} trades)")

    def _check_degradation_trigger(self):
        if self._in_cooldown() or self._is_training:
            return
        degraded, reason = self.performance.is_degraded(
            wr_drop_threshold=self._wr_drop_threshold,
            sharpe_drop_threshold=self._sharpe_drop_threshold,
        )
        if degraded:
            self._trigger_retrain(f"degradation: {reason}")

    def _check_walkforward_trigger(self):
        if self._in_cooldown() or self._is_training:
            return
        # เช็ค min_bars ก่อน — ถ้า buffer ยังไม่พอก็ไม่ต้อง trigger
        if self.walk_forward.buffer_size < self.walk_forward.min_bars:
            return
        if self.walk_forward.has_new_data(self.min_new_bars_for_wf):
            self._trigger_retrain(f"walk-forward ({self.walk_forward.total_bars_seen} bars)")

    def _trigger_retrain(self, reason: str):
        """Kick off non-blocking background retrain"""
        if self._is_training:
            logger.debug(f"[RetrainEngine] Skipped retrain ({reason}) — already training")
            return

        logger.info(f"[RetrainEngine] Retrain triggered: {reason}")
        self._last_retrain_trigger = reason
        thread = threading.Thread(
            target=self._retrain_worker,
            args=(reason,),
            daemon=True,
            name="retrain-worker",
        )
        thread.start()

    def _retrain_worker(self, reason: str):
        """Background worker — runs train_fn with walk-forward data"""
        with self._train_lock:
            self._is_training = True
            start = time.time()
            try:
                train_data = self.walk_forward.get_train_data()
                if train_data is None:
                    logger.warning("[RetrainEngine] Skipped — not enough data")
                    return

                logger.info(f"[RetrainEngine] Starting retrain on {len(train_data)} bars ({reason})")
                success = self.train_fn(train_data)

                elapsed = time.time() - start
                if success:
                    self._retrain_count += 1
                    self._last_retrain_time = datetime.now()
                    self.walk_forward.mark_trained()
                    # Update baseline after successful retrain
                    self.performance.set_baseline()
                    logger.info(
                        f"[RetrainEngine] Retrain #{self._retrain_count} complete "
                        f"in {elapsed:.1f}s ({reason})"
                    )
                else:
                    logger.warning(f"[RetrainEngine] Retrain returned False ({reason})")

            except Exception as e:
                logger.error(f"[RetrainEngine] Retrain error: {e}", exc_info=True)
            finally:
                self._is_training = False

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def get_stats(self) -> Dict:
        cooldown_remaining = 0
        if self._last_retrain_time and self._in_cooldown():
            elapsed = (datetime.now() - self._last_retrain_time).total_seconds() / 60
            cooldown_remaining = max(0, self.cooldown_minutes - elapsed)

        return {
            "retrain_count": self._retrain_count,
            "last_trigger": self._last_retrain_trigger,
            "last_retrain_at": self._last_retrain_time.isoformat() if self._last_retrain_time else None,
            "is_training": self._is_training,
            "in_cooldown": self._in_cooldown(),
            "cooldown_remaining_min": round(cooldown_remaining, 1),
            "trade_count": self._trade_count,
            "performance": self.performance.get_stats(),
            "walk_forward": self.walk_forward.get_stats(),
        }