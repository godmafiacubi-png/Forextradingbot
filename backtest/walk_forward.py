"""Walk-forward and demo-forward validation gates before live trading.

This module is intentionally lightweight: it does not place trades and does not
optimise parameters by itself. It provides reproducible chronological slices and
pass/fail gates that can be used by optimisation scripts before promoting a
strategy profile to demo or live.
"""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """Chronological train/test window used for walk-forward validation."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self):
        return {
            "fold": self.fold,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }


class WalkForwardValidator:
    """Build walk-forward folds and validate out-of-sample backtest results."""

    def __init__(
        self,
        train_bars=500,
        test_bars=120,
        step_bars=None,
        min_folds=3,
        min_profit_factor=1.3,
        max_drawdown_pct=10.0,
        min_trades=20,
    ):
        self.train_bars = int(train_bars)
        self.test_bars = int(test_bars)
        self.step_bars = int(step_bars or test_bars)
        self.min_folds = int(min_folds)
        self.min_profit_factor = float(min_profit_factor)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_trades = int(min_trades)

    def build_splits(self, df, time_col="time"):
        if time_col not in df.columns:
            raise ValueError(f"missing required time column: {time_col}")
        if len(df) < self.train_bars + self.test_bars:
            return []

        times = pd.to_datetime(df[time_col]).reset_index(drop=True)
        splits = []
        fold = 1
        start = 0
        while start + self.train_bars + self.test_bars <= len(df):
            train_start_idx = start
            train_end_idx = start + self.train_bars - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + self.test_bars - 1
            splits.append(
                WalkForwardSplit(
                    fold=fold,
                    train_start=times.iloc[train_start_idx],
                    train_end=times.iloc[train_end_idx],
                    test_start=times.iloc[test_start_idx],
                    test_end=times.iloc[test_end_idx],
                )
            )
            fold += 1
            start += self.step_bars
        return splits

    def evaluate_results(self, fold_results):
        """Return a promotion report for out-of-sample fold summaries."""
        checks = []
        if len(fold_results) < self.min_folds:
            checks.append(f"need at least {self.min_folds} folds, got {len(fold_results)}")

        total_trades = sum(int(r.get("total_trades", 0)) for r in fold_results)
        if total_trades < self.min_trades:
            checks.append(f"need at least {self.min_trades} OOS trades, got {total_trades}")

        failing_pf = [i + 1 for i, r in enumerate(fold_results) if float(r.get("profit_factor", 0.0)) < self.min_profit_factor]
        if failing_pf:
            checks.append(f"profit factor below {self.min_profit_factor} in folds {failing_pf}")

        failing_dd = [i + 1 for i, r in enumerate(fold_results) if float(r.get("max_drawdown_pct", 100.0)) > self.max_drawdown_pct]
        if failing_dd:
            checks.append(f"drawdown above {self.max_drawdown_pct}% in folds {failing_dd}")

        avg_pf = 0.0
        avg_dd = 0.0
        if fold_results:
            avg_pf = sum(float(r.get("profit_factor", 0.0)) for r in fold_results) / len(fold_results)
            avg_dd = sum(float(r.get("max_drawdown_pct", 0.0)) for r in fold_results) / len(fold_results)

        return {
            "passed": not checks,
            "checks": checks,
            "folds": len(fold_results),
            "total_trades": total_trades,
            "avg_profit_factor": round(avg_pf, 2),
            "avg_drawdown_pct": round(avg_dd, 2),
        }


class DemoForwardGate:
    """Promotion gate that must pass before enabling live order routing."""

    def __init__(
        self,
        min_days=14,
        min_trades=20,
        min_profit_factor=1.2,
        max_drawdown_pct=6.0,
        require_positive_expectancy=True,
    ):
        self.min_days = int(min_days)
        self.min_trades = int(min_trades)
        self.min_profit_factor = float(min_profit_factor)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.require_positive_expectancy = bool(require_positive_expectancy)

    def evaluate(self, demo_summary):
        checks = []
        start = pd.to_datetime(demo_summary.get("start_time"))
        end = pd.to_datetime(demo_summary.get("end_time"))
        if pd.isna(start) or pd.isna(end):
            checks.append("demo summary must include start_time and end_time")
            days = 0
        else:
            days = max((end - start) / timedelta(days=1), 0)
            if days < self.min_days:
                checks.append(f"need at least {self.min_days} demo days, got {days:.1f}")

        trades = int(demo_summary.get("total_trades", 0))
        if trades < self.min_trades:
            checks.append(f"need at least {self.min_trades} demo trades, got {trades}")
        if float(demo_summary.get("profit_factor", 0.0)) < self.min_profit_factor:
            checks.append(f"demo PF must be >= {self.min_profit_factor}")
        if float(demo_summary.get("max_drawdown_pct", 100.0)) > self.max_drawdown_pct:
            checks.append(f"demo DD must be <= {self.max_drawdown_pct}%")
        if self.require_positive_expectancy and float(demo_summary.get("expectancy", 0.0)) <= 0:
            checks.append("demo expectancy must be positive")

        return {
            "passed": not checks,
            "checks": checks,
            "demo_days": round(days, 2),
            "total_trades": trades,
        }
