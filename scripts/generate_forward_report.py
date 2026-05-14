"""Generate a forward-performance report from the trade journal.

This report is intentionally dependency-light so it can run in CI and on a VPS
without pandas. It summarizes both edge metrics and execution-quality events.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

TRADE_EXIT_EVENTS = {"CLOSE", "PARTIAL_CLOSE"}
FAILURE_EVENTS = {"ORDER_REJECTED", "ORDER_FAILED", "RISK_BLOCKED", "NEWS_BLOCKED"}


@dataclass
class Report:
    rows: list[dict[str, str]]

    @staticmethod
    def _float(value: str | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def realized_pnls(self) -> list[float]:
        pnls = []
        for row in self.rows:
            if row.get("event_type") in TRADE_EXIT_EVENTS:
                pnl = self._float(row.get("pnl"))
                if pnl is not None:
                    pnls.append(pnl)
        return pnls

    def equity_values(self) -> list[float]:
        values = []
        for row in self.rows:
            equity = self._float(row.get("equity"))
            if equity is not None:
                values.append(equity)
        return values

    def initial_equity(self) -> float | None:
        values = self.equity_values()
        return values[0] if values else None

    def max_drawdown(self) -> float:
        """Realized-PnL drawdown in account currency, for legacy journals."""
        equity = 0.0
        peak = 0.0
        drawdown = 0.0
        for pnl in self.realized_pnls():
            equity += pnl
            peak = max(peak, equity)
            drawdown = min(drawdown, equity - peak)
        return abs(drawdown)

    def max_drawdown_pct(self) -> float | None:
        """Equity-curve drawdown percentage when journal equity snapshots exist."""
        values = self.equity_values()
        if not values:
            return None
        peak = values[0]
        worst_pct = 0.0
        for equity in values:
            peak = max(peak, equity)
            if peak > 0:
                worst_pct = min(worst_pct, (equity - peak) / peak * 100)
        return abs(worst_pct)

    def daily_drawdown_pct(self) -> float | None:
        """Approximate worst intraday equity drawdown from journal dates."""
        by_day: dict[str, list[float]] = defaultdict(list)
        for row in self.rows:
            equity = self._float(row.get("equity"))
            event_time = row.get("event_time") or ""
            if equity is not None and len(event_time) >= 10:
                by_day[event_time[:10]].append(equity)
        worst = 0.0
        found = False
        for values in by_day.values():
            if not values:
                continue
            found = True
            peak = values[0]
            for equity in values:
                peak = max(peak, equity)
                if peak > 0:
                    worst = min(worst, (equity - peak) / peak * 100)
        return abs(worst) if found else None

    def consecutive_losses(self) -> int:
        longest = 0
        current = 0
        for pnl in self.realized_pnls():
            if pnl < 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    def _breakdown_by(self, field: str) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        for row in self.rows:
            if row.get("event_type") in TRADE_EXIT_EVENTS:
                pnl = self._float(row.get("pnl"))
                if pnl is not None:
                    key = row.get(field) or "UNKNOWN"
                    totals[key]["pnl"] += pnl
                    totals[key]["trades"] += 1
        return dict(sorted(totals.items()))

    def symbol_breakdown(self) -> dict[str, dict[str, float]]:
        return self._breakdown_by("symbol")

    def regime_breakdown(self) -> dict[str, dict[str, float]]:
        return self._breakdown_by("regime")

    def session_breakdown(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        for row in self.rows:
            if row.get("event_type") not in TRADE_EXIT_EVENTS:
                continue
            pnl = self._float(row.get("pnl"))
            if pnl is None:
                continue
            session = row.get("session") or ""
            if not session:
                hour = int((row.get("event_time") or "00")[11:13] or 0)
                if 0 <= hour < 7:
                    session = "Asia"
                elif hour < 13:
                    session = "London"
                elif hour < 21:
                    session = "NewYork"
                else:
                    session = "Rollover"
            totals[session]["pnl"] += pnl
            totals[session]["trades"] += 1
        return dict(sorted(totals.items()))

    def average_field(self, field: str) -> float | None:
        values = [self._float(row.get(field)) for row in self.rows]
        clean = [value for value in values if value is not None]
        if not clean:
            return None
        return sum(clean) / len(clean)

    def execution_failure_rate(self) -> float:
        event_counts = Counter(row.get("event_type") or "UNKNOWN" for row in self.rows)
        attempts = event_counts["ORDER_ATTEMPT"]
        failures = sum(event_counts[event] for event in FAILURE_EVENTS)
        if attempts <= 0:
            return 0.0
        return failures / attempts * 100

    def metrics(self) -> dict[str, object]:
        pnls = self.realized_pnls()
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        event_counts = Counter(row.get("event_type") or "UNKNOWN" for row in self.rows)
        failure_count = sum(event_counts[event] for event in FAILURE_EVENTS)
        initial_equity = self.initial_equity()
        max_dd = self.max_drawdown()
        return {
            "total_trades": len(pnls),
            "win_rate": (len(wins) / len(pnls) * 100) if pnls else 0.0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss else None,
            "net_pnl": sum(pnls),
            "max_drawdown": max_dd,
            "max_drawdown_pct": self.max_drawdown_pct() or ((max_dd / initial_equity * 100) if initial_equity else None),
            "daily_drawdown_pct": self.daily_drawdown_pct(),
            "avg_win": (gross_profit / len(wins)) if wins else 0.0,
            "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "expectancy": (sum(pnls) / len(pnls)) if pnls else 0.0,
            "largest_loss": min(pnls) if pnls else 0.0,
            "consecutive_losses": self.consecutive_losses(),
            "symbol_breakdown": self.symbol_breakdown(),
            "session_breakdown": self.session_breakdown(),
            "regime_breakdown": self.regime_breakdown(),
            "execution_failures": failure_count,
            "execution_failure_rate": self.execution_failure_rate(),
            "execution_failure_breakdown": {event: event_counts[event] for event in sorted(FAILURE_EVENTS)},
            "slippage_average": self.average_field("slippage_points"),
            "spread_average": self.average_field("spread"),
            "confidence_average": self.average_field("confidence"),
            "risk_pct_average": self.average_field("risk_pct"),
        }


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt_number(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_breakdown(title: str, values: dict[str, dict[str, float]]) -> list[str]:
    lines = [f"{title}:"]
    if not values:
        lines.append("  n/a")
        return lines
    for key, data in values.items():
        lines.append(f"  {key}: pnl={fmt_number(data['pnl'])}, trades={int(data['trades'])}")
    return lines


def render_report(metrics: dict[str, object]) -> str:
    lines = ["Forward Performance Report", "==========================", ""]
    for key in (
        "total_trades", "win_rate", "profit_factor", "net_pnl", "max_drawdown",
        "max_drawdown_pct", "daily_drawdown_pct", "avg_win", "avg_loss",
        "expectancy", "largest_loss", "consecutive_losses", "execution_failures",
        "execution_failure_rate", "slippage_average", "spread_average",
        "confidence_average", "risk_pct_average",
    ):
        lines.append(f"{key}: {fmt_number(metrics[key])}")
    lines.append("")
    lines.extend(render_breakdown("symbol_breakdown", metrics["symbol_breakdown"]))
    lines.extend(render_breakdown("session_breakdown", metrics["session_breakdown"]))
    lines.extend(render_breakdown("regime_breakdown", metrics["regime_breakdown"]))
    lines.append("execution_failure_breakdown:")
    for event, count in metrics["execution_failure_breakdown"].items():
        lines.append(f"  {event}: {count}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a forward-performance report from journal CSV")
    parser.add_argument("--journal", default="journal/trades.csv", help="Path to journal CSV")
    parser.add_argument("--output", help="Optional path for a text report")
    args = parser.parse_args()

    journal_path = Path(args.journal)
    if not journal_path.exists():
        print(f"Journal CSV not found: {journal_path}")
        return 1

    rows = load_rows(journal_path)
    text = render_report(Report(rows).metrics())
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
