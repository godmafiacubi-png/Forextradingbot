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
        except ValueError:
            return None

    def realized_pnls(self) -> list[float]:
        pnls = []
        for row in self.rows:
            if row.get("event_type") in TRADE_EXIT_EVENTS:
                pnl = self._float(row.get("pnl"))
                if pnl is not None:
                    pnls.append(pnl)
        return pnls

    def max_drawdown(self) -> float:
        equity = 0.0
        peak = 0.0
        drawdown = 0.0
        for pnl in self.realized_pnls():
            equity += pnl
            peak = max(peak, equity)
            drawdown = min(drawdown, equity - peak)
        return abs(drawdown)

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

    def symbol_breakdown(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for row in self.rows:
            if row.get("event_type") in TRADE_EXIT_EVENTS:
                pnl = self._float(row.get("pnl"))
                if pnl is not None:
                    totals[row.get("symbol") or "UNKNOWN"] += pnl
        return dict(sorted(totals.items()))

    def session_breakdown(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for row in self.rows:
            if row.get("event_type") not in TRADE_EXIT_EVENTS:
                continue
            pnl = self._float(row.get("pnl"))
            if pnl is None:
                continue
            hour = int((row.get("event_time") or "00")[11:13] or 0)
            if 0 <= hour < 7:
                session = "Asia"
            elif hour < 13:
                session = "London"
            elif hour < 21:
                session = "NewYork"
            else:
                session = "Rollover"
            totals[session] += pnl
        return dict(sorted(totals.items()))

    def average_slippage(self) -> float | None:
        values = []
        for row in self.rows:
            comment = row.get("comment") or ""
            marker = "slippage_points="
            if marker not in comment:
                continue
            raw = comment.split(marker, 1)[1].split()[0].strip(",;")
            value = self._float(raw)
            if value is not None:
                values.append(value)
        if not values:
            return None
        return sum(values) / len(values)

    def metrics(self) -> dict[str, object]:
        pnls = self.realized_pnls()
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        event_counts = Counter(row.get("event_type") or "UNKNOWN" for row in self.rows)
        failure_count = sum(event_counts[event] for event in FAILURE_EVENTS)
        return {
            "total_trades": len(pnls),
            "win_rate": (len(wins) / len(pnls) * 100) if pnls else 0.0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss else None,
            "max_drawdown": self.max_drawdown(),
            "avg_win": (gross_profit / len(wins)) if wins else 0.0,
            "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "expectancy": (sum(pnls) / len(pnls)) if pnls else 0.0,
            "largest_loss": min(pnls) if pnls else 0.0,
            "consecutive_losses": self.consecutive_losses(),
            "symbol_breakdown": self.symbol_breakdown(),
            "session_breakdown": self.session_breakdown(),
            "execution_failures": failure_count,
            "execution_failure_breakdown": {event: event_counts[event] for event in sorted(FAILURE_EVENTS)},
            "slippage_average": self.average_slippage(),
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


def render_report(metrics: dict[str, object]) -> str:
    lines = ["Forward Performance Report", "==========================", ""]
    for key in (
        "total_trades", "win_rate", "profit_factor", "max_drawdown", "avg_win", "avg_loss",
        "expectancy", "largest_loss", "consecutive_losses", "execution_failures", "slippage_average",
    ):
        lines.append(f"{key}: {fmt_number(metrics[key])}")
    lines.append("")
    lines.append("symbol_breakdown:")
    for symbol, pnl in metrics["symbol_breakdown"].items():
        lines.append(f"  {symbol}: {fmt_number(pnl)}")
    lines.append("session_breakdown:")
    for session, pnl in metrics["session_breakdown"].items():
        lines.append(f"  {session}: {fmt_number(pnl)}")
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
