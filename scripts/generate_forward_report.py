"""Generate a forward-performance report from the trade journal."""

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
        return [
            pnl for row in self.rows
            if row.get("event_type") in TRADE_EXIT_EVENTS
            for pnl in [self._float(row.get("pnl"))]
            if pnl is not None
        ]

    def equity_values(self) -> list[float]:
        return [value for row in self.rows for value in [self._float(row.get("equity"))] if value is not None]

    def max_drawdown(self) -> float:
        equity = 0.0
        peak = 0.0
        drawdown = 0.0
        for pnl in self.realized_pnls():
            equity += pnl
            peak = max(peak, equity)
            drawdown = min(drawdown, equity - peak)
        return abs(drawdown)

    def max_drawdown_pct(self) -> float | None:
        values = self.equity_values()
        if not values:
            return None
        peak = values[0]
        worst = 0.0
        for equity in values:
            peak = max(peak, equity)
            if peak > 0:
                worst = min(worst, (equity - peak) / peak * 100)
        return abs(worst)

    def daily_drawdown_pct(self) -> float | None:
        by_day: dict[str, list[float]] = defaultdict(list)
        for row in self.rows:
            equity = self._float(row.get("equity"))
            event_time = row.get("event_time") or ""
            if equity is not None and len(event_time) >= 10:
                by_day[event_time[:10]].append(equity)
        found = False
        worst = 0.0
        for values in by_day.values():
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

    def _pnl_breakdown(self, field: str) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for row in self.rows:
            if row.get("event_type") in TRADE_EXIT_EVENTS:
                pnl = self._float(row.get("pnl"))
                if pnl is not None:
                    totals[row.get(field) or "UNKNOWN"] += pnl
        return dict(sorted(totals.items()))

    def _trade_counts(self, field: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for row in self.rows:
            if row.get("event_type") in TRADE_EXIT_EVENTS and self._float(row.get("pnl")) is not None:
                counts[row.get(field) or "UNKNOWN"] += 1
        return dict(sorted(counts.items()))

    def symbol_breakdown(self) -> dict[str, float]:
        return self._pnl_breakdown("symbol")

    def regime_breakdown(self) -> dict[str, float]:
        return self._pnl_breakdown("regime")

    def session_breakdown(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for row in self.rows:
            if row.get("event_type") not in TRADE_EXIT_EVENTS:
                continue
            pnl = self._float(row.get("pnl"))
            if pnl is None:
                continue
            session = row.get("session") or ""
            if not session:
                hour = int((row.get("event_time") or "00")[11:13] or 0)
                session = "Asia" if hour < 7 else "London" if hour < 13 else "NewYork" if hour < 21 else "Rollover"
            totals[session] += pnl
        return dict(sorted(totals.items()))

    def _comment_marker_average(self, marker: str) -> float | None:
        values = []
        token = f"{marker}="
        for row in self.rows:
            comment = row.get("comment") or ""
            if token not in comment:
                continue
            raw = comment.split(token, 1)[1].split()[0].strip(",;")
            value = self._float(raw)
            if value is not None:
                values.append(value)
        return sum(values) / len(values) if values else None

    def average_field(self, field: str) -> float | None:
        values = [self._float(row.get(field)) for row in self.rows]
        clean = [value for value in values if value is not None]
        if clean:
            return sum(clean) / len(clean)
        return self._comment_marker_average(field)

    def execution_failure_rate(self) -> float:
        event_counts = Counter(row.get("event_type") or "UNKNOWN" for row in self.rows)
        attempts = event_counts["ORDER_ATTEMPT"]
        failures = sum(event_counts[event] for event in FAILURE_EVENTS)
        return failures / attempts * 100 if attempts else 0.0

    def metrics(self) -> dict[str, object]:
        pnls = self.realized_pnls()
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        event_counts = Counter(row.get("event_type") or "UNKNOWN" for row in self.rows)
        failure_count = sum(event_counts[event] for event in FAILURE_EVENTS)
        initial_equity = self.equity_values()[0] if self.equity_values() else None
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
            "symbol_trade_counts": self._trade_counts("symbol"),
            "session_trade_counts": self._trade_counts("session"),
            "regime_trade_counts": self._trade_counts("regime"),
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


def render_breakdown(title: str, values: dict[str, float], counts: dict[str, int] | None = None) -> list[str]:
    lines = [f"{title}:"]
    if not values:
        lines.append("  n/a")
        return lines
    for key, pnl in values.items():
        suffix = f", trades={counts[key]}" if counts and key in counts else ""
        lines.append(f"  {key}: {fmt_number(pnl)}{suffix}")
    return lines


def render_report(metrics: dict[str, object]) -> str:
    lines = ["Forward Performance Report", "==========================", ""]
    for key in (
        "total_trades", "win_rate", "profit_factor", "net_pnl", "max_drawdown", "max_drawdown_pct",
        "daily_drawdown_pct", "avg_win", "avg_loss", "expectancy", "largest_loss", "consecutive_losses",
        "execution_failures", "execution_failure_rate", "slippage_average", "spread_average",
        "confidence_average", "risk_pct_average",
    ):
        lines.append(f"{key}: {fmt_number(metrics[key])}")
    lines.append("")
    lines.extend(render_breakdown("symbol_breakdown", metrics["symbol_breakdown"], metrics.get("symbol_trade_counts")))
    lines.extend(render_breakdown("session_breakdown", metrics["session_breakdown"], metrics.get("session_trade_counts")))
    lines.extend(render_breakdown("regime_breakdown", metrics["regime_breakdown"], metrics.get("regime_trade_counts")))
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
    text = render_report(Report(load_rows(journal_path)).metrics())
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
