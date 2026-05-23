#!/usr/bin/env python3
"""Analyze forward journal CSV and print execution diagnostics summary."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from math import isfinite


TERMINAL_EVENTS = {"CLOSE", "PARTIAL_CLOSE", "ORDER_REJECTED", "ORDER_FAILED", "ORDER_FILLED", "OPEN"}


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _group_sum(rows, key, value_key):
    grouped = defaultdict(float)
    for row in rows:
        v = _to_float(row.get(value_key))
        if v is None:
            continue
        grouped[row.get(key) or "UNKNOWN"] += v
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="journal/trades.csv")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    events = Counter((r.get("event_type") or "").strip() for r in rows)
    print(f"total events: {len(rows)}")
    print(f"order attempts: {events.get('ORDER_ATTEMPT', 0)}")
    print(f"filled: {events.get('ORDER_FILLED', 0)}")
    print(f"rejected: {events.get('ORDER_REJECTED', 0)}")
    print(f"failed: {events.get('ORDER_FAILED', 0)}")

    realized_events = [r for r in rows if (r.get("event_type") or "").strip() in {"CLOSE", "PARTIAL_CLOSE"}]
    realized_pnl = sum(_to_float(r.get("pnl")) or 0.0 for r in realized_events)
    print(f"realized pnl (close+partial_close): {realized_pnl:.2f}")

    close_events = [r for r in rows if (r.get("event_type") or "").strip() == "CLOSE"]
    close_pnls = [(_to_float(r.get("pnl")) or 0.0) for r in close_events]
    wins = [p for p in close_pnls if p > 0]
    losses = [p for p in close_pnls if p < 0]
    win_rate = (len(wins) / len(close_pnls) * 100.0) if close_pnls else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    pf_str = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    print(f"close win rate: {win_rate:.1f}%")
    print(f"close profit factor: {pf_str}")

    print("\npnl by symbol:")
    for k, v in sorted(_group_sum(realized_events, "symbol", "pnl").items()):
        print(f"  {k}: {v:.2f}")

    for dim in ["side", "regime", "entry_strategy", "market_context"]:
        print(f"\npnl by {dim}:")
        for k, v in sorted(_group_sum(realized_events, dim, "pnl").items()):
            print(f"  {k}: {v:.2f}")

    rejected = [r for r in rows if (r.get("event_type") or "").strip() == "ORDER_REJECTED"]
    reject_counts = Counter((r.get("symbol") or "UNKNOWN", r.get("reason") or r.get("comment") or "UNKNOWN") for r in rejected)
    print("\nrejections by symbol/reason:")
    for (symbol, reason), count in sorted(reject_counts.items()):
        print(f"  {symbol} | {reason}: {count}")

    slip = defaultdict(list)
    for r in rows:
        sp = _to_float(r.get("slippage_points"))
        if sp is not None:
            slip[r.get("symbol") or "UNKNOWN"].append(sp)
    print("\nslippage stats by symbol:")
    for symbol, vals in sorted(slip.items()):
        print(f"  {symbol}: n={len(vals)} avg={sum(vals)/len(vals):.2f} min={min(vals):.2f} max={max(vals):.2f}")

    attempts = [r for r in rows if (r.get("event_type") or "").strip() == "ORDER_ATTEMPT"]
    by_ticket = defaultdict(list)
    for r in rows:
        ticket = (r.get("ticket") or "").strip()
        if ticket:
            by_ticket[ticket].append((r.get("event_type") or "").strip())

    dangling = 0
    for a in attempts:
        ticket = (a.get("ticket") or "").strip()
        if not ticket:
            dangling += 1
            continue
        seen = set(by_ticket.get(ticket, []))
        if not (seen & TERMINAL_EVENTS - {"ORDER_ATTEMPT"}):
            dangling += 1
    print(f"\nattempted orders without terminal event: {dangling}")

    ctx_executed = [r for r in rows if (r.get("event_type") or "").strip() in {"ORDER_FILLED", "OPEN"} and str(r.get("context_allow_trade", "")).strip() != "" and not _truthy(r.get("context_allow_trade"))]
    print(f"context_allow_trade=false but executed: {len(ctx_executed)}")

    planned, executed = [], []
    for r in rows:
        p = _to_float(r.get("planned_rr"))
        e = _to_float(r.get("execution_rr"))
        if p is not None and e is not None:
            planned.append(p)
            executed.append(e)
    if planned:
        deltas = [e - p for p, e in zip(planned, executed)]
        print(
            f"planned_rr vs execution_rr: n={len(planned)} planned_avg={sum(planned)/len(planned):.3f} "
            f"execution_avg={sum(executed)/len(executed):.3f} delta_avg={sum(deltas)/len(deltas):.3f}"
        )
    else:
        print("planned_rr vs execution_rr: no comparable rows")


if __name__ == "__main__":
    main()
