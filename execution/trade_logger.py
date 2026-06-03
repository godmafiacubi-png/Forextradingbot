"""Persistent trading journal for live/demo execution review.

The bot logs are useful for debugging, but journal rows are easier to audit for
edge, drawdown, profit factor, and execution-quality criteria before going live.
"""

import csv
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


JOURNAL_FIELDS = [
    "event_time",
    "event_type",
    "ticket",
    "symbol",
    "side",
    "volume",
    "price",
    "sl",
    "tp",
    "pnl",
    "balance",
    "equity",
    "spread",
    "slippage_points",
    "confidence",
    "risk_pct",
    "regime",
    "session",
    "reason",
    "source",
    "entry_strategy",
    "strategy_confidence",
    "quality_score",
    "quality_grade",
    "ml_prob",
    "ict_score",
    "adx",
    "rsi",
    "planned_rr",
    "execution_rr",
    "rl_reward",
    "q_value",
    "action",
    "comment",
    "market_context",
    "context_bias",
    "context_strategy",
    "context_allow_trade",
    "context_reason",
    "context_risk_mult",
    "context_tp_rr",
    "context_sl_atr_mult",
    "context_min_quality_score",
    "max_slippage_points",
]

EVENT_SIGNAL = "SIGNAL"
EVENT_ORDER_ATTEMPT = "ORDER_ATTEMPT"
EVENT_ORDER_REJECTED = "ORDER_REJECTED"
EVENT_ORDER_FAILED = "ORDER_FAILED"
EVENT_ORDER_FILLED = "ORDER_FILLED"
EVENT_OPEN = "OPEN"
EVENT_SL_MODIFIED = "SL_MODIFIED"
EVENT_PARTIAL_CLOSE = "PARTIAL_CLOSE"
EVENT_CLOSE = "CLOSE"
EVENT_RISK_BLOCKED = "RISK_BLOCKED"
EVENT_NEWS_BLOCKED = "NEWS_BLOCKED"


class ActiveTradeStore:
    """Small JSON-backed store for metadata needed throughout a trade lifecycle."""

    def __init__(self, path="journal/active_trades.json"):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ticket_key(ticket):
        return str(ticket)

    @staticmethod
    def _runtime_ticket(ticket):
        try:
            return int(ticket)
        except (TypeError, ValueError):
            return ticket

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): ActiveTradeStore._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ActiveTradeStore._json_safe(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def load(self):
        if not self.path or not self.path.exists():
            return {}
        try:
            with self.path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {self._runtime_ticket(ticket): metadata for ticket, metadata in payload.items() if isinstance(metadata, dict)}

    def save(self, active_trades):
        if not self.path:
            return
        payload = {self._ticket_key(ticket): self._json_safe(metadata) for ticket, metadata in active_trades.items()}
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temp_path, self.path)

    def upsert(self, ticket, metadata):
        active_trades = self.load()
        current = dict(active_trades.get(self._runtime_ticket(ticket), {}))
        current.update(metadata or {})
        active_trades[self._runtime_ticket(ticket)] = current
        self.save(active_trades)
        return current

    def get(self, ticket, default=None):
        return self.load().get(self._runtime_ticket(ticket), default)

    def remove(self, ticket):
        active_trades = self.load()
        removed = active_trades.pop(self._runtime_ticket(ticket), None)
        self.save(active_trades)
        return removed


class TradeJournal:
    """Append-only CSV/SQLite journal for order and trade lifecycle events."""

    def __init__(self, csv_path="journal/trades.csv", sqlite_path=None):
        self.csv_path = Path(csv_path) if csv_path else None
        self.sqlite_path = Path(sqlite_path) if sqlite_path else None
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, fieldnames=JOURNAL_FIELDS).writeheader()
        if self.sqlite_path:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_sqlite_schema()

    @staticmethod
    def _now_iso():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _ensure_sqlite_schema(self):
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    ticket TEXT,
                    symbol TEXT,
                    side TEXT,
                    volume REAL,
                    price REAL,
                    sl REAL,
                    tp REAL,
                    pnl REAL,
                    balance REAL,
                    equity REAL,
                    spread REAL,
                    slippage_points REAL,
                    confidence REAL,
                    risk_pct REAL,
                    regime TEXT,
                    session TEXT,
                    reason TEXT,
                    source TEXT,
                    entry_strategy TEXT,
                    strategy_confidence REAL,
                    quality_score REAL,
                    quality_grade TEXT,
                    ml_prob REAL,
                    ict_score REAL,
                    adx REAL,
                    rsi REAL,
                    planned_rr REAL,
                    execution_rr REAL,
                    rl_reward REAL,
                    q_value REAL,
                    action TEXT,
                    comment TEXT,
                    market_context TEXT,
                    context_bias TEXT,
                    context_strategy TEXT,
                    context_allow_trade INTEGER,
                    context_reason TEXT,
                    context_risk_mult REAL,
                    context_tp_rr REAL,
                    context_sl_atr_mult REAL,
                    context_min_quality_score INTEGER,
                    max_slippage_points REAL
                )
                """
            )
            existing = {row[1] for row in conn.execute("PRAGMA table_info(trade_journal)")}
            for field, sql_type in (
                ("balance", "REAL"),
                ("equity", "REAL"),
                ("spread", "REAL"),
                ("slippage_points", "REAL"),
                ("confidence", "REAL"),
                ("risk_pct", "REAL"),
                ("regime", "TEXT"),
                ("session", "TEXT"),
                ("reason", "TEXT"),
                ("source", "TEXT"),
                ("entry_strategy", "TEXT"),
                ("strategy_confidence", "REAL"),
                ("quality_score", "REAL"),
                ("quality_grade", "TEXT"),
                ("ml_prob", "REAL"),
                ("ict_score", "REAL"),
                ("adx", "REAL"),
                ("rsi", "REAL"),
                ("planned_rr", "REAL"),
                ("execution_rr", "REAL"),
                ("rl_reward", "REAL"),
                ("q_value", "REAL"),
                ("action", "TEXT"),
                ("market_context", "TEXT"),
                ("context_bias", "TEXT"),
                ("context_strategy", "TEXT"),
                ("context_allow_trade", "INTEGER"),
                ("context_reason", "TEXT"),
                ("context_risk_mult", "REAL"),
                ("context_tp_rr", "REAL"),
                ("context_sl_atr_mult", "REAL"),
                ("context_min_quality_score", "INTEGER"),
                ("max_slippage_points", "REAL"),
            ):
                if field not in existing:
                    conn.execute(f"ALTER TABLE trade_journal ADD COLUMN {field} {sql_type}")

    def has_event(self, ticket, event_type):
        """Return True when the current journal already has an event for a ticket."""
        ticket_value = "" if ticket is None else str(ticket)
        if self.sqlite_path and self.sqlite_path.exists():
            try:
                with sqlite3.connect(self.sqlite_path) as conn:
                    row = conn.execute(
                        """
                        SELECT 1
                        FROM trade_journal
                        WHERE ticket = ? AND event_type = ?
                        LIMIT 1
                        """,
                        (ticket_value, event_type),
                    ).fetchone()
                return row is not None
            except sqlite3.Error:
                # Fall back to CSV below if the SQLite journal is unavailable or
                # from an older/incomplete test fixture.
                pass

        if not self.csv_path or not self.csv_path.exists():
            return False

        with self.csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("ticket") == ticket_value and row.get("event_type") == event_type:
                    return True
        return False

    def append_event(self, event_type, ticket=None, symbol="", side="", volume=None,
                     price=None, sl=None, tp=None, pnl=None, balance=None, equity=None,
                     spread=None, slippage_points=None, confidence=None, risk_pct=None,
                     regime="", session="", reason="", source="", entry_strategy="",
                     strategy_confidence=None, quality_score=None, quality_grade="",
                     ml_prob=None, ict_score=None, adx=None, rsi=None,
                     planned_rr=None, execution_rr=None, rl_reward=None,
                     q_value=None, action="", comment="", market_context="",
                     context_bias="", context_strategy="", context_allow_trade=None,
                     context_reason="", context_risk_mult=None, context_tp_rr=None,
                     context_sl_atr_mult=None, context_min_quality_score=None,
                     max_slippage_points=None):
        row = {
            "event_time": self._now_iso(),
            "event_type": event_type,
            "ticket": "" if ticket is None else str(ticket),
            "symbol": symbol or "",
            "side": side or "",
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
            "pnl": pnl,
            "balance": balance,
            "equity": equity,
            "spread": spread,
            "slippage_points": slippage_points,
            "confidence": confidence,
            "risk_pct": risk_pct,
            "regime": regime or "",
            "session": session or "",
            "reason": reason or "",
            "source": source or "",
            "entry_strategy": entry_strategy or "",
            "strategy_confidence": strategy_confidence,
            "quality_score": quality_score,
            "quality_grade": quality_grade or "",
            "ml_prob": ml_prob,
            "ict_score": ict_score,
            "adx": adx,
            "rsi": rsi,
            "planned_rr": planned_rr,
            "execution_rr": execution_rr,
            "rl_reward": rl_reward,
            "q_value": q_value,
            "action": "" if action is None else str(action),
            "comment": comment or "",
            "market_context": market_context or "",
            "context_bias": context_bias or "",
            "context_strategy": context_strategy or "",
            "context_allow_trade": context_allow_trade,
            "context_reason": context_reason or "",
            "context_risk_mult": context_risk_mult,
            "context_tp_rr": context_tp_rr,
            "context_sl_atr_mult": context_sl_atr_mult,
            "context_min_quality_score": context_min_quality_score,
            "max_slippage_points": max_slippage_points,
        }
        if self.csv_path:
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=JOURNAL_FIELDS).writerow(row)
        if self.sqlite_path:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    f"""
                    INSERT INTO trade_journal ({', '.join(JOURNAL_FIELDS)})
                    VALUES ({', '.join('?' for _ in JOURNAL_FIELDS)})
                    """,
                    tuple(row[field] for field in JOURNAL_FIELDS),
                )
        return row

    def log_signal(self, symbol, side="", price=None, comment="", **context):
        return self.append_event(EVENT_SIGNAL, symbol=symbol, side=side, price=price, comment=comment, **context)

    def log_order_attempt(self, symbol, side, volume, price, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_ORDER_ATTEMPT, symbol=symbol, side=side, volume=volume,
                                 price=price, sl=sl, tp=tp, comment=comment, **context)

    def log_order_rejected(self, symbol, side="", volume=None, price=None, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_ORDER_REJECTED, symbol=symbol, side=side, volume=volume,
                                 price=price, sl=sl, tp=tp, comment=comment, **context)

    def log_order_failed(self, symbol, side="", volume=None, price=None, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_ORDER_FAILED, symbol=symbol, side=side, volume=volume,
                                 price=price, sl=sl, tp=tp, comment=comment, **context)

    def log_order_filled(self, ticket, symbol, side, volume, price, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_ORDER_FILLED, ticket, symbol, side, volume, price, sl, tp, None,
                                 comment=comment, **context)

    def log_open(self, ticket, symbol, side, volume, price, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_OPEN, ticket, symbol, side, volume, price, sl, tp, None,
                                 comment=comment, **context)

    def log_sl_modified(self, ticket, symbol="", side="", price=None, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_SL_MODIFIED, ticket, symbol, side, None, price, sl, tp, None,
                                 comment=comment, **context)

    def log_partial_close(self, ticket, symbol="", side="", volume=None, price=None, pnl=None, comment="", **context):
        return self.append_event(EVENT_PARTIAL_CLOSE, ticket, symbol, side, volume, price, None, None, pnl,
                                 comment=comment, **context)

    def log_close(self, ticket, symbol="", side="", volume=None, price=None, pnl=None, comment="", **context):
        if self.has_event(ticket, EVENT_CLOSE):
            return None
        return self.append_event(EVENT_CLOSE, ticket, symbol, side, volume, price, None, None, pnl,
                                 comment=comment, **context)

    def log_rl_trade_result(self, ticket, symbol="", side="", pnl=None, rl_reward=None, q_value=None,
                            action="", confidence=None, comment="", **context):
        if self.has_event(ticket, "RL_TRADE_RESULT"):
            return None
        return self.append_event("RL_TRADE_RESULT", ticket, symbol, side, pnl=pnl,
                                 confidence=confidence, source="deep_rl", rl_reward=rl_reward,
                                 q_value=q_value, action=action, comment=comment, **context)

    def log_risk_blocked(self, symbol, side="", volume=None, price=None, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_RISK_BLOCKED, symbol=symbol, side=side, volume=volume,
                                 price=price, sl=sl, tp=tp, comment=comment, **context)

    def log_news_blocked(self, ticket=None, symbol="", side="", price=None, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_NEWS_BLOCKED, ticket, symbol, side, None, price, sl, tp, None,
                                 comment=comment, **context)
