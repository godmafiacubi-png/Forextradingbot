"""Persistent trading journal for live/demo execution review.

The bot logs are useful for debugging, but journal rows are easier to audit for
edge, drawdown, profit factor, and execution-quality criteria before going live.
"""

import csv
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
                    comment TEXT
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
            ):
                if field not in existing:
                    conn.execute(f"ALTER TABLE trade_journal ADD COLUMN {field} {sql_type}")

    def append_event(self, event_type, ticket=None, symbol="", side="", volume=None,
                     price=None, sl=None, tp=None, pnl=None, balance=None, equity=None,
                     spread=None, slippage_points=None, confidence=None, risk_pct=None,
                     regime="", session="", reason="", source="", entry_strategy="",
                     strategy_confidence=None, quality_score=None, quality_grade="",
                     ml_prob=None, ict_score=None, adx=None, rsi=None,
                     planned_rr=None, execution_rr=None, rl_reward=None,
                     q_value=None, action="", comment=""):
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
        return self.append_event(EVENT_CLOSE, ticket, symbol, side, volume, price, None, None, pnl,
                                 comment=comment, **context)

    def log_rl_trade_result(self, ticket, symbol="", side="", pnl=None, rl_reward=None, q_value=None,
                            action="", confidence=None, comment="", **context):
        return self.append_event("RL_TRADE_RESULT", ticket, symbol, side, pnl=pnl,
                                 confidence=confidence, source="deep_rl", rl_reward=rl_reward,
                                 q_value=q_value, action=action, comment=comment, **context)

    def log_risk_blocked(self, symbol, side="", volume=None, price=None, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_RISK_BLOCKED, symbol=symbol, side=side, volume=volume,
                                 price=price, sl=sl, tp=tp, comment=comment, **context)

    def log_news_blocked(self, ticket=None, symbol="", side="", price=None, sl=None, tp=None, comment="", **context):
        return self.append_event(EVENT_NEWS_BLOCKED, ticket, symbol, side, None, price, sl, tp, None,
                                 comment=comment, **context)
