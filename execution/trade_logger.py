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
    "comment",
]


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
                    comment TEXT
                )
                """
            )

    def append_event(self, event_type, ticket=None, symbol="", side="", volume=None,
                     price=None, sl=None, tp=None, pnl=None, comment=""):
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
            "comment": comment or "",
        }
        if self.csv_path:
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=JOURNAL_FIELDS).writerow(row)
        if self.sqlite_path:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trade_journal (
                        event_time, event_type, ticket, symbol, side, volume,
                        price, sl, tp, pnl, comment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(row[field] for field in JOURNAL_FIELDS),
                )
        return row

    def log_open(self, ticket, symbol, side, volume, price, sl=None, tp=None, comment=""):
        return self.append_event("OPEN", ticket, symbol, side, volume, price, sl, tp, None, comment)

    def log_close(self, ticket, symbol="", side="", volume=None, price=None, pnl=None, comment=""):
        return self.append_event("CLOSE", ticket, symbol, side, volume, price, None, None, pnl, comment)
