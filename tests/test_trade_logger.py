import csv
import sqlite3

from execution.trade_logger import TradeJournal


def test_trade_journal_writes_csv_and_sqlite(tmp_path):
    csv_path = tmp_path / "trades.csv"
    sqlite_path = tmp_path / "trades.sqlite3"
    journal = TradeJournal(csv_path=csv_path, sqlite_path=sqlite_path)

    journal.log_signal("XAUUSDm", "BUY", 2350.0, comment="signal")
    journal.log_order_attempt("XAUUSDm", "BUY", 0.1, 2350.0, sl=2340.0, tp=2370.0, comment="attempt")
    journal.log_order_filled(123, "XAUUSDm", "BUY", 0.1, 2350.0, sl=2340.0, tp=2370.0, comment="filled")
    journal.log_open(123, "XAUUSDm", "BUY", 0.1, 2350.0, sl=2340.0, tp=2370.0, comment="open")
    journal.log_sl_modified(123, "XAUUSDm", "BUY", sl=2345.0, tp=2370.0, comment="trail")
    journal.log_partial_close(123, "XAUUSDm", "BUY", 0.05, 2355.0, pnl=25.0, comment="partial")
    journal.log_close(123, "XAUUSDm", "BUY", 0.05, 2360.0, pnl=100.0, comment="close")
    journal.log_order_rejected("XAUUSDm", "BUY", comment="bad stops")
    journal.log_order_failed("XAUUSDm", "BUY", comment="retcode=10030")
    journal.log_risk_blocked("XAUUSDm", "BUY", comment="limit")
    journal.log_news_blocked(123, "XAUUSDm", "BUY", comment="news")

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["event_type"] for row in rows] == [
        "SIGNAL",
        "ORDER_ATTEMPT",
        "ORDER_FILLED",
        "OPEN",
        "SL_MODIFIED",
        "PARTIAL_CLOSE",
        "CLOSE",
        "ORDER_REJECTED",
        "ORDER_FAILED",
        "RISK_BLOCKED",
        "NEWS_BLOCKED",
    ]
    assert rows[0]["symbol"] == "XAUUSDm"
    assert rows[2]["ticket"] == "123"

    with sqlite3.connect(sqlite_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
    assert count == 11
