import csv
import sqlite3

from execution.trade_logger import TradeJournal


def test_trade_journal_writes_csv_and_sqlite(tmp_path):
    csv_path = tmp_path / "trades.csv"
    sqlite_path = tmp_path / "trades.sqlite3"
    journal = TradeJournal(csv_path=csv_path, sqlite_path=sqlite_path)

    journal.log_signal(
        "XAUUSDm",
        "BUY",
        2350.0,
        comment="signal",
        balance=10_000,
        equity=10_010,
        spread=120,
        slippage_points=2.5,
        confidence=0.72,
        risk_pct=0.25,
        regime="TREND",
        session="NewYork",
        reason="quality=A",
    )
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
    assert rows[0]["equity"] == "10010"
    assert rows[0]["spread"] == "120"
    assert rows[0]["slippage_points"] == "2.5"
    assert rows[0]["confidence"] == "0.72"
    assert rows[0]["risk_pct"] == "0.25"
    assert rows[0]["regime"] == "TREND"
    assert rows[0]["session"] == "NewYork"
    assert rows[0]["reason"] == "quality=A"
    assert rows[2]["ticket"] == "123"

    with sqlite3.connect(sqlite_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
        row = conn.execute(
            "SELECT equity, spread, slippage_points, confidence, risk_pct, regime, session, reason "
            "FROM trade_journal WHERE event_type='SIGNAL'"
        ).fetchone()
    assert count == 11
    assert row == (10010.0, 120.0, 2.5, 0.72, 0.25, "TREND", "NewYork", "quality=A")
