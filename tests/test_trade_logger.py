import csv
import sqlite3

from execution.trade_logger import TradeJournal


def test_trade_journal_writes_csv_and_sqlite(tmp_path):
    csv_path = tmp_path / "trades.csv"
    sqlite_path = tmp_path / "trades.sqlite3"
    journal = TradeJournal(csv_path=csv_path, sqlite_path=sqlite_path)

    journal.log_open(123, "XAUUSDm", "BUY", 0.1, 2350.0, sl=2340.0, tp=2370.0, comment="test")
    journal.log_close(123, "XAUUSDm", "BUY", 0.1, 2360.0, pnl=100.0, comment="test")

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["event_type"] for row in rows] == ["OPEN", "CLOSE"]
    assert rows[0]["symbol"] == "XAUUSDm"

    with sqlite3.connect(sqlite_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
    assert count == 2
