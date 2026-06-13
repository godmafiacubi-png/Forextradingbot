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
        avg_spread=95,
        slippage_points=2.5,
        confidence=0.72,
        risk_pct=0.25,
        regime="TREND",
        session="NewYork",
        reason="quality=A",
    )
    attempt = journal.log_order_attempt(
        "XAUUSDm", "BUY", 0.1, 2350.0, sl=2340.0, tp=2370.0, comment="attempt", avg_spread=100
    )
    filled = journal.log_order_filled(
        123, "XAUUSDm", "BUY", 0.1, 2350.0, sl=2340.0, tp=2370.0, comment="filled", avg_spread=101
    )
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
    assert rows[0]["avg_spread"] == "95"
    assert rows[0]["slippage_points"] == "2.5"
    assert attempt["avg_spread"] == 100
    assert filled["avg_spread"] == 101
    assert rows[0]["confidence"] == "0.72"
    assert rows[0]["risk_pct"] == "0.25"
    assert rows[0]["regime"] == "TREND"
    assert rows[0]["session"] == "NewYork"
    assert rows[0]["reason"] == "quality=A"
    assert rows[2]["ticket"] == "123"

    with sqlite3.connect(sqlite_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
        row = conn.execute(
            "SELECT equity, spread, avg_spread, slippage_points, confidence, risk_pct, regime, session, reason "
            "FROM trade_journal WHERE event_type='SIGNAL'"
        ).fetchone()
    assert count == 11
    assert row == (10010.0, 120.0, 95.0, 2.5, 0.72, 0.25, "TREND", "NewYork", "quality=A")


def test_sqlite_migration_adds_avg_spread_to_existing_db(tmp_path):
    sqlite_path = tmp_path / "legacy.sqlite3"
    legacy_fields = [
        ("event_time", "TEXT NOT NULL"),
        ("event_type", "TEXT NOT NULL"),
        ("ticket", "TEXT"),
        ("symbol", "TEXT"),
        ("side", "TEXT"),
        ("volume", "REAL"),
        ("price", "REAL"),
        ("sl", "REAL"),
        ("tp", "REAL"),
        ("pnl", "REAL"),
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
        ("comment", "TEXT"),
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
    ]
    columns_sql = ",\n                ".join(f"{name} {sql_type}" for name, sql_type in legacy_fields)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {columns_sql}
            )
            """
        )

    journal = TradeJournal(csv_path=None, sqlite_path=sqlite_path)
    journal.log_order_attempt("EURUSDm", "BUY", 0.1, 1.1002, avg_spread=20)

    with sqlite3.connect(sqlite_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(trade_journal)")}
        avg_spread = conn.execute(
            "SELECT avg_spread FROM trade_journal WHERE event_type='ORDER_ATTEMPT'"
        ).fetchone()[0]

    assert "avg_spread" in columns
    assert avg_spread == 20.0


def test_trade_journal_has_event_checks_sqlite_then_csv(tmp_path):
    csv_path = tmp_path / "trades.csv"
    sqlite_path = tmp_path / "trades.sqlite3"
    journal = TradeJournal(csv_path=csv_path, sqlite_path=sqlite_path)

    journal.log_open(123, "XAUUSDm", "BUY", 0.1, 2350.0)

    assert journal.has_event(123, "OPEN") is True
    assert journal.has_event(123, "CLOSE") is False


def test_active_trade_metadata_survives_restart_load(tmp_path):
    from execution.trade_logger import ActiveTradeStore

    path = tmp_path / "active_trades.json"
    store = ActiveTradeStore(path)
    store.upsert(123, {
        "symbol": "EURUSDm",
        "side": "BUY",
        "volume": 0.2,
        "entry_price": 1.1002,
        "sl": 1.099,
        "tp": 1.104,
        "entry_strategy": "breakout",
        "market_context": "TREND",
    })

    restarted_store = ActiveTradeStore(path)
    active_trades = restarted_store.load()

    assert active_trades[123]["symbol"] == "EURUSDm"
    assert active_trades[123]["entry_strategy"] == "breakout"
    assert active_trades[123]["market_context"] == "TREND"
