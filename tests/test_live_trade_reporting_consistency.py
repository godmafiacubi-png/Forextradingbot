import csv
import sys
import types
import importlib

from execution.trade_logger import TradeJournal
from monitoring.performance_tracker import PerformanceTracker
from scripts.generate_forward_report import Report, render_report


class _Connector:
    def get_symbol_info(self, symbol):
        return {
            "point": 0.00001,
            "digits": 5,
            "bid": 1.10000,
            "ask": 1.10020,
            "spread": 20,
            "volume_min": 0.01,
            "volume_max": 10.0,
            "volume_step": 0.01,
        }


def _load_order_manager(monkeypatch, sent):
    mt5_stub = types.SimpleNamespace(
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        TRADE_ACTION_DEAL=10,
        TRADE_ACTION_SLTP=11,
        ORDER_TIME_GTC=20,
        ORDER_FILLING_IOC=30,
        TRADE_RETCODE_DONE=10009,
        symbol_info=lambda symbol: types.SimpleNamespace(digits=5, point=0.00001, trade_stops_level=10),
        positions_get=lambda *args, **kwargs: [],
        order_send=lambda request: sent.append(request) or types.SimpleNamespace(retcode=10009, order=456, comment="ok"),
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_stub)
    sys.modules.pop("execution.order_manager", None)
    import execution.order_manager as module
    return importlib.reload(module)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_filled_trade_creates_exactly_one_open_event(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    journal_path = tmp_path / "trades.csv"
    journal = TradeJournal(csv_path=journal_path)
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=journal)
    tracker = PerformanceTracker(journal=journal)

    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102, "consistency",
        reference_price=1.10000, max_slippage_points=50,
    )
    tracker.log_trade(ticket, "EURUSDm", "BUY", 1.1002, 0.1)

    tracker.close_trade(ticket, 1.101, actual_pnl=8.0)

    rows = _rows(journal_path)
    lifecycle_rows = [
        row for row in rows
        if row["event_type"] == "ORDER_ATTEMPT" or row["ticket"] == str(ticket)
    ]
    assert [row["event_type"] for row in lifecycle_rows] == ["ORDER_ATTEMPT", "ORDER_FILLED", "OPEN", "CLOSE"]
    open_rows = [row for row in rows if row["event_type"] == "OPEN" and row["ticket"] == str(ticket)]
    assert len(open_rows) == 1
    assert open_rows[0]["source"] == "order_manager"
    assert lifecycle_rows[-1]["source"] == "performance_tracker"


def test_positive_close_pnl_appears_in_forward_report():
    metrics = Report([
        {"event_type": "CLOSE", "pnl": "42.5", "symbol": "EURUSDm", "session": "London"}
    ]).metrics()

    assert metrics["net_pnl"] == 42.5
    assert metrics["total_trades"] == 1
    assert "net_pnl: 42.50" in render_report(metrics)


def test_performance_tracker_uses_injected_shared_journal(tmp_path):
    journal = TradeJournal(csv_path=tmp_path / "trades.csv")
    tracker = PerformanceTracker(journal=journal)

    assert tracker.journal is journal


def test_close_trade_preserves_diagnostics_fields(tmp_path):
    journal_path = tmp_path / "trades.csv"
    journal = TradeJournal(csv_path=journal_path)
    tracker = PerformanceTracker(journal=journal)

    journal.log_open(900, "EURUSDm", "BUY", 0.2, 1.1, source="order_manager")
    tracker.log_trade(900, "EURUSDm", "BUY", 1.1, 0.2)
    tracker.close_trade(
        900,
        1.105,
        actual_pnl=25.0,
        entry_strategy="liquidity_sweep_reversal",
        market_context="TREND",
        planned_rr=2.0,
        execution_rr=1.8,
        quality_score=82,
        regime="TREND",
    )

    close_row = [row for row in _rows(journal_path) if row["event_type"] == "CLOSE"][0]
    assert close_row["entry_strategy"] == "liquidity_sweep_reversal"
    assert close_row["market_context"] == "TREND"
    assert close_row["planned_rr"] == "2.0"
    assert close_row["execution_rr"] == "1.8"
    assert close_row["quality_score"] == "82"
    assert close_row["regime"] == "TREND"


def test_close_trade_backfills_synthetic_open_when_missing(tmp_path):
    journal_path = tmp_path / "trades.csv"
    journal = TradeJournal(csv_path=journal_path)
    tracker = PerformanceTracker(journal=journal)

    tracker.log_trade(901, "EURUSDm", "BUY", 1.1, 0.2)
    tracker.close_trade(
        901,
        1.105,
        actual_pnl=25.0,
        entry_strategy="breakout",
        market_context="RANGE",
        planned_rr=1.5,
        execution_rr=1.4,
        quality_score=76,
        regime="RANGE",
        entry_price=1.1001,
        lots=0.2,
        sl=1.095,
        tp=1.108,
    )

    rows = _rows(journal_path)
    assert [row["event_type"] for row in rows] == ["OPEN", "CLOSE"]
    open_row = rows[0]
    assert open_row["source"] == "lifecycle_backfill"
    assert open_row["comment"] == "synthetic open reconstructed from active trade metadata"
    assert open_row["reason"] == "missing_open_event"
    assert open_row["price"] == "1.1001"
    assert open_row["volume"] == "0.2"
    assert open_row["sl"] == "1.095"
    assert open_row["tp"] == "1.108"
    assert open_row["entry_strategy"] == "breakout"
    assert rows[1]["event_type"] == "CLOSE"


def test_duplicate_open_tracking_does_not_count_as_two_dashboard_trades(tmp_path):
    journal = TradeJournal(csv_path=tmp_path / "trades.csv")
    tracker = PerformanceTracker(journal=journal)

    tracker.log_trade(789, "EURUSDm", "BUY", 1.1, 0.1)
    tracker.log_trade(789, "EURUSDm", "BUY", 1.1, 0.1)
    assert len(tracker.open_trades) == 1

    tracker.close_trade(789, 1.101, actual_pnl=10.0)
    stats = tracker.get_stats()
    assert stats["total_trades"] == 1
    assert stats["total_pnl"] == 10.0


def test_deep_rl_result_logs_pnl_reward_q_action_confidence_together(tmp_path):
    journal_path = tmp_path / "trades.csv"
    journal = TradeJournal(csv_path=journal_path)

    journal.log_rl_trade_result(
        321,
        "EURUSDm",
        "BUY",
        pnl=12.25,
        rl_reward=1.75,
        q_value=0.33,
        action=1,
        confidence=0.81,
        entry_strategy="liquidity_sweep_reversal",
        market_context="TREND",
        planned_rr=2.0,
        execution_rr=1.8,
        quality_score=88,
        regime="TREND",
        comment="pnl=12.25 rl_reward=1.75 q_value=0.33 action=1 confidence=0.81",
    )

    row = _rows(journal_path)[0]
    assert row["event_type"] == "RL_TRADE_RESULT"
    assert row["source"] == "deep_rl"
    assert row["pnl"] == "12.25"
    assert row["rl_reward"] == "1.75"
    assert row["q_value"] == "0.33"
    assert row["action"] == "1"
    assert row["confidence"] == "0.81"
    assert row["entry_strategy"] == "liquidity_sweep_reversal"
    assert row["market_context"] == "TREND"
    assert row["planned_rr"] == "2.0"
    assert row["execution_rr"] == "1.8"
    assert row["quality_score"] == "88"
    assert row["regime"] == "TREND"


def test_duplicate_close_for_ticket_is_not_appended(tmp_path):
    journal_path = tmp_path / "trades.csv"
    journal = TradeJournal(csv_path=journal_path)
    tracker = PerformanceTracker(journal=journal)

    journal.log_open(902, "EURUSDm", "BUY", 0.1, 1.1, source="order_manager")
    journal.log_close(902, "EURUSDm", "BUY", 0.1, 1.101, 10.0, source="order_manager")
    tracker.log_trade(902, "EURUSDm", "BUY", 1.1, 0.1)
    tracker.close_trade(902, 1.101, actual_pnl=10.0)

    close_rows = [row for row in _rows(journal_path) if row["event_type"] == "CLOSE"]
    assert len(close_rows) == 1
    assert close_rows[0]["source"] == "order_manager"


def test_duplicate_rl_trade_result_for_ticket_is_not_appended(tmp_path):
    journal_path = tmp_path / "trades.csv"
    journal = TradeJournal(csv_path=journal_path)

    journal.log_rl_trade_result(903, "EURUSDm", "BUY", pnl=10.0, rl_reward=1.0)
    journal.log_rl_trade_result(903, "EURUSDm", "BUY", pnl=10.0, rl_reward=1.0)

    result_rows = [row for row in _rows(journal_path) if row["event_type"] == "RL_TRADE_RESULT"]
    assert len(result_rows) == 1


def test_performance_tracker_close_removes_active_trade_metadata(tmp_path):
    from execution.trade_logger import ActiveTradeStore

    journal = TradeJournal(csv_path=tmp_path / "trades.csv")
    store = ActiveTradeStore(tmp_path / "active_trades.json")
    store.upsert(904, {"symbol": "EURUSDm", "side": "BUY", "volume": 0.1, "entry_price": 1.1})
    tracker = PerformanceTracker(journal=journal, active_trade_store=store)
    journal.log_open(904, "EURUSDm", "BUY", 0.1, 1.1)
    tracker.log_trade(904, "EURUSDm", "BUY", 1.1, 0.1)

    tracker.close_trade(904, 1.101, actual_pnl=10.0)

    assert store.load() == {}
