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

    rows = _rows(journal_path)
    open_rows = [row for row in rows if row["event_type"] == "OPEN" and row["ticket"] == str(ticket)]
    assert len(open_rows) == 1
    assert open_rows[0]["source"] == "order_manager"


def test_positive_close_pnl_appears_in_forward_report():
    metrics = Report([
        {"event_type": "CLOSE", "pnl": "42.5", "symbol": "EURUSDm", "session": "London"}
    ]).metrics()

    assert metrics["net_pnl"] == 42.5
    assert metrics["total_trades"] == 1
    assert "net_pnl: 42.50" in render_report(metrics)


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
