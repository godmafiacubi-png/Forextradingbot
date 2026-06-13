import csv
import importlib
import sys
import types

import pytest


class _SymbolInfo:
    digits = 5
    point = 0.00001
    trade_stops_level = 10


class _Connector:
    def get_symbol_info(self, symbol):
        return {
            "point": 0.00001,
            "digits": 5,
            "bid": 1.10000,
            "ask": 1.10020,
            "spread": 20,
            "volume_min": 0.001,
            "volume_max": 10.0,
            "volume_step": 0.001,
        }


def _load_order_manager(monkeypatch, sent, *, retcode=10009, positions=None):
    mt5_stub = types.SimpleNamespace(
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        TRADE_ACTION_DEAL=10,
        TRADE_ACTION_SLTP=11,
        ORDER_TIME_GTC=20,
        ORDER_FILLING_IOC=30,
        TRADE_RETCODE_DONE=10009,
        symbol_info=lambda symbol: _SymbolInfo(),
        positions_get=lambda *args, **kwargs: positions if positions is not None else [],
        order_send=lambda request: sent.append(request) or types.SimpleNamespace(retcode=retcode, order=123, comment="ok"),
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_stub)
    sys.modules.pop("execution.order_manager", None)
    import execution.order_manager as module
    return importlib.reload(module)


def test_dry_run_place_order_does_not_send_to_mt5(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    manager = module.OrderManager(_Connector(), dry_run=True, magic=99, deviation=5)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1234, 1.099, 1.102, "test")

    assert ticket is None
    assert sent == []


def test_place_order_rounds_volume_to_broker_step(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    manager = module.OrderManager(_Connector(), dry_run=False, magic=99, deviation=5)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1234, 1.099, 1.102, "test")

    assert ticket == 123
    assert sent[0]["volume"] == 0.123
    assert sent[0]["magic"] == 99
    assert sent[0]["deviation"] == 5


def test_place_order_blocks_when_slippage_exceeds_guard(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    manager = module.OrderManager(_Connector(), dry_run=False, magic=99, deviation=5)

    ticket = manager.place_order(
        "EURUSDm",
        module.mt5.ORDER_TYPE_BUY,
        0.1,
        1.099,
        1.102,
        "test",
        reference_price=1.10000,
        max_slippage_points=10,
    )

    assert ticket is None
    assert sent == []


def test_place_order_sends_to_mt5_when_live_and_valid(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    manager = module.OrderManager(_Connector(), dry_run=False, magic=99, deviation=5)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_SELL, 0.2, 1.101, 1.098, "live-test")

    assert ticket == 123
    assert len(sent) == 1
    assert sent[0]["type"] == module.mt5.ORDER_TYPE_SELL


def test_buy_order_rejects_wrong_side_sl_tp_before_send(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    manager = module.OrderManager(_Connector(), dry_run=False, magic=99, deviation=5)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.101, 1.099, "bad-buy")

    assert ticket is None
    assert sent == []


def test_sell_order_rejects_wrong_side_sl_tp_before_send(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    manager = module.OrderManager(_Connector(), dry_run=False, magic=99, deviation=5)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_SELL, 0.1, 1.099, 1.102, "bad-sell")

    assert ticket is None
    assert sent == []


def _journal_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_place_order_writes_execution_journal_events(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(
        _Connector(), dry_run=False, magic=99, deviation=5,
        trade_journal=TradeJournal(csv_path=journal_path),
    )

    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.2, 1.099, 1.102, "journal-test",
        reference_price=1.10000, max_slippage_points=50,
        diagnostics={
            "entry_strategy": "regime_adaptive_entry",
            "strategy_confidence": 0.77,
            "quality_score": 82,
            "quality_grade": "A",
            "ml_prob": 0.63,
            "ict_score": 3,
            "adx": 29,
            "rsi": 47,
            "regime": "QUIET",
            "session": "London",
            "planned_rr": 2.5,
            "avg_spread": 18,
        },
    )

    assert ticket == 123
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_ATTEMPT", "ORDER_FILLED", "OPEN"]
    assert rows[0]["side"] == "BUY"
    assert rows[1]["ticket"] == "123"
    assert "signal_price=1.1" in rows[0]["comment"]
    assert "execution_price=1.1002" in rows[0]["comment"]
    assert "rr=1.50" in rows[0]["comment"]
    assert rows[0]["entry_strategy"] == "regime_adaptive_entry"
    assert rows[0]["regime"] == "QUIET"
    assert rows[0]["session"] == "London"
    assert rows[0]["quality_score"] == "82"
    assert rows[0]["quality_grade"] == "A"
    assert rows[0]["planned_rr"] == "2.5"
    assert rows[0]["avg_spread"] == "18"
    assert rows[1]["avg_spread"] == "18"
    assert rows[2]["avg_spread"] == "18"
    assert float(rows[0]["execution_rr"]) == pytest.approx(1.5, abs=1e-6)
    assert rows[2]["entry_strategy"] == "regime_adaptive_entry"
    assert all(row["reason"] != "JOURNAL_WRITE_FAILED" for row in rows)
    assert all("JOURNAL_WRITE_FAILED" not in row["comment"] for row in rows)


def test_place_order_journals_rejected_slippage(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10000, max_slippage_points=10,
    )

    assert ticket is None
    assert sent == []
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_REJECTED"]
    assert "slippage guard" in rows[0]["comment"]
    assert rows[0]["slippage_points"] != ""




def test_place_order_journals_rejected_slippage_with_diagnostics_fields(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))
    diagnostics = {
        "entry_strategy": "regime_adaptive_entry",
        "strategy_confidence": 0.81,
        "quality_score": 79,
        "quality_grade": "B",
        "ml_prob": 0.58,
        "ict_score": 2.0,
        "adx": 27,
        "rsi": 49,
        "planned_rr": 2.2,
        "regime": "TREND",
        "session": "NY",
        "market_context": "breakout",
        "context_bias": "bullish",
        "context_strategy": "momentum",
        "context_allow_trade": 1,
        "context_reason": "alignment",
        "context_risk_mult": 0.8,
        "context_tp_rr": 2.4,
        "context_sl_atr_mult": 1.6,
        "context_min_quality_score": 70,
    }

    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10000, max_slippage_points=10, diagnostics=diagnostics,
    )

    assert ticket is None
    rows = _journal_rows(journal_path)
    row = rows[0]
    for key, val in diagnostics.items():
        assert row[key] == str(val)
    assert row["max_slippage_points"] == "10"


def test_btc_spread_gate_blocks_before_order_attempt(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    class _BtcWideSpreadConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.01,
                "digits": 2,
                "bid": 60000.0,
                "ask": 60012.0,
                "spread": 1300,
                "volume_min": 0.01,
                "volume_max": 10.0,
                "volume_step": 0.01,
            }

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(
        _BtcWideSpreadConnector(),
        dry_run=False,
        trade_journal=TradeJournal(csv_path=journal_path),
    )

    ticket = manager.place_order("BTCUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 59800.0, 60500.0)

    assert ticket is None
    assert sent == []
    rows = _journal_rows(journal_path)
    assert [r["event_type"] for r in rows] == ["ORDER_REJECTED"]
    assert "btc spread gate" in rows[0]["comment"]




def test_place_order_slippage_rejection_allows_diagnostics_max_slippage_points(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))
    diagnostics = {
        "strategy_confidence": 0.75,
        "max_slippage_points": 999,
        "entry_strategy": "regime_adaptive_entry",
    }

    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10000, max_slippage_points=10, diagnostics=diagnostics,
    )

    assert ticket is None
    assert sent == []
    row = _journal_rows(journal_path)[0]
    assert row["event_type"] == "ORDER_REJECTED"
    assert row["max_slippage_points"] == "10"
    assert row["entry_strategy"] == "regime_adaptive_entry"
    assert row["strategy_confidence"] == "0.75"


def test_btc_spread_gate_allows_diagnostics_max_slippage_points(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    class _BtcWideSpreadConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.01,
                "digits": 2,
                "bid": 60000.0,
                "ask": 60012.0,
                "spread": 1300,
                "volume_min": 0.01,
                "volume_max": 10.0,
                "volume_step": 0.01,
            }

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(
        _BtcWideSpreadConnector(),
        dry_run=False,
        trade_journal=TradeJournal(csv_path=journal_path),
    )

    diagnostics = {"max_slippage_points": 3, "entry_strategy": "btc_breakout"}
    ticket = manager.place_order(
        "BTCUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 59800.0, 60500.0,
        diagnostics=diagnostics,
    )

    assert ticket is None
    assert sent == []
    row = _journal_rows(journal_path)[0]
    assert row["event_type"] == "ORDER_REJECTED"
    assert row["max_slippage_points"] == str(manager._btc_max_entry_spread_points)
    assert row["entry_strategy"] == "btc_breakout"


def test_forex_order_unchanged_when_spread_high_but_not_btc_gate(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)

    class _ForexWideSpreadConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.00001,
                "digits": 5,
                "bid": 1.10000,
                "ask": 1.10020,
                "spread": 2000,
                "volume_min": 0.001,
                "volume_max": 10.0,
                "volume_step": 0.001,
            }

    manager = module.OrderManager(_ForexWideSpreadConnector(), dry_run=False, magic=99, deviation=5)
    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1234, 1.099, 1.102, "test")

    assert ticket == 123
    assert len(sent) == 1
def test_place_order_journals_broker_failure(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent, retcode=10030)
    from execution.trade_logger import TradeJournal

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_SELL, 0.2, 1.101, 1.098, "fail-test")

    assert ticket is None
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_ATTEMPT", "ORDER_FAILED"]
    assert "retcode=10030" in rows[1]["comment"]


def test_close_order_journals_close_event(monkeypatch, tmp_path):
    position = types.SimpleNamespace(ticket=77, symbol="EURUSDm", type=0, volume=0.1, sl=1.099, tp=1.102)
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])
    from execution.trade_logger import TradeJournal

    class _BtcConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.01,
                "digits": 2,
                "bid": 60010.0,
                "ask": 60010.5,
                "spread": 50,
                "volume_min": 0.01,
                "volume_max": 10.0,
                "volume_step": 0.01,
            }

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_BtcConnector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    assert manager.close_order(77) is True
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["OPEN", "CLOSE"]
    assert rows[0]["source"] == "lifecycle_backfill"
    assert rows[1]["ticket"] == "77"
    assert rows[1]["side"] == "BUY"


class _RiskGuardSpy:
    def __init__(self):
        self.failures = []
        self.successes = 0

    def record_order_failure(self, reason=""):
        self.failures.append(reason)

    def record_order_success(self):
        self.successes += 1


def test_risk_aware_journal_records_broker_failure_from_order_manager(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent, retcode=10030)
    from execution.risk_aware_journal import RiskAwareTradeJournal
    from execution.trade_logger import TradeJournal

    risk_guard = _RiskGuardSpy()
    journal_path = tmp_path / "trades.csv"
    trade_journal = RiskAwareTradeJournal(TradeJournal(csv_path=journal_path), risk_guard)
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=trade_journal)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.2, 1.099, 1.102, "risk-fail")

    assert ticket is None
    assert risk_guard.failures == ["retcode=10030: ok"]
    assert risk_guard.successes == 0
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_ATTEMPT", "ORDER_FAILED"]


def test_risk_aware_journal_records_order_success_from_order_manager(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.risk_aware_journal import RiskAwareTradeJournal
    from execution.trade_logger import TradeJournal

    risk_guard = _RiskGuardSpy()
    journal_path = tmp_path / "trades.csv"
    trade_journal = RiskAwareTradeJournal(TradeJournal(csv_path=journal_path), risk_guard)
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=trade_journal)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.2, 1.099, 1.102, "risk-success")

    assert ticket == 123
    assert risk_guard.failures == []
    assert risk_guard.successes >= 1
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_ATTEMPT", "ORDER_FILLED", "OPEN"]


def test_first_slippage_reject_starts_symbol_side_cooldown(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    manager = module.OrderManager(_Connector(), dry_run=False, slippage_cooldown_seconds=600)

    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10000, max_slippage_points=10,
    )

    assert ticket is None
    assert sent == []
    assert manager._slippage_cooldown_remaining("EURUSDm", "BUY") == 600


def test_slippage_cooldown_blocks_repeated_same_symbol_side_before_attempt(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(
        _Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path),
        slippage_cooldown_seconds=600,
    )

    manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10000, max_slippage_points=10,
    )
    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10020, max_slippage_points=10,
    )

    assert ticket is None
    assert sent == []
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_REJECTED", "ORDER_REJECTED"]
    assert "slippage guard" in rows[0]["comment"]
    assert rows[1]["comment"] == "slippage cooldown"


def test_slippage_cooldown_does_not_block_opposite_side_or_different_symbol(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    manager = module.OrderManager(_Connector(), dry_run=False, slippage_cooldown_seconds=600)

    manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10000, max_slippage_points=10,
    )
    sell_ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_SELL, 0.1, 1.101, 1.098,
        reference_price=1.10000, max_slippage_points=10,
    )
    other_symbol_ticket = manager.place_order(
        "GBPUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10020, max_slippage_points=10,
    )

    assert sell_ticket == 123
    assert other_symbol_ticket == 123
    assert len(sent) == 2
    assert sent[0]["symbol"] == "EURUSDm"
    assert sent[0]["type"] == module.mt5.ORDER_TYPE_SELL
    assert sent[1]["symbol"] == "GBPUSDm"
    assert sent[1]["type"] == module.mt5.ORDER_TYPE_BUY


def test_slippage_cooldown_expires_and_allows_attempt(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(
        _Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path),
        slippage_cooldown_seconds=10,
    )

    manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10000, max_slippage_points=10,
    )
    now[0] = 1005.0
    manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10020, max_slippage_points=10,
    )
    now[0] = 1011.0
    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102,
        reference_price=1.10020, max_slippage_points=10,
    )

    assert ticket == 123
    assert len(sent) == 1
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == [
        "ORDER_REJECTED", "ORDER_REJECTED", "ORDER_ATTEMPT", "ORDER_FILLED", "OPEN"
    ]


def test_place_order_journals_none_result_as_failed(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    monkeypatch.setattr(module.mt5, "order_send", lambda req: None)
    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102, "none-result")

    assert ticket is None
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_ATTEMPT", "ORDER_FAILED"]
    assert "retcode=0: None" in rows[1]["comment"]


def test_calculate_execution_rr_for_buy(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)

    risk, reward, rr = module.OrderManager._calculate_rr("BUY", 1.10020, 1.09900, 1.10200)

    assert risk == pytest.approx(0.00120)
    assert reward == pytest.approx(0.00180)
    assert rr == pytest.approx(1.5)


def test_calculate_execution_rr_for_sell(monkeypatch):
    sent = []
    module = _load_order_manager(monkeypatch, sent)

    risk, reward, rr = module.OrderManager._calculate_rr("SELL", 1.10000, 1.10100, 1.09800)

    assert risk == pytest.approx(0.00100)
    assert reward == pytest.approx(0.00200)
    assert rr == pytest.approx(2.0)

def test_btc_modify_sl_below_min_distance_skips_journal_event(monkeypatch, tmp_path):
    position = types.SimpleNamespace(ticket=88, symbol="BTCUSD", type=0, volume=0.1, sl=60000.0, tp=62000.0, profit=10.0)
    sent = []

    class _BtcInfo:
        digits = 2
        point = 0.01
        trade_stops_level = 50  # min delta = 0.5

    module = _load_order_manager(monkeypatch, sent, positions=[position])
    module.mt5.symbol_info = lambda symbol: _BtcInfo()
    from execution.trade_logger import TradeJournal

    class _BtcConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.01,
                "digits": 2,
                "bid": 60010.0,
                "ask": 60010.5,
                "spread": 50,
                "volume_min": 0.01,
                "volume_max": 10.0,
                "volume_step": 0.01,
            }

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_BtcConnector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    assert manager.modify_sl(88, 60000.2) is False
    assert sent == []
    rows = _journal_rows(journal_path)
    assert rows == []


def test_partial_close_stage_1_cannot_trigger_twice_for_same_ticket(monkeypatch):
    position = types.SimpleNamespace(ticket=91, symbol="EURUSDm", type=0, volume=0.2, sl=1.099, tp=1.102, profit=5.0)
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])
    manager = module.OrderManager(_Connector(), dry_run=False)

    assert manager.partial_close(91, 0.5, stage=1) is True
    assert manager.partial_close(91, 0.5, stage=1) is False
    assert len(sent) == 1
    assert sent[0]["comment"] == "partial_close_s1"


def test_close_event_marks_pnl_unavailable_explicitly(monkeypatch, tmp_path):
    position = types.SimpleNamespace(ticket=101, symbol="EURUSDm", type=0, volume=0.1, sl=1.099, tp=1.102, profit=None)
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])
    from execution.trade_logger import TradeJournal

    class _BtcConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.01,
                "digits": 2,
                "bid": 60010.0,
                "ask": 60010.5,
                "spread": 50,
                "volume_min": 0.01,
                "volume_max": 10.0,
                "volume_step": 0.01,
            }

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_BtcConnector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    assert manager.close_order(101) is True
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["OPEN", "CLOSE"]
    row = rows[1]
    assert row["pnl"] == ""
    assert row["reason"] == "pnl_unavailable"
    assert row["comment"] == "pnl_unavailable"


def test_missing_open_after_successful_fill_is_written_immediately(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    class DropFirstOpenJournal(TradeJournal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.drop_open = True

        def log_open(self, *args, **kwargs):
            if self.drop_open:
                self.drop_open = False
                return None
            return super().log_open(*args, **kwargs)

    journal_path = tmp_path / "trades.csv"
    journal = DropFirstOpenJournal(csv_path=journal_path)
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=journal)

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102)

    assert ticket == 123
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_ATTEMPT", "ORDER_FILLED", "OPEN"]
    assert rows[-1]["source"] == "lifecycle_backfill"
    assert rows[-1]["reason"] == "missing_open_event_after_fill"


def test_partial_close_includes_active_trade_diagnostics(monkeypatch, tmp_path):
    position = types.SimpleNamespace(
        ticket=123, symbol="EURUSDm", type=0, volume=0.2, price_open=1.1,
        price_current=1.101, sl=1.099, tp=1.104, profit=12.0,
    )
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])
    from execution.trade_logger import ActiveTradeStore, TradeJournal

    journal_path = tmp_path / "trades.csv"
    store = ActiveTradeStore(tmp_path / "active_trades.json")
    store.upsert(123, {
        "symbol": "EURUSDm", "side": "BUY", "volume": 0.2, "entry_price": 1.1,
        "sl": 1.099, "tp": 1.104, "entry_strategy": "breakout", "market_context": "TREND",
        "regime": "TREND", "quality_score": 84, "planned_rr": 2.0, "execution_rr": 1.8,
    })
    manager = module.OrderManager(
        _Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path), active_trade_store=store,
    )

    assert manager.partial_close(123, 0.5) is True

    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["OPEN", "PARTIAL_CLOSE"]
    row = rows[-1]
    assert row["entry_strategy"] == "breakout"
    assert row["market_context"] == "TREND"
    assert row["regime"] == "TREND"
    assert row["quality_score"] == "84"
    assert row["planned_rr"] == "2.0"
    assert row["execution_rr"] == "1.8"


def test_sl_modified_includes_active_trade_diagnostics(monkeypatch, tmp_path):
    position = types.SimpleNamespace(
        ticket=123, symbol="EURUSDm", type=0, volume=0.2, price_open=1.1,
        price_current=1.101, sl=1.099, tp=1.104, profit=12.0,
    )
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])
    from execution.trade_logger import ActiveTradeStore, TradeJournal

    journal_path = tmp_path / "trades.csv"
    store = ActiveTradeStore(tmp_path / "active_trades.json")
    store.upsert(123, {
        "symbol": "EURUSDm", "side": "BUY", "volume": 0.2, "entry_price": 1.1,
        "sl": 1.099, "tp": 1.104, "entry_strategy": "breakout", "market_context": "TREND",
        "regime": "TREND", "quality_score": 84, "planned_rr": 2.0, "execution_rr": 1.8,
    })
    manager = module.OrderManager(
        _Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path), active_trade_store=store,
    )

    assert manager.modify_sl_tp(123, new_sl=1.0995) is True

    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["OPEN", "SL_MODIFIED"]
    row = rows[-1]
    assert row["entry_strategy"] == "breakout"
    assert row["market_context"] == "TREND"
    assert row["regime"] == "TREND"
    assert row["quality_score"] == "84"
    assert row["planned_rr"] == "2.0"
    assert row["execution_rr"] == "1.8"


def test_successful_fill_persists_active_trade_metadata(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import ActiveTradeStore, TradeJournal

    journal_path = tmp_path / "trades.csv"
    store_path = tmp_path / "active_trades.json"
    store = ActiveTradeStore(store_path)
    manager = module.OrderManager(
        _Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path), active_trade_store=store,
    )

    ticket = manager.place_order(
        "EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.2, 1.099, 1.102,
        diagnostics={"entry_strategy": "breakout", "market_context": "TREND", "quality_score": 84},
    )

    metadata = ActiveTradeStore(store_path).load()[ticket]
    assert metadata["symbol"] == "EURUSDm"
    assert metadata["side"] == "BUY"
    assert metadata["volume"] == 0.2
    assert metadata["entry_price"] == 1.1002
    assert metadata["sl"] == 1.099
    assert metadata["tp"] == 1.102
    assert metadata["entry_strategy"] == "breakout"
    assert metadata["market_context"] == "TREND"
    assert metadata["quality_score"] == 84


def test_missing_order_filled_after_successful_fill_is_backfilled(monkeypatch, tmp_path):
    sent = []
    module = _load_order_manager(monkeypatch, sent)
    from execution.trade_logger import TradeJournal

    class DropFirstFilledJournal(TradeJournal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.drop_filled = True

        def log_order_filled(self, *args, **kwargs):
            if self.drop_filled:
                self.drop_filled = False
                return None
            return super().log_order_filled(*args, **kwargs)

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(
        _Connector(), dry_run=False, trade_journal=DropFirstFilledJournal(csv_path=journal_path),
    )

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.1, 1.099, 1.102)

    assert ticket == 123
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == [
        "ORDER_ATTEMPT", "ORDER_FAILED", "ORDER_FILLED", "OPEN",
    ]
    assert rows[1]["reason"] == "JOURNAL_WRITE_FAILED"
    assert rows[2]["source"] == "lifecycle_backfill"
    assert rows[2]["reason"] == "missing_filled_event_after_fill"


def test_btc_trailing_cooldown_suppresses_frequent_modify_calls(monkeypatch, tmp_path):
    position = types.SimpleNamespace(
        ticket=188, symbol="BTCUSDm", type=0, volume=0.1, price_open=60000.0,
        sl=60000.0, tp=62000.0, profit=10.0,
    )
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])

    class _BtcInfo:
        digits = 2
        point = 0.01
        trade_stops_level = 50

    module.mt5.symbol_info = lambda symbol: _BtcInfo()
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    from execution.trade_logger import TradeJournal

    class _BtcConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.01, "digits": 2, "bid": 60010.0, "ask": 60010.5,
                "spread": 50, "volume_min": 0.01, "volume_max": 10.0, "volume_step": 0.01,
            }

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_BtcConnector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    assert manager.modify_sl(188, 60003.0) is True
    now[0] += 5.0
    assert manager.modify_sl(188, 60005.5) is False
    now[0] += 5.0
    assert manager.modify_sl(188, 60008.0) is False

    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["OPEN", "SL_MODIFIED"]
    assert len([request for request in sent if request["action"] == module.mt5.TRADE_ACTION_SLTP]) == 1


def test_btc_min_sl_move_suppresses_tiny_sl_moves(monkeypatch, tmp_path):
    position = types.SimpleNamespace(
        ticket=189, symbol="BTCUSDm", type=0, volume=0.1, price_open=60000.0,
        sl=60000.0, tp=62000.0, profit=10.0,
    )
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])

    class _BtcInfo:
        digits = 2
        point = 0.01
        trade_stops_level = 1

    module.mt5.symbol_info = lambda symbol: _BtcInfo()
    from execution.trade_logger import TradeJournal

    class _BtcConnector:
        def get_symbol_info(self, symbol):
            return {
                "point": 0.01, "digits": 2, "bid": 60010.0, "ask": 60010.5,
                "spread": 50, "volume_min": 0.01, "volume_max": 10.0, "volume_step": 0.01,
            }

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_BtcConnector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    assert manager.modify_sl(189, 60001.99) is False
    assert sent == []
    assert _journal_rows(journal_path) == []


def test_news_blocked_includes_active_trade_diagnostics(monkeypatch, tmp_path):
    position = types.SimpleNamespace(
        ticket=190, symbol="EURUSDm", type=0, volume=0.2, price_open=1.1,
        price_current=1.101, sl=1.099, tp=1.104, profit=12.0,
    )
    sent = []
    module = _load_order_manager(monkeypatch, sent, positions=[position])
    from execution.trade_logger import ActiveTradeStore, TradeJournal

    journal_path = tmp_path / "trades.csv"
    store = ActiveTradeStore(tmp_path / "active_trades.json")
    store.upsert(190, {
        "symbol": "EURUSDm", "side": "BUY", "volume": 0.2, "entry_price": 1.1,
        "sl": 1.099, "tp": 1.104, "entry_strategy": "breakout",
        "market_context": "TREND", "regime": "TREND", "planned_rr": 2.0,
        "execution_rr": 1.8, "quality_score": 84,
    })
    manager = module.OrderManager(
        _Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path), active_trade_store=store,
    )

    manager.apply_news_protection("EURUSDm", "BREAKEVEN", atr_value=0.002)

    rows = _journal_rows(journal_path)
    news_row = next(row for row in rows if row["event_type"] == "NEWS_BLOCKED")
    assert news_row["entry_strategy"] == "breakout"
    assert news_row["market_context"] == "TREND"
    assert news_row["regime"] == "TREND"
    assert news_row["planned_rr"] == "2.0"
    assert news_row["execution_rr"] == "1.8"
    assert news_row["quality_score"] == "84"
