import csv
import importlib
import sys
import types


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

    ticket = manager.place_order("EURUSDm", module.mt5.ORDER_TYPE_BUY, 0.2, 1.099, 1.102, "journal-test")

    assert ticket == 123
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["ORDER_ATTEMPT", "ORDER_FILLED", "OPEN"]
    assert rows[0]["side"] == "BUY"
    assert rows[1]["ticket"] == "123"


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

    journal_path = tmp_path / "trades.csv"
    manager = module.OrderManager(_Connector(), dry_run=False, trade_journal=TradeJournal(csv_path=journal_path))

    assert manager.close_order(77) is True
    rows = _journal_rows(journal_path)
    assert [row["event_type"] for row in rows] == ["CLOSE"]
    assert rows[0]["ticket"] == "77"
    assert rows[0]["side"] == "BUY"
