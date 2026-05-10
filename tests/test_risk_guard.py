import importlib
import sys
import types


class _Connector:
    def __init__(self, balance=10_000, equity=10_000, spread=20):
        self.balance = balance
        self.equity = equity
        self.spread = spread

    def get_account_info(self):
        return {"balance": self.balance, "equity": self.equity}

    def get_symbol_info(self, symbol):
        return {
            "spread": self.spread,
            "bid": 1.10000,
            "ask": 1.10020,
        }


def _load_risk_guard(monkeypatch):
    mt5_stub = types.SimpleNamespace(positions_get=lambda *args, **kwargs: [])
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_stub)
    sys.modules.pop("risk_management.risk_guard", None)
    import risk_management.risk_guard as module
    return importlib.reload(module)


def test_daily_loss_limit_uses_equity_not_closed_balance(monkeypatch):
    module = _load_risk_guard(monkeypatch)
    connector = _Connector(balance=10_000, equity=10_000)
    guard = module.RiskGuard(connector, {"DAILY_LOSS_LIMIT_PCT": 2.0})

    connector.balance = 10_000
    connector.equity = 9_750
    guard.update()

    ok, reason = guard.check_daily_limit()

    assert ok is False
    assert "Daily equity loss limit" in reason
    assert guard.daily_pnl == 0
    assert guard.daily_equity_pnl == -250


def test_spread_filter_blocks_absolute_symbol_cap(monkeypatch):
    module = _load_risk_guard(monkeypatch)
    connector = _Connector(spread=31)
    guard = module.RiskGuard(
        connector,
        {
            "MAX_SPREAD_POINTS": {"EURUSDm": 30},
            "DEFAULT_MAX_SPREAD_POINTS": 50,
            "SPREAD_AVG_PERIOD": 50,
            "MAX_SPREAD_MULTIPLIER": 3.0,
        },
    )

    ok, spread, avg = guard.check_spread("EURUSDm")

    assert ok is False
    assert spread == 31
    assert avg == 0
