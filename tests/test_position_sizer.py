import importlib
import sys
import types


class _SymbolInfo:
    point = 0.00001
    digits = 5
    trade_tick_value = 1.0
    trade_tick_size = 0.00001
    trade_contract_size = 100000
    volume_min = 0.01
    volume_max = 1.0
    volume_step = 0.01


def _install_mt5_stub(monkeypatch, symbol_info=None):
    mt5_stub = types.SimpleNamespace(symbol_info=lambda symbol: symbol_info)
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_stub)
    module = importlib.import_module("risk_management.position_sizer")
    return importlib.reload(module)


def test_atr_position_sizing_uses_mt5_tick_value_and_volume_bounds(monkeypatch):
    module = _install_mt5_stub(monkeypatch, _SymbolInfo())
    sizer = module.PositionSizer(method="ATR", account_risk=1.0)

    lot_size = sizer.calculate_position_size(
        account_balance=10_000,
        atr_value=0.001,
        symbol_point=0.00001,
        confidence=1.0,
        symbol="EURUSDm",
    )

    assert lot_size == 1.0
    assert sizer._symbol_cache["EURUSDm"]["digits"] == 5


def test_position_sizer_blocks_when_drawdown_limit_is_exceeded(monkeypatch):
    module = _install_mt5_stub(monkeypatch, None)
    sizer = module.PositionSizer(method="FIXED_PERCENT", account_risk=1.0, max_drawdown=10.0)

    assert sizer.calculate_position_size(10_000, 0.001, 0.00001) == 0.1
    assert sizer.calculate_position_size(8_500, 0.001, 0.00001) == 0.0


def test_atr_position_sizing_matches_actual_sl_multiplier(monkeypatch):
    module = _install_mt5_stub(monkeypatch, _SymbolInfo())
    sizer = module.PositionSizer(method="ATR", account_risk=1.0, max_lot_size=10.0)

    one_atr_lot = sizer.calculate_position_size(
        account_balance=10_000,
        atr_value=0.001,
        symbol_point=0.00001,
        confidence=1.0,
        symbol="EURUSDm",
        sl_multiplier=1.0,
    )
    wider_sl_lot = sizer.calculate_position_size(
        account_balance=10_000,
        atr_value=0.001,
        symbol_point=0.00001,
        confidence=1.0,
        symbol="EURUSDm",
        sl_multiplier=2.0,
    )

    assert one_atr_lot == 1.0
    assert wider_sl_lot == 0.5


def _symbol_info(point, digits, tick_value, tick_size, volume_min=0.01, volume_max=10.0, volume_step=0.01):
    return types.SimpleNamespace(
        point=point,
        digits=digits,
        trade_tick_value=tick_value,
        trade_tick_size=tick_size,
        trade_contract_size=100000,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
    )


def test_xauusd_position_sizing_uses_gold_tick_value(monkeypatch):
    module = _install_mt5_stub(monkeypatch, _symbol_info(0.001, 3, 1.0, 0.01))
    sizer = module.PositionSizer(method="ATR", account_risk=1.0, max_lot_size=10.0)

    lot_size = sizer.calculate_position_size(10_000, 2.0, 0.001, confidence=1.0, symbol="XAUUSDm")

    assert lot_size == 0.5


def test_btcusd_position_sizing_uses_crypto_tick_value(monkeypatch):
    module = _install_mt5_stub(monkeypatch, _symbol_info(0.01, 2, 0.01, 0.01))
    sizer = module.PositionSizer(method="ATR", account_risk=1.0, max_lot_size=10.0)

    lot_size = sizer.calculate_position_size(10_000, 500.0, 0.01, confidence=1.0, symbol="BTCUSDm")

    assert lot_size == 0.2


def test_jpy_position_sizing_uses_jpy_tick_value_and_step(monkeypatch):
    module = _install_mt5_stub(monkeypatch, _symbol_info(0.001, 3, 0.9, 0.001))
    sizer = module.PositionSizer(method="ATR", account_risk=1.0, max_lot_size=10.0)

    lot_size = sizer.calculate_position_size(10_000, 0.15, 0.001, confidence=1.0, symbol="USDJPYm")

    assert lot_size == 0.74
