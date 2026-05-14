import importlib
import sys
import types


def _load_settings(monkeypatch):
    dotenv_stub = types.SimpleNamespace(load_dotenv=lambda: None)
    mt5_stub = types.SimpleNamespace(
        TIMEFRAME_M1=1,
        TIMEFRAME_M5=5,
        TIMEFRAME_M15=15,
        TIMEFRAME_M30=30,
        TIMEFRAME_H1=60,
        TIMEFRAME_H4=240,
        TIMEFRAME_D1=1440,
    )
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_stub)
    sys.modules.pop("config.settings", None)
    import config.settings as settings
    return importlib.reload(settings)


def test_bot_mode_can_be_configured_from_environment(monkeypatch):
    monkeypatch.setenv("BOT_MODE", "default")

    settings = _load_settings(monkeypatch)

    assert settings.BOT_MODE == "DEFAULT"
    assert settings.TRADING_MODE == "DEFAULT"


def test_dashboard_port_defaults_to_documented_port(monkeypatch):
    monkeypatch.delenv("DASHBOARD_PORT", raising=False)

    settings = _load_settings(monkeypatch)

    assert settings.DASHBOARD_PORT == 5001


def test_live_trading_requires_explicit_confirmation(monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)

    settings = _load_settings(monkeypatch)

    assert settings.DRY_RUN_REQUESTED is True
    assert settings.LIVE_TRADING_CONFIRMED is False
    assert settings.DRY_RUN is True

    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")

    settings = _load_settings(monkeypatch)

    assert settings.DRY_RUN_REQUESTED is False
    assert settings.LIVE_TRADING_CONFIRMED is True
    assert settings.DRY_RUN is False


def test_execution_safety_env_values_are_parsed(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("ORDER_MAGIC", "98765")
    monkeypatch.setenv("ORDER_DEVIATION", "7")
    monkeypatch.setenv("MAX_LOT_SIZE", "0.25")

    settings = _load_settings(monkeypatch)

    assert settings.DRY_RUN_REQUESTED is True
    assert settings.LIVE_TRADING_CONFIRMED is False
    assert settings.DRY_RUN is True
    assert settings.ORDER_MAGIC == 98765
    assert settings.ORDER_DEVIATION == 7
    assert settings.MAX_LOT_SIZE == 0.25
    assert settings.MAX_SPREAD_POINTS["EURUSDm"] > 0
    assert settings.MAX_SLIPPAGE_POINTS["EURUSDm"] > 0
    assert settings.SYMBOL_POINTS["USDJPYm"] == 0.001


def test_symbol_settings_cover_all_configured_symbols(monkeypatch):
    settings = _load_settings(monkeypatch)

    configured_symbols = {
        symbol
        for group_symbols in settings.SYMBOLS.values()
        for symbol in group_symbols
    }

    assert set(settings.SYMBOL_SETTINGS) == configured_symbols

    for symbol in configured_symbols:
        cfg = settings.get_symbol_config(symbol)
        assert 0 < cfg["risk_pct"] <= settings.ACCOUNT_RISK_PERCENT
        assert cfg["sl_atr_mult"] > 0
        assert cfg["tp_atr_mult"] > cfg["sl_atr_mult"]
        assert 0 < cfg["ml_sell_threshold"] < cfg["ml_buy_threshold"] < 1
        assert 0 < cfg["min_confidence"] < 1
        assert cfg["max_per_symbol"] == 1


def test_xauusd_spread_cap_matches_broker_points(monkeypatch):
    settings = _load_settings(monkeypatch)

    assert settings.SYMBOL_POINTS["XAUUSDm"] == 0.001
    assert settings.MAX_SPREAD_POINTS["XAUUSDm"] >= 300


def test_live_profile_still_fails_closed_without_confirmation(monkeypatch):
    monkeypatch.setenv("SETTINGS_PROFILE", "live")
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)

    settings = _load_settings(monkeypatch)

    assert settings.PROFILE_NAME == "live"
    assert settings.DRY_RUN is True


def test_demo_profile_forces_dry_run(monkeypatch):
    monkeypatch.setenv("SETTINGS_PROFILE", "demo")
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")

    settings = _load_settings(monkeypatch)

    assert settings.PROFILE_NAME == "demo"
    assert settings.DRY_RUN is True
