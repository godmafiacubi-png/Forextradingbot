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


def test_execution_safety_env_values_are_parsed(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("ORDER_MAGIC", "98765")
    monkeypatch.setenv("ORDER_DEVIATION", "7")
    monkeypatch.setenv("MAX_LOT_SIZE", "0.25")

    settings = _load_settings(monkeypatch)

    assert settings.DRY_RUN is True
    assert settings.ORDER_MAGIC == 98765
    assert settings.ORDER_DEVIATION == 7
    assert settings.MAX_LOT_SIZE == 0.25
    assert settings.SYMBOL_POINTS["USDJPYm"] == 0.001
