"""Import smoke checks for modules that should stay lightweight in CI."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_dependency_stubs() -> None:
    sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
    sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))
    sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=lambda *args, **kwargs: None, to_datetime=lambda *args, **kwargs: None))


def install_mt5_stub() -> None:
    if "MetaTrader5" in sys.modules:
        return
    sys.modules["MetaTrader5"] = types.SimpleNamespace(
        TIMEFRAME_M1=1,
        TIMEFRAME_M5=5,
        TIMEFRAME_M15=15,
        TIMEFRAME_M30=30,
        TIMEFRAME_H1=60,
        TIMEFRAME_H4=240,
        TIMEFRAME_D1=1440,
        positions_get=lambda *args, **kwargs: [],
    )


def main() -> int:
    install_dependency_stubs()
    install_mt5_stub()
    modules = [
        "config.settings_default",
        "config.settings_demo",
        "config.settings_live",
        "config.settings",
        "execution.trade_logger",
        "risk_management.risk_guard",
        "data_layer.mt5_connector",
    ]
    for module in modules:
        importlib.import_module(module)
        print(f"import ok: {module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
