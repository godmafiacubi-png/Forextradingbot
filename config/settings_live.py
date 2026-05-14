"""Live trading profile with explicit fail-closed gates.

Even when ``SETTINGS_PROFILE=live`` is selected, live order routing stays blocked
unless the environment also sets ``DRY_RUN=false`` and
``LIVE_TRADING_CONFIRMED=true``. This keeps accidental live execution
fail-closed.
"""

import os


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


PROFILE_NAME = "live"
_DRY_RUN_REQUESTED = _env_bool("DRY_RUN", True)
_LIVE_TRADING_CONFIRMED = _env_bool("LIVE_TRADING_CONFIRMED", False)

PROFILE_OVERRIDES = {
    "PROFILE_NAME": PROFILE_NAME,
    "DRY_RUN_REQUESTED": _DRY_RUN_REQUESTED,
    "LIVE_TRADING_CONFIRMED": _LIVE_TRADING_CONFIRMED,
    "DRY_RUN": _DRY_RUN_REQUESTED or not _LIVE_TRADING_CONFIRMED,
    "ACCOUNT_RISK_PERCENT": 0.35,
    "MAX_OPEN_TRADES": 2,
    "MAX_TRADES_PER_SYMBOL": 1,
    "MAX_INTRADAY_EQUITY_DRAWDOWN_PCT": 2.0,
    "MAX_CONSECUTIVE_ORDER_FAILURES": 2,
}
