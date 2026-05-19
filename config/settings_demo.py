"""Demo forward-testing profile.

Demo remains fail-safe by default.  It uses conservative demo parameters but does
not enable broker order routing by itself; real order routing still requires a
separate explicit operator-controlled execution gate.
"""

PROFILE_NAME = "demo"
PROFILE_OVERRIDES = {
    "PROFILE_NAME": PROFILE_NAME,
    "DRY_RUN": True,
    "LIVE_TRADING_CONFIRMED": False,
    "ACCOUNT_RISK_PERCENT": 0.5,
    "MAX_OPEN_TRADES": 5,
    "MAX_TRADES_PER_SYMBOL": 1,
    "MAX_INTRADAY_EQUITY_DRAWDOWN_PCT": 3.0,
    "MAX_CONSECUTIVE_ORDER_FAILURES": 3,
}
