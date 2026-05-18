"""Demo forward-testing profile.

Demo remains fail-safe by default: it keeps dry-run enabled unless a caller has a
separate MT5 demo execution path and explicitly disables DRY_RUN in code/tests.
"""

PROFILE_NAME = "demo"
PROFILE_OVERRIDES = {
    "PROFILE_NAME": PROFILE_NAME,
    "DRY_RUN": False,
    "LIVE_TRADING_CONFIRMED": True,
    "ACCOUNT_RISK_PERCENT": 0.5,
    "MAX_OPEN_TRADES": 5,
    "MAX_TRADES_PER_SYMBOL": 1,
    "MAX_INTRADAY_EQUITY_DRAWDOWN_PCT": 3.0,
    "MAX_CONSECUTIVE_ORDER_FAILURES": 3,
}
