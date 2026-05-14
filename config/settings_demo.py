"""Demo forward-testing profile.

Demo remains fail-safe by default: it keeps dry-run enabled unless a caller has a
separate MT5 demo execution path and explicitly disables DRY_RUN in code/tests.
"""

PROFILE_NAME = "demo"
PROFILE_OVERRIDES = {
    "PROFILE_NAME": PROFILE_NAME,
    "DRY_RUN": True,
    "LIVE_TRADING_CONFIRMED": False,
    "ACCOUNT_RISK_PERCENT": 0.5,
    "MAX_OPEN_TRADES": 3,
    "MAX_TRADES_PER_SYMBOL": 1,
}
