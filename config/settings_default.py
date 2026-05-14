"""Default fail-safe configuration profile.

This profile relies on ``config.settings`` fail-closed environment parsing:
DRY_RUN defaults to true and LIVE_TRADING_CONFIRMED defaults to false.
"""

PROFILE_NAME = "default"
PROFILE_OVERRIDES = {
    "PROFILE_NAME": PROFILE_NAME,
    "ACCOUNT_RISK_PERCENT": 0.7,
    "MAX_OPEN_TRADES": 5,
    "MAX_TRADES_PER_SYMBOL": 1,
}
