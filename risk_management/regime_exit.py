"""Regime-aware exit policy helpers for SL/TP and trade management."""

from copy import deepcopy


REGIME_EXIT_POLICIES = {
    "TRENDING": {
        "sl_atr_mult": 1.6,
        "tp_atr_mult": 3.5,
        "breakeven_atr": 1.2,
        "partial_stages": [(1.5, 0.30), (2.5, 0.30)],
        "trail_stages": [(3.0, 1.0), (2.0, 1.2)],
        "risk_mult": 1.0,
    },
    "RANGING": {
        "sl_atr_mult": 1.1,
        "tp_atr_mult": 1.6,
        "breakeven_atr": 0.7,
        "partial_stages": [(0.8, 0.35), (1.3, 0.35)],
        "trail_stages": [(1.4, 0.5), (1.0, 0.7)],
        "risk_mult": 0.8,
    },
    "VOLATILE": {
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "breakeven_atr": 1.5,
        "partial_stages": [(1.5, 0.30), (2.5, 0.30)],
        "trail_stages": [(2.5, 1.2), (1.8, 1.5)],
        "risk_mult": 0.6,
    },
    "QUIET": {
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 1.2,
        "breakeven_atr": 0.6,
        "partial_stages": [(0.7, 0.40), (1.1, 0.30)],
        "trail_stages": [(1.2, 0.4), (0.8, 0.6)],
        "risk_mult": 0.5,
    },
    "GLOBAL": {
        "risk_mult": 1.0,
    },
}


def normalize_regime(regime):
    """Return a canonical regime name used by the exit policy table."""
    if regime is None:
        return "GLOBAL"
    normalized = str(regime).strip().upper()
    if not normalized or normalized in {"NAN", "NONE", "NULL"}:
        return "GLOBAL"
    return normalized


def get_regime_exit_policy(
    regime,
    base_sl_atr_mult,
    base_tp_atr_mult,
    base_breakeven_atr=1.0,
    base_partial_stages=None,
    base_trail_stages=None,
    policies=None,
    confidence=0.5,
):
    """
    Build the effective exit policy for a trade.

    The symbol/backtest defaults remain the fallback, while a matching market
    regime can override SL/TP, break-even, partial close, trailing, and risk.
    """
    regime_name = normalize_regime(regime)
    policy_table = policies or REGIME_EXIT_POLICIES

    effective = {
        "regime": regime_name,
        "sl_atr_mult": float(base_sl_atr_mult),
        "tp_atr_mult": float(base_tp_atr_mult),
        "breakeven_atr": float(base_breakeven_atr),
        "partial_stages": list(base_partial_stages or []),
        "trail_stages": list(base_trail_stages or []),
        "risk_mult": 1.0,
    }

    global_policy = deepcopy(policy_table.get("GLOBAL", {}))
    regime_policy = deepcopy(policy_table.get(regime_name, {}))
    effective.update(global_policy)
    effective.update(regime_policy)

    if confidence >= 0.75 and regime_name == "TRENDING":
        effective["tp_atr_mult"] = float(effective["tp_atr_mult"]) + 0.5

    effective["sl_atr_mult"] = max(float(effective["sl_atr_mult"]), 0.1)
    effective["tp_atr_mult"] = max(float(effective["tp_atr_mult"]), 0.1)
    effective["breakeven_atr"] = max(float(effective["breakeven_atr"]), 0.0)
    effective["risk_mult"] = min(max(float(effective.get("risk_mult", 1.0)), 0.0), 1.0)
    effective["partial_stages"] = [tuple(stage) for stage in effective.get("partial_stages", [])]
    effective["trail_stages"] = [tuple(stage) for stage in effective.get("trail_stages", [])]

    return effective
