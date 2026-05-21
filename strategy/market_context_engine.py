"""Pure market context engine for strategy, exit, and risk routing."""

from dataclasses import asdict, dataclass
from math import isfinite, isnan


@dataclass(frozen=True)
class MarketContextDecision:
    market_regime: str
    direction_bias: int
    preferred_strategy: str
    allow_trade: bool
    reason: str
    sl_mode: str
    tp_mode: str
    sl_atr_mult: float
    tp_rr: float
    risk_mult: float
    min_quality_score: int

    def to_dict(self):
        return asdict(self)


def _num(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value is None:
            return default
        value = float(value)
        if isnan(value) or not isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _flag(row, key):
    return bool(_num(row, key, 0.0))


def _regime(row):
    raw = row.get("regime", row.get("market_regime", row.get("htf_regime", "GLOBAL")))
    value = str(raw or "GLOBAL").strip().upper()
    return "GLOBAL" if value in {"", "NAN", "NONE", "NULL"} else value


class MarketContextEngine:
    """Turn one market feature row into an explicit trade plan.

    The engine does not send orders and does not mutate state.  It only answers:
    current context, direction bias, preferred strategy, SL/TP style, and risk
    multiplier.
    """

    def __init__(self, quiet_adx=20.0, trend_adx=26.0, volatile_atr_pct=0.006, max_spread_ratio=1.5):
        self.quiet_adx = quiet_adx
        self.trend_adx = trend_adx
        self.volatile_atr_pct = volatile_atr_pct
        self.max_spread_ratio = max_spread_ratio

    def analyze_row(self, row, symbol_config=None):
        cfg = symbol_config or {}
        regime = _regime(row)
        adx = _num(row, "adx", 0.0)
        rsi = _num(row, "rsi", 50.0)
        atr_pct = abs(_num(row, "atr_pct", 0.0))
        spread = _num(row, "spread", _num(row, "cur_spread", 0.0))
        avg_spread = max(_num(row, "avg_spread", spread or 1.0), 1e-9)
        structure = int(_num(row, "structure", 0))
        htf = int(_num(row, "htf_trend", 0))
        ml = _num(row, "ml_probability", _num(row, "ml_prob", 0.5))
        ict = int(_num(row, "ict_score", 0))
        min_quality = int(cfg.get("min_quality_score", 70))

        if spread > 0 and spread / avg_spread > self.max_spread_ratio:
            return MarketContextDecision("BAD_EXECUTION", 0, "no_trade", False, "spread too wide", "none", "none", 0.0, 0.0, 0.0, min_quality)

        if atr_pct >= self.volatile_atr_pct or regime == "VOLATILE":
            return self._volatile(row, cfg, adx, structure, htf, ml, ict, min_quality)
        if regime == "QUIET" or adx < self.quiet_adx:
            return self._quiet(row, cfg, rsi, min_quality)
        if regime == "RANGING" or adx < self.trend_adx:
            return self._range(row, cfg, rsi, min_quality)
        return self._trend(row, cfg, structure, htf, ml, ict, min_quality)

    def _trend(self, row, cfg, structure, htf, ml, ict, min_quality):
        bull_break = _flag(row, "bos_bullish") or _flag(row, "choch_bullish")
        bear_break = _flag(row, "bos_bearish") or _flag(row, "choch_bearish")
        if (structure > 0 or htf > 0 or ml >= 0.55) and not bear_break:
            bias = 1
        elif (structure < 0 or htf < 0 or ml <= 0.45) and not bull_break:
            bias = -1
        else:
            bias = 0
        allow = bias != 0 and ict >= int(cfg.get("min_ict_score", 2))
        strategy = "breakout_retest" if bull_break or bear_break else "trend_following"
        return MarketContextDecision("TRENDING", bias, strategy, allow, "trend context", "structure_or_atr", "rr", float(cfg.get("trend_sl_atr_mult", 1.5)), float(cfg.get("trend_tp_rr", 3.0)), float(cfg.get("trend_risk_mult", 1.0)), min_quality)

    def _range(self, row, cfg, rsi, min_quality):
        if _flag(row, "liq_sweep_low") and rsi <= 48:
            bias = 1
        elif _flag(row, "liq_sweep_high") and rsi >= 52:
            bias = -1
        else:
            bias = 0
        return MarketContextDecision("RANGING", bias, "ranging_mean_reversion", bias != 0, "range requires sweep reversal", "sweep_structure", "range_mid_or_opposite_liquidity", float(cfg.get("range_sl_atr_mult", 1.0)), float(cfg.get("range_tp_rr", 1.5)), float(cfg.get("range_risk_mult", 0.6)), max(min_quality, 70))

    def _quiet(self, row, cfg, rsi, min_quality):
        if _flag(row, "liq_sweep_low") and rsi <= 42:
            bias = 1
        elif _flag(row, "liq_sweep_high") and rsi >= 58:
            bias = -1
        else:
            bias = 0
        return MarketContextDecision("QUIET", bias, "ranging_mean_reversion", bias != 0, "quiet only allows high-quality sweep reversal", "tight_sweep_structure", "short_rr_or_mid_range", float(cfg.get("quiet_sl_atr_mult", 0.9)), float(cfg.get("quiet_tp_rr", 1.2)), float(cfg.get("quiet_risk_mult", 0.35)), max(min_quality + 5, 75))

    def _volatile(self, row, cfg, adx, structure, htf, ml, ict, min_quality):
        bull = (_flag(row, "bos_bullish") or structure > 0) and htf >= 0 and ml >= 0.56
        bear = (_flag(row, "bos_bearish") or structure < 0) and htf <= 0 and ml <= 0.44
        bias = 1 if bull and not bear else -1 if bear and not bull else 0
        allow = bias != 0 and adx >= 28 and ict >= int(cfg.get("min_ict_score", 2))
        return MarketContextDecision("VOLATILE", bias, "breakout_retest", allow, "volatile requires stronger breakout confirmation", "wider_atr_or_structure", "rr_high", float(cfg.get("volatile_sl_atr_mult", 1.8)), float(cfg.get("volatile_tp_rr", 3.5)), float(cfg.get("volatile_risk_mult", 0.45)), max(min_quality, 78))
