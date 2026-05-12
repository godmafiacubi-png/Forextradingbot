"""Meta strategy selection for regime-adaptive entries.

The selector keeps the existing ICT/ML signal as the baseline and adds
specialised entry models that are safe for live/backtest use because they only
read the current bar plus already-confirmed historical bars.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EntryCandidate:
    """A directional vote from an entry model."""

    signal: int
    confidence: float
    ict_score: int
    strategy: str
    reason: str

    def clipped(self):
        return EntryCandidate(
            signal=1 if self.signal > 0 else -1 if self.signal < 0 else 0,
            confidence=float(np.clip(self.confidence, 0.0, 1.0)),
            ict_score=max(int(self.ict_score), 0),
            strategy=self.strategy,
            reason=self.reason,
        )


def normalize_regime_name(value):
    """Return a stable uppercase regime label."""
    if value is None:
        return "GLOBAL"
    regime = str(value).strip().upper()
    if not regime or regime in {"NAN", "NONE", "NULL"}:
        return "GLOBAL"
    return regime


def _num(row, name, default=0.0):
    try:
        value = row.get(name, default)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(row, name):
    return bool(_num(row, name, 0.0))


def _near(row, *names, threshold=0.007):
    return any(_num(row, name, 1.0) <= threshold for name in names)


def _window_any(window, name):
    if name not in window.columns:
        return False
    return bool(window[name].astype(bool).any())


class RangingMeanReversionStrategy:
    """Fade liquidity sweeps at range extremes in low/moderate ADX regimes."""

    name = "ranging_mean_reversion"

    def __init__(self, max_adx=24, buy_rsi_max=46, sell_rsi_min=54):
        self.max_adx = max_adx
        self.buy_rsi_max = buy_rsi_max
        self.sell_rsi_min = sell_rsi_min

    def evaluate(self, df, index):
        row = df.iloc[index]
        regime = normalize_regime_name(row.get("regime", row.get("market_regime", row.get("htf_regime", "GLOBAL"))))
        adx = _num(row, "adx", 0.0)
        rsi = _num(row, "rsi", 50.0)

        if regime not in {"RANGING", "QUIET", "GLOBAL"} and adx > self.max_adx:
            return None
        if adx > self.max_adx and regime != "RANGING":
            return None

        buy_context = (
            _flag(row, "liq_sweep_low")
            and rsi <= self.buy_rsi_max
            and (
                _near(row, "near_demand_ob", "near_bull_fvg")
                or _flag(row, "ob_demand")
                or _flag(row, "fvg_bull_unfilled")
            )
        )
        sell_context = (
            _flag(row, "liq_sweep_high")
            and rsi >= self.sell_rsi_min
            and (
                _near(row, "near_supply_ob", "near_bear_fvg")
                or _flag(row, "ob_supply")
                or _flag(row, "fvg_bear_unfilled")
            )
        )

        if buy_context and not sell_context:
            confidence = 0.58 + min((self.buy_rsi_max - rsi) * 0.006, 0.10)
            return EntryCandidate(1, confidence, 2, self.name, "range low sweep reversal").clipped()
        if sell_context and not buy_context:
            confidence = 0.58 + min((rsi - self.sell_rsi_min) * 0.006, 0.10)
            return EntryCandidate(-1, confidence, 2, self.name, "range high sweep reversal").clipped()
        return None


class BreakoutRetestStrategy:
    """Trade confirmed BOS/CHoCH continuation when price retests an imbalance."""

    name = "breakout_retest"

    def __init__(self, lookback=5, min_adx=22):
        self.lookback = lookback
        self.min_adx = min_adx

    def evaluate(self, df, index):
        row = df.iloc[index]
        start = max(0, index - self.lookback)
        window = df.iloc[start : index + 1]
        adx = _num(row, "adx", 0.0)
        if adx < self.min_adx:
            return None

        bull_break = _window_any(window, "bos_bullish") or _window_any(window, "choch_bullish")
        bear_break = _window_any(window, "bos_bearish") or _window_any(window, "choch_bearish")
        structure = int(_num(row, "structure", 0))
        htf = int(_num(row, "htf_trend", 0))

        buy_retest = (
            bull_break
            and structure >= 0
            and htf >= 0
            and (
                _near(row, "near_demand_ob", "near_bull_fvg")
                or _flag(row, "in_ote_buy_zone")
                or _flag(row, "fvg_bull_unfilled")
            )
        )
        sell_retest = (
            bear_break
            and structure <= 0
            and htf <= 0
            and (
                _near(row, "near_supply_ob", "near_bear_fvg")
                or _flag(row, "in_ote_sell_zone")
                or _flag(row, "fvg_bear_unfilled")
            )
        )

        if buy_retest and not sell_retest:
            confidence = 0.60 + min(max(adx - self.min_adx, 0) * 0.004, 0.10)
            ict_score = 3 if (_flag(row, "in_ote_buy_zone") or _near(row, "near_demand_ob")) else 2
            return EntryCandidate(1, confidence, ict_score, self.name, "bullish breakout retest").clipped()
        if sell_retest and not buy_retest:
            confidence = 0.60 + min(max(adx - self.min_adx, 0) * 0.004, 0.10)
            ict_score = 3 if (_flag(row, "in_ote_sell_zone") or _near(row, "near_supply_ob")) else 2
            return EntryCandidate(-1, confidence, ict_score, self.name, "bearish breakout retest").clipped()
        return None


class RegimeAdaptiveEntryStrategy:
    """Route entry generation to the strategy best suited for the current regime."""

    name = "regime_adaptive_entry"

    def __init__(self, ranging_strategy=None, breakout_strategy=None):
        self.ranging_strategy = ranging_strategy or RangingMeanReversionStrategy()
        self.breakout_strategy = breakout_strategy or BreakoutRetestStrategy()

    def evaluate(self, df, index):
        row = df.iloc[index]
        regime = normalize_regime_name(row.get("regime", row.get("market_regime", row.get("htf_regime", "GLOBAL"))))
        adx = _num(row, "adx", 0.0)

        if regime in {"RANGING", "QUIET"} or adx < 22:
            candidate = self.ranging_strategy.evaluate(df, index)
            if candidate is not None:
                return EntryCandidate(
                    candidate.signal,
                    candidate.confidence,
                    candidate.ict_score,
                    self.name,
                    f"{regime.lower()} routed to {candidate.strategy}",
                ).clipped()
            return None

        candidate = self.breakout_strategy.evaluate(df, index)
        if candidate is not None:
            confidence = candidate.confidence
            if regime == "VOLATILE":
                confidence -= 0.04
            elif regime == "TRENDING":
                confidence += 0.03
            return EntryCandidate(
                candidate.signal,
                confidence,
                candidate.ict_score,
                self.name,
                f"{regime.lower()} routed to {candidate.strategy}",
            ).clipped()
        return None


class MetaStrategySelector:
    """Blend baseline ICT/ML signals with specialised regime-aware entries."""

    def __init__(self, min_alt_confidence=0.57, override_margin=0.12):
        self.min_alt_confidence = min_alt_confidence
        self.override_margin = override_margin
        self.ranging = RangingMeanReversionStrategy()
        self.breakout = BreakoutRetestStrategy()
        self.regime_adaptive = RegimeAdaptiveEntryStrategy(self.ranging, self.breakout)

    def _score(self, candidate):
        return candidate.confidence + min(candidate.ict_score, 4) * 0.03

    def _ml_allows_candidate(self, row, candidate):
        """Require ML confirmation for adaptive candidates when ML data is present.

        The baseline generator treats ML as a confirmation layer, so adaptive
        entries must not re-introduce trades that are clearly against the ML
        probability.  If a historical/backtest frame does not provide
        ``ml_probability`` we keep the legacy behaviour for compatibility.
        """
        if candidate.strategy == "ict_ml_baseline" or "ml_probability" not in row.index:
            return True

        ml_prob = _num(row, "ml_probability", 0.5)
        buy_threshold = _num(row, "ml_threshold_buy", 0.54)
        sell_threshold = _num(row, "ml_threshold_sell", 0.46)

        if candidate.signal > 0:
            return ml_prob > buy_threshold or (candidate.ict_score >= 3 and ml_prob > 0.52)
        if candidate.signal < 0:
            return ml_prob < sell_threshold or (candidate.ict_score >= 3 and ml_prob < 0.48)
        return True

    def _base_candidate(self, row):
        signal = int(_num(row, "signal", 0))
        if signal == 0:
            return None
        return EntryCandidate(
            signal=signal,
            confidence=_num(row, "confidence", 0.0),
            ict_score=int(_num(row, "ict_score", 0)),
            strategy="ict_ml_baseline",
            reason="existing ICT/ML decision tree",
        ).clipped()

    def select(self, df, index):
        row = df.iloc[index]
        base = self._base_candidate(row)
        candidates = [c for c in (
            base,
            self.regime_adaptive.evaluate(df, index),
            self.ranging.evaluate(df, index),
            self.breakout.evaluate(df, index),
        ) if c is not None and c.signal != 0 and self._ml_allows_candidate(row, c)]

        if not candidates:
            return EntryCandidate(0, 0.0, 0, "none", "no entry candidate")

        best = max(candidates, key=self._score)
        same_direction_votes = [c for c in candidates if c.signal == best.signal]

        if base is not None and best.signal == base.signal:
            boosted = min(max(base.confidence, best.confidence) + 0.04 * (len(same_direction_votes) - 1), 1.0)
            return EntryCandidate(
                base.signal,
                boosted,
                max(base.ict_score, best.ict_score),
                best.strategy if best.strategy != "ict_ml_baseline" else base.strategy,
                f"{len(same_direction_votes)} aligned entry votes",
            ).clipped()

        if base is not None and best.signal != base.signal:
            if best.confidence >= base.confidence + self.override_margin and best.confidence >= self.min_alt_confidence:
                return EntryCandidate(
                    best.signal,
                    best.confidence,
                    best.ict_score,
                    best.strategy,
                    f"override baseline: {best.reason}",
                ).clipped()
            return EntryCandidate(
                0,
                0.0,
                max(base.ict_score, best.ict_score),
                "meta_conflict_filter",
                "baseline and adaptive entries conflict",
            )

        if best.confidence >= self.min_alt_confidence:
            return best.clipped()
        return EntryCandidate(0, 0.0, best.ict_score, "meta_low_confidence", "adaptive entry below confidence floor")

    def apply(self, df):
        result = df.copy()
        defaults = {
            "base_signal": 0,
            "base_confidence": 0.0,
            "entry_strategy": "none",
            "strategy_confidence": 0.0,
            "strategy_reason": "",
        }
        for col, default in defaults.items():
            if col not in result.columns:
                result[col] = default

        for i in range(len(result)):
            result.iat[i, result.columns.get_loc("base_signal")] = int(_num(result.iloc[i], "signal", 0))
            result.iat[i, result.columns.get_loc("base_confidence")] = float(_num(result.iloc[i], "confidence", 0.0))
            selected = self.select(result, i)
            result.iat[i, result.columns.get_loc("signal")] = selected.signal
            result.iat[i, result.columns.get_loc("confidence")] = selected.confidence
            result.iat[i, result.columns.get_loc("ict_score")] = max(int(_num(result.iloc[i], "ict_score", 0)), selected.ict_score)
            result.iat[i, result.columns.get_loc("entry_strategy")] = selected.strategy
            result.iat[i, result.columns.get_loc("strategy_confidence")] = selected.confidence
            result.iat[i, result.columns.get_loc("strategy_reason")] = selected.reason
        return result
