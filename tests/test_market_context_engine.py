from strategy.market_context_engine import MarketContextEngine


def test_trending_context_selects_breakout_when_structure_confirms():
    engine = MarketContextEngine()
    decision = engine.analyze_row(
        {
            "regime": "TRENDING",
            "adx": 32,
            "structure": 1,
            "htf_trend": 1,
            "ml_probability": 0.62,
            "ict_score": 3,
            "bos_bullish": 1,
            "spread": 8,
            "avg_spread": 8,
        },
        {"min_ict_score": 2, "min_quality_score": 70},
    )

    assert decision.allow_trade is True
    assert decision.direction_bias == 1
    assert decision.market_regime == "TRENDING"
    assert decision.preferred_strategy == "breakout_retest"
    assert decision.tp_rr >= 3.0
    assert decision.risk_mult == 1.0


def test_range_context_requires_liquidity_sweep():
    engine = MarketContextEngine()
    decision = engine.analyze_row(
        {
            "regime": "RANGING",
            "adx": 21,
            "rsi": 50,
            "spread": 8,
            "avg_spread": 8,
        },
        {"min_quality_score": 70},
    )

    assert decision.allow_trade is False
    assert decision.preferred_strategy == "ranging_mean_reversion"
    assert decision.risk_mult < 1.0


def test_quiet_context_allows_only_strong_sweep_reversal_with_low_risk():
    engine = MarketContextEngine()
    decision = engine.analyze_row(
        {
            "regime": "QUIET",
            "adx": 16,
            "rsi": 39,
            "liq_sweep_low": 1,
            "spread": 8,
            "avg_spread": 8,
        },
        {"min_quality_score": 70},
    )

    assert decision.allow_trade is True
    assert decision.direction_bias == 1
    assert decision.market_regime == "QUIET"
    assert decision.risk_mult <= 0.35
    assert decision.min_quality_score >= 75


def test_bad_execution_blocks_trading():
    engine = MarketContextEngine(max_spread_ratio=1.5)
    decision = engine.analyze_row(
        {
            "regime": "TRENDING",
            "adx": 35,
            "structure": 1,
            "htf_trend": 1,
            "ml_probability": 0.7,
            "ict_score": 4,
            "spread": 25,
            "avg_spread": 10,
        },
        {"min_quality_score": 70},
    )

    assert decision.allow_trade is False
    assert decision.market_regime == "BAD_EXECUTION"
    assert decision.risk_mult == 0.0


def test_volatile_context_requires_strong_confirmation_and_reduces_risk():
    engine = MarketContextEngine()
    decision = engine.analyze_row(
        {
            "regime": "VOLATILE",
            "atr_pct": 0.008,
            "adx": 30,
            "structure": -1,
            "htf_trend": -1,
            "ml_probability": 0.39,
            "ict_score": 3,
            "bos_bearish": 1,
            "spread": 8,
            "avg_spread": 8,
        },
        {"min_ict_score": 2, "min_quality_score": 75},
    )

    assert decision.allow_trade is True
    assert decision.direction_bias == -1
    assert decision.preferred_strategy == "breakout_retest"
    assert decision.risk_mult < 0.5
    assert decision.tp_rr >= 3.5
