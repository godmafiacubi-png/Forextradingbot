import pytest

pd = pytest.importorskip("pandas")

from strategy.meta_strategy_selector import (
    BreakoutRetestStrategy,
    MetaStrategySelector,
    RangingMeanReversionStrategy,
)
from backtest.walk_forward import DemoForwardGate, WalkForwardValidator


def _frame(rows):
    base = {
        "time": pd.date_range("2024-01-01", periods=len(rows), freq="h"),
        "signal": [0] * len(rows),
        "confidence": [0.0] * len(rows),
        "ict_score": [0] * len(rows),
        "adx": [20] * len(rows),
        "rsi": [50] * len(rows),
        "regime": ["GLOBAL"] * len(rows),
        "near_demand_ob": [1.0] * len(rows),
        "near_supply_ob": [1.0] * len(rows),
        "near_bull_fvg": [1.0] * len(rows),
        "near_bear_fvg": [1.0] * len(rows),
        "bos_bullish": [0] * len(rows),
        "bos_bearish": [0] * len(rows),
        "choch_bullish": [0] * len(rows),
        "choch_bearish": [0] * len(rows),
        "structure": [0] * len(rows),
        "htf_trend": [0] * len(rows),
    }
    df = pd.DataFrame(base)
    for idx, values in enumerate(rows):
        for key, value in values.items():
            df.loc[idx, key] = value
    return df


def test_ranging_mean_reversion_buys_liquidity_sweep_low():
    df = _frame([
        {
            "regime": "RANGING",
            "adx": 16,
            "rsi": 38,
            "liq_sweep_low": 1,
            "near_demand_ob": 0.002,
        }
    ])

    candidate = RangingMeanReversionStrategy().evaluate(df, 0)

    assert candidate.signal == 1
    assert candidate.strategy == "ranging_mean_reversion"
    assert candidate.confidence >= 0.58


def test_breakout_retest_sells_after_confirmed_bearish_break():
    df = _frame([
        {"bos_bearish": 1, "adx": 28, "structure": -1, "htf_trend": -1},
        {"adx": 28, "structure": -1, "htf_trend": -1, "near_supply_ob": 0.001},
    ])

    candidate = BreakoutRetestStrategy().evaluate(df, 1)

    assert candidate.signal == -1
    assert candidate.strategy == "breakout_retest"
    assert "bearish" in candidate.reason


def test_meta_selector_adds_adaptive_signal_when_baseline_is_flat():
    df = _frame([
        {
            "regime": "RANGING",
            "adx": 18,
            "rsi": 60,
            "liq_sweep_high": 1,
            "near_supply_ob": 0.002,
        }
    ])

    selected = MetaStrategySelector().apply(df).iloc[0]

    assert selected["base_signal"] == 0
    assert selected["signal"] == -1
    assert selected["entry_strategy"] in {"regime_adaptive_entry", "ranging_mean_reversion"}
    assert selected["strategy_confidence"] >= 0.57


def test_walk_forward_validator_builds_splits_and_blocks_weak_demo():
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=30, freq="h")})
    validator = WalkForwardValidator(train_bars=10, test_bars=5, step_bars=5, min_folds=2)

    splits = validator.build_splits(df)
    report = validator.evaluate_results([
        {"total_trades": 12, "profit_factor": 1.4, "max_drawdown_pct": 4.0},
        {"total_trades": 11, "profit_factor": 1.5, "max_drawdown_pct": 5.0},
    ])

    assert len(splits) == 4
    assert report["passed"] is True

    demo = DemoForwardGate(min_days=14, min_trades=20).evaluate(
        {
            "start_time": "2024-01-01",
            "end_time": "2024-01-05",
            "total_trades": 6,
            "profit_factor": 0.9,
            "max_drawdown_pct": 8.0,
            "expectancy": -1.0,
        }
    )
    assert demo["passed"] is False
    assert len(demo["checks"]) >= 4


def test_meta_selector_blocks_adaptive_sell_against_bullish_ml():
    df = _frame([
        {"bos_bearish": 1, "adx": 42, "structure": -1, "htf_trend": -1},
        {
            "adx": 42,
            "structure": -1,
            "htf_trend": -1,
            "near_supply_ob": 0.001,
            "ml_probability": 0.69,
            "ml_threshold_buy": 0.52,
            "ml_threshold_sell": 0.48,
        },
    ])

    selected = MetaStrategySelector().apply(df).iloc[1]

    assert selected["base_signal"] == 0
    assert selected["signal"] == 0
    assert selected["entry_strategy"] == "none"


def test_meta_selector_allows_adaptive_sell_when_ml_confirms():
    df = _frame([
        {"bos_bearish": 1, "adx": 42, "structure": -1, "htf_trend": -1},
        {
            "adx": 42,
            "structure": -1,
            "htf_trend": -1,
            "near_supply_ob": 0.001,
            "ml_probability": 0.31,
            "ml_threshold_buy": 0.52,
            "ml_threshold_sell": 0.48,
        },
    ])

    selected = MetaStrategySelector().apply(df).iloc[1]

    assert selected["base_signal"] == 0
    assert selected["signal"] == -1
    assert selected["entry_strategy"] in {"regime_adaptive_entry", "breakout_retest"}
