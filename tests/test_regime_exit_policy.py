import pytest

from risk_management.regime_exit import get_regime_exit_policy


class _PassthroughSignalGenerator:
    def generate_signals(self, df, *args, **kwargs):
        return df.copy()


def _bars(pd, n=205, regime="TRENDING"):
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "time": times,
            "c": [1.1000] * n,
            "h": [1.1005] * n,
            "l": [1.0995] * n,
            "atr": [0.0010] * n,
            "signal": [0] * n,
            "confidence": [0.60] * n,
            "adx": [30] * n,
            "rsi": [50] * n,
            "ict_score": [3] * n,
            "regime": [regime] * n,
        }
    )
    df.loc[200, "signal"] = 1
    df.loc[200, "confidence"] = 0.80
    return df


def _engine_config(**overrides):
    cfg = {
        "initial_balance": 10000,
        "spread_pips": 0,
        "slippage_pips": 0,
        "commission_per_lot": 0,
        "risk_pct": 1.0,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
        "min_confidence": 0.1,
        "min_adx": 0,
        "min_ict_score": 0,
        "require_htf": False,
        "require_pullback": False,
        "partial_close": False,
        "entry_cooldown": 999,
        "max_trades": 5,
        "max_per_symbol": 5,
        "compounding": False,
        "conf_scaling": False,
    }
    cfg.update(overrides)
    return cfg


def test_exit_policy_applies_trending_tp_bonus_for_high_confidence():
    policy = get_regime_exit_policy(
        "TRENDING",
        base_sl_atr_mult=1.5,
        base_tp_atr_mult=3.0,
        confidence=0.80,
    )

    assert policy["regime"] == "TRENDING"
    assert policy["sl_atr_mult"] == 1.6
    assert policy["tp_atr_mult"] == 4.0
    assert policy["risk_mult"] == 1.0


def test_exit_policy_falls_back_to_symbol_defaults_for_unknown_regime():
    policy = get_regime_exit_policy(
        "UNKNOWN",
        base_sl_atr_mult=1.25,
        base_tp_atr_mult=2.75,
        base_breakeven_atr=0.9,
        confidence=0.60,
    )

    assert policy["regime"] == "UNKNOWN"
    assert policy["sl_atr_mult"] == 1.25
    assert policy["tp_atr_mult"] == 2.75
    assert policy["breakeven_atr"] == 0.9
    assert policy["risk_mult"] == 1.0


def test_backtest_trade_uses_regime_exit_policy_for_sl_tp_and_reporting():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(_engine_config())
    df_h1 = _bars(pd, regime="TRENDING")
    df_h4 = _bars(pd, n=20, regime="TRENDING")

    results = engine.run(
        "EURUSDm",
        df_h1,
        df_h4,
        ml_model=None,
        signal_generator=_PassthroughSignalGenerator(),
    )

    trade = results["trades"][0]
    assert trade["regime"] == "TRENDING"
    assert trade["sl_atr_mult"] == 1.6
    assert trade["tp_atr_mult"] == 4.0
    assert trade["sl"] == 1.0984
    assert trade["tp"] == 1.104


def test_regime_risk_multiplier_reduces_lot_size():
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(_engine_config(conf_scaling=False, compounding=False))

    global_lot = engine.calculate_lot_size("EURUSDm", 1.1000, 1.0985, 0.8, risk_mult=1.0)
    quiet_lot = engine.calculate_lot_size("EURUSDm", 1.1000, 1.0985, 0.8, risk_mult=0.5)

    assert quiet_lot == round(global_lot * 0.5, 2)
