import pytest
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from strategy.signal_generator import SignalGenerator


class _StubModel:
    def predict(self, df):
        return np.array([0.7] * len(df))


def _base_row():
    return {
        'near_demand_ob': 0.001,
        'near_supply_ob': 1.0,
        'near_bull_fvg': 1.0,
        'near_bear_fvg': 1.0,
        'fvg_bullish': 0,
        'fvg_bearish': 0,
        'fvg_bull_unfilled': 0,
        'fvg_bear_unfilled': 0,
        'fvg_bull_retest': 0,
        'fvg_bear_retest': 0,
        'fvg_bull_mitigated': 0,
        'fvg_bear_mitigated': 0,
        'bos_bullish': 1,
        'bos_bearish': 0,
        'choch_bullish': 0,
        'choch_bearish': 0,
        'in_ote_buy_zone': 0,
        'in_ote_sell_zone': 0,
        'structure': 1,
        'liq_sweep_low': 0,
        'liq_sweep_high': 0,
        'rsi': 45,
        'macd_hist': 0.1,
        'stoch_k': 35,
        'adx': 25,
        'ema_cross': 1,
        'zz_direction': 1,
    }


def test_fvg_newly_created_only_does_not_trigger_entry_score():
    row = _base_row()
    row['near_demand_ob'] = 1.0
    row['bos_bullish'] = 0
    row['structure'] = 0
    row['fvg_bullish'] = 1  # creation only
    df = pd.DataFrame([row])
    out = SignalGenerator(_StubModel(), use_meta_strategy_selector=False).generate_signals(df)
    assert int(out.iloc[0]['signal']) == 0
