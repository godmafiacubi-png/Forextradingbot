import numpy as np

from ml_models.deep_rl_agent_v22 import DeepRLTradingAgent


def test_build_state_uses_normalized_symbol_performance_for_broker_suffixes(tmp_path):
    agent = DeepRLTradingAgent(model_dir=str(tmp_path), batch_size=4)
    agent.symbol_performance["EURUSD"].update(
        {"trades": 4, "wins": 3, "total_pnl": 12.5, "rewards": [1.0, 2.0, -1.0, 4.0]}
    )

    state = agent.build_state(
        market_data={"rsi": 55, "adx": 20, "session": "LONDON"},
        signal_data={"ml_prob": 0.62, "confidence": 0.7},
        has_position=False,
        pnl_pct=0.0,
        symbol="EURUSDm",
    )

    assert len(state) == agent.STATE_SIZE
    assert state[21] == 0.75
    assert np.isclose(state[22], 1.5)


def test_symbol_head_warmup_uses_normalized_broker_symbol_keys(tmp_path):
    agent = DeepRLTradingAgent(model_dir=str(tmp_path), batch_size=4)

    assert agent._inference_symbol_key("XAUUSD.r") == "__default__"

    agent.symbol_performance["XAUUSD"].update(
        {"trades": agent._min_symbol_head_trades, "wins": 6, "total_pnl": 0.0, "rewards": []}
    )

    assert agent._inference_symbol_key("XAUUSD.r") == "XAUUSD"
