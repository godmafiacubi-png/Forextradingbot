from pathlib import Path


def test_main_sig_data_and_diagnostics_include_market_context_fields():
    content = Path("main.py").read_text(encoding="utf-8")
    required_fields = [
        "'market_context'",
        "'context_bias'",
        "'context_strategy'",
        "'context_allow_trade'",
        "'context_reason'",
        "'context_risk_mult'",
        "'context_tp_rr'",
        "'context_sl_atr_mult'",
        "'context_min_quality_score'",
    ]
    for field in required_fields:
        assert content.count(field) >= 2, f"{field} should exist in sig_data and diagnostics"


def test_main_market_context_is_advisory_only_not_used_as_blocking_condition():
    content = Path("main.py").read_text(encoding="utf-8")
    assert "market_context = self.market_context_engine.analyze_row(context_row, sym_cfg)" in content
    assert "blocked = f\"Context" not in content
    assert "if not market_context.allow_trade" not in content
    assert "context_allow_trade" in content
