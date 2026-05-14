from execution.risk_aware_journal import RiskAwareTradeJournal
from execution.trade_logger import TradeJournal


class _RiskGuardSpy:
    def __init__(self):
        self.failures = []
        self.successes = 0

    def record_order_failure(self, reason=""):
        self.failures.append(reason)

    def record_order_success(self):
        self.successes += 1


def test_risk_aware_journal_records_failures_and_successes(tmp_path):
    journal = TradeJournal(csv_path=tmp_path / "trades.csv")
    risk_guard = _RiskGuardSpy()
    adapter = RiskAwareTradeJournal(journal, risk_guard)

    adapter.log_order_failed("XAUUSDm", "BUY", comment="retcode=10030")
    adapter.log_order_filled(123, "XAUUSDm", "BUY", 0.1, 2350.0, sl=2340.0, tp=2370.0)

    assert risk_guard.failures == ["retcode=10030"]
    assert risk_guard.successes == 1


def test_risk_aware_journal_delegates_other_events(tmp_path):
    journal = TradeJournal(csv_path=tmp_path / "trades.csv")
    risk_guard = _RiskGuardSpy()
    adapter = RiskAwareTradeJournal(journal, risk_guard)

    row = adapter.log_risk_blocked("EURUSDm", "SELL", comment="daily limit")

    assert row["event_type"] == "RISK_BLOCKED"
    assert risk_guard.failures == []
    assert risk_guard.successes == 0
