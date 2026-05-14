"""Bridge trade-journal events into RiskGuard execution-failure counters.

OrderManager already writes structured journal events. This adapter keeps that
journal behavior, while also notifying RiskGuard when broker/execution outcomes
matter for emergency halt logic.
"""


class RiskAwareTradeJournal:
    """Forward journal writes and update RiskGuard on execution outcomes."""

    def __init__(self, journal, risk_guard=None):
        self.journal = journal
        self.risk_guard = risk_guard

    def __getattr__(self, name):
        return getattr(self.journal, name)

    def _record_failure(self, reason):
        if self.risk_guard is not None:
            self.risk_guard.record_order_failure(reason or "order execution failed")

    def _record_success(self):
        if self.risk_guard is not None:
            self.risk_guard.record_order_success()

    def log_order_failed(self, *args, **kwargs):
        comment = kwargs.get("comment") or (args[-1] if args else "")
        self._record_failure(str(comment))
        return self.journal.log_order_failed(*args, **kwargs)

    def log_order_filled(self, *args, **kwargs):
        self._record_success()
        return self.journal.log_order_filled(*args, **kwargs)

    def log_open(self, *args, **kwargs):
        # Some brokers/adapters may only log OPEN after a successful fill, so
        # keep this idempotent enough for safety by resetting failure counters.
        self._record_success()
        return self.journal.log_open(*args, **kwargs)
