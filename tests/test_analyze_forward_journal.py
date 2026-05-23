import csv
import io
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path


def _run_script(csv_path: Path) -> str:
    old_argv = sys.argv[:]
    sys.argv = ["analyze_forward_journal.py", str(csv_path)]
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            runpy.run_path("scripts/analyze_forward_journal.py", run_name="__main__")
    finally:
        sys.argv = old_argv
    return out.getvalue()


def _write_csv(path: Path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_old_journal_without_market_context_columns_still_works(tmp_path):
    path = tmp_path / "old.csv"
    headers = ["event_type", "symbol", "pnl", "ticket", "slippage_points", "planned_rr", "execution_rr"]
    _write_csv(path, headers, [{"event_type": "CLOSE", "symbol": "EURUSD", "pnl": "5.0"}])
    out = _run_script(path)
    assert "total events: 1" in out
    assert "pnl by market_context:" in out


def test_rejected_events_grouped_by_symbol(tmp_path):
    path = tmp_path / "rej.csv"
    headers = ["event_type", "symbol", "reason", "comment"]
    _write_csv(path, headers, [
        {"event_type": "ORDER_REJECTED", "symbol": "EURUSD", "reason": "spread"},
        {"event_type": "ORDER_REJECTED", "symbol": "EURUSD", "reason": "spread"},
    ])
    out = _run_script(path)
    assert "EURUSD | spread: 2" in out


def test_realized_pnl_includes_close_and_partial_close(tmp_path):
    path = tmp_path / "pnl.csv"
    headers = ["event_type", "symbol", "pnl"]
    _write_csv(path, headers, [
        {"event_type": "CLOSE", "symbol": "EURUSD", "pnl": "10"},
        {"event_type": "PARTIAL_CLOSE", "symbol": "EURUSD", "pnl": "-3"},
    ])
    out = _run_script(path)
    assert "realized pnl (close+partial_close): 7.00" in out


def test_attempt_without_terminal_event_detected(tmp_path):
    path = tmp_path / "dangling.csv"
    headers = ["event_type", "symbol", "ticket"]
    _write_csv(path, headers, [
        {"event_type": "ORDER_ATTEMPT", "symbol": "EURUSD", "ticket": ""},
    ])
    out = _run_script(path)
    assert "attempted orders without terminal event: 1" in out
