import csv

from execution.trade_logger import JOURNAL_FIELDS
from scripts.generate_forward_report import Report, render_report


def test_forward_report_summarizes_edge_and_execution_metrics():
    rows = [
        {
            "event_time": "2026-01-01T08:00:00+00:00",
            "event_type": "CLOSE",
            "ticket": "1",
            "symbol": "EURUSDm",
            "side": "BUY",
            "volume": "0.1",
            "price": "1.1",
            "sl": "",
            "tp": "",
            "pnl": "100",
            "comment": "slippage_points=1.5",
        },
        {
            "event_time": "2026-01-01T14:00:00+00:00",
            "event_type": "CLOSE",
            "ticket": "2",
            "symbol": "XAUUSDm",
            "side": "SELL",
            "volume": "0.1",
            "price": "2350",
            "sl": "",
            "tp": "",
            "pnl": "-40",
            "comment": "slippage_points=2.5",
        },
        {
            "event_time": "2026-01-01T15:00:00+00:00",
            "event_type": "ORDER_FAILED",
            "ticket": "",
            "symbol": "XAUUSDm",
            "side": "SELL",
            "volume": "0.1",
            "price": "2350",
            "sl": "",
            "tp": "",
            "pnl": "",
            "comment": "retcode=10030",
        },
    ]

    metrics = Report(rows).metrics()

    assert metrics["total_trades"] == 2
    assert metrics["win_rate"] == 50.0
    assert metrics["profit_factor"] == 2.5
    assert metrics["max_drawdown"] == 40.0
    assert metrics["expectancy"] == 30.0
    assert metrics["largest_loss"] == -40.0
    assert metrics["execution_failures"] == 1
    assert metrics["symbol_breakdown"] == {"EURUSDm": 100.0, "XAUUSDm": -40.0}
    assert metrics["session_breakdown"] == {"London": 100.0, "NewYork": -40.0}
    assert metrics["slippage_average"] == 2.0
    assert "Forward Performance Report" in render_report(metrics)


def test_forward_report_loads_trade_journal_csv(tmp_path):
    journal_path = tmp_path / "trades.csv"
    with journal_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=JOURNAL_FIELDS)
        writer.writeheader()
        writer.writerow({field: "" for field in JOURNAL_FIELDS} | {"event_type": "CLOSE", "pnl": "5"})

    from scripts.generate_forward_report import load_rows

    assert load_rows(journal_path)[0]["pnl"] == "5"
