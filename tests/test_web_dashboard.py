import importlib
import sys


def test_default_dashboard_imports_without_flask(monkeypatch):
    """Default dashboard must be importable in dependency-light environments."""
    monkeypatch.setitem(sys.modules, "flask", None)
    module = importlib.import_module("monitoring.web_dashboard")

    assert module.dashboard_state["mode"] == "DEFAULT"
    assert callable(module.update_dashboard)
    assert callable(module.add_log)
    assert callable(module.start_dashboard)


def test_default_dashboard_state_updates_and_bounds_logs():
    from monitoring import web_dashboard

    web_dashboard.update_dashboard("bot_status", "RUNNING")
    assert web_dashboard.dashboard_state["bot_status"] == "RUNNING"
    assert web_dashboard.dashboard_state["last_update"]

    web_dashboard.dashboard_state["log_messages"] = []
    for idx in range(205):
        web_dashboard.add_log(f"message-{idx}")

    assert len(web_dashboard.dashboard_state["log_messages"]) == 200
    assert web_dashboard.dashboard_state["log_messages"][0]["msg"] == "message-5"
    assert web_dashboard.dashboard_state["log_messages"][-1]["msg"] == "message-204"


def test_default_dashboard_renders_improved_sections_and_json_api():
    from urllib.request import urlopen

    from monitoring import web_dashboard

    web_dashboard.dashboard_state["open_positions"] = [
        {
            "ticket": 101,
            "symbol": "EURUSDm",
            "side": "BUY",
            "volume": 0.1,
            "entry": 1.08234,
            "current_price": 1.08334,
            "sl": 1.079,
            "pnl": 12.5,
        }
    ]
    web_dashboard.dashboard_state["signals"] = {"EURUSDm": {"signal": "BUY", "confidence": 0.72}}
    web_dashboard.dashboard_state["symbols"] = {"EURUSDm": {"price": 1.08334, "adx": 24.125}}
    web_dashboard.dashboard_state["risk_guard"] = {"status": "OK", "drawdown_pct": -0.1399}

    server = web_dashboard.start_dashboard(port=0, host="127.0.0.1", open_browser=False)
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as response:
            payload = response.read().decode("utf-8")
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            health = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        web_dashboard._server = None
        web_dashboard._server_thread = None

    assert "Trading Bot Dashboard" in html
    assert "Open Positions (1)" in html
    assert "Risk Guard" in html
    assert "API State" in html
    assert "kv-chip" in html
    assert "72.00%" in html
    assert "-0.14%" in html
    assert '"EURUSDm"' in payload
    assert '"ok": true' in health
