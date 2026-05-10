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
