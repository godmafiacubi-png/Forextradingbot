import importlib
import sys
import types


def _load_connector(monkeypatch, initialize_results=None, login_results=None):
    initialize_results = list(initialize_results or [True])
    login_results = list(login_results or [True])

    def initialize(*args, **kwargs):
        return initialize_results.pop(0) if initialize_results else False

    def login(*args, **kwargs):
        return login_results.pop(0) if login_results else False

    mt5_stub = types.SimpleNamespace(
        initialize=initialize,
        login=login,
        shutdown=lambda: None,
        last_error=lambda: (1, "stub failure"),
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5_stub)
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(ndarray=object))
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        types.SimpleNamespace(DataFrame=lambda *args, **kwargs: None, to_datetime=lambda *args, **kwargs: None),
    )
    sys.modules.pop("data_layer.mt5_connector", None)
    import data_layer.mt5_connector as module
    return importlib.reload(module)


def test_connector_fails_closed_when_initialize_fails(monkeypatch):
    module = _load_connector(monkeypatch, initialize_results=[False])

    connector = module.MT5Connector(login=0, password="", server="", path="")

    assert connector.connected is False


def test_connector_reconnects_after_drop(monkeypatch):
    module = _load_connector(monkeypatch, initialize_results=[True, True])
    connector = module.MT5Connector(login=0, password="", server="", path="")
    connector.connected = False

    assert connector.ensure_connected(max_reconnects=1) is True
    assert connector.connected is True


def test_connector_reconnect_stops_after_configured_failures(monkeypatch):
    module = _load_connector(monkeypatch, initialize_results=[True, False, False])
    connector = module.MT5Connector(login=0, password="", server="", path="")
    connector.connected = False

    assert connector.ensure_connected(max_reconnects=2) is False
    assert connector.connected is False
