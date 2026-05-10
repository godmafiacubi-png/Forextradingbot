"""
Default web dashboard compatibility module.

``main.py`` imports this module whenever ``BOT_MODE`` is not ``AGGRESSIVE``.
The aggressive dashboard depends on Flask and renders a high-risk UI; this
module intentionally keeps the default dashboard import lightweight so config
validation, tests, and non-aggressive startup checks do not fail before runtime
services are needed.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from typing import Any

logger = logging.getLogger(__name__)


dashboard_state: dict[str, Any] = {
    'account': {'balance': 0, 'equity': 0, 'margin': 0, 'free_margin': 0, 'profit': 0, 'growth_pct': 0},
    'symbols': {},
    'signals': {},
    'open_positions': [],
    'closed_trades': [],
    'tracker_stats': None,
    'bot_status': 'STOPPED',
    'iteration': 0,
    'last_update': '',
    'log_messages': [],
    'rl_stats': {},
    'regime_stats': {},
    'news_data': {},
    'risk_guard': {},
    'm30_stats': {},
    'quality_stats': {},
    'adaptive': {},
    'streak': {},
    'auto_adjust': {},
    'daily_pnl': 0,
    'dashboard_port': 5001,
    'mode': 'DEFAULT',
}

_server: ThreadingHTTPServer | None = None
_server_thread: threading.Thread | None = None


def update_dashboard(key: str, value: Any) -> None:
    """Update one dashboard state key and refresh the last-update timestamp."""
    dashboard_state[key] = value
    dashboard_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def add_log(msg: str) -> None:
    """Append a dashboard log message, keeping the in-memory log bounded."""
    dashboard_state['log_messages'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'msg': str(msg),
    })
    if len(dashboard_state['log_messages']) > 200:
        dashboard_state['log_messages'] = dashboard_state['log_messages'][-200:]


class _DashboardHandler(BaseHTTPRequestHandler):
    """Tiny stdlib HTTP handler for the default dashboard."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path not in ('/', '/index.html'):
            self.send_error(404)
            return

        body = _render_html().encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("Default dashboard: " + format, *args)


def _render_html() -> str:
    account = dashboard_state.get('account') or {}
    logs = dashboard_state.get('log_messages') or []
    recent_logs = logs[-25:]
    log_rows = ''.join(
        f"<li><span>{escape(str(item.get('time', '')))}</span> {escape(str(item.get('msg', '')))}</li>"
        for item in recent_logs
    ) or '<li>No log messages yet.</li>'

    symbols = dashboard_state.get('symbols') or {}
    symbol_rows = ''.join(
        f"<tr><td>{escape(str(symbol))}</td><td>{escape(str(data))}</td></tr>"
        for symbol, data in symbols.items()
    ) or '<tr><td colspan="2">No symbol data yet.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>Trading Bot Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #101820; color: #eef2f7; }}
    .card {{ background: #172331; border: 1px solid #2f4054; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .75rem; }}
    .metric {{ background: #0f1722; border-radius: 8px; padding: .75rem; }}
    .label {{ color: #93a4b8; font-size: .8rem; text-transform: uppercase; }}
    .value {{ font-size: 1.3rem; font-weight: 700; margin-top: .25rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ border-bottom: 1px solid #2f4054; padding: .5rem; text-align: left; vertical-align: top; }}
    li {{ margin: .35rem 0; }}
    li span {{ color: #93a4b8; font-family: monospace; }}
  </style>
</head>
<body>
  <h1>Trading Bot Dashboard <small>({escape(str(dashboard_state.get('mode', 'DEFAULT')))} mode)</small></h1>
  <p>Status: <strong>{escape(str(dashboard_state.get('bot_status', 'STOPPED')))}</strong> |
     Iteration: {escape(str(dashboard_state.get('iteration', 0)))} |
     Last update: {escape(str(dashboard_state.get('last_update', '')))}</p>
  <section class="card">
    <h2>Account</h2>
    <div class="grid">
      <div class="metric"><div class="label">Balance</div><div class="value">{escape(str(account.get('balance', 0)))}</div></div>
      <div class="metric"><div class="label">Equity</div><div class="value">{escape(str(account.get('equity', 0)))}</div></div>
      <div class="metric"><div class="label">Profit</div><div class="value">{escape(str(account.get('profit', 0)))}</div></div>
      <div class="metric"><div class="label">Daily PnL</div><div class="value">{escape(str(dashboard_state.get('daily_pnl', 0)))}</div></div>
    </div>
  </section>
  <section class="card">
    <h2>Symbols</h2>
    <table><tbody>{symbol_rows}</tbody></table>
  </section>
  <section class="card">
    <h2>Recent Logs</h2>
    <ul>{log_rows}</ul>
  </section>
</body>
</html>"""


def start_dashboard(port: int | None = None, host: str = '0.0.0.0') -> ThreadingHTTPServer:
    """
    Start the lightweight default dashboard in a daemon thread.

    The function is idempotent for a running process and returns the active
    server object, which is useful for smoke tests and operational diagnostics.
    """
    global _server, _server_thread

    if _server is not None:
        return _server

    selected_port = int(port or dashboard_state.get('dashboard_port', 5001))
    dashboard_state['dashboard_port'] = selected_port
    _server = ThreadingHTTPServer((host, selected_port), _DashboardHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, name='default-dashboard', daemon=True)
    _server_thread.start()
    logger.info("Default dashboard started on http://%s:%s", host, selected_port)
    return _server
