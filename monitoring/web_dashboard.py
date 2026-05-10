"""
Default web dashboard compatibility module.

``main.py`` imports this module whenever ``BOT_MODE`` is not ``AGGRESSIVE``.
The aggressive dashboard depends on Flask and renders a high-risk UI; this
module intentionally keeps the default dashboard import lightweight so config
validation, tests, and non-aggressive startup checks do not fail before runtime
services are needed.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

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
_opened_browser = False


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


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that can restart quickly on the same dashboard port."""

    allow_reuse_address = True
    daemon_threads = True


class _DashboardHandler(BaseHTTPRequestHandler):
    """Tiny stdlib HTTP handler for the default dashboard."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            self._send_text(_render_html(), 'text/html; charset=utf-8')
            return
        if parsed.path == '/api/state':
            self._send_text(json.dumps(dashboard_state, default=str), 'application/json; charset=utf-8')
            return
        if parsed.path == '/health':
            self._send_text(json.dumps({'ok': True, 'status': dashboard_state.get('bot_status')}), 'application/json; charset=utf-8')
            return
        if parsed.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def _send_text(self, content: str, content_type: str) -> None:
        body = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("Default dashboard: " + format, *args)


def _format_number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_dashboard_value(key: Any, value: Any) -> str:
    """Format scalar dashboard values so operators can scan them quickly."""
    key_text = str(key).lower()
    if value in (None, ''):
        return '—'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if key_text.endswith(('pct', 'percent')):
            return f"{float(value):,.2f}%"
        if 'rate' in key_text or 'confidence' in key_text:
            numeric = float(value)
            if abs(numeric) <= 1:
                numeric *= 100
            return f"{numeric:,.2f}%"
        if any(token in key_text for token in ('price', 'entry', 'current', 'sl', 'tp')):
            return _format_number(value, 5)
        if any(token in key_text for token in ('adx', 'rsi', 'spread', 'atr', 'profit', 'pnl', 'balance', 'equity')):
            return _format_number(value, 2)
        if float(value).is_integer():
            return f"{int(value):,}"
        return _format_number(value, 4)
    return str(value)


def _json_summary(value: Any) -> str:
    if value in (None, '', [], {}):
        return '—'
    try:
        return json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except TypeError:
        return str(value)


def _badge_class(value: Any) -> str:
    text = str(value).upper()
    if any(word in text for word in ('RUNNING', 'BUY', 'PROFIT', 'PASS', 'OK')):
        return 'positive'
    if any(word in text for word in ('STOP', 'SELL', 'LOSS', 'BLOCK', 'ERROR', 'FAIL')):
        return 'negative'
    return 'neutral'


def _render_kv_table(data: Any, empty: str = 'No data yet.') -> str:
    if not isinstance(data, dict) or not data:
        return f'<tr><td colspan="2" class="muted">{escape(empty)}</td></tr>'
    rows = []
    for key, value in data.items():
        label = escape(str(key).replace('_', ' ').title())
        if isinstance(value, dict) and value:
            compact_items = ''.join(
                '<div class="kv-chip">'
                f'<span>{escape(str(child_key).replace("_", " "))}</span>'
                f'<strong>{escape(_format_dashboard_value(child_key, child_value))}</strong>'
                '</div>'
                for child_key, child_value in value.items()
                if not isinstance(child_value, (dict, list, tuple))
            )
            details = escape(_json_summary(value))
            value_html = (
                f'<div class="kv-chips">{compact_items}</div>'
                f'<details><summary>Raw data</summary><pre>{details}</pre></details>'
            ) if compact_items else f'<pre>{details}</pre>'
        elif isinstance(value, (list, tuple)):
            value_html = f'<pre>{escape(_json_summary(value))}</pre>'
        else:
            value_html = f'<span class="scalar {_badge_class(value)}">{escape(_format_dashboard_value(key, value))}</span>'
        rows.append(
            '<tr>'
            f'<th>{label}</th>'
            f'<td>{value_html}</td>'
            '</tr>'
        )
    return ''.join(rows)


def _render_positions() -> str:
    positions = dashboard_state.get('open_positions') or []
    if not positions:
        return '<tr><td colspan="8" class="muted">No open positions.</td></tr>'
    rows = []
    for pos in positions:
        side = pos.get('side', '') if isinstance(pos, dict) else ''
        pnl = pos.get('pnl', 0) if isinstance(pos, dict) else 0
        try:
            pnl_value = float(pnl or 0)
        except (TypeError, ValueError):
            pnl_value = 0.0
        rows.append(
            '<tr>'
            f'<td>{escape(str(pos.get("ticket", "")))}</td>'
            f'<td>{escape(str(pos.get("symbol", "")))}</td>'
            f'<td><span class="badge {_badge_class(side)}">{escape(str(side))}</span></td>'
            f'<td>{escape(_format_number(pos.get("volume", 0), 2))}</td>'
            f'<td>{escape(_format_number(pos.get("entry", 0), 5))}</td>'
            f'<td>{escape(_format_number(pos.get("current_price", 0), 5))}</td>'
            f'<td>{escape(_format_number(pos.get("sl", 0), 5))}</td>'
            f'<td class="{_badge_class("PROFIT" if pnl_value >= 0 else "LOSS")}">{escape(_format_number(pnl, 2))}</td>'
            '</tr>'
        )
    return ''.join(rows)


def _render_html() -> str:
    account = dashboard_state.get('account') or {}
    logs = dashboard_state.get('log_messages') or []
    recent_logs = logs[-30:]
    log_rows = ''.join(
        f"<li><span>{escape(str(item.get('time', '')))}</span> {escape(str(item.get('msg', '')))}</li>"
        for item in recent_logs
    ) or '<li class="muted">No log messages yet.</li>'

    status = dashboard_state.get('bot_status', 'STOPPED')
    profit = account.get('profit', 0)
    daily_pnl = dashboard_state.get('daily_pnl', 0)
    try:
        profit_value = float(profit or 0)
    except (TypeError, ValueError):
        profit_value = 0.0
    try:
        daily_pnl_value = float(daily_pnl or 0)
    except (TypeError, ValueError):
        daily_pnl_value = 0.0
    open_positions = dashboard_state.get('open_positions') or []

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>Trading Bot Dashboard</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0f1b2d; --panel2:#13243a; --line:#263a56; --text:#eef5ff; --muted:#93a8c2; --green:#2ee59d; --red:#ff6b7a; --amber:#ffd166; --blue:#60a5fa; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; min-height:100vh; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif; background: radial-gradient(circle at top left, #12355f 0, transparent 32rem), var(--bg); color: var(--text); }}
    .wrap {{ width:min(1400px, calc(100% - 32px)); margin:0 auto; padding:28px 0 48px; }}
    .hero {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; margin-bottom:22px; }}
    h1 {{ margin:0; font-size:clamp(1.8rem, 4vw, 3.2rem); letter-spacing:-.04em; }}
    h2 {{ margin:0 0 14px; font-size:1rem; letter-spacing:.08em; text-transform:uppercase; color:#cfe2ff; }}
    .subtitle {{ color:var(--muted); margin-top:8px; }}
    .pill, .badge {{ display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:7px 12px; font-weight:700; font-size:.85rem; border:1px solid var(--line); background:rgba(255,255,255,.04); }}
    .positive {{ color:var(--green); }} .negative {{ color:var(--red); }} .neutral {{ color:var(--amber); }} .muted {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap:16px; }}
    .card {{ grid-column: span 12; background:linear-gradient(180deg, rgba(19,36,58,.92), rgba(12,24,40,.92)); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 18px 50px rgba(0,0,0,.25); min-width:0; }}
    .span-3 {{ grid-column: span 3; }} .span-4 {{ grid-column: span 4; }} .span-6 {{ grid-column: span 6; }} .span-8 {{ grid-column: span 8; }}
    .metric {{ min-height:128px; display:flex; flex-direction:column; justify-content:space-between; }}
    .label {{ color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; }}
    .value {{ font-size:clamp(1.5rem, 3vw, 2.5rem); font-weight:800; letter-spacing:-.04em; margin-top:8px; }}
    .hint {{ color:var(--muted); font-size:.86rem; margin-top:10px; }}
    .table-scroll {{ overflow:auto; max-height:520px; }}
    table {{ width:100%; border-collapse:collapse; overflow:hidden; }}
    th, td {{ border-bottom:1px solid rgba(147,168,194,.18); padding:12px 10px; text-align:left; vertical-align:top; }}
    th {{ color:#bdd2ee; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }}
    pre {{ white-space:pre-wrap; word-break:break-word; margin:0; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color:#dcecff; }}
    details {{ margin-top:8px; }} details summary {{ cursor:pointer; color:var(--muted); font-size:.78rem; }}
    .kv-chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .kv-chip {{ display:flex; gap:6px; align-items:baseline; border:1px solid rgba(147,168,194,.16); background:rgba(255,255,255,.035); border-radius:10px; padding:6px 8px; }}
    .kv-chip span {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; }}
    .kv-chip strong, .scalar {{ font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    ul {{ padding-left:1.1rem; margin:0; max-height:420px; overflow:auto; }}
    li {{ margin:.48rem 0; }} li span {{ color:var(--blue); font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .toolbar {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:10px; }}
    a {{ color:#93c5fd; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    @media (max-width: 980px) {{ .span-3, .span-4, .span-6, .span-8 {{ grid-column: span 12; }} .hero {{ flex-direction:column; }} .toolbar {{ justify-content:flex-start; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <div>
        <h1>Trading Bot Dashboard</h1>
        <div class="subtitle">Mode: <strong>{escape(str(dashboard_state.get('mode', 'DEFAULT')))}</strong> · Iteration {escape(str(dashboard_state.get('iteration', 0)))} · Last update {escape(str(dashboard_state.get('last_update') or 'waiting for data'))}</div>
      </div>
      <div class="toolbar">
        <span class="pill {_badge_class(status)}">● {escape(str(status))}</span>
        <a class="pill" href="/api/state">API State</a>
        <a class="pill" href="/health">Health</a>
      </div>
    </header>

    <section class="grid">
      <article class="card metric span-3"><div><div class="label">Balance</div><div class="value">{escape(_format_number(account.get('balance', 0)))}</div></div><div class="hint">Account cash balance</div></article>
      <article class="card metric span-3"><div><div class="label">Equity</div><div class="value">{escape(_format_number(account.get('equity', 0)))}</div></div><div class="hint">Live equity including floating PnL</div></article>
      <article class="card metric span-3"><div><div class="label">Profit</div><div class="value {_badge_class('PROFIT' if profit_value >= 0 else 'LOSS')}">{escape(_format_number(profit))}</div></div><div class="hint">Current account profit</div></article>
      <article class="card metric span-3"><div><div class="label">Daily PnL</div><div class="value {_badge_class('PROFIT' if daily_pnl_value >= 0 else 'LOSS')}">{escape(_format_number(daily_pnl))}</div></div><div class="hint">Today’s realized bot PnL</div></article>

      <article class="card span-8">
        <h2>Open Positions ({len(open_positions)})</h2>
        <div class="table-scroll"><table><thead><tr><th>Ticket</th><th>Symbol</th><th>Side</th><th>Lots</th><th>Entry</th><th>Current</th><th>SL</th><th>PnL</th></tr></thead><tbody>{_render_positions()}</tbody></table></div>
      </article>
      <article class="card span-4">
        <h2>Risk Guard</h2>
        <div class="table-scroll"><table><tbody>{_render_kv_table(dashboard_state.get('risk_guard'), 'Risk status has not been reported yet.')}</tbody></table></div>
      </article>

      <article class="card span-6">
        <h2>Signals</h2>
        <div class="table-scroll"><table><tbody>{_render_kv_table(dashboard_state.get('signals'), 'No signal data yet.')}</tbody></table></div>
      </article>
      <article class="card span-6">
        <h2>Symbols</h2>
        <div class="table-scroll"><table><tbody>{_render_kv_table(dashboard_state.get('symbols'), 'No symbol data yet.')}</tbody></table></div>
      </article>

      <article class="card span-4">
        <h2>Performance</h2>
        <div class="table-scroll"><table><tbody>{_render_kv_table(dashboard_state.get('tracker_stats'), 'No closed-trade statistics yet.')}</tbody></table></div>
      </article>
      <article class="card span-4">
        <h2>Quality / M30</h2>
        <div class="table-scroll"><table><tbody>{_render_kv_table({'quality': dashboard_state.get('quality_stats'), 'm30': dashboard_state.get('m30_stats')})}</tbody></table></div>
      </article>
      <article class="card span-4">
        <h2>AI / Regime</h2>
        <div class="table-scroll"><table><tbody>{_render_kv_table({'rl': dashboard_state.get('rl_stats'), 'regime': dashboard_state.get('regime_stats')})}</tbody></table></div>
      </article>

      <article class="card span-6">
        <h2>Adaptive Controls</h2>
        <div class="table-scroll"><table><tbody>{_render_kv_table({'adaptive': dashboard_state.get('adaptive'), 'streak': dashboard_state.get('streak'), 'auto_adjust': dashboard_state.get('auto_adjust')})}</tbody></table></div>
      </article>
      <article class="card span-6">
        <h2>Recent Logs</h2>
        <ul>{log_rows}</ul>
      </article>
    </section>
  </main>
</body>
</html>"""


def start_dashboard(port: int | None = None, host: str = '0.0.0.0', open_browser: bool = True) -> ThreadingHTTPServer:
    """
    Start the lightweight default dashboard in a daemon thread.

    The function is idempotent for a running process and returns the active
    server object, which is useful for smoke tests and operational diagnostics.
    By default it also opens the local dashboard URL once, matching the
    aggressive dashboard behavior and making manual launches easier.
    """
    global _server, _server_thread, _opened_browser

    if _server is not None:
        return _server

    selected_port = int(port or dashboard_state.get('dashboard_port', 5001))
    dashboard_state['dashboard_port'] = selected_port
    _server = _ReusableThreadingHTTPServer((host, selected_port), _DashboardHandler)
    actual_port = int(_server.server_address[1])
    dashboard_state['dashboard_port'] = actual_port
    _server_thread = threading.Thread(target=_server.serve_forever, name='default-dashboard', daemon=True)
    _server_thread.start()

    url = f"http://localhost:{actual_port}"
    logger.info("Default dashboard started on %s (bind host: %s)", url, host)
    if open_browser and not _opened_browser:
        try:
            webbrowser.open(url)
            _opened_browser = True
        except Exception as exc:  # browser launch is best-effort; server is already running
            logger.warning("Default dashboard is running; open manually: %s (%s)", url, exc)
    return _server
