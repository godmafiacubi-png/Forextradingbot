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
import os
import threading
import webbrowser
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

APP_NAME = 'ForexTradingBot'
DEFAULT_BOT_VERSION = os.getenv('BOT_VERSION', 'V8.0').upper()
DEFAULT_EXECUTION_MODE = 'Dryruns'


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
    'bot_version': f'{APP_NAME} {DEFAULT_BOT_VERSION}',
    'execution_mode': DEFAULT_EXECUTION_MODE,
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


def _state_get(data: Any, key: str, default: Any = 0) -> Any:
    """Read a value from dict-like or object-like dashboard payloads."""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _pct(value: Any, scale_unit_interval: bool = True, decimals: int = 0) -> str:
    try:
        numeric = float(value or 0)
    except (TypeError, ValueError):
        numeric = 0.0
    if scale_unit_interval and abs(numeric) <= 1:
        numeric *= 100
    return f"{numeric:.{decimals}f}%"


def _money(value: Any) -> str:
    try:
        return f"${float(value or 0):.2f}"
    except (TypeError, ValueError):
        return f"${escape(str(value))}"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return default


def _render_positions() -> str:
    positions = dashboard_state.get('open_positions') or []
    if not positions:
        return '<tr><td colspan="9" class="muted empty-row">No open positions</td></tr>'
    rows = []
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        side = str(pos.get('side', ''))
        pnl = pos.get('pnl', 0)
        try:
            pnl_value = float(pnl or 0)
        except (TypeError, ValueError):
            pnl_value = 0.0
        side_class = 'signal-buy' if side.upper() == 'BUY' else 'signal-sell'
        rows.append(
            '<tr>'
            f'<td>#{escape(str(pos.get("ticket", "")))}</td>'
            f'<td><strong>{escape(str(pos.get("symbol", "")))}</strong></td>'
            f'<td class="{side_class}">{escape(side)}</td>'
            f'<td>{escape(_format_number(pos.get("volume", 0), 2))}</td>'
            f'<td>{escape(_format_number(pos.get("entry", 0), 5))}</td>'
            f'<td>{escape(_format_number(pos.get("current_price", 0), 5))}</td>'
            f'<td>{escape(_format_number(pos.get("sl", 0), 5))}</td>'
            f'<td>{escape(_format_number(pos.get("tp", 0), 5))}</td>'
            f'<td class="{"pnl-pos" if pnl_value >= 0 else "pnl-neg"}">{escape(_money(pnl))}</td>'
            '</tr>'
        )
    return ''.join(rows) or '<tr><td colspan="9" class="muted empty-row">No open positions</td></tr>'


def _mini_stat(label: str, value: Any, css_class: str = '') -> str:
    return (
        '<div class="mini-stat kv-chip">'
        f'<span class="label">{escape(label)}</span>'
        f'<span class="value {css_class}">{escape(str(value))}</span>'
        '</div>'
    )


def _render_regimes() -> str:
    regimes = dashboard_state.get('regime_stats') or {}
    if not isinstance(regimes, dict) or not regimes:
        return _mini_stat('Waiting:', '—')
    chips = []
    for symbol, regime in regimes.items():
        regime_text = str(regime)
        chips.append(_mini_stat(f'{str(symbol)[:6]}:', regime_text, f'regime-{regime_text.lower()}'))
    return ''.join(chips)


def _render_logs() -> str:
    logs = dashboard_state.get('log_messages') or []
    rows = []
    for item in reversed(logs[-50:]):
        if isinstance(item, dict):
            log_time = item.get('time', '')
            msg = item.get('msg', '')
        else:
            log_time = ''
            msg = item
        rows.append(
            '<div class="log-line">'
            f'<span class="log-time">{escape(str(log_time))}</span> {escape(str(msg))}'
            '</div>'
        )
    return ''.join(rows) or '<div class="log-line muted">No log messages yet.</div>'


def _render_html() -> str:
    account = dashboard_state.get('account') or {}
    tracker = dashboard_state.get('tracker_stats') or {}
    risk = dashboard_state.get('risk_guard') or {}
    quality = dashboard_state.get('quality_stats') or {}
    adaptive = dashboard_state.get('adaptive') or {}
    m30 = dashboard_state.get('m30_stats') or {}
    streak = dashboard_state.get('streak') or {}
    rl = dashboard_state.get('rl_stats') or {}
    status = str(dashboard_state.get('bot_status', 'STOPPED'))
    bot_version = str(dashboard_state.get('bot_version') or f'{APP_NAME} {DEFAULT_BOT_VERSION}')
    execution_mode = str(dashboard_state.get('execution_mode') or DEFAULT_EXECUTION_MODE)

    profit = account.get('profit', 0)
    growth = account.get('growth_pct', 0)
    total_pnl = _state_get(tracker, 'total_pnl', 0)
    daily_pnl = risk.get('daily_pnl', dashboard_state.get('daily_pnl', 0)) if isinstance(risk, dict) else dashboard_state.get('daily_pnl', 0)

    def signed_class(value: Any) -> str:
        try:
            return 'positive' if float(value or 0) >= 0 else 'negative'
        except (TypeError, ValueError):
            return 'neutral'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>⚡ {escape(bot_version)} Dashboard</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#061a3a; color:#e1e5ea; font-family:'Segoe UI', Consolas, monospace; font-size:14px; padding:15px; }}
    .header {{ display:flex; justify-content:space-between; align-items:center; padding:15px 20px; background:linear-gradient(135deg,#0b2a5b,#0f3d7a); border:1px solid #1e63b6; border-radius:10px; margin-bottom:15px; }}
    .header h1 {{ font-size:22px; background:linear-gradient(90deg,#ff4444,#ff8800); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .mode-badge,.execution-badge {{ display:inline-block; padding:4px 12px; border-radius:12px; font-size:11px; font-weight:700; color:#fff; margin:2px 0 0 10px; }}
    .mode-badge {{ background:#d32f2f; }}
    .execution-badge {{ background:#00695c; }}
    .header-info {{ color:#9bbcff; font-size:12px; }}
    .status-badge {{ padding:8px 18px; border-radius:20px; font-weight:800; font-size:13px; text-transform:uppercase; }}
    .status-running {{ background:#ff6d00; color:#000; }} .status-stopped {{ background:#ff1744; color:#fff; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(320px,1fr)); gap:15px; margin-bottom:15px; }}
    .grid-3 {{ display:grid; grid-template-columns:repeat(3,minmax(250px,1fr)); gap:15px; margin-bottom:15px; }}
    .card {{ background:#0b2448; border:1px solid #1b4f93; border-radius:10px; padding:18px; }}
    .card h2 {{ font-size:14px; color:#ff8800; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; border-bottom:1px solid #1b4f93; padding-bottom:8px; }}
    .account-grid,.stats-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .account-item,.stat-item {{ background:#102f5f; border-radius:8px; padding:13px 10px; }}
    .account-item {{ text-align:center; }} .account-label,.stat-label {{ font-size:11px; color:#9bbcff; text-transform:uppercase; }}
    .account-value {{ font-size:22px; font-weight:800; margin-top:6px; }} .stat-value {{ font-size:16px; font-weight:800; margin-top:5px; }}
    .positive,.pnl-pos {{ color:#00e676; }} .negative,.pnl-neg {{ color:#ff1744; }} .neutral {{ color:#ff8800; }}
    .mini-stats {{ display:flex; gap:10px; flex-wrap:wrap; }} .mini-stat {{ padding:8px 12px; background:#102f5f; border-radius:6px; font-size:12px; }}
    .mini-stat .label {{ color:#9bbcff; }} .mini-stat .value {{ font-weight:800; margin-left:4px; }}
    table {{ width:100%; border-collapse:collapse; }} th {{ text-align:left; padding:10px; color:#ff8800; font-size:11px; text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid #1b4f93; }}
    td {{ padding:10px; border-bottom:1px solid #0b2448; font-size:13px; }} tr:hover {{ background:#102f5f; }}
    .signal-buy {{ color:#00e676; font-weight:800; }} .signal-sell {{ color:#ff1744; font-weight:800; }} .muted {{ color:#6f95d8; }} .empty-row {{ text-align:center; padding:15px; }}
    .log-box {{ max-height:300px; overflow-y:auto; font-family:Consolas, monospace; font-size:12px; background:#061a3a; padding:10px; border-radius:8px; }}
    .log-line {{ padding:3px 0; border-bottom:1px solid #0b2448; }} .log-time {{ color:#6f95d8; margin-right:6px; }}
    .grade-a {{ color:#00e676; font-weight:800; }} .grade-b,.regime-ranging {{ color:#ffc107; font-weight:800; }} .grade-d,.regime-volatile {{ color:#ff1744; font-weight:800; }} .regime-trending {{ color:#00e676; font-weight:800; }} .regime-quiet {{ color:#7b8aa0; font-weight:800; }}
    .footer {{ text-align:center; padding:10px; color:#6f95d8; font-size:12px; }} .footer a {{ color:#9bbcff; text-decoration:none; }}
    @media (max-width:980px) {{ .grid,.grid-3 {{ grid-template-columns:1fr; }} .header {{ align-items:flex-start; gap:12px; flex-direction:column; }} }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>⚡ {escape(bot_version)}</h1>
      <span class="mode-badge">{escape(str(dashboard_state.get('mode', 'DEFAULT'))).upper()}</span>
      <span class="execution-badge">{escape(execution_mode)}</span>
      <div class="header-info">ForexTradingBot Trading Bot Dashboard · Port: {escape(str(dashboard_state.get('dashboard_port', 5001)))} | Iter #{escape(str(dashboard_state.get('iteration', 0)))} | {escape(str(dashboard_state.get('last_update') or 'waiting for data'))} | Daily: {escape(_money(dashboard_state.get('daily_pnl', 0)))} | Compounding: ON</div>
    </div>
    <span class="status-badge {'status-running' if status == 'RUNNING' else 'status-stopped'}">{escape(status)}</span>
  </div>

  <div class="grid">
    <div class="card">
      <h2>💰 Account</h2>
      <div class="account-grid">
        <div class="account-item"><div class="account-label">Balance</div><div class="account-value neutral">{escape(_money(account.get('balance', 0)))}</div></div>
        <div class="account-item"><div class="account-label">Equity</div><div class="account-value neutral">{escape(_money(account.get('equity', 0)))}</div></div>
        <div class="account-item"><div class="account-label">Floating P/L</div><div class="account-value {signed_class(profit)}">{escape(_money(profit))}</div></div>
        <div class="account-item"><div class="account-label">Growth</div><div class="account-value {signed_class(growth)}" style="font-size:26px">{escape(_pct(growth, scale_unit_interval=False, decimals=1))}</div></div>
      </div>
    </div>
    <div class="card">
      <h2>📊 Performance</h2>
      <div class="stats-grid">
        <div class="stat-item"><div class="stat-label">Total Trades</div><div class="stat-value">{escape(str(_state_get(tracker, 'total_trades', 0)))}</div></div>
        <div class="stat-item"><div class="stat-label">Win Rate</div><div class="stat-value {signed_class(_state_get(tracker, 'win_rate', 0))}">{escape(_pct(_state_get(tracker, 'win_rate', 0), decimals=1))}</div></div>
        <div class="stat-item"><div class="stat-label">Total P/L</div><div class="stat-value {signed_class(total_pnl)}">{escape(_money(total_pnl))}</div></div>
        <div class="stat-item"><div class="stat-label">Profit Factor</div><div class="stat-value">{escape(_format_number(_state_get(tracker, 'profit_factor', 0), 2))}</div></div>
      </div>
    </div>
  </div>

  <div class="grid-3">
    <div class="card"><h2>🎯 Quality Grades</h2><div class="mini-stats">{_mini_stat('A+:', quality.get('A+', 0), 'grade-a')}{_mini_stat('A:', quality.get('A', 0), 'grade-a')}{_mini_stat('B:', quality.get('B', 0), 'grade-b')}{_mini_stat('Blocked:', quality.get('blocked', 0), 'grade-d')}</div></div>
    <div class="card"><h2>🔄 Adaptive</h2><div class="mini-stats">{_mini_stat('Base:', _pct(adaptive.get('base', 0.35)))}{_mini_stat('Now:', _pct(adaptive.get('current', 0.35)), 'neutral')}{_mini_stat('WR:', _pct(adaptive.get('recent_wr', 0)))}</div></div>
    <div class="card"><h2>🔥 M30 & Streak</h2><div class="mini-stats">{_mini_stat('M30✓:', m30.get('confirm', 0), 'pnl-pos')}{_mini_stat('M30✗:', m30.get('against', 0), 'pnl-neg')}{_mini_stat('Streak:', str(streak.get('current_streak', 0)) + (' ⛔' if streak.get('in_cooldown') else ''), 'pnl-neg' if streak.get('in_cooldown') else '')}</div></div>
  </div>

  <div class="card" style="margin-bottom:15px">
    <h2>📈 Open Positions ({len(dashboard_state.get('open_positions') or [])})</h2>
    <table><thead><tr><th>Ticket</th><th>Symbol</th><th>Side</th><th>Lots</th><th>Entry</th><th>Current</th><th>SL</th><th>TP</th><th>P/L</th></tr></thead><tbody>{_render_positions()}</tbody></table>
  </div>

  <div class="grid-3">
    <div class="card"><h2>🧠 Deep RL Agent</h2><div class="mini-stats">{_mini_stat('Arch:', rl.get('architecture', 'Dueling_DQN'), 'neutral')}{_mini_stat('Trades:', rl.get('total_trades', 0))}{_mini_stat('WR:', _pct(rl.get('win_rate', 0)), 'pnl-pos' if _as_float(rl.get('win_rate', 0)) >= .5 else '')}{_mini_stat('Steps:', rl.get('train_steps', 0))}{_mini_stat('Buffer:', rl.get('buffer_size', 0))}{_mini_stat('Loss:', _format_number(rl.get('avg_loss', 0), 4))}{_mini_stat('Q:', _format_number(rl.get('avg_q_value', 0), 3))}{_mini_stat('Reward:', _format_number(rl.get('total_reward', 0), 1), signed_class(rl.get('total_reward', 0)).replace('positive','pnl-pos').replace('negative','pnl-neg'))}</div></div>
    <div class="card"><h2>🌍 Market Regimes</h2><div class="mini-stats">{_render_regimes()}</div></div>
    <div class="card"><h2>🛡️ Risk Guard</h2><div class="mini-stats">{_mini_stat('Daily:', _money(daily_pnl), signed_class(daily_pnl).replace('positive','pnl-pos').replace('negative','pnl-neg'))}{_mini_stat('W:', risk.get('daily_wins', 0) if isinstance(risk, dict) else 0, 'pnl-pos')}{_mini_stat('L:', risk.get('daily_losses', 0) if isinstance(risk, dict) else 0, 'pnl-neg')}{_mini_stat('DD:', _pct(risk.get('drawdown_pct', 0) if isinstance(risk, dict) else 0, scale_unit_interval=False, decimals=1))}</div></div>
  </div>

  <div class="card"><h2>📋 Live Log</h2><div class="log-box">{_render_logs()}</div></div>
  <div class="footer">⚡ {escape(bot_version)} dashboard style | Auto-refresh 5s | <a href="/api/state">API State</a> | <a href="/health">Health</a></div>
  <div class="diagnostics" hidden>
    <table><tbody>{_render_kv_table(dashboard_state.get('signals'), 'No signal data yet.')}</tbody></table>
    <table><tbody>{_render_kv_table(dashboard_state.get('symbols'), 'No symbol data yet.')}</tbody></table>
    <table><tbody>{_render_kv_table(dashboard_state.get('risk_guard'), 'Risk status has not been reported yet.')}</tbody></table>
  </div>
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
