"""
ForexTradingBot V8.0 Web Dashboard — AGGRESSIVE Edition
สี: แดง/ส้ม | High Risk High Reward | Deep RL Support
"""

import threading
import logging
import os
import json
import webbrowser
import time as _time
from datetime import datetime

from flask import Flask, render_template_string

logger = logging.getLogger(__name__)

APP_NAME = 'ForexTradingBot'
DEFAULT_BOT_VERSION = os.getenv('BOT_VERSION', 'V8.0').upper()
DEFAULT_EXECUTION_MODE = 'Dryruns'

dashboard_state = {
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
    'mode': 'AGGRESSIVE',
    'bot_version': f'{APP_NAME} {DEFAULT_BOT_VERSION}',
    'execution_mode': DEFAULT_EXECUTION_MODE,
}


def update_dashboard(key, value):
    dashboard_state[key] = value
    dashboard_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def add_log(msg):
    dashboard_state['log_messages'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'msg': msg
    })
    if len(dashboard_state['log_messages']) > 200:
        dashboard_state['log_messages'] = dashboard_state['log_messages'][-200:]


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚡ {{ state.bot_version }} — Port {{ port }}</title>
<meta http-equiv="refresh" content="5">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #061a3a;
    color: #e1e5ea;
    font-family: 'Segoe UI', Consolas, monospace;
    font-size: 14px;
    padding: 15px;
}
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 20px;
    background: linear-gradient(135deg, #0b2a5b, #0f3d7a);
    border-radius: 10px;
    margin-bottom: 15px;
    border: 1px solid #1e63b6;
}
.header h1 {
    font-size: 22px;
    background: linear-gradient(90deg, #ff4444, #ff8800);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mode-badge, .execution-badge {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: bold;
    color: #fff;
    margin-left: 10px;
    animation: pulse 2s infinite;
}
.mode-badge { background: #d32f2f; }
.execution-badge { background: #00695c; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
.header-info { color: #9bbcff; font-size: 12px; }
.status-badge {
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 13px;
}
.status-running { background: #ff6d00; color: #000; }
.status-stopped { background: #ff1744; color: #fff; }

.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 15px; margin-bottom: 15px; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 15px; }

.card {
    background: #0b2448;
    border-radius: 10px;
    padding: 18px;
    border: 1px solid #1b4f93;
}
.card h2 {
    font-size: 14px;
    color: #ff8800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    border-bottom: 1px solid #1b4f93;
    padding-bottom: 8px;
}

.account-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.account-item { text-align: center; padding: 10px; background: #102f5f; border-radius: 8px; }
.account-label { font-size: 11px; color: #9bbcff; text-transform: uppercase; }
.account-value { font-size: 22px; font-weight: bold; margin-top: 4px; }
.positive { color: #00e676; }
.negative { color: #ff1744; }
.neutral { color: #ff8800; }

.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-item { padding: 8px; background: #102f5f; border-radius: 6px; }
.stat-label { font-size: 11px; color: #9bbcff; }
.stat-value { font-size: 16px; font-weight: bold; margin-top: 2px; }

.mini-stats { display: flex; gap: 10px; flex-wrap: wrap; }
.mini-stat { padding: 6px 12px; background: #102f5f; border-radius: 6px; font-size: 12px; }
.mini-stat .label { color: #9bbcff; }
.mini-stat .value { font-weight: bold; margin-left: 4px; }

table { width: 100%; border-collapse: collapse; }
th {
    text-align: left; padding: 8px 10px; color: #ff8800;
    font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid #1b4f93;
}
td { padding: 8px 10px; border-bottom: 1px solid #0b2448; font-size: 13px; }
tr:hover { background: #102f5f; }

.signal-buy { color: #00e676; font-weight: bold; }
.signal-sell { color: #ff1744; font-weight: bold; }
.pnl-pos { color: #00e676; }
.pnl-neg { color: #ff1744; }

.log-box {
    max-height: 300px; overflow-y: auto;
    font-family: Consolas, monospace; font-size: 12px;
    background: #061a3a; padding: 10px; border-radius: 8px;
}
.log-line { padding: 2px 0; border-bottom: 1px solid #0b2448; }
.log-time { color: #6f95d8; }

.grade-a { color: #00e676; font-weight: bold; }
.grade-b { color: #ffc107; font-weight: bold; }
.grade-c { color: #ff9800; font-weight: bold; }
.grade-d { color: #ff1744; font-weight: bold; }

.regime-trending { color: #00e676; font-weight: bold; }
.regime-ranging { color: #ffc107; font-weight: bold; }
.regime-volatile { color: #ff1744; font-weight: bold; }
.regime-quiet { color: #7b8aa0; }

.footer { text-align: center; padding: 10px; color: #6f95d8; font-size: 12px; }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div>
        <h1>⚡ {{ state.bot_version }}</h1>
        <span class="mode-badge">{{ state.mode }}</span>
        <span class="execution-badge">{{ state.execution_mode }}</span>
        <div class="header-info">
            Port: {{ port }} | Iter #{{ state.iteration }} | {{ state.last_update }}
            | Daily: ${{ "%.2f"|format(state.daily_pnl) }}
            | Compounding: ON
        </div>
    </div>
    <span class="status-badge {{ 'status-running' if state.bot_status == 'RUNNING' else 'status-stopped' }}">
        {{ state.bot_status }}
    </span>
</div>

<!-- ACCOUNT + PERFORMANCE -->
<div class="grid">
    <div class="card">
        <h2>💰 Account</h2>
        <div class="account-grid">
            <div class="account-item">
                <div class="account-label">Balance</div>
                <div class="account-value neutral">${{ "%.2f"|format(state.account.balance) }}</div>
            </div>
            <div class="account-item">
                <div class="account-label">Equity</div>
                <div class="account-value neutral">${{ "%.2f"|format(state.account.equity) }}</div>
            </div>
            <div class="account-item">
                <div class="account-label">Floating P/L</div>
                <div class="account-value {{ 'positive' if state.account.profit >= 0 else 'negative' }}">
                    ${{ "%.2f"|format(state.account.profit) }}
                </div>
            </div>
            <div class="account-item">
                <div class="account-label">Growth</div>
                <div class="account-value {{ 'positive' if state.account.get('growth_pct', 0) >= 0 else 'negative' }}" style="font-size:26px">
                    {{ "%.1f"|format(state.account.get('growth_pct', 0)) }}%
                </div>
            </div>
        </div>
    </div>

    {% if state.tracker_stats %}
    <div class="card">
        <h2>📊 Performance</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{{ state.tracker_stats.total_trades }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value {{ 'positive' if state.tracker_stats.win_rate >= 0.5 else 'negative' }}">
                    {{ "%.1f"|format(state.tracker_stats.win_rate * 100) }}%
                </div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total P/L</div>
                <div class="stat-value {{ 'pnl-pos' if state.tracker_stats.total_pnl >= 0 else 'pnl-neg' }}" style="font-size:20px">
                    ${{ "%.2f"|format(state.tracker_stats.total_pnl) }}
                </div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Profit Factor</div>
                <div class="stat-value">{{ "%.2f"|format(state.tracker_stats.profit_factor) }}</div>
            </div>
        </div>
    </div>
    {% else %}
    <div class="card">
        <h2>📊 Performance</h2>
        <p style="color:#6f95d8;text-align:center;padding:20px">Waiting for trades...</p>
    </div>
    {% endif %}
</div>

<!-- SMART FILTERS -->
<div class="grid-3">
    <div class="card">
        <h2>🎯 Quality Grades</h2>
        <div class="mini-stats">
            <div class="mini-stat"><span class="label">A+:</span><span class="value grade-a">{{ state.quality_stats.get('A+', 0) }}</span></div>
            <div class="mini-stat"><span class="label">A:</span><span class="value grade-a">{{ state.quality_stats.get('A', 0) }}</span></div>
            <div class="mini-stat"><span class="label">B:</span><span class="value grade-b">{{ state.quality_stats.get('B', 0) }}</span></div>
            <div class="mini-stat"><span class="label">Blocked:</span><span class="value grade-d">{{ state.quality_stats.get('blocked', 0) }}</span></div>
        </div>
    </div>
    <div class="card">
        <h2>🔄 Adaptive</h2>
        <div class="mini-stats">
            <div class="mini-stat"><span class="label">Base:</span><span class="value">{{ "%.0f"|format(state.adaptive.get('base', 0.35) * 100) }}%</span></div>
            <div class="mini-stat"><span class="label">Now:</span><span class="value neutral">{{ "%.0f"|format(state.adaptive.get('current', 0.35) * 100) }}%</span></div>
            <div class="mini-stat"><span class="label">WR:</span><span class="value">{{ "%.0f"|format(state.adaptive.get('recent_wr', 0) * 100) }}%</span></div>
        </div>
    </div>
    <div class="card">
        <h2>🔥 M30 & Streak</h2>
        <div class="mini-stats">
            <div class="mini-stat"><span class="label">M30✓:</span><span class="value pnl-pos">{{ state.m30_stats.get('confirm', 0) }}</span></div>
            <div class="mini-stat"><span class="label">M30✗:</span><span class="value pnl-neg">{{ state.m30_stats.get('against', 0) }}</span></div>
            <div class="mini-stat"><span class="label">Streak:</span><span class="value {{ 'pnl-neg' if state.streak.get('in_cooldown', False) else '' }}">
                {{ state.streak.get('current_streak', 0) }}{{ ' ⛔' if state.streak.get('in_cooldown', False) else '' }}
            </span></div>
        </div>
    </div>
</div>

<!-- OPEN POSITIONS -->
<div class="card" style="margin-bottom:15px">
    <h2>📈 Open Positions ({{ state.open_positions|length }})</h2>
    {% if state.open_positions %}
    <table>
        <thead><tr><th>Ticket</th><th>Symbol</th><th>Side</th><th>Lots</th><th>Entry</th><th>Current</th><th>SL</th><th>TP</th><th>P/L</th></tr></thead>
        <tbody>
        {% for p in state.open_positions %}
        <tr>
            <td>#{{ p.ticket }}</td>
            <td><strong>{{ p.symbol }}</strong></td>
            <td class="signal-{{ 'buy' if p.side == 'BUY' else 'sell' }}">{{ p.side }}</td>
            <td>{{ p.volume }}</td>
            <td>{{ "%.5f"|format(p.entry) }}</td>
            <td>{{ "%.5f"|format(p.current_price) }}</td>
            <td>{{ "%.5f"|format(p.sl) }}</td>
            <td>{{ "%.5f"|format(p.tp) }}</td>
            <td class="{{ 'pnl-pos' if p.pnl >= 0 else 'pnl-neg' }}">${{ "%.2f"|format(p.pnl) }}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="color:#6f95d8;text-align:center;padding:15px">No open positions</p>
    {% endif %}
</div>

<!-- DEEP RL + REGIME + RISK -->
<div class="grid-3">
    <div class="card">
        <h2>🧠 Deep RL Agent</h2>
        <div class="mini-stats">
            <div class="mini-stat"><span class="label">Arch:</span><span class="value neutral">{{ state.rl_stats.get('architecture', 'Dueling_DQN') }}</span></div>
            <div class="mini-stat"><span class="label">Trades:</span><span class="value">{{ state.rl_stats.get('total_trades', 0) }}</span></div>
            <div class="mini-stat"><span class="label">WR:</span><span class="value {{ 'pnl-pos' if state.rl_stats.get('win_rate', 0) >= 0.5 else '' }}">{{ "%.0f"|format(state.rl_stats.get('win_rate', 0) * 100) }}%</span></div>
            <div class="mini-stat"><span class="label">Steps:</span><span class="value">{{ state.rl_stats.get('train_steps', 0) }}</span></div>
            <div class="mini-stat"><span class="label">Buffer:</span><span class="value">{{ state.rl_stats.get('buffer_size', 0) }}</span></div>
            <div class="mini-stat"><span class="label">Loss:</span><span class="value">{{ "%.4f"|format(state.rl_stats.get('avg_loss', 0)) }}</span></div>
            <div class="mini-stat"><span class="label">Q:</span><span class="value">{% if state.rl_stats.get('avg_q_value') is none %}N/A{% else %}{{ "%.3f"|format(state.rl_stats.get('avg_q_value')) }}{% endif %}</span></div>
            <div class="mini-stat"><span class="label">RL Reward:</span><span class="value {{ 'pnl-pos' if state.rl_stats.get('total_reward', 0) >= 0 else 'pnl-neg' }}">{{ "%.1f"|format(state.rl_stats.get('total_reward', 0)) }}</span></div>
        </div>
    </div>
    <div class="card">
        <h2>🌍 Market Regimes</h2>
        <div class="mini-stats">
            {% for sym, regime in state.regime_stats.items() %}
            <div class="mini-stat">
                <span class="label">{{ sym[:6] }}:</span>
                <span class="value regime-{{ regime|lower }}">{{ regime }}</span>
            </div>
            {% endfor %}
            {% if not state.regime_stats %}
            <div class="mini-stat"><span class="label">Waiting...</span></div>
            {% endif %}
        </div>
    </div>
    <div class="card">
        <h2>🛡️ Risk Guard</h2>
        <div class="mini-stats">
            <div class="mini-stat"><span class="label">Daily:</span><span class="value {{ 'pnl-pos' if state.risk_guard.get('daily_pnl', 0) >= 0 else 'pnl-neg' }}">${{ "%.2f"|format(state.risk_guard.get('daily_pnl', 0)) }}</span></div>
            <div class="mini-stat"><span class="label">W:</span><span class="value pnl-pos">{{ state.risk_guard.get('daily_wins', 0) }}</span></div>
            <div class="mini-stat"><span class="label">L:</span><span class="value pnl-neg">{{ state.risk_guard.get('daily_losses', 0) }}</span></div>
            <div class="mini-stat"><span class="label">DD:</span><span class="value">{{ "%.1f"|format(state.risk_guard.get('drawdown_pct', 0)) }}%</span></div>
            {% if state.risk_guard.get('recovery_mode', False) %}
            <div class="mini-stat" style="background:#ff174433"><span class="value pnl-neg">⚠️ RECOVERY</span></div>
            {% endif %}
        </div>
    </div>
</div>

<!-- LOG -->
<div class="card">
    <h2>📋 Live Log</h2>
    <div class="log-box">
    {% for log in state.log_messages[-50:]|reverse %}
        <div class="log-line"><span class="log-time">{{ log.time }}</span> {{ log.msg }}</div>
    {% endfor %}
    </div>
</div>

<div class="footer">⚡ {{ state.bot_version }} Aggressive + Deep RL | Compounding ON | Auto-refresh 5s | Port {{ port }}</div>
</body>
</html>
"""

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aggressive-bot'

flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.WARNING)


@app.route('/')
def index():
    class DotDict(dict):
        __getattr__ = dict.get

    def to_dot(d):
        if isinstance(d, dict):
            return DotDict({k: to_dot(v) for k, v in d.items()})
        if isinstance(d, list):
            return [to_dot(i) for i in d]
        return d

    state = to_dot(dashboard_state)
    port = dashboard_state.get('dashboard_port', 5001)
    return render_template_string(HTML_TEMPLATE, state=state, port=port)


@app.route('/api/state')
def api_state():
    return json.dumps(dashboard_state, default=str)


def start_dashboard(host='0.0.0.0', port=5001):
    def run():
        try:
            app.run(host=host, port=port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"[DASHBOARD] Error: {e}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    _time.sleep(1.5)
    url = f"http://localhost:{port}"
    try:
        webbrowser.open(url)
        logger.info(f"[DASHBOARD] ⚡ Aggressive: {url}")
    except Exception as e:
        logger.warning(f"[DASHBOARD] Open manually: {url}")

    return thread