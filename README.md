# Forextradingbot

Multi-symbol MetaTrader 5 trading bot with ICT-style feature engineering, ML/RL model hooks, risk controls, order execution, monitoring dashboards, and backtest artifacts.

> **Risk warning:** This repository can place live trades through MetaTrader 5. Use a demo account first, verify every configuration value, and treat all bundled model artifacts as trusted local artifacts only.

## Current entry points

- `python main.py` — primary live bot entry point.
- `python run_live.py` — legacy entry point retained for compatibility; prefer `main.py` for the current ML stack.

## Prerequisites

1. Python 3.11.x.
2. MetaTrader 5 terminal installed and logged in on the machine that will run live trading.
3. Broker-specific symbol names matching `config/settings.py` (for example `EURUSDm`, `XAUUSDm`).
4. The Python dependencies in `requirements.txt`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your MT5 credentials and notification settings. Do not commit `.env`.

## Important environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `BOT_MODE` | Optional | `AGGRESSIVE` uses the aggressive Flask dashboard. Any other value uses the lightweight default dashboard. |
| `MT5_LOGIN` | Live trading | MetaTrader 5 account number. |
| `MT5_PASSWORD` | Live trading | MetaTrader 5 account password. |
| `MT5_SERVER` | Live trading | Broker server name exactly as shown in MT5. |
| `MT5_PATH` | Optional | Explicit terminal path if auto-discovery is not enough. |
| `TELEGRAM_TOKEN` | Optional | Telegram bot token for alerts. |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat ID for alerts. |
| `DASHBOARD_PORT` | Optional | Dashboard port; default is `5001`. |
| `DRY_RUN` | Recommended for smoke tests | Defaults to `true`. Set `false` only when intentionally enabling live order routing. |
| `LIVE_TRADING_CONFIRMED` | Live trading | Must be `true` together with `DRY_RUN=false`; otherwise the bot stays in dry-run mode as a fail-safe. |
| `ORDER_MAGIC` | Optional | MT5 magic number used on bot-managed orders; default `123456`. |
| `ORDER_DEVIATION` | Optional | Maximum order price deviation in points; default `20`. |
| `MAX_LOT_SIZE` | Optional | Global hard cap for calculated lot size; default `2.0`. |
| `MAX_SPREAD_POINTS` | Code config | Per-symbol absolute spread caps in `config/settings.py`; trades are blocked above the cap. |
| `MAX_SLIPPAGE_POINTS` | Code config | Per-symbol max execution slippage caps in `config/settings.py`; order send is skipped above the cap. |

## Running

```bash
python main.py
```

For non-aggressive modes, `monitoring.web_dashboard` is intentionally dependency-light and starts a minimal standard-library HTTP dashboard. For `BOT_MODE=AGGRESSIVE`, install the full monitoring dependencies from `requirements.txt` because `monitoring.web_dashboard_aggressive` uses Flask.

## Validation and smoke checks

Run these checks before using a live account:

```bash
python -m compileall -q .
python -m pytest -q
python - <<'PY'
import importlib
for name in ['MetaTrader5', 'pandas', 'numpy', 'tensorflow', 'torch', 'flask', 'dash']:
    try:
        importlib.import_module(name)
        print(f'{name}: OK')
    except Exception as exc:
        print(f'{name}: MISSING ({exc})')
PY
```

`pytest` currently focuses on dependency-light smoke coverage for the default dashboard and position sizing. Add MT5-mocked tests for order execution, risk guard decisions, signal generation, and dashboard integration before relying on automated coverage for production behavior.

## Backtesting

Backtest utilities live in `backtest/`, and historical report artifacts are stored in `backtest_results/`. Use them to validate symbol-specific assumptions before enabling live execution.

## Operational hardening notes

- Many runtime paths intentionally fall back on exceptions to keep the bot alive. For critical live incidents, prefer logging full error context and tracebacks when tightening those paths.
- Model files loaded through pickle/joblib/PyTorch should be treated as trusted artifacts only.
- Review risk settings in `config/settings.py`, especially dry-run/live confirmation, equity-based daily loss limits, max open trades, spread/slippage filters, session filters, max lot size, and per-symbol settings.
- Start with demo trading plus `DRY_RUN=true`, then validate fills, spreads, SL/TP placement, and Telegram/dashboard telemetry manually.
