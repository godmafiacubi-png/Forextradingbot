import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed — no charts")


class BacktestReport:
    """Generate visual reports from backtest results"""

    def __init__(self, output_dir='./backtest_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, results, name='backtest'):
        """Generate full report: console + charts + HTML"""
        self.print_console(results, name)
        if HAS_MATPLOTLIB:
            self.plot_equity_curve(results, name)
            self.plot_trade_distribution(results, name)
            self.plot_monthly_pnl(results, name)
            self.plot_drawdown(results, name)
        self.save_html(results, name)
        self.save_json(results, name)
        logger.info(f"\nReport saved to: {self.output_dir}/{name}_report.html")

    def print_console(self, r, name=''):
        """Print results to console"""
        print("\n" + "=" * 70)
        print(f"  BACKTEST RESULTS: {name}")
        print("=" * 70)
        print(f"  Initial Balance:     ${r['initial_balance']:.2f}")
        print(f"  Final Balance:       ${r['final_balance']:.2f}")
        print(f"  Total Return:        {r['total_return']:.2f}%")
        print(f"  Total P/L:           ${r['total_pnl']:.2f}")
        print("-" * 70)
        print(f"  Total Trades:        {r['total_trades']}")
        print(f"  Wins / Losses:       {r['win_count']} / {r['loss_count']}")
        print(f"  Win Rate:            {r['win_rate']:.2%}")
        print(f"  Avg Win:             ${r['avg_win']:.2f}")
        print(f"  Avg Loss:            ${r['avg_loss']:.2f}")
        print(f"  Best Trade:          ${r['best_trade']:.2f}")
        print(f"  Worst Trade:         ${r['worst_trade']:.2f}")
        print("-" * 70)
        print(f"  Profit Factor:       {r['profit_factor']:.2f}")
        print(f"  Expectancy:          ${r['expectancy']:.2f}")
        print(f"  R:R Realized:        1:{r['rr_realized']:.2f}")
        print(f"  Sharpe Ratio:        {r['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:       {r['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:        {r['calmar_ratio']:.2f}")
        print("-" * 70)
        print(f"  Max Drawdown:        ${r['max_drawdown']:.2f} ({r['max_drawdown_pct']:.2f}%)")
        print(f"  Max Consec Wins:     {r['max_consec_win']}")
        print(f"  Max Consec Losses:   {r['max_consec_loss']}")
        print(f"  Avg Trade Duration:  {r['avg_duration_bars']:.1f} bars")
        print("-" * 70)
        print(f"  BUY P/L:             ${r['buy_pnl']:.2f}  (WR: {r['buy_win_rate']:.2%})")
        print(f"  SELL P/L:            ${r['sell_pnl']:.2f}  (WR: {r['sell_win_rate']:.2%})")
        print("-" * 70)
        print(f"  Exit Reasons:")
        for reason, count in r.get('exit_reasons', {}).items():
            pct = count / r['total_trades'] * 100
            print(f"    {reason:12} {count:5} ({pct:.1f}%)")
        print("-" * 70)
        if r.get('monthly'):
            print(f"  Monthly Breakdown:")
            for m in r['monthly']:
                print(f"    {m['month']}  PnL: ${m['pnl']:.2f}  Trades: {m['trades']}  WR: {m['win_rate']:.1%}")
        print("=" * 70)

    def plot_equity_curve(self, results, name):
        try:
            eq = results.get('equity_curve', [])
            if not eq:
                return

            times = list(range(len(eq)))
            balances = [e['balance'] for e in eq]
            equities = [e['equity'] for e in eq]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(times, balances, label='Balance', color='#00d4ff', linewidth=1.5)
            ax.plot(times, equities, label='Equity', color='#7b68ee', linewidth=1, alpha=0.7)
            ax.axhline(y=results['initial_balance'], color='gray', linestyle='--', alpha=0.5, label='Initial')

            ax.fill_between(times, equities, balances, alpha=0.1, color='#7b68ee')

            ax.set_title(f'Equity Curve — {name}', fontsize=14, color='white')
            ax.set_xlabel('Bar')
            ax.set_ylabel('USD')
            ax.legend()
            ax.grid(True, alpha=0.2)

            fig.patch.set_facecolor('#0a0e17')
            ax.set_facecolor('#0a0e17')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#333')

            path = os.path.join(self.output_dir, f'{name}_equity.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0e17')
            plt.close()
            logger.info(f"Saved: {path}")
        except Exception as e:
            logger.error(f"Plot equity error: {e}")

    def plot_trade_distribution(self, results, name):
        try:
            trades = results.get('trades', [])
            if not trades:
                return

            pnls = [t['pnl'] for t in trades]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # PnL histogram
            colors = ['#00e676' if p > 0 else '#ff1744' for p in pnls]
            axes[0].bar(range(len(pnls)), pnls, color=colors, width=1)
            axes[0].axhline(y=0, color='white', linewidth=0.5)
            axes[0].set_title('Trade P/L', color='white')
            axes[0].set_xlabel('Trade #')
            axes[0].set_ylabel('USD')

            # Distribution
            axes[1].hist(pnls, bins=30, color='#7b68ee', alpha=0.7, edgecolor='#333')
            axes[1].axvline(x=0, color='white', linewidth=0.5)
            axes[1].axvline(x=np.mean(pnls), color='#00d4ff', linewidth=1, linestyle='--', label=f'Mean: ${np.mean(pnls):.2f}')
            axes[1].set_title('P/L Distribution', color='white')
            axes[1].set_xlabel('USD')
            axes[1].legend()

            for ax in axes:
                ax.set_facecolor('#0a0e17')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.2)
                for spine in ax.spines.values():
                    spine.set_color('#333')

            fig.patch.set_facecolor('#0a0e17')
            path = os.path.join(self.output_dir, f'{name}_trades.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0e17')
            plt.close()
        except Exception as e:
            logger.error(f"Plot trades error: {e}")

    def plot_monthly_pnl(self, results, name):
        try:
            monthly = results.get('monthly', [])
            if not monthly:
                return

            fig, ax = plt.subplots(figsize=(12, 5))
            months = [str(m['month']) for m in monthly]
            pnls = [m['pnl'] for m in monthly]
            colors = ['#00e676' if p > 0 else '#ff1744' for p in pnls]

            ax.bar(months, pnls, color=colors)
            ax.axhline(y=0, color='white', linewidth=0.5)
            ax.set_title(f'Monthly P/L — {name}', color='white')
            ax.set_ylabel('USD')
            plt.xticks(rotation=45)

            fig.patch.set_facecolor('#0a0e17')
            ax.set_facecolor('#0a0e17')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2)

            path = os.path.join(self.output_dir, f'{name}_monthly.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0e17')
            plt.close()
        except Exception as e:
            logger.error(f"Plot monthly error: {e}")

    def plot_drawdown(self, results, name):
        try:
            eq = results.get('equity_curve', [])
            if not eq:
                return

            equities = [e['equity'] for e in eq]
            peak = equities[0]
            dd = []
            for e in equities:
                if e > peak:
                    peak = e
                dd.append((peak - e) / peak * 100)

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.fill_between(range(len(dd)), dd, color='#ff1744', alpha=0.4)
            ax.plot(range(len(dd)), dd, color='#ff1744', linewidth=0.8)
            ax.set_title(f'Drawdown % — {name}', color='white')
            ax.set_ylabel('%')
            ax.invert_yaxis()

            fig.patch.set_facecolor('#0a0e17')
            ax.set_facecolor('#0a0e17')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2)

            path = os.path.join(self.output_dir, f'{name}_drawdown.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0e17')
            plt.close()
        except Exception as e:
            logger.error(f"Plot drawdown error: {e}")

    def save_html(self, r, name):
        try:
            has_charts = HAS_MATPLOTLIB

            html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Backtest: {name}</title>
<style>
body {{ background:#0a0e17; color:#e1e5ea; font-family:'Segoe UI',monospace; padding:20px; }}
h1 {{ color:#00d4ff; }} h2 {{ color:#7b8aa0; border-bottom:1px solid #1e2636; padding-bottom:8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:15px 0; }}
.card {{ background:#141824; border-radius:8px; padding:15px; border:1px solid #1e2636; text-align:center; }}
.label {{ color:#7b8aa0; font-size:12px; text-transform:uppercase; }}
.value {{ font-size:20px; font-weight:bold; margin-top:4px; }}
.pos {{ color:#00e676; }} .neg {{ color:#ff1744; }} .neu {{ color:#00d4ff; }}
table {{ width:100%; border-collapse:collapse; margin:10px 0; }}
th {{ text-align:left; padding:8px; color:#7b8aa0; font-size:11px; border-bottom:1px solid #1e2636; }}
td {{ padding:8px; border-bottom:1px solid #111622; }}
img {{ max-width:100%; border-radius:8px; margin:10px 0; }}
</style></head><body>
<h1>Backtest Report: {name}</h1>

<div class="grid">
<div class="card"><div class="label">Initial</div><div class="value neu">${r['initial_balance']:.2f}</div></div>
<div class="card"><div class="label">Final</div><div class="value {'pos' if r['total_return']>=0 else 'neg'}">${r['final_balance']:.2f}</div></div>
<div class="card"><div class="label">Return</div><div class="value {'pos' if r['total_return']>=0 else 'neg'}">{r['total_return']:.2f}%</div></div>
<div class="card"><div class="label">Total P/L</div><div class="value {'pos' if r['total_pnl']>=0 else 'neg'}">${r['total_pnl']:.2f}</div></div>
</div>

<div class="grid">
<div class="card"><div class="label">Trades</div><div class="value">{r['total_trades']}</div></div>
<div class="card"><div class="label">Win Rate</div><div class="value {'pos' if r['win_rate']>=0.5 else 'neg'}">{r['win_rate']:.1%}</div></div>
<div class="card"><div class="label">Profit Factor</div><div class="value {'pos' if r['profit_factor']>=1.5 else 'neg'}">{r['profit_factor']:.2f}</div></div>
<div class="card"><div class="label">Sharpe</div><div class="value">{r['sharpe_ratio']:.2f}</div></div>
<div class="card"><div class="label">Max DD</div><div class="value neg">{r['max_drawdown_pct']:.2f}%</div></div>
<div class="card"><div class="label">R:R Realized</div><div class="value">1:{r['rr_realized']:.2f}</div></div>
<div class="card"><div class="label">Expectancy</div><div class="value {'pos' if r['expectancy']>=0 else 'neg'}">${r['expectancy']:.2f}</div></div>
<div class="card"><div class="label">Avg Duration</div><div class="value">{r['avg_duration_bars']:.0f} bars</div></div>
</div>

{'<h2>Equity Curve</h2><img src="'+name+'_equity.png">' if has_charts else ''}
{'<h2>Drawdown</h2><img src="'+name+'_drawdown.png">' if has_charts else ''}
{'<h2>Trade Distribution</h2><img src="'+name+'_trades.png">' if has_charts else ''}
{'<h2>Monthly P/L</h2><img src="'+name+'_monthly.png">' if has_charts else ''}

<h2>Exit Reasons</h2>
<table><tr><th>Reason</th><th>Count</th><th>%</th></tr>"""

            for reason, count in r.get('exit_reasons', {}).items():
                pct = count / r['total_trades'] * 100
                html += f"<tr><td>{reason}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"

            html += "</table>"

            if r.get('monthly'):
                html += "<h2>Monthly</h2><table><tr><th>Month</th><th>P/L</th><th>Trades</th><th>Win Rate</th></tr>"
                for m in r['monthly']:
                    cls = 'pos' if m['pnl'] >= 0 else 'neg'
                    html += f"<tr><td>{m['month']}</td><td class='{cls}'>${m['pnl']:.2f}</td><td>{m['trades']}</td><td>{m['win_rate']:.1%}</td></tr>"
                html += "</table>"

            html += "</body></html>"

            path = os.path.join(self.output_dir, f'{name}_report.html')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            logger.error(f"HTML report error: {e}")

    def save_json(self, results, name):
        try:
            # Remove equity_curve and trades for smaller JSON
            summary = {k: v for k, v in results.items() if k not in ['equity_curve', 'trades']}
            path = os.path.join(self.output_dir, f'{name}_summary.json')
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"JSON save error: {e}")