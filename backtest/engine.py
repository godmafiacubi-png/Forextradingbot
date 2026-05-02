import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

CONTRACT_SPECS = {
    'EURUSDm': (0.0001, 10.0, 100000),
    'GBPUSDm': (0.0001, 10.0, 100000),
    'USDJPYm': (0.01, 6.67, 100000),
    'BTCUSDm': (1.0, 1.0, 1),
    'XAUUSDm': (0.01, 1.0, 100),
}


def get_contract_spec(symbol):
    if symbol in CONTRACT_SPECS:
        return CONTRACT_SPECS[symbol]
    return (0.0001, 10.0, 100000)


class BacktestTrade:
    def __init__(self, ticket, symbol, side, entry_price, volume, sl, tp,
                 entry_time, entry_bar, confidence=0, atr=0):
        self.ticket = ticket
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time
        self.entry_bar = entry_bar
        self.confidence = confidence
        self.atr = atr
        self.exit_price = None
        self.exit_time = None
        self.exit_bar = None
        self.exit_reason = ''
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        self.duration_bars = 0
        self.partial_pnl = 0.0
        self.pip_size, self.pip_value_per_lot, self.contract_size = get_contract_spec(symbol)

    def close(self, exit_price, exit_time, exit_bar, reason=''):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_bar = exit_bar
        self.exit_reason = reason
        self.duration_bars = exit_bar - self.entry_bar
        if self.side == 'BUY':
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price
        pips = price_diff / self.pip_size
        self.pnl = pips * self.pip_value_per_lot * self.volume + self.partial_pnl
        self.pnl_pct = self.pnl / (self.entry_price * self.volume * self.contract_size + 1e-10) * 100

    def calc_floating_pnl(self, current_price):
        if self.side == 'BUY':
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price
        pips = price_diff / self.pip_size
        return pips * self.pip_value_per_lot * self.volume

    def to_dict(self):
        return {
            'ticket': self.ticket, 'symbol': self.symbol, 'side': self.side,
            'entry_price': self.entry_price, 'exit_price': self.exit_price,
            'volume': self.volume, 'sl': self.sl, 'tp': self.tp,
            'entry_time': str(self.entry_time), 'exit_time': str(self.exit_time),
            'exit_reason': self.exit_reason, 'pnl': round(self.pnl, 2),
            'pnl_pct': round(self.pnl_pct, 4), 'confidence': self.confidence,
            'max_favorable': round(self.max_favorable, 5),
            'max_adverse': round(self.max_adverse, 5),
            'duration_bars': self.duration_bars,
        }


class BacktestEngine:
    def __init__(self, config):
        self.cfg = config
        self.initial_balance = config.get('initial_balance', 10000)
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_balance = self.initial_balance

        self.spread_pips = config.get('spread_pips', 1.5)
        self.commission_per_lot = config.get('commission_per_lot', 3.5)
        self.slippage_pips = config.get('slippage_pips', 0.5)

        self.risk_pct = config.get('risk_pct', 2.0)
        self.sl_atr_mult = config.get('sl_atr_mult', 1.5)
        self.tp_atr_mult = config.get('tp_atr_mult', 2.5)
        self.max_trades = config.get('max_trades', 5)
        self.max_per_symbol = config.get('max_per_symbol', 2)

        self.breakeven_atr = config.get('breakeven_atr', 0.8)
        self.trail_stages = [(2.0, 0.5), (1.5, 0.8)]

        self.partial_enabled = config.get('partial_close', True)
        self.partial_stages = [
            (config.get('partial_1_atr', 0.6), config.get('partial_1_pct', 0.40)),
            (config.get('partial_2_atr', 1.2), config.get('partial_2_pct', 0.40)),
        ]

        self.min_confidence = config.get('min_confidence', 0.35)
        self.min_adx = config.get('min_adx', 20)
        self.require_htf = config.get('require_htf', True)
        self.require_pullback = config.get('require_pullback', True)
        self.pullback_rsi_buy_max = config.get('pullback_rsi_buy_max', 58)
        self.pullback_rsi_sell_min = config.get('pullback_rsi_sell_min', 42)
        self.ml_buy_threshold = config.get('ml_buy_threshold', 0.57)
        self.ml_sell_threshold = config.get('ml_sell_threshold', 0.43)
        self.min_ict_score = config.get('min_ict_score', 2)
        self.entry_cooldown = config.get('entry_cooldown', 2)
        self.last_entry_bar = {}

        self.session_filter = config.get('session_filter', False)
        self.symbol_best_sessions = config.get('symbol_best_sessions', {})

        self.use_m30 = config.get('use_m30', False)
        self.m30_confirmation = config.get('m30_confirmation', 'signal')
        self.m30_conf_boost = config.get('m30_conf_boost', 0.12)
        self.m30_conf_penalty = config.get('m30_conf_penalty', 0.10)

        # Compounding
        self.compounding = config.get('compounding', True)

        # Confidence scaling
        self.conf_scaling = config.get('conf_scaling', True)
        self.conf_scale_min = config.get('conf_scale_min', 0.5)
        self.conf_scale_max = config.get('conf_scale_max', 1.5)

        # Max lot cap per symbol
        self.max_lot = config.get('max_lot', 5.0)

        # Daily loss limit
        self.daily_loss_limit_pct = config.get('daily_loss_limit_pct', 6.0)
        self.daily_pnl = 0.0
        self.current_day = None

        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = []
        self.trade_counter = 0
        self.partial_done = {}
        self.trades_blocked = defaultdict(int)

    def _get_sizing_base(self):
        if self.compounding:
            return max(self.equity, self.initial_balance * 0.3)
        return self.initial_balance

    def _get_conf_multiplier(self, confidence):
        if not self.conf_scaling:
            return 1.0
        t = np.clip((confidence - 0.3) / 0.5, 0, 1)
        return self.conf_scale_min + t * (self.conf_scale_max - self.conf_scale_min)

    def calculate_lot_size(self, symbol, entry_price, sl_price, confidence=0.5):
        pip_size, pip_value_per_lot, _ = get_contract_spec(symbol)

        base = self._get_sizing_base()
        risk_amount = base * (self.risk_pct / 100)

        # Confidence scaling
        conf_mult = self._get_conf_multiplier(confidence)
        risk_amount *= conf_mult

        sl_pips = abs(entry_price - sl_price) / pip_size
        if sl_pips <= 0 or pip_value_per_lot <= 0:
            return 0.01

        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        # Safety: max risk per trade
        max_risk = base * (self.risk_pct * 3 / 100)
        potential_loss = sl_pips * pip_value_per_lot * lot_size
        if potential_loss > max_risk:
            lot_size = max_risk / (sl_pips * pip_value_per_lot + 1e-10)

        # ⚡ Max lot cap — ป้องกัน compounding runaway
        lot_size = max(min(round(lot_size, 2), self.max_lot), 0.01)

        return lot_size

    def _check_daily_limit(self, bar_time):
        try:
            day = bar_time.date() if hasattr(bar_time, 'date') else pd.Timestamp(bar_time).date()
        except Exception:
            return False
        if self.current_day != day:
            self.daily_pnl = 0.0
            self.current_day = day
        limit = self._get_sizing_base() * (self.daily_loss_limit_pct / 100)
        if self.daily_pnl < -limit:
            return True
        return False

    def _get_bar_sessions(self, bar_time):
        try:
            h = bar_time.hour if hasattr(bar_time, 'hour') else pd.Timestamp(bar_time).hour
        except Exception:
            return ['UNKNOWN']
        sessions = []
        if 0 <= h < 8: sessions.append('ASIAN')
        if 7 <= h < 16: sessions.append('LONDON')
        if 13 <= h < 22: sessions.append('NY')
        if 13 <= h < 16: sessions.append('OVERLAP')
        if not sessions: sessions.append('OFF_HOURS')
        return sessions

    def _build_m30_lookup(self, df_m30, signal_generator):
        lookup = {}
        if df_m30 is None or len(df_m30) < 50:
            return lookup
        df_m30 = signal_generator.generate_signals(df_m30, self.ml_buy_threshold, self.ml_sell_threshold)
        c = df_m30['c']
        sma20 = c.rolling(20, min_periods=10).mean()
        sma50 = c.rolling(50, min_periods=20).mean()
        for i in range(50, len(df_m30)):
            row = df_m30.iloc[i]
            t = row['time']
            p, s20, s50 = float(c.iloc[i]), float(sma20.iloc[i]), float(sma50.iloc[i])
            trend = 1 if p > s20 and s20 > s50 else (-1 if p < s20 and s20 < s50 else 0)
            lookup[t] = {
                'signal': int(row.get('signal', 0)),
                'confidence': float(row.get('confidence', 0)),
                'ict_score': int(row.get('ict_score', 0)),
                'rsi': float(row.get('rsi', 50)),
                'macd_hist': float(row.get('macd_hist', 0)),
                'trend': trend,
            }
        return lookup

    def _get_m30_data(self, m30_lookup, h1_time):
        if not m30_lookup: return None
        best_data, best_time = None, None
        for t, data in m30_lookup.items():
            if t <= h1_time:
                if best_time is None or t > best_time:
                    best_time = t; best_data = data
        return best_data

    def _check_m30_confirmation(self, m30_data, h1_signal):
        if m30_data is None:
            return True, 0, "no_m30_data"
        mode = self.m30_confirmation
        m30_signal, m30_trend = m30_data['signal'], m30_data['trend']
        m30_rsi, m30_macd = m30_data['rsi'], m30_data['macd_hist']
        boost, penalty = self.m30_conf_boost, self.m30_conf_penalty

        if mode == 'signal':
            if m30_signal == h1_signal:
                return True, boost, "m30_signal_confirm"
            elif m30_signal == -h1_signal:
                return False, -penalty, "m30_signal_against"
            else:
                if h1_signal == 1 and m30_rsi < 55 and m30_macd > 0:
                    return True, boost * 0.5, "m30_indicators_bullish"
                elif h1_signal == -1 and m30_rsi > 45 and m30_macd < 0:
                    return True, boost * 0.5, "m30_indicators_bearish"
                return True, 0, "m30_neutral"
        elif mode == 'trend':
            if m30_trend == h1_signal: return True, boost, "m30_trend_confirm"
            elif m30_trend == -h1_signal: return False, -penalty, "m30_trend_against"
            else: return True, 0, "m30_trend_neutral"
        elif mode == 'both':
            signal_ok = m30_signal == h1_signal or m30_signal == 0
            trend_ok = m30_trend == h1_signal or m30_trend == 0
            if signal_ok and trend_ok:
                extra = 1.5 if (m30_signal == h1_signal and m30_trend == h1_signal) else 1.0
                return True, boost * extra, "m30_both_confirm"
            elif m30_signal == -h1_signal or m30_trend == -h1_signal:
                return False, -penalty, "m30_against"
            else: return True, 0, "m30_partial"
        return True, 0, "unknown"

    def run(self, symbol, df_h1, df_h4, ml_model, signal_generator, symbol_point=0.0001, df_m30=None):
        logger.info(f"\n{'='*70}")
        logger.info(f"BACKTESTING: {symbol}")
        logger.info(f"Bars: {len(df_h1)} | Balance: ${self.initial_balance} | Risk: {self.risk_pct}% | MaxLot: {self.max_lot}")
        pip_size, pip_value, contract_size = get_contract_spec(symbol)
        m30_status = f"ON({self.m30_confirmation})" if self.use_m30 else "OFF"
        comp = "ON" if self.compounding else "OFF"
        cscale = "ON" if self.conf_scaling else "OFF"
        logger.info(f"SL={self.sl_atr_mult}x TP={self.tp_atr_mult}x M30={m30_status} Compound={comp} ConfScale={cscale}")
        logger.info(f"{'='*70}")

        df_h1 = signal_generator.generate_signals(df_h1, self.ml_buy_threshold, self.ml_sell_threshold)
        htf_trend = self._build_htf_trend(df_h4)
        m30_lookup = {}
        if self.use_m30 and df_m30 is not None:
            m30_lookup = self._build_m30_lookup(df_m30, signal_generator)

        warmup = 200
        m30_stats = {'confirm': 0, 'against': 0, 'neutral': 0, 'boost_total': 0}

        for i in range(warmup, len(df_h1)):
            bar = df_h1.iloc[i]
            bar_time = bar['time']
            price = float(bar['c'])
            high = float(bar['h'])
            low = float(bar['l'])
            atr = float(bar.get('atr', 0))
            if atr <= 0: continue

            self._process_open_trades(symbol, high, low, price, atr, bar_time, i)

            floating = sum(t.calc_floating_pnl(price) for t in self.open_trades)
            self.equity = self.balance + floating
            if self.equity > self.peak_balance:
                self.peak_balance = self.equity

            self.equity_curve.append({
                'bar': i, 'time': str(bar_time),
                'balance': round(self.balance, 2),
                'equity': round(self.equity, 2),
                'open_trades': len(self.open_trades),
            })

            signal = int(bar.get('signal', 0))
            confidence = float(bar.get('confidence', 0))
            adx = float(bar.get('adx', 0))
            rsi = float(bar.get('rsi', 50))
            ict_score = int(bar.get('ict_score', 0))

            if signal == 0: continue
            if ict_score < self.min_ict_score:
                self.trades_blocked['low_ict'] += 1; continue
            if confidence < self.min_confidence:
                self.trades_blocked['low_conf'] += 1; continue
            if adx < self.min_adx:
                self.trades_blocked['low_adx'] += 1; continue
            if len(self.open_trades) >= self.max_trades:
                self.trades_blocked['max_trades'] += 1; continue
            if len([t for t in self.open_trades if t.symbol == symbol]) >= self.max_per_symbol:
                self.trades_blocked['max_per_sym'] += 1; continue
            if self.balance <= self.initial_balance * 0.2:
                self.trades_blocked['low_bal'] += 1; continue
            if self._check_daily_limit(bar_time):
                self.trades_blocked['daily_limit'] += 1; continue

            side_key = f"{symbol}_{'B' if signal == 1 else 'S'}"
            if i - self.last_entry_bar.get(side_key, 0) < self.entry_cooldown:
                self.trades_blocked['cooldown'] += 1; continue

            if self.require_htf:
                htf = self._get_htf_trend(htf_trend, bar_time)
                if (signal == 1 and htf == -1) or (signal == -1 and htf == 1):
                    self.trades_blocked['htf'] += 1; continue

            if self.require_pullback:
                if (signal == 1 and rsi > self.pullback_rsi_buy_max) or \
                   (signal == -1 and rsi < self.pullback_rsi_sell_min):
                    self.trades_blocked['pullback'] += 1; continue

            if self.session_filter:
                curr_sess = self._get_bar_sessions(bar_time)
                best = self.symbol_best_sessions.get(symbol, [])
                if best and not any(s in best for s in curr_sess):
                    self.trades_blocked['session'] += 1; continue

            # M30
            if self.use_m30 and m30_lookup:
                m30_data = self._get_m30_data(m30_lookup, bar_time)
                confirmed, conf_adj, reason = self._check_m30_confirmation(m30_data, signal)
                confidence += conf_adj
                confidence = np.clip(confidence, 0.0, 1.0)
                if 'confirm' in reason: m30_stats['confirm'] += 1
                elif 'against' in reason: m30_stats['against'] += 1
                else: m30_stats['neutral'] += 1
                if not confirmed:
                    self.trades_blocked['m30'] += 1; continue
                if confidence < self.min_confidence:
                    self.trades_blocked['m30_conf'] += 1; continue

            # SL/TP
            spread_cost = self.spread_pips * pip_size + self.slippage_pips * pip_size
            if signal == 1:
                entry = price + spread_cost / 2
                sl = entry - atr * self.sl_atr_mult
                tp = entry + atr * self.tp_atr_mult
            else:
                entry = price - spread_cost / 2
                sl = entry + atr * self.sl_atr_mult
                tp = entry - atr * self.tp_atr_mult

            if signal == 1 and (sl >= entry or tp <= entry): continue
            if signal == -1 and (sl <= entry or tp >= entry): continue

            # ⚡ Calculate lot with confidence scaling + max_lot cap
            volume = self.calculate_lot_size(symbol, entry, sl, confidence)
            self.balance -= self.commission_per_lot * volume * 2

            self.trade_counter += 1
            trade = BacktestTrade(
                self.trade_counter, symbol,
                'BUY' if signal == 1 else 'SELL',
                entry, volume, sl, tp,
                bar_time, i, confidence, atr
            )
            self.open_trades.append(trade)
            self.partial_done[trade.ticket] = [False] * len(self.partial_stages)
            self.last_entry_bar[side_key] = i

        # Close remaining
        last_price = float(df_h1.iloc[-1]['c'])
        last_time = df_h1.iloc[-1]['time']
        for trade in list(self.open_trades):
            trade.close(last_price, last_time, len(df_h1) - 1, 'END')
            self.balance += trade.pnl
            self.closed_trades.append(trade)
        self.open_trades.clear()

        logger.info(f"Complete: {len(self.closed_trades)} trades | MaxLot cap: {self.max_lot}")
        if self.trades_blocked:
            logger.info(f"Blocked: {dict(self.trades_blocked)}")

        results = self.get_results()
        results['m30_stats'] = m30_stats
        return results

    def _process_open_trades(self, symbol, high, low, current_price, atr, bar_time, bar_idx):
        to_close = []
        for trade in self.open_trades:
            pip_size = trade.pip_size
            if trade.side == 'BUY':
                profit_dist = current_price - trade.entry_price
                trade.max_favorable = max(trade.max_favorable, high - trade.entry_price)
                trade.max_adverse = max(trade.max_adverse, trade.entry_price - low)
            else:
                profit_dist = trade.entry_price - current_price
                trade.max_favorable = max(trade.max_favorable, trade.entry_price - low)
                trade.max_adverse = max(trade.max_adverse, high - trade.entry_price)

            if trade.side == 'BUY' and low <= trade.sl:
                trade.close(trade.sl, bar_time, bar_idx, 'SL')
                self.balance += trade.pnl; self.daily_pnl += trade.pnl
                to_close.append(trade); continue
            elif trade.side == 'SELL' and high >= trade.sl:
                trade.close(trade.sl, bar_time, bar_idx, 'SL')
                self.balance += trade.pnl; self.daily_pnl += trade.pnl
                to_close.append(trade); continue

            if trade.side == 'BUY' and high >= trade.tp:
                trade.close(trade.tp, bar_time, bar_idx, 'TP')
                self.balance += trade.pnl; self.daily_pnl += trade.pnl
                to_close.append(trade); continue
            elif trade.side == 'SELL' and low <= trade.tp:
                trade.close(trade.tp, bar_time, bar_idx, 'TP')
                self.balance += trade.pnl; self.daily_pnl += trade.pnl
                to_close.append(trade); continue

            t_atr = trade.atr
            if t_atr <= 0: continue
            profit_in_atr = profit_dist / t_atr

            if self.partial_enabled and trade.ticket in self.partial_done:
                stages = self.partial_done[trade.ticket]
                for s_idx, (atr_level, close_pct) in enumerate(self.partial_stages):
                    if profit_in_atr >= atr_level and not stages[s_idx]:
                        close_vol = trade.volume * close_pct
                        pips = profit_dist / pip_size
                        partial_pnl = pips * trade.pip_value_per_lot * close_vol
                        self.balance += partial_pnl; self.daily_pnl += partial_pnl
                        trade.partial_pnl += partial_pnl
                        trade.volume = max(round(trade.volume * (1 - close_pct), 2), 0.01)
                        stages[s_idx] = True
                        if s_idx == 0:
                            if trade.side == 'BUY':
                                new_sl = trade.entry_price + t_atr * 0.1
                                if new_sl > trade.sl: trade.sl = new_sl
                            else:
                                new_sl = trade.entry_price - t_atr * 0.1
                                if new_sl < trade.sl: trade.sl = new_sl

            new_sl = None
            if trade.side == 'BUY':
                for atr_level, trail_dist in self.trail_stages:
                    if profit_in_atr >= atr_level:
                        potential = current_price - t_atr * trail_dist
                        if potential > trade.sl: new_sl = potential
                        break
                if new_sl is None and profit_in_atr >= self.breakeven_atr:
                    be = trade.entry_price + t_atr * 0.1
                    if be > trade.sl: new_sl = be
            else:
                for atr_level, trail_dist in self.trail_stages:
                    if profit_in_atr >= atr_level:
                        potential = current_price + t_atr * trail_dist
                        if potential < trade.sl: new_sl = potential
                        break
                if new_sl is None and profit_in_atr >= self.breakeven_atr:
                    be = trade.entry_price - t_atr * 0.1
                    if be < trade.sl: new_sl = be

            if new_sl is not None: trade.sl = new_sl

        for trade in to_close:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)

    def _build_htf_trend(self, df_h4):
        trends = {}
        if df_h4 is None or len(df_h4) < 50: return trends
        c = df_h4['c']
        sma50 = c.rolling(50, min_periods=20).mean()
        sma20 = c.rolling(20, min_periods=10).mean()
        for i in range(len(df_h4)):
            t = df_h4.iloc[i]['time']
            p, s50, s20 = c.iloc[i], sma50.iloc[i], sma20.iloc[i]
            if pd.isna(s50) or pd.isna(s20): trends[t] = 0
            elif p > s50 and s20 > s50: trends[t] = 1
            elif p < s50 and s20 < s50: trends[t] = -1
            else: trends[t] = 0
        return trends

    def _get_htf_trend(self, htf_trends, current_time):
        if not htf_trends: return 0
        best, best_time = 0, None
        for t, trend in htf_trends.items():
            if t <= current_time:
                if best_time is None or t > best_time:
                    best_time = t; best = trend
        return best

    def get_results(self):
        if not self.closed_trades:
            return {'error': 'No trades'}

        trades_df = pd.DataFrame([t.to_dict() for t in self.closed_trades])
        total_trades = len(trades_df)
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_count, loss_count = len(wins), len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        total_pnl = trades_df['pnl'].sum()
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        best_trade = trades_df['pnl'].max()
        worst_trade = trades_df['pnl'].min()

        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        rr_realized = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        max_dd, max_dd_pct, peak = 0, 0, self.initial_balance
        for e in self.equity_curve:
            eq = e['equity']
            if eq > peak: peak = eq
            dd = peak - eq
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd: max_dd = dd; max_dd_pct = dd_pct

        max_cw, max_cl, cw, cl = 0, 0, 0, 0
        for pnl in trades_df['pnl']:
            if pnl > 0: cw += 1; cl = 0; max_cw = max(max_cw, cw)
            else: cl += 1; cw = 0; max_cl = max(max_cl, cl)

        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            downside = returns[returns < 0]
            sortino = np.mean(returns) / (np.std(downside) + 1e-10) * np.sqrt(252) if len(downside) > 0 else 0
        else: sharpe = sortino = 0

        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        calmar = total_return / max_dd_pct if max_dd_pct > 0 else 0
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        buy_trades = trades_df[trades_df['side'] == 'BUY']
        sell_trades = trades_df[trades_df['side'] == 'SELL']

        monthly_data = []
        try:
            trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
            monthly = trades_df.groupby('month')['pnl'].agg(['sum', 'count', lambda x: (x > 0).sum()]).reset_index()
            monthly.columns = ['month', 'pnl', 'trades', 'wins']
            monthly['win_rate'] = monthly['wins'] / monthly['trades']
            monthly['return_pct'] = monthly['pnl'] / self.initial_balance * 100
            monthly_data = monthly.to_dict('records')
        except Exception: pass

        return {
            'initial_balance': self.initial_balance,
            'final_balance': round(self.balance, 2),
            'total_return': round(total_return, 2),
            'total_pnl': round(total_pnl, 2),
            'total_trades': total_trades,
            'win_count': win_count, 'loss_count': loss_count, 'win_rate': win_rate,
            'avg_pnl': round(total_pnl / total_trades, 2),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'best_trade': round(best_trade, 2), 'worst_trade': round(worst_trade, 2),
            'gross_profit': round(gross_profit, 2), 'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2), 'expectancy': round(expectancy, 2),
            'rr_realized': round(rr_realized, 2),
            'max_drawdown': round(max_dd, 2), 'max_drawdown_pct': round(max_dd_pct, 2),
            'max_consec_win': max_cw, 'max_consec_loss': max_cl,
            'sharpe_ratio': round(sharpe, 2), 'sortino_ratio': round(sortino, 2),
            'calmar_ratio': round(calmar, 2),
            'avg_duration_bars': round(trades_df['duration_bars'].mean(), 1),
            'exit_reasons': exit_reasons,
            'buy_pnl': round(buy_trades['pnl'].sum(), 2) if len(buy_trades) > 0 else 0,
            'sell_pnl': round(sell_trades['pnl'].sum(), 2) if len(sell_trades) > 0 else 0,
            'buy_win_rate': round(buy_trades['pnl'].gt(0).mean(), 4) if len(buy_trades) > 0 else 0,
            'sell_win_rate': round(sell_trades['pnl'].gt(0).mean(), 4) if len(sell_trades) > 0 else 0,
            'blocked_trades': dict(self.trades_blocked),
            'monthly': monthly_data, 'equity_curve': self.equity_curve,
            'trades': [t.to_dict() for t in self.closed_trades],
        }