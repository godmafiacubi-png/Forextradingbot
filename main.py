import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import time
import os
import sys
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import tensorflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import MetaTrader5 as mt5
from config.settings import *
from risk_management.regime_exit import REGIME_EXIT_POLICIES, get_regime_exit_policy
from data_layer.mt5_connector import MT5Connector
from feature_engineering.ict_features import ICTFeatures
from feature_engineering.ml_features import MLFeatures
from ml_models.ensemble import EnsembleModel
from ml_models.model_manager import ModelManager
from ml_models.deep_rl_agent import DeepRLTradingAgent
from ml_models.ml_hub import MLHub
from strategy.signal_generator import SignalGenerator
from strategy.news_filter import NewsFilter
from strategy.smart_filters import (
    SignalQualityScorer, AdaptiveThreshold, LossStreakManager,
    TimeFilter, SmartTrailingV2, EnhancedPartialClose, PerformanceAutoAdjust
)
from risk_management.position_sizer import PositionSizer
from risk_management.risk_guard import RiskGuard
from execution.order_manager import OrderManager
from execution.trade_logger import TradeJournal
from execution.risk_aware_journal import RiskAwareTradeJournal
from monitoring.simple_dashboard import SimpleMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.telegram_alerts import TelegramAlerts
from monitoring.expectancy_tracker import ExpectancyTracker

if BOT_MODE == 'AGGRESSIVE':
    from monitoring.web_dashboard_aggressive import update_dashboard, add_log, start_dashboard
else:
    from monitoring.web_dashboard import update_dashboard, add_log, start_dashboard

RISK_CONFIG = {
    'CORRELATION_FILTER_ENABLED': CORRELATION_FILTER_ENABLED,
    'CORRELATION_GROUPS': CORRELATION_GROUPS,
    'SYMBOL_CURRENCIES': SYMBOL_CURRENCIES,
    'MAX_SAME_DIRECTION_CORRELATED': MAX_SAME_DIRECTION_CORRELATED,
    'MAX_CURRENCY_EXPOSURE': MAX_CURRENCY_EXPOSURE,
    'MAX_SPREAD_MULTIPLIER': MAX_SPREAD_MULTIPLIER,
    'SPREAD_AVG_PERIOD': SPREAD_AVG_PERIOD,
    'SESSION_FILTER_ENABLED': SESSION_FILTER_ENABLED,
    'SYMBOL_BEST_SESSIONS': SYMBOL_BEST_SESSIONS,
    'DAILY_LOSS_LIMIT_PCT': DAILY_LOSS_LIMIT_PCT,
    'DAILY_PROFIT_TARGET_PCT': DAILY_PROFIT_TARGET_PCT,
    'CONSECUTIVE_LOSS_COOLDOWN': CONSECUTIVE_LOSS_COOLDOWN,
    'COOLDOWN_MINUTES': COOLDOWN_MINUTES,
    'RECOVERY_DRAWDOWN_TRIGGER': RECOVERY_DRAWDOWN_TRIGGER,
    'RECOVERY_RISK_PERCENT': RECOVERY_RISK_PERCENT,
    'RECOVERY_MAX_TRADES': RECOVERY_MAX_TRADES,
    'PARTIAL_CLOSE_ENABLED': PARTIAL_CLOSE_ENABLED,
    'PARTIAL_CLOSE_1_ATR': PARTIAL_CLOSE_1_ATR,
    'PARTIAL_CLOSE_1_PCT': PARTIAL_CLOSE_1_PCT,
    'PARTIAL_CLOSE_2_ATR': PARTIAL_CLOSE_2_ATR,
    'PARTIAL_CLOSE_2_PCT': PARTIAL_CLOSE_2_PCT,
    'ACCOUNT_RISK_PERCENT': ACCOUNT_RISK_PERCENT,
    'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
    'MAX_LOT_SIZE': MAX_LOT_SIZE,
    'MAX_SPREAD_POINTS': MAX_SPREAD_POINTS,
    'DEFAULT_MAX_SPREAD_POINTS': DEFAULT_MAX_SPREAD_POINTS,
}


# Quality threshold จาก config (ถ้ามี)
MIN_QUALITY_SCORE = getattr(
    __import__('config.settings', fromlist=['MIN_QUALITY_SCORE']),
    'MIN_QUALITY_SCORE',
    40 if BOT_MODE == 'AGGRESSIVE' else 55
)


class M30Analyzer:
    def __init__(self, mt5_conn, ml_model):
        self.mt5 = mt5_conn
        self.ml_model = ml_model
        self.signal_gen = SignalGenerator(ml_model, use_meta_strategy_selector=globals().get('USE_META_STRATEGY_SELECTOR', True))
        self.m30_cache = {}
        self.cache_ttl = 120

    def _fetch_m30(self, symbol):
        now = time.time()
        cached = self.m30_cache.get(symbol)
        if cached and (now - cached['updated']) < self.cache_ttl:
            return cached['data']
        try:
            df = self.mt5.get_ohlcv(symbol, TIMEFRAMES.get('M30', mt5.TIMEFRAME_M30), bars=200)
            if df is None or len(df) < 50:
                return None
            df = ICTFeatures(df).get_ict_features()
            df = MLFeatures(df).get_ml_features()
            sym_cfg = get_symbol_config(symbol)
            df = self.signal_gen.generate_signals(df, sym_cfg.get('ml_buy_threshold', ML_THRESHOLD_BUY), sym_cfg.get('ml_sell_threshold', ML_THRESHOLD_SELL))
            c = df['c']
            sma20 = c.rolling(20, min_periods=10).mean()
            sma50 = c.rolling(50, min_periods=20).mean()
            df['m30_trend'] = 0
            for i in range(len(df)):
                p = float(c.iloc[i])
                s20 = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else p
                s50 = float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else p
                if p > s20 and s20 > s50:
                    df.iloc[i, df.columns.get_loc('m30_trend')] = 1
                elif p < s20 and s20 < s50:
                    df.iloc[i, df.columns.get_loc('m30_trend')] = -1
            self.m30_cache[symbol] = {'data': df, 'updated': now}
            return df
        except Exception as e:
            logger.debug(f"M30 error {symbol}: {e}")
            return None

    def get_m30_data(self, symbol):
        df = self._fetch_m30(symbol)
        if df is None or len(df) < 2:
            return None
        latest = df.iloc[-1]
        return {
            'signal': int(latest.get('signal', 0)),
            'confidence': float(latest.get('confidence', 0)),
            'rsi': float(latest.get('rsi', 50)),
            'macd_hist': float(latest.get('macd_hist', 0)),
            'trend': int(latest.get('m30_trend', 0)),
        }

    def check_confirmation(self, symbol, h1_signal):
        sym_cfg = get_symbol_config(symbol)
        if not sym_cfg.get('use_m30', False):
            return True, 0, "m30_disabled", None
        m30 = self.get_m30_data(symbol)
        if m30 is None:
            return True, 0, "no_m30_data", None

        mode = sym_cfg.get('m30_confirmation', 'signal')
        boost = sym_cfg.get('m30_conf_boost', 0.12)
        penalty = sym_cfg.get('m30_conf_penalty', 0.10)

        if mode == 'signal':
            if m30['signal'] == h1_signal:
                return True, boost, "m30_signal_confirm", m30
            elif m30['signal'] == -h1_signal:
                return False, -penalty, "m30_signal_against", m30
            else:
                if h1_signal == 1 and m30['rsi'] < 55 and m30['macd_hist'] > 0:
                    return True, boost * 0.5, "m30_bullish", m30
                elif h1_signal == -1 and m30['rsi'] > 45 and m30['macd_hist'] < 0:
                    return True, boost * 0.5, "m30_bearish", m30
                return True, 0, "m30_neutral", m30
        elif mode == 'trend':
            if m30['trend'] == h1_signal:
                return True, boost, "m30_trend_confirm", m30
            elif m30['trend'] == -h1_signal:
                return False, -penalty, "m30_trend_against", m30
            else:
                return True, 0, "m30_trend_neutral", m30
        elif mode == 'both':
            sig_ok = m30['signal'] == h1_signal or m30['signal'] == 0
            tr_ok = m30['trend'] == h1_signal or m30['trend'] == 0
            if sig_ok and tr_ok:
                extra = 1.5 if (m30['signal'] == h1_signal and m30['trend'] == h1_signal) else 1.0
                return True, boost * extra, "m30_both_confirm", m30
            elif m30['signal'] == -h1_signal or m30['trend'] == -h1_signal:
                return False, -penalty, "m30_against", m30
            else:
                return True, 0, "m30_partial", m30
        return True, 0, "unknown", m30


class TradingBot:
    def __init__(self):
        logger.info("=" * 80)
        logger.info(f"TRADING BOT {BOT_VERSION} Deep RL | {BOT_MODE} | Port: {DASHBOARD_PORT}")
        logger.info(f"Execution safety: DRY_RUN={DRY_RUN} | LIVE_TRADING_CONFIRMED={LIVE_TRADING_CONFIRMED}")
        logger.info("Dueling DQN + PER + N-step + Market Regime")
        logger.info("=" * 80)

        try:
            self.mt5 = MT5Connector(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH)
            if not self.mt5.connected:
                raise Exception("MT5 not connected")
            ai = self.mt5.get_account_info()
            self.start_balance = ai['balance']
            logger.info(f"[OK] Balance: ${ai['balance']:.2f} | Equity: ${ai['equity']:.2f}")

            # ---- ML Hub (รวม Ensemble + DeepRL + Temporal + Meta + Retrain) ----
            self.hub = MLHub(
                model_dir='./models',
                use_temporal_encoder=True,
                use_meta_learner=True,
                use_rl=True,
                use_regime_models=True,
                seq_len=32,
                retrain_every_n_trades=100,
                meta_activity_threshold=0.40,
            )
            self.hub.load()

            # Backward-compat aliases — โค้ดส่วนอื่นที่ยังใช้ชื่อเดิมทำงานได้ทันที
            self.ml_model = self.hub.ensemble
            self.rl_agent = self.hub.rl
            self.model_manager = ModelManager('./models')

            if not getattr(self.ml_model, 'is_trained', False):
                self.ml_model.is_trained = False
            if self.ml_model.is_trained and getattr(self.ml_model, 'feature_cols', None) is None:
                self._derive_feature_cols()

            rl_stats = self.rl_agent.get_stats() if self.rl_agent else {}
            logger.info(f"[OK] ML Hub v1.0 — Regime-Aware + Cross-Symbol + Temporal + Meta")
            logger.info(f"[OK] Deep RL: {rl_stats.get('architecture','N/A')} | Device: {rl_stats.get('device','N/A')}")
            logger.info(f"     Trades: {rl_stats.get('total_trades',0)} | Steps: {rl_stats.get('train_steps',0)} | Buffer: {rl_stats.get('buffer_size',0)}")

            self.m30_analyzer = M30Analyzer(self.mt5, self.ml_model)
            logger.info("[OK] M30 Multi-TF")

            self.quality_scorer = SignalQualityScorer()
            self.adaptive_threshold = AdaptiveThreshold(base_threshold=MIN_CONFIDENCE, window=20)
            self.loss_streak = LossStreakManager(max_streak=2, cooldown_minutes=120)
            self.time_filter = TimeFilter()
            self.smart_trailing = SmartTrailingV2()
            self.perf_adjuster = PerformanceAutoAdjust(window=30)
            logger.info(f"[OK] Smart Filters + SmartTrailingV2 (BE + Trail) | Quality >= {MIN_QUALITY_SCORE}")

            self.news_filter = NewsFilter(30, 30)
            logger.info(f"[OK] News ({len(self.news_filter.events)} events)")

            self.risk_guard = RiskGuard(self.mt5, RISK_CONFIG)
            logger.info("[OK] Risk Guard")

            base_journal = TradeJournal(csv_path="journal/trades.csv", sqlite_path="journal/trades.sqlite3")
            self.trade_journal = RiskAwareTradeJournal(base_journal, self.risk_guard)

            self.signal_gen = SignalGenerator(self.ml_model, use_meta_strategy_selector=globals().get('USE_META_STRATEGY_SELECTOR', True))
            self.position_sizer = PositionSizer(POSITION_SIZING_METHOD, ACCOUNT_RISK_PERCENT, MAX_DRAWDOWN_PERCENT, max_lot_size=MAX_LOT_SIZE)
            self.order_manager = OrderManager(
                self.mt5,
                MAX_OPEN_TRADES,
                MAX_TRADES_PER_SYMBOL,
                dry_run=DRY_RUN,
                magic=ORDER_MAGIC,
                deviation=ORDER_DEVIATION,
                trade_journal=self.trade_journal,
            )
            self.monitor = SimpleMonitor()
            self.tracker = PerformanceTracker()
            self.telegram = TelegramAlerts(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            self.expectancy_tracker = ExpectancyTracker()

            start_dashboard(port=DASHBOARD_PORT)
            update_dashboard('bot_status', 'RUNNING')
            update_dashboard('dashboard_port', DASHBOARD_PORT)
            update_dashboard('mode', BOT_MODE)
            update_dashboard('bot_version', f'ForexTradingBot {BOT_VERSION}')
            update_dashboard('execution_mode', 'Dryruns' if DRY_RUN else 'Livetrade')

            self.htf_trend = {}
            self.active_trades = {}
            self.prev_positions = {}
            self.symbol_data_cache = {}
            self.atr_cache = {}
            self.iteration = 0
            self.m30_stats = {'confirm': 0, 'against': 0, 'neutral': 0, 'disabled': 0}
            self.quality_stats = {'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'blocked': 0}
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_day = None

            # Trailing thread control
            self._trailing_running = False
            self._trailing_thread = None

            logger.info("")
            logger.info("Per-Symbol Settings:")
            logger.info(f"  {'Symbol':<12} {'SL':>4} {'TP':>4} {'Risk':>5} {'Conf':>5} {'ADX':>4} {'ICT':>4} {'Sess':>5} {'M30':>8} {'HTF':>4}")
            logger.info(f"  {'-'*70}")
            for sym, cfg in SYMBOL_SETTINGS.items():
                m30_mode = cfg.get('m30_confirmation', '-') if cfg.get('use_m30', False) else 'OFF'
                sess = 'ON' if cfg.get('session_filter', True) else 'OFF'
                htf = 'ON' if cfg.get('require_htf', True) else 'OFF'
                logger.info(f"  {sym:<12} {cfg.get('sl_atr_mult', 2.0):>4.1f} {cfg.get('tp_atr_mult', 2.5):>4.1f} "
                           f"{cfg.get('risk_pct', 0.5):>4.1f}% {cfg.get('min_confidence', 0.45):>4.0%} "
                           f"{cfg.get('min_adx', 25):>4} {cfg.get('min_ict_score', 2):>3}+ "
                           f"{sess:>5} {m30_mode:>8} {htf:>4}")

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[OK] BOT {BOT_VERSION} READY — {BOT_MODE} + Deep RL")
            logger.info(f"[OK] BE_ATR={BREAKEVEN_ATR} TRAIL_ATR={TRAILING_STOP_ATR} | Trailing: 5s thread")
            logger.info(f"[OK] Dashboard: http://localhost:{DASHBOARD_PORT}")
            logger.info(f"[OK] Execution: {'DRY RUN (no orders sent)' if DRY_RUN else 'LIVE ORDERS ENABLED'} | Magic={ORDER_MAGIC}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Init error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)

    def _derive_feature_cols(self):
        try:
            df = self.mt5.get_ohlcv(SYMBOLS['FOREX'][0], TIMEFRAMES.get(PRIMARY_TIMEFRAME, mt5.TIMEFRAME_H1), bars=LOOKBACK_PERIOD)
            if df is not None and len(df) > 0:
                df = MLFeatures(ICTFeatures(df).get_ict_features()).get_ml_features()
                exclude = {
                    'time', 'o', 'h', 'l', 'c', 'v', 'signal', 'confidence',
                    'regime', 'market_regime', 'htf_regime',
                    'symbol', 'date', 'datetime', 'timestamp', 'index',
                }
                self.ml_model.feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
                self.ml_model.n_features = len(self.ml_model.feature_cols)
                # sync ไปที่ regime_ensemble ด้วยถ้ามี
                if hasattr(self.hub, 'ensemble') and hasattr(self.hub.ensemble, 'regime_ensemble'):
                    if self.hub.ensemble.regime_ensemble:
                        self.hub.ensemble.regime_ensemble.feature_cols = self.ml_model.feature_cols
                self.model_manager.save_model(self.ml_model, 'trading_model')
        except Exception as e:
            logger.error(f"Derive error: {e}")

    def _auto_train(self):
        try:
            all_dfs = []
            for s in SYMBOLS['FOREX'] + SYMBOLS.get('CRYPTO', []) + SYMBOLS.get('GOLD', []):
                df = self.mt5.get_ohlcv(s, TIMEFRAMES.get(PRIMARY_TIMEFRAME, mt5.TIMEFRAME_H1), bars=LOOKBACK_PERIOD * TRAIN_BARS_MULTIPLIER)
                if df is not None and len(df) >= 100:
                    all_dfs.append(MLFeatures(ICTFeatures(df).get_ict_features()).get_ml_features())
            if not all_dfs:
                return False
            combined = pd.concat(all_dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
            if len(combined) < MIN_TRAINING_SAMPLES:
                return False

            # ---- ใช้ hub.train() แทน — train ทั้ง global + regime models ----
            if self.hub.train(combined):
                self.hub.save()
                add_log(f"[TRAIN] Done ({len(combined)} samples) — Regime-Aware models updated")
                return True
            return False
        except Exception as e:
            logger.error(f"Train error: {e}")
            return False

    def _update_htf_trend(self, symbol):
        try:
            df = self.mt5.get_ohlcv(symbol, TIMEFRAMES.get(HIGHER_TIMEFRAME, mt5.TIMEFRAME_H4), bars=100)
            if df is None or len(df) < 50:
                return 0
            c = df['c']
            p, s50, s20 = c.iloc[-1], c.rolling(50).mean().iloc[-1], c.rolling(20).mean().iloc[-1]
            t = 1 if p > s50 and s20 > s50 else (-1 if p < s50 and s20 < s50 else 0)
            self.htf_trend[symbol] = t
            return t
        except Exception:
            return 0

    def _get_session_name(self):
        h = datetime.now().hour
        sessions = []
        if 0 <= h < 8:
            sessions.append("ASIAN")
        if 7 <= h < 16:
            sessions.append("LONDON")
        if 13 <= h < 22:
            sessions.append("NY")
        if 13 <= h < 16:
            sessions.append("OVERLAP")
        return sessions if sessions else ["OFF"]

    def _check_session_filter(self, symbol):
        if not SESSION_FILTER_ENABLED:
            return True, "global_off"
        sym_cfg = get_symbol_config(symbol)
        if not sym_cfg.get('session_filter', True):
            return True, "off"
        current = self._get_session_name()
        best = SYMBOL_BEST_SESSIONS.get(symbol, [])
        if not best:
            return True, "no_restrict"
        if any(s in best for s in current):
            return True, "ok"
        return False, f"blocked ({','.join(current)})"

    def _detect_closed_trades(self):
        try:
            current = {p.ticket: p for p in (self.order_manager.get_open_positions() or [])}
            for ticket, prev in self.prev_positions.items():
                if ticket not in current:
                    pnl = prev.profit
                    side = 'BUY' if prev.type == 0 else 'SELL'
                    is_win = pnl > 0

                    self.daily_pnl += pnl

                    exit_price = prev.price_current if hasattr(prev, 'price_current') else prev.price_open
                    self.tracker.close_trade(ticket, exit_price, actual_pnl=pnl)

                    if ticket not in self.active_trades:
                        continue

                    active_meta = self.active_trades.pop(ticket)

                    self.risk_guard.record_trade_result(pnl)
                    self.adaptive_threshold.record_result(is_win)
                    self.loss_streak.record_result(is_win)
                    self.perf_adjuster.record_trade(pnl)
                    self.expectancy_tracker.record(prev.symbol, pnl)

                    sym_data_c = self.symbol_data_cache.get(prev.symbol, {})
                    sig_data_c = self.symbol_data_cache.get(f'{prev.symbol}_signal', {})
                    pnl_pct = pnl / (prev.price_open * prev.volume * 100000 + 1e-10)
                    ai_now = self.mt5.get_account_info()

                    # ถ้าไม่มี cache (เช่น trade เปิดก่อน restart) สร้าง minimal market_data
                    if not sym_data_c:
                        sym_data_c = {
                            'price': prev.price_current if hasattr(prev, 'price_current') else prev.price_open,
                            'atr_pct': 0.005,
                            'vol_ratio': 1.0,
                        }

                    # ---- hub.on_trade_close() — อัปเดต RL + Meta + Retrain ทีเดียว ----
                    rl_result = self.hub.on_trade_close(
                        symbol=prev.symbol,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        equity=ai_now['equity'],
                        market_data=sym_data_c,
                        signal_data=sig_data_c,
                    )
                    if rl_result:
                        self.trade_journal.log_rl_trade_result(
                            ticket, prev.symbol, side, pnl=pnl,
                            rl_reward=rl_result.get('rl_reward'),
                            q_value=rl_result.get('q_value'),
                            action=rl_result.get('action'),
                            confidence=active_meta.get('confidence'),
                            comment=(
                                f"pnl={pnl:.2f} rl_reward={rl_result.get('rl_reward')} "
                                f"q_value={rl_result.get('q_value')} action={rl_result.get('action')} "
                                f"confidence={active_meta.get('confidence')}"
                            ),
                        )

                    result = "WIN" if is_win else "LOSS"
                    ai = self.mt5.get_account_info()
                    growth = (ai['equity'] - self.start_balance) / self.start_balance * 100
                    at = self.adaptive_threshold.get_stats()
                    ls = self.loss_streak.get_stats()
                    regime_info = self.rl_agent.regime_detector.current_regime.get(prev.symbol, {})
                    regime_name = regime_info.get('regime', '?')

                    add_log(f"[CLOSED] #{ticket} {prev.symbol} {side} ${pnl:.2f} {result} [{regime_name}]")
                    logger.info(f"[CLOSED] #{ticket} {prev.symbol} {side} ${pnl:.2f} {result} | Eq=${ai['equity']:.2f} ({growth:+.1f}%) | Regime: {regime_name}")
                    logger.info(f"  Adaptive: thresh={at['current']:.0%} WR={at['recent_wr']:.0%} | Streak: {ls['current_streak']}")

                    try:
                        self.telegram.send_message(
                            f"{'✅' if is_win else '❌'} #{ticket} {prev.symbol} {side}\n"
                            f"PnL: ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}\n"
                            f"Equity: ${ai['equity']:.2f} ({growth:+.1f}%)\n"
                            f"Regime: {regime_name}\n"
                            f"Adaptive: {at['current']:.0%} | Streak: {ls['current_streak']}"
                        )
                    except Exception:
                        pass

            self.prev_positions = current
        except Exception as e:
            logger.debug(f"Detect closed error: {e}")

    # ===== TRAILING THREAD (เช็คทุก 5 วินาที) =====
    def _trailing_thread_loop(self):
        """Thread แยก — เช็ค BE/Trail ทุก 5 วินาที"""
        logger.info("[TRAIL-THREAD] Started — checking every 5s")
        while self._trailing_running:
            try:
                positions = self.order_manager.get_open_positions() or []
                if not positions:
                    time.sleep(5)
                    continue

                for pos in positions:
                    atr = self.atr_cache.get(pos.symbol, 0)
                    if atr <= 0:
                        continue
                    si = self.mt5.get_symbol_info(pos.symbol)
                    if si is None:
                        continue

                    current_price = si['bid'] if pos.type == 0 else si['ask']
                    side = 'BUY' if pos.type == 0 else 'SELL'

                    new_sl = self.smart_trailing.calculate_new_sl(
                        side=side,
                        entry=pos.price_open,
                        current_price=current_price,
                        current_sl=pos.sl,
                        atr=atr,
                        ticket=pos.ticket
                    )

                    if new_sl != pos.sl:
                        if side == 'BUY' and new_sl > pos.sl:
                            success = self.order_manager.modify_sl(pos.ticket, new_sl)
                            if success:
                                profit_atr = (current_price - pos.price_open) / atr
                                add_log(f"[BE/TRAIL] #{pos.ticket} {pos.symbol} SL->{new_sl:.5f} (+{profit_atr:.1f}ATR)")
                        elif side == 'SELL' and (new_sl < pos.sl or pos.sl == 0):
                            success = self.order_manager.modify_sl(pos.ticket, new_sl)
                            if success:
                                profit_atr = (pos.price_open - current_price) / atr
                                add_log(f"[BE/TRAIL] #{pos.ticket} {pos.symbol} SL->{new_sl:.5f} (+{profit_atr:.1f}ATR)")

                active_tickets = {p.ticket for p in positions}
                self.smart_trailing.cleanup(active_tickets)

            except Exception as e:
                logger.debug(f"Trailing thread error: {e}")

            time.sleep(5)

        logger.info("[TRAIL-THREAD] Stopped")

    def _start_trailing_thread(self):
        self._trailing_running = True
        self._trailing_thread = threading.Thread(
            target=self._trailing_thread_loop,
            daemon=True,
            name="TrailingThread"
        )
        self._trailing_thread.start()

    def _stop_trailing_thread(self):
        self._trailing_running = False
        if self._trailing_thread and self._trailing_thread.is_alive():
            self._trailing_thread.join(timeout=10)

    def _apply_news_protection_all(self):
        try:
            for pos in (self.order_manager.get_open_positions() or []):
                si = self.mt5.get_symbol_info(pos.symbol)
                if si is None:
                    continue
                cp = si['bid'] if pos.type == 0 else si['ask']
                profit = (cp - pos.price_open) if pos.type == 0 else (pos.price_open - cp)
                atr = self.atr_cache.get(pos.symbol, abs(si['ask'] - si['bid']) * 50)
                action, details = self.news_filter.get_position_action(pos.symbol, profit, atr, pos.price_open)
                if action != 'NORMAL':
                    logger.info(f"[NEWS] {pos.symbol} #{pos.ticket}: {action}")
                    self.order_manager.apply_news_protection(pos.symbol, action, atr)
        except Exception as e:
            logger.debug(f"News protection error: {e}")

    def _apply_partial_closes(self):
        try:
            positions = self.order_manager.get_open_positions()
            if not positions:
                return
            open_tickets = {p.ticket for p in positions}
            self.risk_guard.cleanup_partial_tracking(open_tickets)
            actions = self.risk_guard.check_partial_close(positions, self.atr_cache)
            for ticket, close_pct, reason in actions:
                stage = None
                if isinstance(reason, str) and reason.startswith("Stage"):
                    stage = reason.split(":", 1)[0].replace("Stage", "")
                if self.order_manager.partial_close(ticket, close_pct, stage=stage):
                    add_log(f"[PARTIAL] #{ticket} {close_pct:.0%} — {reason}")
        except Exception as e:
            logger.debug(f"Partial close error: {e}")

    def _check_mt5_health(self):
        if not AUTO_RESTART_ENABLED:
            return True
        try:
            ai = self.mt5.get_account_info()
            if ai and ai.get('balance', 0) > 0:
                return True
        except Exception:
            pass
        logger.warning("[HEALTH] MT5 lost — reconnecting...")
        for attempt in range(MAX_RESTART_ATTEMPTS):
            try:
                time.sleep(RESTART_DELAY_SECONDS)
                self.mt5 = MT5Connector(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH)
                if self.mt5.connected:
                    self.order_manager.mt5 = self.mt5
                    self.risk_guard.mt5 = self.mt5
                    self.m30_analyzer.mt5 = self.mt5
                    logger.info(f"[HEALTH] Reconnected ({attempt + 1})")
                    return True
            except Exception:
                pass
        return False

    def _update_dashboard_all(self):
        try:
            ai = self.mt5.get_account_info()
            growth = (ai['equity'] - self.start_balance) / self.start_balance * 100
            update_dashboard('account', {
                'balance': ai.get('balance', 0),
                'equity': ai.get('equity', 0),
                'profit': ai.get('profit', 0),
                'free_margin': ai.get('free_margin', 0),
                'growth_pct': round(growth, 2),
            })
            self.monitor.log_account(ai.get('balance', 0), ai.get('equity', 0))

            pos_list = []
            for p in (self.order_manager.get_open_positions() or []):
                si = self.mt5.get_symbol_info(p.symbol)
                cur = (si['bid'] if p.type == 0 else si['ask']) if si else 0
                pos_list.append({
                    'ticket': p.ticket, 'symbol': p.symbol,
                    'side': 'BUY' if p.type == 0 else 'SELL',
                    'volume': p.volume,
                    'entry': p.price_open, 'current_price': cur,
                    'sl': p.sl, 'tp': p.tp, 'pnl': p.profit
                })
            update_dashboard('open_positions', pos_list)

            symbol_snapshot = {
                key: value
                for key, value in self.symbol_data_cache.items()
                if not str(key).endswith('_signal')
            }
            signal_snapshot = {
                str(key)[:-7]: value
                for key, value in self.symbol_data_cache.items()
                if str(key).endswith('_signal')
            }
            update_dashboard('symbols', symbol_snapshot)
            update_dashboard('signals', signal_snapshot)

            tracker_stats = self.tracker.get_stats()
            if tracker_stats:
                update_dashboard('tracker_stats', tracker_stats)

            rl_stats = self.rl_agent.get_stats()
            update_dashboard('rl_stats', rl_stats)
            update_dashboard('regime_stats', rl_stats.get('regimes', {}))

            update_dashboard('news_data', self.news_filter.get_dashboard_data())
            update_dashboard('risk_guard', self.risk_guard.get_status())
            update_dashboard('m30_stats', self.m30_stats)
            update_dashboard('quality_stats', self.quality_stats)
            update_dashboard('adaptive', self.adaptive_threshold.get_stats())
            update_dashboard('streak', self.loss_streak.get_stats())
            update_dashboard('auto_adjust', self.perf_adjuster.get_stats())
            update_dashboard('daily_pnl', round(self.daily_pnl, 2))
        except Exception:
            pass

    def fetch_and_process(self, symbol, timeframe):
        try:
            df = self.mt5.get_ohlcv(symbol, TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1), bars=LOOKBACK_PERIOD * 2)
            if df is None or len(df) == 0:
                return None
            return MLFeatures(ICTFeatures(df).get_ict_features()).get_ml_features()
        except Exception as e:
            logger.error(f"Process {symbol}: {e}")
            return None

    def run_strategy(self, symbol, timeframe):
        try:
            sym_cfg = get_symbol_config(symbol)
            sl_mult = sym_cfg['sl_atr_mult']
            tp_mult = sym_cfg['tp_atr_mult']
            sym_risk = sym_cfg['risk_pct']
            sym_min_conf = sym_cfg['min_confidence']
            sym_min_adx = sym_cfg['min_adx']
            sym_min_ict = sym_cfg['min_ict_score']
            sym_ml_buy = sym_cfg['ml_buy_threshold']
            sym_ml_sell = sym_cfg['ml_sell_threshold']
            sym_rsi_buy_max = sym_cfg['pullback_rsi_buy_max']
            sym_rsi_sell_min = sym_cfg['pullback_rsi_sell_min']
            sym_require_htf = sym_cfg['require_htf']
            sym_require_pb = sym_cfg['require_pullback']
            sym_use_m30 = sym_cfg.get('use_m30', False)
            sym_m30_mode = sym_cfg.get('m30_confirmation', 'signal')

            df = self.fetch_and_process(symbol, timeframe)
            if df is None or len(df) == 0:
                return

            latest = df.iloc[-1]
            price = float(latest['c'])
            atr = float(latest.get('atr', 0))
            self.atr_cache[symbol] = atr

            sma20 = float(latest.get('sma_20', 0))
            sma50 = float(latest.get('sma_50', 0))
            sma200 = float(latest.get('sma_200', 0))
            h1_trend = "BULL" if price > sma200 and sma20 > sma50 else ("BEAR" if price < sma200 and sma20 < sma50 else "MIXED")
            htf = self._update_htf_trend(symbol)
            htf_str = "BULL" if htf == 1 else ("BEAR" if htf == -1 else "MIXED")

            rsi = float(latest.get('rsi', 50))
            adx = float(latest.get('adx', 0))
            structure = int(latest.get('structure', 0))
            vol_spike = int(latest.get('vol_spike', 0))
            session = self._get_session_name()
            session_str = ",".join(session)

            try:
                ml_prob = float(self.ml_model.predict(df)[-1])
                # Debug: ตรวจ feature coverage — ถ้าน้อยกว่า 50% prob จะไม่น่าเชื่อถือ
                if self.ml_model.feature_cols:
                    matched = sum(1 for c in self.ml_model.feature_cols if c in df.columns)
                    coverage = matched / len(self.ml_model.feature_cols)
                    if coverage < 0.5:
                        logger.warning(f"  ML feature coverage low: {matched}/{len(self.ml_model.feature_cols)} ({coverage:.0%}) — prob unreliable")
                        ml_prob = 0.5
            except Exception:
                ml_prob = 0.5

            ai = self.mt5.get_account_info()
            sym_data = {
                'price': price, 'atr': atr, 'h1_trend': h1_trend, 'htf_trend': htf_str,
                'rsi': rsi, 'adx': adx, 'structure': structure, 'vol_spike': vol_spike,
                'session': session_str,
                'atr_pct': float(latest.get('atr_pct', 0)),
                'macd_hist': float(latest.get('macd_hist', 0)),
                'stoch_k': float(latest.get('stoch_k', 50)),
                'price_vs_sma200': float(latest.get('price_vs_sma200', 0)),
                'bb_pct': float(latest.get('bb_percent_b', 0.5)),
                'bb_width': float(latest.get('bb_width', 0.02)),
                'vol_ratio': float(latest.get('vol_ratio', 1.0)),
                'equity': ai['equity'],
            }
            self.symbol_data_cache[symbol] = sym_data

            is_blackout, news_reason, _, _ = self.news_filter.is_blackout(symbol)
            news_ok, news_msg, news_mult = self.news_filter.get_trade_permission(symbol)
            spread_ok, cur_spread, avg_spread = self.risk_guard.check_spread(symbol)

            df = self.signal_gen.generate_signals(df, sym_ml_buy, sym_ml_sell)
            latest = df.iloc[-1]
            base_signal = int(latest['signal'])
            base_conf = float(np.clip(latest['confidence'], 0.0, 0.95))  # cap ที่ 95% ป้องกัน overflow
            ict_score = int(latest.get('ict_score', 0))
            raw_base_signal = int(latest.get('base_signal', base_signal))
            raw_base_conf = float(np.clip(latest.get('base_confidence', base_conf), 0.0, 0.95))
            entry_strategy = str(latest.get('entry_strategy', 'ict_ml_baseline') or 'ict_ml_baseline')
            strategy_reason = str(latest.get('strategy_reason', '') or '')
            strategy_conf = float(np.clip(latest.get('strategy_confidence', base_conf), 0.0, 1.0))

            has_pos = len(self.order_manager.get_open_positions(symbol)) > 0
            pnl_pct = 0.0
            if has_pos:
                pos = self.order_manager.get_open_positions(symbol)
                if pos:
                    pnl_pct = pos[0].profit / (pos[0].price_open * pos[0].volume * 100000 + 1e-10)

            sig_data = {
                'ml_prob': ml_prob,
                'confidence': base_conf,
                'base_signal': 'BUY' if base_signal == 1 else ('SELL' if base_signal == -1 else 'HOLD'),
                'raw_base_signal': 'BUY' if raw_base_signal == 1 else ('SELL' if raw_base_signal == -1 else 'HOLD'),
                'raw_base_confidence': round(raw_base_conf, 4),
                'entry_strategy': entry_strategy,
                'strategy_confidence': round(strategy_conf, 4),
                'strategy_reason': strategy_reason,
            }

            # ---- คำนวณ Swing Low / Swing High (20 bars) สำหรับ near-SR check ----
            lookback = min(100, len(df))
            swing_low  = float(df['l'].iloc[-lookback:].min()) if 'l' in df.columns else price * 0.99
            swing_high = float(df['h'].iloc[-lookback:].max()) if 'h' in df.columns else price * 1.01
            dist_to_support  = (price - swing_low)  / (atr + 1e-10)   # หน่วย ATR
            dist_to_resist   = (swing_high - price) / (atr + 1e-10)
            # ถ้า SELL แต่ราคาอยู่ใกล้แนวรับ < 1.5 ATR = อันตราย
            # ถ้า BUY  แต่ราคาอยู่ใกล้แนวต้าน < 1.5 ATR = อันตราย
            NEAR_SR_ATR = 1.5
            self.symbol_data_cache[f'{symbol}_signal'] = sig_data

            # ---- hub.on_bar() — feed temporal buffer + retrain engine ----
            self.hub.on_bar(symbol, sym_data, sig_data)

            # ---- hub.get_signal() — RL + Meta + Temporal + Regime ----
            hub_result = self.hub.get_signal(
                symbol=symbol,
                base_signal=base_signal,
                base_confidence=base_conf,
                market_data=sym_data,
                signal_data=sig_data,
                has_position=has_pos,
                pnl_pct=pnl_pct,
            )

            adj_sig  = hub_result.signal
            adj_conf = hub_result.confidence
            rl_act   = hub_result.rl_action
            rl_src   = hub_result.source

            # rl_state ยังต้องการสำหรับ record_trade_open ข้างล่าง
            # ใช้ state ที่ hub.get_signal() สร้างไว้แล้วเพื่อหลีกเลี่ยงการเรียกซ้ำ
            rl_state = self.hub.get_last_rl_state(symbol)
            if rl_state is None and self.rl_agent:
                logger.warning(f"[run_strategy] rl_state cache miss for {symbol} — building fallback state")
                rl_state = self.rl_agent.build_state(sym_data, sig_data, has_pos, pnl_pct, symbol)

            rl_name = ["HOLD", "BUY", "SELL"][rl_act]
            regime_name = hub_result.regime
            meta_tag = f"Meta:{hub_result.meta_activity:.0%}" if hub_result.meta_trade else f"Meta:SKIP({hub_result.meta_activity:.0%})"

            adj_conf *= news_mult
            signal = adj_sig
            confidence = adj_conf
            exit_policy = get_regime_exit_policy(
                regime_name,
                sl_mult,
                tp_mult,
                base_breakeven_atr=BREAKEVEN_ATR,
                policies=sym_cfg.get('regime_exit_policies', REGIME_EXIT_POLICIES),
                confidence=confidence,
            )
            sl_mult = exit_policy['sl_atr_mult']
            tp_mult = exit_policy['tp_atr_mult']
            signal_name = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")

            perf_adj = self.perf_adjuster.get_adjustments()
            adjusted_conf_thresh = sym_min_conf + perf_adj.get('conf_adj', 0)
            adjusted_adx_thresh = sym_min_adx + perf_adj.get('adx_adj', 0)

            adaptive_thresh = self.adaptive_threshold.get_threshold()
            effective_conf_thresh = max(adjusted_conf_thresh, adaptive_thresh)

            m30_tag = f"M30:{sym_m30_mode}" if sym_use_m30 else "M30:OFF"

            logger.info(f"")
            logger.info(f"  ┌─── {symbol} [R:{sym_risk}% SL:{sl_mult}x TP:{tp_mult}x {m30_tag}]")
            max_spread = MAX_SPREAD_POINTS.get(symbol, DEFAULT_MAX_SPREAD_POINTS)
            cap_tag = f" cap:{max_spread:.1f}" if max_spread is not None else " cap:OFF"
            logger.info(f"  │ Price: {price:.5f}  ATR: {atr:.5f}  Spread: {cur_spread:.1f}(avg:{avg_spread:.1f}{cap_tag})  Sess: {session_str}")
            logger.info(f"  │ DistSup: {dist_to_support:.1f}ATR  DistRes: {dist_to_resist:.1f}ATR  SwL: {swing_low:.5f}  SwH: {swing_high:.5f}")
            logger.info(f"  │ H1: {h1_trend}  H4: {htf_str}  RSI={rsi:.1f}  ADX={adx:.1f}  ML: {ml_prob:.4f}")
            logger.info(f"  │ DeepRL: {rl_name} | Regime: {regime_name} | {meta_tag} | Temporal: {'Y' if hub_result.temporal_enriched else 'N'} | Src: {rl_src}")
            logger.info(
                f"  │ Strategy: {entry_strategy} ({strategy_conf:.2%}) | "
                f"Reason: {strategy_reason or 'n/a'}"
            )
            if is_blackout:
                logger.info(f"  │ NEWS: {news_reason}")
            if perf_adj:
                logger.info(f"  │ AutoAdj: {perf_adj.get('reason', '')} conf>={effective_conf_thresh:.0%} adx>={adjusted_adx_thresh}")
            logger.info(
                f"  │ ICT={ict_score} RawBase: {'BUY' if raw_base_signal==1 else ('SELL' if raw_base_signal==-1 else 'HOLD')} "
                f"({raw_base_conf:.2%}) -> StrategyBase: {'BUY' if base_signal==1 else ('SELL' if base_signal==-1 else 'HOLD')} "
                f"({base_conf:.2%}) -> {signal_name} ({confidence:.2%})"
            )

            self.monitor.log_signal(symbol, signal, confidence)

            risk_pct, max_trades = self.risk_guard.get_risk_adjusted_params()
            effective_risk = min(risk_pct, sym_risk) if self.risk_guard.recovery_mode else sym_risk
            risk_mult = perf_adj.get('risk_mult', 1.0) * exit_policy.get('risk_mult', 1.0)
            effective_risk *= risk_mult
            self.order_manager.set_limits(max_trades)
            self.position_sizer.account_risk = effective_risk

            can_trade = self.order_manager.can_open_trade(symbol)

            blocked = None

            if signal == 0:
                blocked = "HOLD"
            elif not spread_ok:
                max_spread = MAX_SPREAD_POINTS.get(symbol, DEFAULT_MAX_SPREAD_POINTS)
                if max_spread is not None and cur_spread > max_spread:
                    blocked = f"Spread {cur_spread:.1f} > cap {max_spread:.1f} (avg:{avg_spread:.1f})"
                else:
                    blocked = f"Spread {cur_spread:.1f} > {avg_spread*MAX_SPREAD_MULTIPLIER:.1f} (avg:{avg_spread:.1f})"
            elif not hub_result.meta_trade:
                blocked = f"Meta filter ({hub_result.meta_activity:.0%} < threshold)"
            elif ict_score < sym_min_ict:
                blocked = f"ICT {ict_score}<{sym_min_ict}"
            # Quiet market kill switch — no edge in flat market
            elif regime_name == 'QUIET' and adx < QUIET_MARKET_ADX_THRESHOLD:
                blocked = f"QUIET_MARKET (ADX={adx:.1f}<{QUIET_MARKET_ADX_THRESHOLD})"
            elif confidence < effective_conf_thresh:
                blocked = f"conf {confidence:.2%}<{effective_conf_thresh:.0%}"
            elif adx < adjusted_adx_thresh:
                blocked = f"ADX {adx:.1f}<{adjusted_adx_thresh}"
            elif not can_trade:
                blocked = "position limit"
            elif not news_ok:
                blocked = f"NEWS: {news_msg}"

            if not blocked:
                rg_ok, rg_reasons = self.risk_guard.can_trade(symbol, signal)
                if not rg_ok:
                    blocked = " | ".join(rg_reasons)

            if not blocked and sym_require_htf:
                if (signal == 1 and htf == -1) or (signal == -1 and htf == 1):
                    blocked = "H4 against"

            if not blocked and sym_require_pb:
                if (signal == 1 and rsi > sym_rsi_buy_max) or (signal == -1 and rsi < sym_rsi_sell_min):
                    blocked = "RSI pullback"

            # ---- Near Support/Resistance check ----
            # SELL ใกล้แนวรับ หรือ BUY ใกล้แนวต้าน → risk/reward แย่มาก
            if not blocked:
                if signal == -1 and dist_to_support < NEAR_SR_ATR:
                    blocked = f"Near support ({dist_to_support:.1f} ATR < {NEAR_SR_ATR})"
                elif signal == 1 and dist_to_resist < NEAR_SR_ATR:
                    blocked = f"Near resist ({dist_to_resist:.1f} ATR < {NEAR_SR_ATR})"

            if not blocked:
                sess_ok, sess_reason = self._check_session_filter(symbol)
                if not sess_ok:
                    blocked = f"Session: {sess_reason}"

            if not blocked:
                time_ok, time_reason = self.time_filter.can_trade(symbol)
                if not time_ok:
                    blocked = f"Time: {time_reason}"

            if not blocked:
                streak_ok, streak_reason = self.loss_streak.can_trade()
                if not streak_ok:
                    blocked = f"Streak: {streak_reason}"

            m30_reason = ""
            m30_confirmed = True
            if not blocked and sym_use_m30:
                m30_confirmed, m30_adj, m30_reason, m30_data = self.m30_analyzer.check_confirmation(symbol, signal)
                confidence += m30_adj
                confidence = np.clip(confidence, 0.0, 1.0)
                if 'confirm' in m30_reason:
                    self.m30_stats['confirm'] += 1
                elif 'against' in m30_reason:
                    self.m30_stats['against'] += 1
                else:
                    self.m30_stats['neutral'] += 1
                if not m30_confirmed:
                    blocked = f"M30: {m30_reason}"
                elif confidence < effective_conf_thresh:
                    blocked = f"M30 adj conf {confidence:.2%}"
                if m30_data:
                    m30_sig = "BUY" if m30_data['signal'] == 1 else ("SELL" if m30_data['signal'] == -1 else "HOLD")
                    m30_tr = "BULL" if m30_data['trend'] == 1 else ("BEAR" if m30_data['trend'] == -1 else "FLAT")
                    logger.info(f"  │ M30: {m30_sig} trend={m30_tr} RSI={m30_data['rsi']:.1f} -> {m30_reason} ({m30_adj:+.2f})")
            elif not blocked:
                self.m30_stats['disabled'] += 1

            quality_score = 0
            quality_grade = 'D'
            quality_icon = '🔴'
            if not blocked:
                quality_score, q_details = self.quality_scorer.calculate(
                    signal, confidence, ict_score, adx, rsi, htf,
                    m30_confirmed, structure, vol_spike
                )
                quality_grade, quality_icon = self.quality_scorer.get_grade(quality_score)
                self.quality_stats[quality_grade] = self.quality_stats.get(quality_grade, 0) + 1

                if quality_score < MIN_QUALITY_SCORE:
                    blocked = f"Quality {quality_icon} {quality_grade} ({quality_score}<{MIN_QUALITY_SCORE})"
                    self.quality_stats['blocked'] += 1

                logger.info(f"  │ Quality: {quality_icon} {quality_grade} ({quality_score}/100) min={MIN_QUALITY_SCORE}")

            logger.info(f"  │ Final: {signal_name} ({confidence:.2%})")

            sig_data.update({
                'signal': signal_name,
                'confidence': round(float(confidence), 4),
                'base_confidence': round(float(base_conf), 4),
                'blocked': blocked or '',
                'regime': regime_name,
                'quality_grade': quality_grade,
                'quality_score': quality_score,
                'm30': m30_reason,
                'rl_action': rl_name,
                'source': rl_src,
                'entry_strategy': entry_strategy,
                'strategy_confidence': round(strategy_conf, 4),
                'strategy_reason': strategy_reason,
                'raw_base_signal': 'BUY' if raw_base_signal == 1 else ('SELL' if raw_base_signal == -1 else 'HOLD'),
                'raw_base_confidence': round(raw_base_conf, 4),
                'spread': round(float(cur_spread), 2),
                'avg_spread': round(float(avg_spread), 2),
                'adx': round(float(adx), 2),
                'rsi': round(float(rsi), 2),
                'updated': datetime.now().strftime('%H:%M:%S'),
            })

            if blocked:
                logger.info(f"  │ >> NO TRADE: {blocked}")
                add_log(f"{symbol}: {blocked} [{regime_name}]")
                self.hub.on_hold(symbol, had_signal=base_signal != 0,
                                 market_data=sym_data, signal_data=sig_data)
            else:
                logger.info(f"  │ >> TRADE! {quality_icon}{quality_grade} [{regime_name}]")

            logger.info(f"  └──────────────────────────────────────────────")

            if blocked:
                return

            sp = SYMBOL_POINTS.get(symbol, 0.0001)
            sl, tp = self.position_sizer.calculate_sl_tp(price, signal, atr, sl_mult, tp_mult, symbol=symbol)
            if sl is None:
                return

            lot = self.position_sizer.calculate_position_size(
                ai['equity'], atr, sp, confidence, symbol=symbol,
                sl_multiplier=sl_mult
            )
            if lot <= 0:
                return

            ot = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
            planned_rr = tp_mult / sl_mult

            logger.info(
                f"  [{symbol}] ORDER_REQUEST: {signal_name} {lot}lots signal_price={price:.5f} "
                f"SL={sl:.5f} TP={tp:.5f} planned_R:R=1:{planned_rr:.1f}; "
                f"execution RR uses broker bid/ask in OrderManager [{regime_name}]"
            )

            max_slippage = MAX_SLIPPAGE_POINTS.get(symbol, DEFAULT_MAX_SLIPPAGE_POINTS)
            ticket = self.order_manager.place_order(
                symbol, ot, lot, sl, tp, f"v71_{signal_name}_{quality_grade}_{regime_name[:3]}",
                reference_price=price, max_slippage_points=max_slippage
            )

            if ticket:
                self.daily_trades += 1
                self.active_trades[ticket] = {
                    'symbol': symbol, 'signal': signal_name, 'price': price,
                    'sl': sl, 'tp': tp, 'lots': lot, 'confidence': confidence,
                    'time': datetime.now(), 'quality': quality_score, 'grade': quality_grade,
                    'regime': regime_name,
                    'exit_policy': exit_policy,
                    'risk_mult': exit_policy.get('risk_mult', 1.0),
                }
                self.tracker.log_trade(ticket, symbol, signal_name, price, lot)
                self.hub.on_trade_open(symbol, action=rl_act,
                                       market_data=sym_data, signal_data=sig_data)
                self.time_filter.record_trade(symbol)

                add_log(f"[TRADE] #{ticket} {signal_name} {symbol} {lot}lots {quality_icon}{quality_grade} [{regime_name}]")

                try:
                    growth = (ai['equity'] - self.start_balance) / self.start_balance * 100
                    self.telegram.send_message(
                        f"<b>Trade {quality_icon}{quality_grade}</b>\n"
                        f"{signal_name} {symbol} | {lot} lots\n"
                        f"Price: {price:.5f}\n"
                        f"SL: {sl:.5f} | TP: {tp:.5f}\n"
                        f"Conf: {confidence:.1%} | Planned R:R=1:{planned_rr:.1f}\n"
                        f"Quality: {quality_score}/100 | ICT:{ict_score}\n"
                        f"Regime: {regime_name} | M30: {m30_reason}\n"
                        f"Equity: ${ai['equity']:.2f} ({growth:+.1f}%)"
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"[ERROR] {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def live_trading(self):
        logger.info(f"STARTING {BOT_VERSION} {BOT_MODE} + Deep RL | Port: {DASHBOARD_PORT}")
        logger.info(f"Dashboard: http://localhost:{DASHBOARD_PORT}")
        logger.info(f"Trailing Thread: 5s | Main Loop: {UPDATE_INTERVAL}s")
        logger.info("=" * 80)

        all_symbols = SYMBOLS['FOREX'] + SYMBOLS.get('CRYPTO', []) + SYMBOLS.get('GOLD', [])

        if not self.ml_model.is_trained:
            self._auto_train()

        self.prev_positions = {p.ticket: p for p in (self.order_manager.get_open_positions() or [])}

        # เริ่ม trailing thread แยก — เช็คทุก 5 วินาที
        self._start_trailing_thread()

        try:
            while True:
                self.iteration += 1
                update_dashboard('iteration', self.iteration)
                update_dashboard('bot_status', 'RUNNING')

                if self.iteration % max(1, HEALTH_CHECK_INTERVAL // UPDATE_INTERVAL) == 0:
                    if not self._check_mt5_health():
                        break

                today = datetime.now().date()
                if self.current_day != today:
                    if self.current_day is not None:
                        logger.info(f"[DAILY] Yesterday: PnL=${self.daily_pnl:.2f} Trades={self.daily_trades}")
                    self.daily_pnl = 0.0
                    self.daily_trades = 0
                    self.current_day = today

                ai = self.mt5.get_account_info()
                growth = (ai['equity'] - self.start_balance) / self.start_balance * 100
                at = self.adaptive_threshold.get_stats()
                ls = self.loss_streak.get_stats()
                rl = self.rl_agent.get_stats()

                logger.info("")
                logger.info("=" * 80)
                logger.info(f"[ITER {self.iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {BOT_MODE}  Sess: {','.join(self._get_session_name())}")
                logger.info(f"  Equity: ${ai['equity']:.2f} ({growth:+.1f}%) | Daily: ${self.daily_pnl:.2f} T:{self.daily_trades}")
                q_value = rl.get('avg_q_value')
                q_text = "N/A" if q_value is None else f"{q_value:.3f}"
                logger.info(
                    "  DeepRL: steps=%s buf=%s loss=%.6f Q=%s power=%.0f%%",
                    rl.get('train_steps', 0),
                    rl.get('buffer_size', 0),
                    float(rl.get('avg_loss') or 0.0),
                    q_text,
                    float(rl.get('rl_power', 0) or 0.0) * 100,
                )
                logger.info(f"  Regimes: {rl.get('regimes', {})}")
                logger.info(f"  Adaptive: thresh={at['current']:.0%} WR={at['recent_wr']:.0%} | Streak: {ls['current_streak']} {'COOLDOWN' if ls['in_cooldown'] else ''}")
                logger.info(f"  Quality: A+:{self.quality_stats.get('A+',0)} A:{self.quality_stats.get('A',0)} B:{self.quality_stats.get('B',0)} blocked:{self.quality_stats.get('blocked',0)} (min={MIN_QUALITY_SCORE})")

                pa = self.perf_adjuster.get_stats()
                if pa.get('adjustments'):
                    logger.info(f"  AutoAdj: {pa['adjustments'].get('reason', '')}")

                self.risk_guard.update()
                rg = self.risk_guard.get_status()
                if rg['recovery_mode']:
                    logger.info(f"  RECOVERY (DD:{rg['drawdown_pct']:.1f}%)")
                logger.info("=" * 80)

                self.news_filter.refresh_if_needed()
                self._detect_closed_trades()
                self._apply_news_protection_all()
                # trailing ย้ายไป thread แล้ว — ไม่ต้องเรียกที่นี่
                self._apply_partial_closes()
                self._update_dashboard_all()

                for symbol in all_symbols:
                    self.run_strategy(symbol, PRIMARY_TIMEFRAME)

                if self.iteration % 10 == 0:
                    logger.info(f"[DeepRL] trades={rl['total_trades']} WR={rl['win_rate']:.1%} reward={rl['total_reward']:.1f}")
                    sym_perf = rl.get('symbol_performance', {})
                    for sym, sp in sym_perf.items():
                        logger.info(f"  {sym}: {sp['trades']}T WR={sp['win_rate']:.0%} PnL=${sp['total_pnl']:.2f}")
                    self.monitor.print_status()
                    self.expectancy_tracker.report()

                if self.iteration % 50 == 0:
                    self.hub.save()  # save ทุก component ในครั้งเดียว

                if self.iteration % RETRAIN_INTERVAL == 0:
                    self._auto_train()

                logger.info(f"[INFO] Waiting {UPDATE_INTERVAL}s...")
                time.sleep(UPDATE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n[STOP] Stopped by user")
        except Exception as e:
            logger.error(f"[FATAL] {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self._shutdown()

    def _shutdown(self):
        self._stop_trailing_thread()
        update_dashboard('bot_status', 'STOPPED')
        try:
            ai = self.mt5.get_account_info()
            growth = (ai['equity'] - self.start_balance) / self.start_balance * 100

            self.hub.save()  # save ทุก component — ensemble + RL + temporal + meta

            rl = self.rl_agent.get_stats()

            logger.info(f"\n{'='*60}")
            logger.info(f"  SHUTDOWN {BOT_VERSION} {BOT_MODE} + Deep RL | Port: {DASHBOARD_PORT}")
            logger.info(f"{'='*60}")
            logger.info(f"  Start:      ${self.start_balance:.2f}")
            logger.info(f"  Final:      ${ai['equity']:.2f} ({growth:+.1f}%)")
            logger.info(f"  Daily:      ${self.daily_pnl:.2f} | {self.daily_trades} trades")
            logger.info(f"  DeepRL:     {rl['total_trades']}T WR={rl['win_rate']:.1%} steps={rl['train_steps']} power={rl.get('rl_power',0):.0%}")
            logger.info(f"  Quality:    A+:{self.quality_stats.get('A+',0)} A:{self.quality_stats.get('A',0)} B:{self.quality_stats.get('B',0)} blocked:{self.quality_stats.get('blocked',0)}")
            logger.info(f"  Regimes:    {rl.get('regimes', {})}")

            tracker = self.tracker.get_stats()
            if tracker:
                logger.info(f"  Performance: {tracker['total_trades']}T WR={tracker['win_rate']:.1%} PnL=${tracker['total_pnl']:.2f} PF={tracker['profit_factor']:.2f}")

            logger.info(f"{'='*60}")
            self.mt5.disconnect()
            logger.info("[OK] Shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.live_trading()
    except Exception as e:
        logger.error(f"[FATAL] {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)