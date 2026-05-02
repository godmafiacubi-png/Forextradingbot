import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class RiskGuard:
    """
    Master risk guard — combines all safety filters:
    1. Correlation Filter
    2. Spread Filter
    3. Session Filter
    4. Daily P/L Limit
    5. Partial Close
    6. Drawdown Recovery
    7. Consecutive Loss Cooldown
    """

    def __init__(self, mt5_connector, config):
        self.mt5 = mt5_connector
        self.cfg = config

        # Daily tracking
        self.daily_start_balance = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.last_reset_date = None

        # Consecutive loss tracking
        self.consecutive_losses = 0
        self.cooldown_until = None

        # Drawdown
        self.peak_balance = 0
        self.recovery_mode = False

        # Spread history
        self.spread_history = defaultdict(list)

        # Partial close tracking
        self.partial_close_done = {}  # {ticket: {'stage1': bool, 'stage2': bool}}

        # Initialize
        self._reset_daily()

    def _reset_daily(self):
        """Reset daily counters at midnight"""
        today = datetime.now().date()
        if self.last_reset_date != today:
            try:
                ai = self.mt5.get_account_info()
                self.daily_start_balance = ai.get('balance', 0)
            except Exception:
                pass
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self.last_reset_date = today
            logger.info(f"[RISK] Daily reset — start balance: ${self.daily_start_balance:.2f}")

    def update(self):
        """Call every iteration to update state"""
        self._reset_daily()
        self._update_drawdown()
        self._update_daily_pnl()

    def _update_drawdown(self):
        """Track drawdown and trigger recovery mode"""
        try:
            ai = self.mt5.get_account_info()
            equity = ai.get('equity', 0)
            balance = ai.get('balance', 0)

            if balance > self.peak_balance:
                self.peak_balance = balance

            if self.peak_balance > 0:
                dd_pct = (self.peak_balance - equity) / self.peak_balance * 100

                if dd_pct >= self.cfg.get('RECOVERY_DRAWDOWN_TRIGGER', 5.0):
                    if not self.recovery_mode:
                        self.recovery_mode = True
                        logger.warning(f"[RISK] RECOVERY MODE ON — drawdown {dd_pct:.1f}%")
                elif dd_pct < self.cfg.get('RECOVERY_DRAWDOWN_TRIGGER', 5.0) * 0.5:
                    if self.recovery_mode:
                        self.recovery_mode = False
                        logger.info(f"[RISK] Recovery mode OFF — drawdown {dd_pct:.1f}%")
        except Exception:
            pass

    def _update_daily_pnl(self):
        """Update daily P/L"""
        try:
            ai = self.mt5.get_account_info()
            current_balance = ai.get('balance', 0)
            if self.daily_start_balance > 0:
                self.daily_pnl = current_balance - self.daily_start_balance
        except Exception:
            pass

    def record_trade_result(self, pnl):
        """Record a closed trade result"""
        self.daily_trades += 1
        self.daily_pnl += pnl

        if pnl >= 0:
            self.daily_wins += 1
            self.consecutive_losses = 0
            self.cooldown_until = None
        else:
            self.daily_losses += 1
            self.consecutive_losses += 1

            max_consec = self.cfg.get('CONSECUTIVE_LOSS_COOLDOWN', 2)
            if self.consecutive_losses >= max_consec:
                cooldown_min = self.cfg.get('COOLDOWN_MINUTES', 120)
                self.cooldown_until = datetime.now() + timedelta(minutes=cooldown_min)
                logger.warning(
                    f"[RISK] {self.consecutive_losses} consecutive losses — "
                    f"cooldown until {self.cooldown_until.strftime('%H:%M')}"
                )

    # ================================================================
    # 1. CORRELATION FILTER
    # ================================================================
    def check_correlation(self, symbol, signal):
        """
        Check if opening this trade would create too much correlated exposure.
        Returns: (allowed, reason)
        """
        if not self.cfg.get('CORRELATION_FILTER_ENABLED', True):
            return True, ''

        try:
            positions = mt5.positions_get()
            if not positions:
                return True, ''

            # Check correlation groups
            corr_groups = self.cfg.get('CORRELATION_GROUPS', {})
            for group_name, group_symbols in corr_groups.items():
                if symbol in group_symbols:
                    same_dir_count = 0
                    for pos in positions:
                        if pos.symbol in group_symbols:
                            pos_dir = 1 if pos.type == 0 else -1
                            if pos_dir == signal:
                                same_dir_count += 1

                    max_same = self.cfg.get('MAX_SAME_DIRECTION_CORRELATED', 1)
                    if same_dir_count >= max_same:
                        return False, f"Correlated: {group_name} already has {same_dir_count} same-direction"

            # Check currency exposure
            sym_currencies = self.cfg.get('SYMBOL_CURRENCIES', {})
            target_currencies = sym_currencies.get(symbol, [])

            currency_count = defaultdict(int)
            for pos in positions:
                pos_currencies = sym_currencies.get(pos.symbol, [])
                for cur in pos_currencies:
                    currency_count[cur] += 1

            max_exposure = self.cfg.get('MAX_CURRENCY_EXPOSURE', 2)
            for cur in target_currencies:
                if currency_count[cur] >= max_exposure:
                    return False, f"Currency exposure: {cur} already has {currency_count[cur]} positions"

            return True, ''

        except Exception as e:
            logger.debug(f"Correlation check error: {e}")
            return True, ''

    # ================================================================
    # 2. SPREAD FILTER
    # ================================================================
    def check_spread(self, symbol):
        """
        Check if current spread is acceptable.
        Returns: (allowed, current_spread, avg_spread)
        """
        try:
            si = self.mt5.get_symbol_info(symbol)
            if si is None:
                return True, 0, 0

            current_spread = si.get('spread', 0)
            ask = si.get('ask', 0)
            bid = si.get('bid', 0)
            if ask > 0 and bid > 0:
                current_spread_price = ask - bid
            else:
                current_spread_price = 0

            # Track spread history
            self.spread_history[symbol].append(current_spread)
            if len(self.spread_history[symbol]) > self.cfg.get('SPREAD_AVG_PERIOD', 50):
                self.spread_history[symbol] = self.spread_history[symbol][-50:]

            # Calculate average
            history = self.spread_history[symbol]
            if len(history) < 5:
                return True, current_spread, current_spread

            avg_spread = sum(history) / len(history)

            max_mult = self.cfg.get('MAX_SPREAD_MULTIPLIER', 3.0)
            if avg_spread > 0 and current_spread > avg_spread * max_mult:
                return False, current_spread, avg_spread

            return True, current_spread, avg_spread

        except Exception as e:
            logger.debug(f"Spread check error: {e}")
            return True, 0, 0

    # ================================================================
    # 3. SESSION FILTER
    # ================================================================
    def check_session(self, symbol):
        """
        Check if current session is good for this symbol.
        Returns: (allowed, current_session, reason)
        """
        if not self.cfg.get('SESSION_FILTER_ENABLED', True):
            return True, 'ANY', ''

        try:
            hour = datetime.utcnow().hour
            current_sessions = []
            if 0 <= hour < 8:
                current_sessions.append('ASIAN')
            if 7 <= hour < 16:
                current_sessions.append('LONDON')
            if 13 <= hour < 22:
                current_sessions.append('NY')
            if 13 <= hour < 16:
                current_sessions.append('OVERLAP')

            session_str = ', '.join(current_sessions) if current_sessions else 'OFF_HOURS'

            best_sessions = self.cfg.get('SYMBOL_BEST_SESSIONS', {}).get(symbol, [])
            if not best_sessions:
                return True, session_str, ''

            for sess in current_sessions:
                if sess in best_sessions:
                    return True, session_str, ''

            return False, session_str, f"{symbol} best in {best_sessions}, now={session_str}"

        except Exception:
            return True, 'UNKNOWN', ''

    # ================================================================
    # 4. DAILY P/L LIMIT
    # ================================================================
    def check_daily_limit(self):
        """
        Check daily loss/profit limits.
        Returns: (allowed, reason)
        """
        try:
            if self.daily_start_balance <= 0:
                return True, ''

            daily_pnl_pct = (self.daily_pnl / self.daily_start_balance) * 100

            # Daily loss limit
            loss_limit = self.cfg.get('DAILY_LOSS_LIMIT_PCT', 3.0)
            if daily_pnl_pct <= -loss_limit:
                return False, f"Daily loss limit: {daily_pnl_pct:.1f}% (max -{loss_limit}%)"

            return True, ''
        except Exception:
            return True, ''

    # ================================================================
    # 5. CONSECUTIVE LOSS COOLDOWN
    # ================================================================
    def check_cooldown(self):
        """
        Check if we're in cooldown after consecutive losses.
        Returns: (allowed, reason)
        """
        if self.cooldown_until is not None:
            if datetime.now() < self.cooldown_until:
                remaining = (self.cooldown_until - datetime.now()).total_seconds() / 60
                return False, f"Cooldown: {remaining:.0f}min left ({self.consecutive_losses} losses)"
            else:
                self.cooldown_until = None
                self.consecutive_losses = 0
                logger.info("[RISK] Cooldown ended — resuming trading")

        return True, ''

    # ================================================================
    # 6. DRAWDOWN RECOVERY
    # ================================================================
    def get_risk_adjusted_params(self):
        """
        Get risk parameters adjusted for current state.
        Returns: (risk_pct, max_trades)
        """
        base_risk = self.cfg.get('ACCOUNT_RISK_PERCENT', 1.0)
        base_max = self.cfg.get('MAX_OPEN_TRADES', 3)

        if self.recovery_mode:
            risk = self.cfg.get('RECOVERY_RISK_PERCENT', 0.5)
            max_trades = self.cfg.get('RECOVERY_MAX_TRADES', 1)
            return risk, max_trades

        # Reduce risk after daily profit target
        if self.daily_start_balance > 0:
            daily_pct = (self.daily_pnl / self.daily_start_balance) * 100
            profit_target = self.cfg.get('DAILY_PROFIT_TARGET_PCT', 5.0)
            if daily_pct >= profit_target:
                return base_risk * 0.5, max(base_max - 1, 1)

        return base_risk, base_max

    # ================================================================
    # 7. PARTIAL CLOSE
    # ================================================================
    def check_partial_close(self, positions, atr_values):
        """
        Check if any positions should be partially closed.
        Returns: list of (ticket, close_pct, reason)
        """
        if not self.cfg.get('PARTIAL_CLOSE_ENABLED', True):
            return []

        actions = []

        for pos in positions:
            symbol = pos.symbol
            ticket = pos.ticket
            atr = atr_values.get(symbol, 0)
            if atr <= 0:
                continue

            try:
                si = self.mt5.get_symbol_info(symbol)
                if si is None:
                    continue

                current_price = si['bid'] if pos.type == 0 else si['ask']
                entry = pos.price_open
                profit_dist = (current_price - entry) if pos.type == 0 else (entry - current_price)
                profit_in_atr = profit_dist / atr

                if ticket not in self.partial_close_done:
                    self.partial_close_done[ticket] = {'stage1': False, 'stage2': False}

                state = self.partial_close_done[ticket]

                # Stage 1: profit >= 1x ATR → close 50%
                stage1_atr = self.cfg.get('PARTIAL_CLOSE_1_ATR', 1.0)
                stage1_pct = self.cfg.get('PARTIAL_CLOSE_1_PCT', 0.50)
                if profit_in_atr >= stage1_atr and not state['stage1']:
                    actions.append((ticket, stage1_pct, f"Stage1: {profit_in_atr:.1f}x ATR profit"))
                    state['stage1'] = True

                # Stage 2: profit >= 2x ATR → close 25% (of original, 50% of remaining)
                stage2_atr = self.cfg.get('PARTIAL_CLOSE_2_ATR', 2.0)
                stage2_pct = self.cfg.get('PARTIAL_CLOSE_2_PCT', 0.50)  # 50% of remaining
                if profit_in_atr >= stage2_atr and state['stage1'] and not state['stage2']:
                    actions.append((ticket, stage2_pct, f"Stage2: {profit_in_atr:.1f}x ATR profit"))
                    state['stage2'] = True

            except Exception as e:
                logger.debug(f"Partial close check error for #{ticket}: {e}")

        return actions

    def cleanup_partial_tracking(self, open_tickets):
        """Remove tracking for closed positions"""
        closed = [t for t in self.partial_close_done if t not in open_tickets]
        for t in closed:
            del self.partial_close_done[t]

    # ================================================================
    # MASTER CHECK
    # ================================================================
    def can_trade(self, symbol, signal):
        """
        Master check — run ALL filters.
        Returns: (allowed, reasons_list)
        """
        reasons = []

        # Daily limit
        ok, reason = self.check_daily_limit()
        if not ok:
            reasons.append(reason)

        # Cooldown
        ok, reason = self.check_cooldown()
        if not ok:
            reasons.append(reason)

        # Correlation
        ok, reason = self.check_correlation(symbol, signal)
        if not ok:
            reasons.append(reason)

        # Spread
        ok, spread, avg = self.check_spread(symbol)
        if not ok:
            reasons.append(f"Spread too wide: {spread} vs avg {avg:.0f}")

        # Session
        ok, session, reason = self.check_session(symbol)
        if not ok:
            reasons.append(reason)

        # Max drawdown
        if self.recovery_mode:
            positions = mt5.positions_get()
            max_recovery = self.cfg.get('RECOVERY_MAX_TRADES', 1)
            if positions and len(positions) >= max_recovery:
                reasons.append(f"Recovery mode: max {max_recovery} trades")

        allowed = len(reasons) == 0
        return allowed, reasons

    def get_status(self):
        """Get status for dashboard"""
        try:
            ai = self.mt5.get_account_info()
            equity = ai.get('equity', 0)
            dd = (self.peak_balance - equity) / self.peak_balance * 100 if self.peak_balance > 0 else 0
        except Exception:
            dd = 0

        daily_pct = (self.daily_pnl / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0
        risk_pct, max_trades = self.get_risk_adjusted_params()

        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': daily_pct,
            'daily_trades': self.daily_trades,
            'daily_wins': self.daily_wins,
            'daily_losses': self.daily_losses,
            'consecutive_losses': self.consecutive_losses,
            'cooldown_until': self.cooldown_until.strftime('%H:%M') if self.cooldown_until else None,
            'recovery_mode': self.recovery_mode,
            'drawdown_pct': dd,
            'peak_balance': self.peak_balance,
            'current_risk_pct': risk_pct,
            'current_max_trades': max_trades,
        }