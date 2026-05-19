from decimal import Decimal, ROUND_HALF_UP
import logging
import time

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class OrderManager:
    """Execute and manage orders — trailing + news protection + partial close."""

    def __init__(self, mt5_connector, max_open_trades=3, max_per_symbol=1,
                 dry_run=True, magic=123456, deviation=20, trade_journal=None,
                 slippage_cooldown_seconds=None):
        self.mt5 = mt5_connector
        self.max_open_trades = max_open_trades
        self.max_per_symbol = max_per_symbol
        self.dry_run = dry_run
        self.magic = magic
        self.deviation = deviation
        self.trade_journal = trade_journal
        if slippage_cooldown_seconds is None:
            try:
                from config.settings import SLIPPAGE_REJECTION_COOLDOWN_SECONDS
                slippage_cooldown_seconds = SLIPPAGE_REJECTION_COOLDOWN_SECONDS
            except Exception:
                slippage_cooldown_seconds = 600
        self.slippage_cooldown_seconds = max(0, float(slippage_cooldown_seconds))
        self._slippage_cooldowns = {}
        self._partial_close_stages = {}

    def _journal_event(self, method_name, *args, **kwargs):
        kwargs.setdefault("source", "order_manager")
        if self.trade_journal is None:
            return None
        try:
            return getattr(self.trade_journal, method_name)(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"Trade journal write failed: {exc}")
            return None

    @staticmethod
    def _side_label(order_type):
        return "BUY" if OrderManager._is_buy(order_type) else "SELL"

    @staticmethod
    def _position_side(pos):
        return "BUY" if getattr(pos, "type", None) == 0 else "SELL"

    @staticmethod
    def _cooldown_key(symbol, side):
        return (str(symbol).upper(), str(side).upper())

    @staticmethod
    def _calculate_rr(side, request_price, sl, tp):
        if str(side).upper() == "BUY":
            risk = request_price - sl
            reward = tp - request_price
        else:
            risk = sl - request_price
            reward = request_price - tp
        rr = reward / risk if risk > 0 and reward > 0 else None
        return risk, reward, rr

    @staticmethod
    def _with_execution_context(comment, signal_price, execution_price, rr=None):
        parts = []
        if comment:
            parts.append(str(comment))
        if signal_price is not None:
            parts.append(f"signal_price={signal_price}")
        parts.append(f"execution_price={execution_price}")
        if rr is not None:
            parts.append(f"rr={rr:.2f}")
        return " | ".join(parts)

    def _slippage_cooldown_remaining(self, symbol, side):
        expires_at = self._slippage_cooldowns.get(self._cooldown_key(symbol, side))
        if expires_at is None:
            return 0
        remaining = expires_at - time.time()
        if remaining <= 0:
            self._slippage_cooldowns.pop(self._cooldown_key(symbol, side), None)
            return 0
        return remaining

    def _start_slippage_cooldown(self, symbol, side):
        if self.slippage_cooldown_seconds <= 0:
            return
        self._slippage_cooldowns[self._cooldown_key(symbol, side)] = time.time() + self.slippage_cooldown_seconds

    def set_limits(self, max_open_trades, max_per_symbol=None):
        self.max_open_trades = max_open_trades
        if max_per_symbol is not None:
            self.max_per_symbol = max_per_symbol

    def _get_digits(self, symbol):
        """ดึง digits ของ symbol สำหรับ round ราคา"""
        try:
            info = mt5.symbol_info(symbol)
            if info:
                return info.digits
        except Exception:
            pass
        symbol_upper = symbol.upper()
        if 'JPY' in symbol_upper:
            return 3
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            return 3
        if 'BTC' in symbol_upper:
            return 2
        return 5

    def _round_price(self, price, symbol):
        """Round ราคาตาม digits ของ symbol"""
        return round(price, self._get_digits(symbol))

    @staticmethod
    def _round_volume(volume, step):
        """Round volume to broker step without assuming two decimal places."""
        if step <= 0:
            step = 0.01
        volume_dec = Decimal(str(volume))
        step_dec = Decimal(str(step))
        steps = (volume_dec / step_dec).to_integral_value(rounding=ROUND_HALF_UP)
        rounded = steps * step_dec
        decimals = max(0, -step_dec.as_tuple().exponent)
        return float(round(rounded, decimals))

    @staticmethod
    def _is_buy(order_type):
        return order_type == mt5.ORDER_TYPE_BUY or order_type == 0

    def _validate_sl_tp_side(self, symbol, price, sl, tp, order_type):
        """Reject SL/TP that are on the wrong side before any broker stop-level adjustment."""
        if self._is_buy(order_type):
            if sl <= 0 or tp <= 0 or sl >= price or tp <= price:
                logger.error(f"Invalid BUY SL/TP for {symbol}: price={price} sl={sl} tp={tp}")
                return False
        else:
            if sl <= 0 or tp <= 0 or sl <= price or tp >= price:
                logger.error(f"Invalid SELL SL/TP for {symbol}: price={price} sl={sl} tp={tp}")
                return False
        return True

    def _check_stop_level(self, symbol, price, sl, tp, order_type):
        """เช็คว่า SL/TP อยู่ห่างจากราคาพอตาม broker stop level"""
        try:
            info = mt5.symbol_info(symbol)
            if not info:
                return sl, tp

            stop_level = getattr(info, 'trade_stops_level', 0)
            point = getattr(info, 'point', 0)
            if stop_level <= 0 or point <= 0:
                return sl, tp

            min_distance = stop_level * point
            if self._is_buy(order_type):
                if sl > 0 and (price - sl) < min_distance:
                    sl = price - min_distance
                    logger.debug(f"[STOP_LEVEL] {symbol} BUY SL adjusted to {sl:.5f} (min_dist={min_distance:.5f})")
                if tp > 0 and (tp - price) < min_distance:
                    tp = price + min_distance
            else:
                if sl > 0 and (sl - price) < min_distance:
                    sl = price + min_distance
                    logger.debug(f"[STOP_LEVEL] {symbol} SELL SL adjusted to {sl:.5f}")
                if tp > 0 and (price - tp) < min_distance:
                    tp = price - min_distance

            digits = getattr(info, 'digits', self._get_digits(symbol))
            return round(sl, digits), round(tp, digits)
        except Exception:
            return sl, tp

    def can_open_trade(self, symbol):
        try:
            positions = mt5.positions_get()
            if positions is None:
                return True
            if len(positions) >= self.max_open_trades:
                return False
            if len([p for p in positions if p.symbol == symbol]) >= self.max_per_symbol:
                return False
            return True
        except Exception:
            return False

    def get_open_positions(self, symbol=None):
        try:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            return positions if positions else []
        except Exception:
            return []

    def _send_order(self, request, dry_run_message):
        if self.dry_run:
            logger.info(f"[DRY_RUN] {dry_run_message}: {request}")
            return None
        return mt5.order_send(request)

    def place_order(self, symbol, order_type, volume, stop_loss, take_profit, comment="",
                    reference_price=None, max_slippage_points=None, diagnostics=None):
        try:
            side = self._side_label(order_type)
            cooldown_remaining = self._slippage_cooldown_remaining(symbol, side)
            if cooldown_remaining > 0:
                message = "slippage cooldown"
                logger.warning(f"[SKIP] {symbol} {side} {message} ({cooldown_remaining:.0f}s remaining)")
                self._journal_event(
                    "log_order_rejected", symbol, side=side, volume=volume, sl=stop_loss, tp=take_profit,
                    reason=message, comment=message, source="order_manager",
                )
                return None

            if not self.can_open_trade(symbol):
                logger.info(f"[SKIP] {symbol} position limit reached")
                self._journal_event(
                    "log_risk_blocked", symbol, side=side, volume=volume, sl=stop_loss, tp=take_profit,
                    comment="position limit reached",
                )
                return None

            si = self.mt5.get_symbol_info(symbol)
            if si is None:
                self._journal_event(
                    "log_order_failed", symbol, side=side, volume=volume, sl=stop_loss, tp=take_profit,
                    comment="symbol info unavailable",
                )
                return None

            price = si['ask'] if order_type == mt5.ORDER_TYPE_BUY else si['bid']
            point = si.get('point', 0)
            if reference_price is not None and max_slippage_points is not None and point > 0:
                slippage_points = abs(price - reference_price) / point
                if slippage_points > max_slippage_points:
                    message = (
                        f"slippage guard: {slippage_points:.1f}pts > {max_slippage_points}pts "
                        f"(ref={reference_price}, exec={price})"
                    )
                    logger.warning(f"[SKIP] {symbol} {message}")
                    self._start_slippage_cooldown(symbol, side)
                    self._journal_event(
                        "log_order_rejected", symbol, side=side, volume=volume, price=price,
                        sl=stop_loss, tp=take_profit, comment=message, source="order_manager",
                    )
                    return None
            digits = self._get_digits(symbol)
            stop_loss = round(stop_loss, digits)
            take_profit = round(take_profit, digits)
            if not self._validate_sl_tp_side(symbol, price, stop_loss, take_profit, order_type):
                self._journal_event(
                    "log_order_rejected", symbol, side=side, volume=volume, price=price,
                    sl=stop_loss, tp=take_profit, comment="invalid SL/TP side",
                )
                return None

            stop_loss, take_profit = self._check_stop_level(symbol, price, stop_loss, take_profit, order_type)
            if not self._validate_sl_tp_side(symbol, price, stop_loss, take_profit, order_type):
                self._journal_event(
                    "log_order_rejected", symbol, side=side, volume=volume, price=price,
                    sl=stop_loss, tp=take_profit, comment="invalid SL/TP after stop-level adjustment",
                )
                return None

            risk, reward, rr = self._calculate_rr(side, price, stop_loss, take_profit)
            execution_comment = self._with_execution_context(comment, reference_price, price, rr)
            if risk <= 0 or reward <= 0:
                message = (
                    f"invalid execution RR: risk={risk:.10f} reward={reward:.10f} "
                    f"signal_price={reference_price} execution_price={price}"
                )
                logger.error(f"[SKIP] {symbol} {side} {message}")
                self._journal_event(
                    "log_order_rejected", symbol, side=side, volume=volume, price=price,
                    sl=stop_loss, tp=take_profit, comment=message, source="order_manager",
                )
                return None

            vol_min = si.get('volume_min', 0.01)
            vol_max = si.get('volume_max', 100.0)
            vol_step = si.get('volume_step', 0.01)
            volume = max(min(volume, vol_max), vol_min)
            volume = self._round_volume(volume, vol_step)
            if volume < vol_min or volume > vol_max:
                message = f"Invalid volume after rounding: {volume} not in [{vol_min}, {vol_max}]"
                logger.error(message)
                self._journal_event(
                    "log_order_rejected", symbol, side=side, volume=volume, price=price,
                    sl=stop_loss, tp=take_profit, comment=message, source="order_manager",
                )
                return None

            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                "volume": volume, "type": order_type, "price": price,
                "sl": stop_loss, "tp": take_profit, "deviation": self.deviation,
                "magic": self.magic, "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }

            logger.info(
                f"[EXECUTION] {symbol} {side} signal_price={reference_price} "
                f"execution_price={price:.{digits}f} SL={stop_loss:.{digits}f} "
                f"TP={take_profit:.{digits}f} R:R=1:{rr:.2f}"
            )
            journal_context = dict(diagnostics or {})
            journal_context.setdefault("execution_rr", rr)
            self._journal_event(
                "log_order_attempt", symbol, side, volume, price,
                sl=stop_loss, tp=take_profit, comment=execution_comment, source="order_manager", **journal_context
            )
            result = self._send_order(request, f"would place {side} {symbol} {volume} lots")
            if self.dry_run:
                return None
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = result.comment if result else 'None'
                retcode = result.retcode if result else 0
                message = f"retcode={retcode}: {error_msg}"
                logger.error(f"Order failed [{retcode}]: {error_msg}")
                self._journal_event(
                    "log_order_failed", symbol, side=side, volume=volume, price=price,
                    sl=stop_loss, tp=take_profit, comment=message, source="order_manager",
                )
                return None

            logger.info(f"[TRADE] {side} {symbol} {volume}lots @{price:.{digits}f} SL={stop_loss:.{digits}f} TP={take_profit:.{digits}f}")
            self._journal_event(
                "log_order_filled", result.order, symbol, side, volume, price,
                sl=stop_loss, tp=take_profit, comment=execution_comment, source="order_manager", **journal_context
            )
            self._journal_event(
                "log_open", result.order, symbol, side, volume, price,
                sl=stop_loss, tp=take_profit, comment=execution_comment, source="order_manager", **journal_context
            )
            return result.order
        except Exception as e:
            logger.error(f"Place order error: {e}")
            return None

    def _min_sl_modify_distance(self, symbol):
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return 0.0
            point = float(getattr(info, "point", 0) or 0)
            stops = float(getattr(info, "trade_stops_level", 0) or 0)
            if point <= 0:
                return 0.0
            return max(point, stops * point)
        except Exception:
            return 0.0

    def modify_sl(self, ticket, new_sl):
        """Modify SL — ใช้โดย SmartTrailingV2"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.warning(f"[MODIFY] #{ticket} not found")
                return False

            pos = position[0]
            digits = self._get_digits(pos.symbol)
            new_sl = round(new_sl, digits)

            if pos.type == 0:
                if new_sl <= pos.sl and pos.sl > 0:
                    logger.debug(f"[MODIFY] #{ticket} BUY new_sl={new_sl:.{digits}f} <= old_sl={pos.sl:.{digits}f} SKIP")
                    return False
            else:
                if new_sl >= pos.sl and pos.sl > 0:
                    logger.debug(f"[MODIFY] #{ticket} SELL new_sl={new_sl:.{digits}f} >= old_sl={pos.sl:.{digits}f} SKIP")
                    return False

            si = self.mt5.get_symbol_info(pos.symbol)
            if si:
                current_price = si['bid'] if pos.type == 0 else si['ask']
                new_sl, _ = self._check_stop_level(pos.symbol, current_price, new_sl, pos.tp, pos.type)

            min_distance = self._min_sl_modify_distance(pos.symbol)
            if pos.sl > 0 and abs(new_sl - pos.sl) < min_distance:
                logger.debug(
                    f"[MODIFY] #{ticket} {pos.symbol} delta={abs(new_sl - pos.sl):.{digits}f} "
                    f"below min_distance={min_distance:.{digits}f} SKIP"
                )
                return False

            return self._modify_sl(pos, new_sl)
        except Exception as e:
            logger.error(f"Modify SL error: {e}")
            return False

    def modify_sl_tp(self, ticket, new_sl=None, new_tp=None):
        """Modify SL and/or TP"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            pos = position[0]
            digits = self._get_digits(pos.symbol)
            sl = round(new_sl, digits) if new_sl is not None else pos.sl
            tp = round(new_tp, digits) if new_tp is not None else pos.tp

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": sl,
                "tp": tp,
            }
            result = self._send_order(request, f"would modify #{ticket} SL={sl} TP={tp}")
            if self.dry_run:
                self._journal_event(
                    "log_sl_modified", ticket, pos.symbol, self._position_side(pos), sl=sl, tp=tp,
                    comment="dry-run SL/TP modify",
                )
                return True
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[MODIFY] #{ticket} {pos.symbol} SL={sl:.{digits}f} TP={tp:.{digits}f}")
                self._journal_event(
                    "log_sl_modified", ticket, pos.symbol, self._position_side(pos), sl=sl, tp=tp,
                    comment="SL/TP modified",
                )
                return True
            error_msg = result.comment if result else 'None'
            retcode = result.retcode if result else 0
            logger.warning(f"[MODIFY] #{ticket} failed [{retcode}]: {error_msg}")
            self._journal_event(
                "log_order_failed", pos.symbol, side=self._position_side(pos), sl=sl, tp=tp,
                comment=f"SL/TP modify failed retcode={retcode}: {error_msg}",
            )
            return False
        except Exception as e:
            logger.error(f"Modify SL/TP error: {e}")
            return False

    def _modify_sl(self, pos, new_sl):
        """Internal: ส่ง SLTP request ไป MT5"""
        try:
            digits = self._get_digits(pos.symbol)
            new_sl = round(new_sl, digits)
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }
            result = self._send_order(request, f"would move SL #{pos.ticket} to {new_sl}")
            if self.dry_run:
                self._journal_event(
                    "log_sl_modified", pos.ticket, pos.symbol, self._position_side(pos),
                    sl=new_sl, tp=pos.tp, comment="dry-run SL modify",
                )
                return True
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[SL] #{pos.ticket} {pos.symbol} SL -> {new_sl:.{digits}f}")
                self._journal_event(
                    "log_sl_modified", pos.ticket, pos.symbol, self._position_side(pos),
                    sl=new_sl, tp=pos.tp, comment="SL modified",
                )
                return True
            error_msg = result.comment if result else 'None'
            retcode = result.retcode if result else 0
            logger.warning(f"[SL] #{pos.ticket} failed [{retcode}]: {error_msg}")
            self._journal_event(
                "log_order_failed", pos.symbol, side=self._position_side(pos), sl=new_sl, tp=pos.tp,
                comment=f"SL modify failed retcode={retcode}: {error_msg}",
            )
            return False
        except Exception as e:
            logger.error(f"_modify_sl error: {e}")
            return False

    def partial_close(self, ticket, close_pct, stage=None):
        """Close a percentage of an open position"""
        try:
            stage_label = None if stage is None else str(stage)
            if stage_label:
                ticket_key = str(ticket)
                done = self._partial_close_stages.setdefault(ticket_key, set())
                if stage_label in done:
                    logger.debug(f"[PARTIAL] #{ticket} stage {stage_label} already executed, skip")
                    return False

            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            pos = position[0]
            close_volume = round(pos.volume * close_pct, 8)

            vol_min = 0.01
            vol_step = 0.01
            si = self.mt5.get_symbol_info(pos.symbol)
            if si:
                vol_min = si.get('volume_min', 0.01)
                vol_step = si.get('volume_step', 0.01)
                close_volume = max(self._round_volume(close_volume, vol_step), vol_min)

            if close_volume < vol_min:
                return False
            if close_volume >= pos.volume:
                return self.close_order(ticket)

            order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            price = (si['bid'] if pos.type == 0 else si['ask']) if si else 0
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
                "volume": close_volume, "type": order_type,
                "position": ticket, "price": price, "deviation": self.deviation,
                "magic": self.magic, "comment": f"partial_close_s{stage_label}" if stage_label else "partial_close",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = self._send_order(request, f"would partial close #{ticket} {close_volume} lots")
            if self.dry_run:
                pnl = getattr(pos, "profit", None)
                context = {"comment": f"dry-run partial close stage={stage_label}" if stage_label else "dry-run partial close", "pnl": pnl}
                if pnl is None:
                    context["reason"] = "pnl_unavailable"
                    context["comment"] = "pnl_unavailable"
                self._journal_event(
                    "log_partial_close", ticket, pos.symbol, self._position_side(pos), close_volume, price,
                    **context,
                )
                if stage_label:
                    self._partial_close_stages.setdefault(str(ticket), set()).add(stage_label)
                return True
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[PARTIAL] #{ticket} {pos.symbol} closed {close_volume}/{pos.volume} lots ({close_pct:.0%})")
                pnl = getattr(pos, "profit", None)
                context = {"comment": f"partial close stage={stage_label} {close_pct:.0%}" if stage_label else f"partial close {close_pct:.0%}", "pnl": pnl}
                if pnl is None:
                    context["reason"] = "pnl_unavailable"
                    context["comment"] = "pnl_unavailable"
                self._journal_event(
                    "log_partial_close", ticket, pos.symbol, self._position_side(pos), close_volume, price,
                    **context,
                )
                if stage_label:
                    self._partial_close_stages.setdefault(str(ticket), set()).add(stage_label)
                return True
            error_msg = result.comment if result else 'None'
            retcode = result.retcode if result else 0
            logger.warning(f"[PARTIAL] #{ticket} failed: {error_msg}")
            self._journal_event(
                "log_order_failed", pos.symbol, side=self._position_side(pos), volume=close_volume, price=price,
                comment=f"partial close failed retcode={retcode}: {error_msg}",
            )
            return False
        except Exception as e:
            logger.error(f"Partial close error: {e}")
            return False

    def apply_news_protection(self, symbol, action, atr_value):
        try:
            for pos in self.get_open_positions(symbol):
                si = self.mt5.get_symbol_info(symbol)
                if si is None:
                    continue
                cp = si['bid'] if pos.type == 0 else si['ask']
                entry = pos.price_open

                if action == 'FORCE_CLOSE':
                    self._journal_event(
                        "log_news_blocked", pos.ticket, symbol, self._position_side(pos), price=cp,
                        sl=pos.sl, tp=pos.tp, comment="news protection force close",
                    )
                    self.close_order(pos.ticket)
                    continue

                new_sl = None
                if pos.type == 0:
                    if action == 'BREAKEVEN':
                        p = entry + atr_value * 0.05
                        if p > pos.sl:
                            new_sl = p
                    elif action == 'LOCK_PROFIT':
                        p = cp - atr_value * 0.3
                        if p > pos.sl and p > entry:
                            new_sl = p
                    elif action == 'TIGHTEN_SL':
                        d = cp - pos.sl
                        p = cp - d * 0.5
                        if p > pos.sl:
                            new_sl = p
                else:
                    if action == 'BREAKEVEN':
                        p = entry - atr_value * 0.05
                        if pos.sl == 0 or p < pos.sl:
                            new_sl = p
                    elif action == 'LOCK_PROFIT':
                        p = cp + atr_value * 0.3
                        if (pos.sl == 0 or p < pos.sl) and p < entry:
                            new_sl = p
                    elif action == 'TIGHTEN_SL':
                        d = pos.sl - cp
                        p = cp + d * 0.5
                        if pos.sl == 0 or p < pos.sl:
                            new_sl = p

                if new_sl and self._modify_sl(pos, new_sl):
                    self._journal_event(
                        "log_news_blocked", pos.ticket, symbol, self._position_side(pos), price=cp,
                        sl=new_sl, tp=pos.tp, comment=f"news protection {action}",
                    )
                    logger.info(f"[NEWS] {action} #{pos.ticket} {symbol} SL -> {new_sl:.5f}")
        except Exception as e:
            logger.error(f"News protection error: {e}")

    def close_order(self, ticket):
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            pos = position[0]
            ot = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            si = self.mt5.get_symbol_info(pos.symbol)
            if si is None:
                return False
            price = si['bid'] if pos.type == 0 else si['ask']
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
                "volume": pos.volume, "type": ot, "position": ticket,
                "price": price, "deviation": self.deviation, "magic": self.magic,
                "comment": "close", "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            r = self._send_order(request, f"would close #{ticket}")
            if self.dry_run:
                pnl = getattr(pos, "profit", None)
                context = {"comment": "dry-run close", "pnl": pnl}
                if pnl is None:
                    context["reason"] = "pnl_unavailable"
                    context["comment"] = "pnl_unavailable"
                self._journal_event(
                    "log_close", ticket, pos.symbol, self._position_side(pos), pos.volume, price,
                    **context,
                )
                return True
            if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[CLOSE] #{ticket} {pos.symbol}")
                pnl = getattr(pos, "profit", None)
                context = {"comment": "closed", "pnl": pnl}
                if pnl is None:
                    context["reason"] = "pnl_unavailable"
                    context["comment"] = "pnl_unavailable"
                self._journal_event(
                    "log_close", ticket, pos.symbol, self._position_side(pos), pos.volume, price,
                    **context,
                )
                return True
            error_msg = r.comment if r else 'None'
            retcode = r.retcode if r else 0
            self._journal_event(
                "log_order_failed", pos.symbol, side=self._position_side(pos), volume=pos.volume, price=price,
                comment=f"close failed retcode={retcode}: {error_msg}",
            )
            return False
        except Exception as e:
            logger.error(f"Close error: {e}")
            return False

    def close_all(self, symbol=None):
        closed = 0
        for pos in self.get_open_positions(symbol):
            if self.close_order(pos.ticket):
                closed += 1
        return closed
