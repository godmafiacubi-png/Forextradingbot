from decimal import Decimal, ROUND_HALF_UP
import logging
import time

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class OrderManager:
    """Execute and manage orders — trailing + news protection + partial close."""

    def __init__(self, mt5_connector, max_open_trades=3, max_per_symbol=1,
                 dry_run=True, magic=123456, deviation=20):
        self.mt5 = mt5_connector
        self.max_open_trades = max_open_trades
        self.max_per_symbol = max_per_symbol
        self.dry_run = dry_run
        self.magic = magic
        self.deviation = deviation

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
                    reference_price=None, max_slippage_points=None):
        try:
            if not self.can_open_trade(symbol):
                logger.info(f"[SKIP] {symbol} position limit reached")
                return None

            si = self.mt5.get_symbol_info(symbol)
            if si is None:
                return None

            price = si['ask'] if order_type == mt5.ORDER_TYPE_BUY else si['bid']
            point = si.get('point', 0)
            if reference_price is not None and max_slippage_points is not None and point > 0:
                slippage_points = abs(price - reference_price) / point
                if slippage_points > max_slippage_points:
                    logger.warning(
                        f"[SKIP] {symbol} slippage guard: {slippage_points:.1f}pts > {max_slippage_points}pts "
                        f"(ref={reference_price}, exec={price})"
                    )
                    return None
            digits = self._get_digits(symbol)
            stop_loss = round(stop_loss, digits)
            take_profit = round(take_profit, digits)
            if not self._validate_sl_tp_side(symbol, price, stop_loss, take_profit, order_type):
                return None

            stop_loss, take_profit = self._check_stop_level(symbol, price, stop_loss, take_profit, order_type)
            if not self._validate_sl_tp_side(symbol, price, stop_loss, take_profit, order_type):
                return None

            vol_min = si.get('volume_min', 0.01)
            vol_max = si.get('volume_max', 100.0)
            vol_step = si.get('volume_step', 0.01)
            volume = max(min(volume, vol_max), vol_min)
            volume = self._round_volume(volume, vol_step)
            if volume < vol_min or volume > vol_max:
                logger.error(f"Invalid volume after rounding: {volume} not in [{vol_min}, {vol_max}]")
                return None

            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                "volume": volume, "type": order_type, "price": price,
                "sl": stop_loss, "tp": take_profit, "deviation": self.deviation,
                "magic": self.magic, "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }

            d = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
            result = self._send_order(request, f"would place {d} {symbol} {volume} lots")
            if self.dry_run:
                return None
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = result.comment if result else 'None'
                retcode = result.retcode if result else 0
                logger.error(f"Order failed [{retcode}]: {error_msg}")
                return None

            logger.info(f"[TRADE] {d} {symbol} {volume}lots @{price:.{digits}f} SL={stop_loss:.{digits}f} TP={take_profit:.{digits}f}")
            return result.order
        except Exception as e:
            logger.error(f"Place order error: {e}")
            return None

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
                return True
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[MODIFY] #{ticket} {pos.symbol} SL={sl:.{digits}f} TP={tp:.{digits}f}")
                return True
            error_msg = result.comment if result else 'None'
            retcode = result.retcode if result else 0
            logger.warning(f"[MODIFY] #{ticket} failed [{retcode}]: {error_msg}")
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
                return True
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[SL] #{pos.ticket} {pos.symbol} SL -> {new_sl:.{digits}f}")
                return True
            error_msg = result.comment if result else 'None'
            retcode = result.retcode if result else 0
            logger.warning(f"[SL] #{pos.ticket} failed [{retcode}]: {error_msg}")
            return False
        except Exception as e:
            logger.error(f"_modify_sl error: {e}")
            return False

    def partial_close(self, ticket, close_pct):
        """Close a percentage of an open position"""
        try:
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
                "magic": self.magic, "comment": "partial_close",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = self._send_order(request, f"would partial close #{ticket} {close_volume} lots")
            if self.dry_run:
                return True
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[PARTIAL] #{ticket} {pos.symbol} closed {close_volume}/{pos.volume} lots ({close_pct:.0%})")
                return True
            error_msg = result.comment if result else 'None'
            logger.warning(f"[PARTIAL] #{ticket} failed: {error_msg}")
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
                return True
            if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[CLOSE] #{ticket} {pos.symbol}")
                return True
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
