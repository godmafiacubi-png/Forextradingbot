from decimal import Decimal, ROUND_HALF_UP
import logging

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculate position size — auto-detect symbol info from MT5"""

    def __init__(self, method='ATR', account_risk=1.0, max_drawdown=10.0, max_lot_size=2.0):
        self.method = method
        self.account_risk = account_risk
        self.max_drawdown = max_drawdown
        self.max_lot_size = max_lot_size
        self.peak_balance = 0
        self._symbol_cache = {}

    def _get_symbol_info(self, symbol):
        """ดึงข้อมูล symbol จาก MT5 + cache"""
        if symbol and symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        if symbol:
            try:
                info = mt5.symbol_info(symbol)
                if info:
                    data = {
                        'point': info.point,
                        'digits': info.digits,
                        'trade_tick_value': info.trade_tick_value,
                        'trade_tick_size': info.trade_tick_size,
                        'trade_contract_size': info.trade_contract_size,
                        'volume_min': info.volume_min,
                        'volume_max': info.volume_max,
                        'volume_step': info.volume_step,
                    }
                    self._symbol_cache[symbol] = data
                    return data
            except Exception as exc:
                logger.debug(f"Unable to load MT5 symbol info for {symbol}: {exc}")
        return None

    @staticmethod
    def _round_volume(volume, step):
        if step <= 0:
            step = 0.01
        volume_dec = Decimal(str(volume))
        step_dec = Decimal(str(step))
        steps = (volume_dec / step_dec).to_integral_value(rounding=ROUND_HALF_UP)
        rounded = steps * step_dec
        decimals = max(0, -step_dec.as_tuple().exponent)
        return float(round(rounded, decimals))

    def calculate_position_size(self, account_balance, atr_value, symbol_point,
                                confidence=1.0, method=None, symbol=None):
        """
        คำนวณ lot size
        ถ้ามี symbol → ดึง point + tick_value จาก MT5 โดยตรง
        """
        if method is None:
            method = self.method

        # ดึงข้อมูลจริงจาก MT5
        sym_info = self._get_symbol_info(symbol)
        if sym_info:
            symbol_point = sym_info['point']

        # Drawdown check
        if account_balance > self.peak_balance:
            self.peak_balance = account_balance

        if self.peak_balance > 0:
            drawdown_pct = (self.peak_balance - account_balance) / self.peak_balance * 100
            if drawdown_pct > self.max_drawdown:
                logger.warning(f"Max drawdown reached ({drawdown_pct:.1f}%). Lot = 0")
                return 0.0

        # คำนวณ lot
        if method == 'FIXED_PERCENT':
            lot_size = self.fixed_percent_sizing(account_balance)
        elif method == 'KELLY':
            lot_size = self.kelly_sizing(account_balance, win_rate=0.55)
        elif method == 'ATR':
            lot_size = self.atr_sizing(account_balance, atr_value, symbol_point, sym_info)
        else:
            lot_size = 0.01

        # Confidence scaling
        confidence_factor = 0.5 + (confidence * 0.5)
        lot_size *= confidence_factor

        # Volume constraints จาก MT5
        vol_min = 0.01
        vol_max = 2.0
        vol_step = 0.01
        if sym_info:
            vol_min = sym_info.get('volume_min', 0.01)
            vol_max = min(sym_info.get('volume_max', 100.0), self.max_lot_size)
            vol_step = sym_info.get('volume_step', 0.01)

        # Round to volume step
        if lot_size < vol_min:
            logger.warning(
                f"Calculated lot {lot_size:.4f} is below broker minimum {vol_min}; skip to avoid over-risking"
            )
            return 0.0

        lot_size = self._round_volume(min(lot_size, vol_max), vol_step)
        if lot_size < vol_min:
            return 0.0

        digits = sym_info['digits'] if sym_info else 5
        logger.info(f"Position: {lot_size} lots (risk={self.account_risk}%, "
                    f"conf={confidence:.2f}, point={symbol_point}, "
                    f"sym={symbol or '?'})")
        return lot_size

    def fixed_percent_sizing(self, account_balance, lot_size=0.1):
        return lot_size

    def kelly_sizing(self, account_balance, win_rate, avg_win=3, avg_loss=1):
        if win_rate >= 1.0 or win_rate <= 0:
            return 0.01

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b
        safe_fraction = max(kelly_fraction * 0.20, 0.005)
        lot_size = account_balance * safe_fraction / 100000
        return max(min(lot_size, 1.0), 0.01)

    def atr_sizing(self, account_balance, atr_value, symbol_point, sym_info=None):
        """
        ATR-based position sizing
        ใช้ tick_value จาก MT5 ถ้ามี → แม่นยำทุก symbol
        """
        risk_amount = account_balance * (self.account_risk / 100)

        sl_distance = atr_value * 1.0  # 1x ATR

        if sl_distance <= 0 or symbol_point <= 0:
            return 0.0

        # วิธี 1: ใช้ tick_value จาก MT5 (แม่นยำที่สุด)
        if sym_info and sym_info.get('trade_tick_value', 0) > 0:
            tick_value = sym_info['trade_tick_value']
            tick_size = sym_info.get('trade_tick_size', symbol_point)

            if tick_size <= 0:
                tick_size = symbol_point

            # จำนวน ticks ของ SL
            sl_ticks = sl_distance / tick_size

            # Loss per lot ถ้าโดน SL
            loss_per_lot = sl_ticks * tick_value

            if loss_per_lot <= 0:
                return 0.0

            lot_size = risk_amount / loss_per_lot
            logger.debug(f"ATR sizing: risk=${risk_amount:.2f} sl_dist={sl_distance:.5f} "
                        f"ticks={sl_ticks:.0f} tick_val={tick_value} loss/lot=${loss_per_lot:.2f} "
                        f"-> {lot_size:.4f} lots")
            return max(min(lot_size, self.max_lot_size), 0.0)

        # วิธี 2: Fallback — ประมาณจาก pip value
        sl_pips = sl_distance / symbol_point

        # ประมาณ pip value ต่อ lot
        if symbol_point >= 0.01:
            # JPY pairs, BTC, Gold (3 digits)
            pip_value_per_lot = 1.0
        elif symbol_point >= 0.001:
            # 3-digit symbols
            pip_value_per_lot = 1.0
        else:
            # Standard forex (5 digits)
            pip_value_per_lot = 10.0

        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        return max(min(lot_size, self.max_lot_size), 0.0)

    def calculate_sl_tp(self, price, signal, atr_value, sl_multiplier=1.0,
                        tp_multiplier=3.0, symbol=None):
        """SL/TP — round ตาม digits ของ symbol"""
        sl_distance = atr_value * sl_multiplier
        tp_distance = atr_value * tp_multiplier

        if signal == 1:  # BUY
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        elif signal == -1:  # SELL
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
        else:
            return None, None

        # Round ตาม digits
        digits = 5  # default
        sym_info = self._get_symbol_info(symbol)
        if sym_info:
            digits = sym_info['digits']

        return round(stop_loss, digits), round(take_profit, digits)