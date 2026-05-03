import pandas as pd
import numpy as np
import logging

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)


class ICTFeatures:
    """ICT Trading Concepts - Vectorized for performance"""

    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.sort_values('time').reset_index(drop=True)

    def identify_order_blocks(self, lookback=10):
        """Identify Order Blocks (vectorized) — ICT-correct logic, no lookahead.

        At bar i:
          Demand OB = bar (i-1) was bearish AND bar i is a bullish impulse.
          Supply OB = bar (i-1) was bullish AND bar i is a bearish impulse.
        Impulse = candle whose body > 0.5 × ATR(14).

        Using shift(+1) on the previous-candle condition (rather than shift(-1)
        on the impulse condition) ensures no future data is read at training time.
        """
        h = self.df['h'].values
        l = self.df['l'].values
        o = self.df['o'].values
        c = self.df['c'].values

        bullish_candle = c > o
        bearish_candle = c < o

        # Impulse detection: body larger than 0.5 × ATR(14)
        atr_series = pd.Series(h - l).rolling(14, min_periods=1).mean()
        body_series = pd.Series(np.abs(c - o))
        impulse_threshold = atr_series * 0.5

        bullish_impulse = bullish_candle & (body_series > impulse_threshold)
        bearish_impulse = bearish_candle & (body_series > impulse_threshold)

        # Demand OB at bar i = bar (i-1) was bearish + bar i is the bullish impulse
        ob_demand = (
            pd.Series(bearish_candle).shift(1).fillna(False).infer_objects(copy=False).astype(bool) &
            pd.Series(bullish_impulse)
        ).astype(int)

        # Supply OB at bar i = bar (i-1) was bullish + bar i is the bearish impulse
        ob_supply = (
            pd.Series(bullish_candle).shift(1).fillna(False).infer_objects(copy=False).astype(bool) &
            pd.Series(bearish_impulse)
        ).astype(int)

        self.df['ob_supply'] = ob_supply
        self.df['ob_demand'] = ob_demand

        self.df['ob_supply_strength'] = np.where(ob_supply == 1, (h - c) / (h - l + 1e-10), 0)
        self.df['ob_demand_strength'] = np.where(ob_demand == 1, (c - l) / (h - l + 1e-10), 0)

        # Proximity to recent OB (within last N bars)
        ob_supply_price = pd.Series(np.where(ob_supply == 1, h, np.nan))
        ob_demand_price = pd.Series(np.where(ob_demand == 1, l, np.nan))
        self.df['near_supply_ob'] = (
            (pd.Series(c) - ob_supply_price.ffill()) / (pd.Series(c) + 1e-10)
        ).abs().fillna(1.0).values
        self.df['near_demand_ob'] = (
            (pd.Series(c) - ob_demand_price.ffill()) / (pd.Series(c) + 1e-10)
        ).abs().fillna(1.0).values

        return self.df

    def identify_fvg(self):
        """Identify Fair Value Gaps (vectorized, adaptive threshold)"""
        h = self.df['h'].values
        l = self.df['l'].values
        atr = pd.Series(h - l).rolling(14).mean().values

        fvg_bullish = np.zeros(len(self.df), dtype=int)
        fvg_bearish = np.zeros(len(self.df), dtype=int)
        fvg_size = np.zeros(len(self.df))

        for i in range(2, len(self.df)):
            min_gap = atr[i] * 0.2 if not np.isnan(atr[i]) else 0.0001

            gap_bull = l[i] - h[i - 2]
            if gap_bull > min_gap:
                fvg_bullish[i] = 1
                fvg_size[i] = gap_bull

            gap_bear = l[i - 2] - h[i]
            if gap_bear > min_gap:
                fvg_bearish[i] = 1
                fvg_size[i] = -gap_bear

        self.df['fvg_bullish'] = fvg_bullish
        self.df['fvg_bearish'] = fvg_bearish
        self.df['fvg_size'] = fvg_size

        # FVG unfilled / filled detection
        # "unfilled" = the gap still exists and price has NOT returned to close it
        # "filled"   = price has since re-entered the gap zone
        c = self.df['c'].values
        fvg_bull_unfilled = np.zeros(len(self.df), dtype=int)
        fvg_bear_unfilled = np.zeros(len(self.df), dtype=int)
        fvg_bull_filled = np.zeros(len(self.df), dtype=int)
        fvg_bear_filled = np.zeros(len(self.df), dtype=int)
        last_bull_fvg_low = np.nan   # h[i-2]: lower boundary of the most recent bullish FVG gap
        last_bear_fvg_high = np.nan  # l[i-2]: upper boundary of the most recent bearish FVG gap

        for i in range(2, len(self.df)):
            if fvg_bullish[i] == 1:
                last_bull_fvg_low = h[i - 2]   # gap lower edge (high of the i-2 reference candle)
            if fvg_bearish[i] == 1:
                last_bear_fvg_high = l[i - 2]  # gap upper edge (low of the i-2 reference candle)

            # Unfilled: price is still below/above the gap → gap is intact
            if not np.isnan(last_bull_fvg_low) and c[i] < last_bull_fvg_low:
                fvg_bull_unfilled[i] = 1        # gap is above current price
            if not np.isnan(last_bear_fvg_high) and c[i] > last_bear_fvg_high:
                fvg_bear_unfilled[i] = 1        # gap is below current price

            # Filled: price has returned into (or through) the gap zone
            if not np.isnan(last_bull_fvg_low) and c[i] >= last_bull_fvg_low:
                fvg_bull_filled[i] = 1
            if not np.isnan(last_bear_fvg_high) and c[i] <= last_bear_fvg_high:
                fvg_bear_filled[i] = 1

        self.df['fvg_bull_unfilled'] = fvg_bull_unfilled
        self.df['fvg_bear_unfilled'] = fvg_bear_unfilled
        self.df['fvg_bull_filled'] = fvg_bull_filled
        self.df['fvg_bear_filled'] = fvg_bear_filled

        return self.df

    def identify_market_structure(self, lookback=5, depth=10, deviation=5):
        """
        Improved: Zigzag-based Pivot Structure
        Uses true swing pivot detection instead of rolling window.
        Ported from Pine Script signalLib zigzag logic.

        Parameters:
            lookback   : kept for backward compatibility (unused internally)
            depth      : bars to look left/right for pivot confirmation (default 10)
            deviation  : minimum % price move to confirm a new swing (default 5)
        """
        h = self.df['h'].values
        l = self.df['l'].values
        n = len(h)

        # --- True Pivot High/Low Detection ---
        pivot_high = np.zeros(n)
        pivot_low  = np.zeros(n)

        for i in range(depth, n):
            window_h = h[max(0, i - depth): i + 1]
            window_l = l[max(0, i - depth): i + 1]
            if h[i] == max(window_h):
                pivot_high[i] = h[i]
            if l[i] == min(window_l):
                pivot_low[i] = l[i]

        # --- Zigzag Direction (inspired by signalLib) ---
        zz_direction = np.zeros(n)
        last_ph  = np.nan
        last_pl  = np.nan
        current_dir = 0

        for i in range(n):
            if pivot_high[i] > 0:
                if not np.isnan(last_pl) and last_pl > 0:
                    move_pct = (pivot_high[i] - last_pl) / (last_pl + 1e-10) * 100
                    if move_pct >= deviation:
                        current_dir = 1
                last_ph = pivot_high[i]

            if pivot_low[i] > 0:
                if not np.isnan(last_ph) and last_ph > 0:
                    move_pct = (last_ph - pivot_low[i]) / (last_ph + 1e-10) * 100
                    if move_pct >= deviation:
                        current_dir = -1
                last_pl = pivot_low[i]

            zz_direction[i] = current_dir

        zz_series  = pd.Series(zz_direction)
        prev_dir   = zz_series.shift(1).fillna(0)

        # --- BOS: direction change (more reliable than rolling window) ---
        self.df['bos_bullish']   = ((zz_series == 1)  & (prev_dir != 1)).astype(int).values
        self.df['bos_bearish']   = ((zz_series == -1) & (prev_dir != -1)).astype(int).values

        # --- CHoCH: full reversal ---
        self.df['choch_bullish'] = ((zz_series == 1)  & (prev_dir == -1)).astype(int).values
        self.df['choch_bearish'] = ((zz_series == -1) & (prev_dir ==  1)).astype(int).values

        # --- Structure score (same column name, backward compatible) ---
        self.df['structure']   = zz_direction.astype(int)
        self.df['swing_high']  = (pivot_high > 0).astype(int)
        self.df['swing_low']   = (pivot_low  > 0).astype(int)

        # --- New column: zigzag direction for signal_generator filter ---
        self.df['zz_direction'] = zz_direction.astype(int)

        return self.df

    def identify_liquidity_levels(self, lookback=20):
        """Identify Liquidity Levels"""
        h = pd.Series(self.df['h'].values)
        l = pd.Series(self.df['l'].values)
        c = pd.Series(self.df['c'].values)

        highest = h.rolling(lookback).max()
        lowest = l.rolling(lookback).min()

        atr = (h - l).rolling(14).mean()
        atr = atr.replace(0, np.nan).ffill().fillna(1e-10)

        self.df['ls_high'] = ((highest - c) / atr).values
        self.df['ls_low'] = ((c - lowest) / atr).values

        self.df['liq_sweep_high'] = (
            (h > highest.shift(1)) & (c < highest.shift(1))
        ).astype(int).values

        self.df['liq_sweep_low'] = (
            (l < lowest.shift(1)) & (c > lowest.shift(1))
        ).astype(int).values

        return self.df

    def identify_breaker_blocks(self, lookback=10):
        """Identify Breaker Blocks"""
        structure = self.df['structure'].values

        breaker = np.zeros(len(self.df))
        prev_structure = np.roll(structure, 1)
        prev_structure[0] = 0

        change = (structure != prev_structure) & (structure != 0)
        breaker[change & (structure == 1)] = 1
        breaker[change & (structure == -1)] = -1

        self.df['breaker_blocks'] = breaker

        return self.df

    def identify_sessions(self):
        """Add trading session features"""
        if 'time' not in self.df.columns:
            return self.df

        times = pd.to_datetime(self.df['time'])
        hours = times.dt.hour

        self.df['session_asian'] = ((hours >= 0) & (hours < 8)).astype(int)
        self.df['session_london'] = ((hours >= 7) & (hours < 16)).astype(int)
        self.df['session_newyork'] = ((hours >= 13) & (hours < 22)).astype(int)
        self.df['session_overlap'] = ((hours >= 13) & (hours < 16)).astype(int)

        self.df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        dow = times.dt.dayofweek
        self.df['dow_sin'] = np.sin(2 * np.pi * dow / 5)
        self.df['dow_cos'] = np.cos(2 * np.pi * dow / 5)

        return self.df

    def identify_optimal_trade_entry(self):
        """OTE — Optimal Trade Entry zone (61.8%-78.6% Fibonacci retracement)"""
        h = pd.Series(self.df['h'].values)
        l = pd.Series(self.df['l'].values)
        c = pd.Series(self.df['c'].values)

        swing_h = h.rolling(20).max()
        swing_l = l.rolling(20).min()
        fib_range = swing_h - swing_l

        # Where price is in the Fibonacci retracement
        fib_level = (swing_h - c) / (fib_range + 1e-10)

        # OTE zone: 0.618 to 0.786 retracement for BUY
        self.df['in_ote_buy_zone'] = ((fib_level >= 0.618) & (fib_level <= 0.786)).astype(int)
        # OTE zone: 0.214 to 0.382 for SELL (inverse)
        self.df['in_ote_sell_zone'] = ((fib_level >= 0.214) & (fib_level <= 0.382)).astype(int)
        self.df['fib_level'] = fib_level.values

        return self.df

    def get_ict_features(self):
        """Generate all ICT features"""
        try:
            logger.debug("Generating ICT features...")

            self.identify_order_blocks(lookback=10)
            self.identify_fvg()
            self.identify_market_structure(lookback=5)
            self.identify_liquidity_levels(lookback=20)
            self.identify_breaker_blocks(lookback=10)
            self.identify_sessions()
            self.identify_optimal_trade_entry()

            self.df = self.df.fillna(0)

            logger.debug("ICT features generated successfully")
            return self.df

        except Exception as e:
            logger.error(f"Error generating ICT features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self.df