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

    @staticmethod
    def _zone_distance_pct(price, zone_low, zone_high):
        """Return 0 when price is inside a zone, otherwise percentage distance."""
        if np.isnan(zone_low) or np.isnan(zone_high):
            return 1.0
        lower = min(zone_low, zone_high)
        upper = max(zone_low, zone_high)
        if lower <= price <= upper:
            return 0.0
        if price < lower:
            return (lower - price) / (abs(price) + 1e-10)
        return (price - upper) / (abs(price) + 1e-10)

    def identify_order_blocks(self, lookback=10):
        """Identify confirmed ICT order-block zones without repainting.

        A demand OB is confirmed only after a bullish displacement candle forms
        immediately after a bearish candle and closes through recent buy-side
        structure. The active demand zone is the *previous bearish candle*.
        Supply OB uses the inverse rule.  This avoids marking the displacement
        candle as the order block and prevents training-time lookahead: the OB
        becomes tradable on the confirmation bar, using only completed bars.
        """
        h = self.df['h'].astype(float).values
        l = self.df['l'].astype(float).values
        o = self.df['o'].astype(float).values
        c = self.df['c'].astype(float).values
        n = len(self.df)

        ranges = pd.Series(h - l)
        atr = ranges.rolling(14, min_periods=1).mean().replace(0, np.nan).ffill().fillna(1e-10).values
        bodies = np.abs(c - o)

        ob_supply = np.zeros(n, dtype=int)
        ob_demand = np.zeros(n, dtype=int)
        ob_supply_low = np.full(n, np.nan)
        ob_supply_high = np.full(n, np.nan)
        ob_demand_low = np.full(n, np.nan)
        ob_demand_high = np.full(n, np.nan)
        near_supply_ob = np.ones(n)
        near_demand_ob = np.ones(n)
        ob_supply_mitigated = np.zeros(n, dtype=int)
        ob_demand_mitigated = np.zeros(n, dtype=int)

        active_supply = []
        active_demand = []

        for i in range(n):
            # First update mitigation state for zones created on prior bars.
            for zone in active_demand:
                if not zone['mitigated'] and i > zone['created_at'] and l[i] <= zone['high'] and h[i] >= zone['low']:
                    zone['mitigated'] = True
                    ob_demand_mitigated[i] = 1
            for zone in active_supply:
                if not zone['mitigated'] and i > zone['created_at'] and h[i] >= zone['low'] and l[i] <= zone['high']:
                    zone['mitigated'] = True
                    ob_supply_mitigated[i] = 1

            if i >= 1:
                recent_start = max(0, i - lookback)
                prev_recent_high = np.max(h[recent_start:i]) if i > recent_start else h[i - 1]
                prev_recent_low = np.min(l[recent_start:i]) if i > recent_start else l[i - 1]
                bullish_displacement = c[i] > o[i] and bodies[i] >= atr[i] * 0.6 and c[i] > prev_recent_high
                bearish_displacement = c[i] < o[i] and bodies[i] >= atr[i] * 0.6 and c[i] < prev_recent_low

                if c[i - 1] < o[i - 1] and bullish_displacement:
                    zone_low = float(l[i - 1])
                    zone_high = float(max(o[i - 1], c[i - 1]))
                    zone = {'low': zone_low, 'high': zone_high, 'created_at': i, 'mitigated': False}
                    active_demand.append(zone)
                    ob_demand[i] = 1
                    ob_demand_low[i] = zone_low
                    ob_demand_high[i] = zone_high

                if c[i - 1] > o[i - 1] and bearish_displacement:
                    zone_low = float(min(o[i - 1], c[i - 1]))
                    zone_high = float(h[i - 1])
                    zone = {'low': zone_low, 'high': zone_high, 'created_at': i, 'mitigated': False}
                    active_supply.append(zone)
                    ob_supply[i] = 1
                    ob_supply_low[i] = zone_low
                    ob_supply_high[i] = zone_high

            demand_candidates = [z for z in active_demand if not z['mitigated']]
            supply_candidates = [z for z in active_supply if not z['mitigated']]
            if demand_candidates:
                nearest = min(demand_candidates, key=lambda z: self._zone_distance_pct(c[i], z['low'], z['high']))
                near_demand_ob[i] = self._zone_distance_pct(c[i], nearest['low'], nearest['high'])
                if np.isnan(ob_demand_low[i]):
                    ob_demand_low[i] = nearest['low']
                    ob_demand_high[i] = nearest['high']
            if supply_candidates:
                nearest = min(supply_candidates, key=lambda z: self._zone_distance_pct(c[i], z['low'], z['high']))
                near_supply_ob[i] = self._zone_distance_pct(c[i], nearest['low'], nearest['high'])
                if np.isnan(ob_supply_low[i]):
                    ob_supply_low[i] = nearest['low']
                    ob_supply_high[i] = nearest['high']

        self.df['ob_supply'] = ob_supply
        self.df['ob_demand'] = ob_demand
        self.df['ob_supply_low'] = ob_supply_low
        self.df['ob_supply_high'] = ob_supply_high
        self.df['ob_demand_low'] = ob_demand_low
        self.df['ob_demand_high'] = ob_demand_high
        self.df['ob_supply_strength'] = np.where(ob_supply == 1, bodies / (atr + 1e-10), 0)
        self.df['ob_demand_strength'] = np.where(ob_demand == 1, bodies / (atr + 1e-10), 0)
        self.df['ob_supply_mitigated'] = ob_supply_mitigated
        self.df['ob_demand_mitigated'] = ob_demand_mitigated
        self.df['near_supply_ob'] = near_supply_ob
        self.df['near_demand_ob'] = near_demand_ob

        return self.df

    def identify_fvg(self):
        """Identify ICT Fair Value Gap zones and mitigation state.

        Bullish FVG: high two candles ago is below the current low.
        Bearish FVG: low two candles ago is above the current high.
        The full zone is retained and considered unfilled until a later candle
        trades back into the zone. Formation bars are not marked as filled.
        """
        h = self.df['h'].astype(float).values
        l = self.df['l'].astype(float).values
        c = self.df['c'].astype(float).values
        atr = pd.Series(h - l).rolling(14, min_periods=1).mean().replace(0, np.nan).ffill().fillna(1e-10).values
        n = len(self.df)

        fvg_bullish = np.zeros(n, dtype=int)
        fvg_bearish = np.zeros(n, dtype=int)
        fvg_size = np.zeros(n)
        fvg_bull_low = np.full(n, np.nan)
        fvg_bull_high = np.full(n, np.nan)
        fvg_bear_low = np.full(n, np.nan)
        fvg_bear_high = np.full(n, np.nan)
        fvg_bull_unfilled = np.zeros(n, dtype=int)
        fvg_bear_unfilled = np.zeros(n, dtype=int)
        fvg_bull_filled = np.zeros(n, dtype=int)
        fvg_bear_filled = np.zeros(n, dtype=int)
        near_bull_fvg = np.ones(n)
        near_bear_fvg = np.ones(n)

        active_bull = []
        active_bear = []

        for i in range(n):
            # Check mitigation before adding newly formed gaps, so a gap cannot
            # be filled on the same candle that created it.
            for gap in active_bull:
                if not gap['filled'] and i > gap['created_at'] and l[i] <= gap['high']:
                    gap['filled'] = True
                    fvg_bull_filled[i] = 1
            for gap in active_bear:
                if not gap['filled'] and i > gap['created_at'] and h[i] >= gap['low']:
                    gap['filled'] = True
                    fvg_bear_filled[i] = 1

            if i >= 2:
                min_gap = max(atr[i] * 0.1, abs(c[i]) * 0.00002)
                bull_gap = l[i] - h[i - 2]
                bear_gap = l[i - 2] - h[i]

                if bull_gap > min_gap:
                    zone_low = float(h[i - 2])
                    zone_high = float(l[i])
                    active_bull.append({'low': zone_low, 'high': zone_high, 'created_at': i, 'filled': False})
                    fvg_bullish[i] = 1
                    fvg_size[i] = bull_gap
                    fvg_bull_low[i] = zone_low
                    fvg_bull_high[i] = zone_high

                if bear_gap > min_gap:
                    zone_low = float(h[i])
                    zone_high = float(l[i - 2])
                    active_bear.append({'low': zone_low, 'high': zone_high, 'created_at': i, 'filled': False})
                    fvg_bearish[i] = 1
                    fvg_size[i] = -bear_gap
                    fvg_bear_low[i] = zone_low
                    fvg_bear_high[i] = zone_high

            bull_candidates = [g for g in active_bull if not g['filled']]
            bear_candidates = [g for g in active_bear if not g['filled']]
            if bull_candidates:
                nearest = min(bull_candidates, key=lambda g: self._zone_distance_pct(c[i], g['low'], g['high']))
                fvg_bull_unfilled[i] = 1
                near_bull_fvg[i] = self._zone_distance_pct(c[i], nearest['low'], nearest['high'])
                if np.isnan(fvg_bull_low[i]):
                    fvg_bull_low[i] = nearest['low']
                    fvg_bull_high[i] = nearest['high']
            if bear_candidates:
                nearest = min(bear_candidates, key=lambda g: self._zone_distance_pct(c[i], g['low'], g['high']))
                fvg_bear_unfilled[i] = 1
                near_bear_fvg[i] = self._zone_distance_pct(c[i], nearest['low'], nearest['high'])
                if np.isnan(fvg_bear_low[i]):
                    fvg_bear_low[i] = nearest['low']
                    fvg_bear_high[i] = nearest['high']

        self.df['fvg_bullish'] = fvg_bullish
        self.df['fvg_bearish'] = fvg_bearish
        self.df['fvg_size'] = fvg_size
        self.df['fvg_bull_low'] = fvg_bull_low
        self.df['fvg_bull_high'] = fvg_bull_high
        self.df['fvg_bear_low'] = fvg_bear_low
        self.df['fvg_bear_high'] = fvg_bear_high
        self.df['fvg_bull_unfilled'] = fvg_bull_unfilled
        self.df['fvg_bear_unfilled'] = fvg_bear_unfilled
        self.df['fvg_bull_filled'] = fvg_bull_filled
        self.df['fvg_bear_filled'] = fvg_bear_filled
        self.df['near_bull_fvg'] = near_bull_fvg
        self.df['near_bear_fvg'] = near_bear_fvg

        return self.df

    def identify_market_structure(self, lookback=5, depth=5, deviation=0.05):
        """Confirm swing structure, BOS and CHoCH without future leakage.

        Pivots are confirmed only after ``depth`` bars have elapsed to the
        right of a candidate pivot. The event is stamped on the confirmation
        bar, not on the historical pivot bar, so downstream models never see a
        future-confirmed swing before it was knowable in live trading.
        """
        h = self.df['h'].astype(float).values
        l = self.df['l'].astype(float).values
        c = self.df['c'].astype(float).values
        n = len(h)
        depth = max(2, int(depth or lookback or 5))
        min_break_pct = float(deviation) / 100.0

        swing_high = np.zeros(n, dtype=int)
        swing_low = np.zeros(n, dtype=int)
        confirmed_swing_high = np.full(n, np.nan)
        confirmed_swing_low = np.full(n, np.nan)
        bos_bullish = np.zeros(n, dtype=int)
        bos_bearish = np.zeros(n, dtype=int)
        choch_bullish = np.zeros(n, dtype=int)
        choch_bearish = np.zeros(n, dtype=int)
        structure = np.zeros(n, dtype=int)
        last_swing_high_series = np.full(n, np.nan)
        last_swing_low_series = np.full(n, np.nan)

        last_swing_high = np.nan
        last_swing_low = np.nan
        last_broken_high = np.nan
        last_broken_low = np.nan
        current_structure = 0

        for i in range(n):
            candidate = i - depth
            if candidate >= depth:
                left = candidate - depth
                right = candidate + depth + 1
                if h[candidate] >= np.max(h[left:right]):
                    last_swing_high = float(h[candidate])
                    confirmed_swing_high[i] = last_swing_high
                    swing_high[i] = 1
                if l[candidate] <= np.min(l[left:right]):
                    last_swing_low = float(l[candidate])
                    confirmed_swing_low[i] = last_swing_low
                    swing_low[i] = 1

            if not np.isnan(last_swing_high):
                threshold = last_swing_high * (1.0 + min_break_pct)
                if c[i] > threshold and (np.isnan(last_broken_high) or last_swing_high != last_broken_high):
                    if current_structure == -1:
                        choch_bullish[i] = 1
                    else:
                        bos_bullish[i] = 1
                    current_structure = 1
                    last_broken_high = last_swing_high

            if not np.isnan(last_swing_low):
                threshold = last_swing_low * (1.0 - min_break_pct)
                if c[i] < threshold and (np.isnan(last_broken_low) or last_swing_low != last_broken_low):
                    if current_structure == 1:
                        choch_bearish[i] = 1
                    else:
                        bos_bearish[i] = 1
                    current_structure = -1
                    last_broken_low = last_swing_low

            structure[i] = current_structure
            last_swing_high_series[i] = last_swing_high
            last_swing_low_series[i] = last_swing_low

        self.df['bos_bullish'] = bos_bullish
        self.df['bos_bearish'] = bos_bearish
        self.df['choch_bullish'] = choch_bullish
        self.df['choch_bearish'] = choch_bearish
        self.df['structure'] = structure
        self.df['swing_high'] = swing_high
        self.df['swing_low'] = swing_low
        self.df['confirmed_swing_high'] = confirmed_swing_high
        self.df['confirmed_swing_low'] = confirmed_swing_low
        self.df['last_swing_high'] = last_swing_high_series
        self.df['last_swing_low'] = last_swing_low_series
        self.df['zz_direction'] = structure

        return self.df

    def identify_liquidity_levels(self, lookback=20):
        """Identify prior liquidity pools and one-bar sweep/reclaim events."""
        h = pd.Series(self.df['h'].astype(float).values)
        l = pd.Series(self.df['l'].astype(float).values)
        c = pd.Series(self.df['c'].astype(float).values)

        highest_prev = h.shift(1).rolling(lookback, min_periods=max(3, lookback // 2)).max()
        lowest_prev = l.shift(1).rolling(lookback, min_periods=max(3, lookback // 2)).min()

        atr = (h - l).rolling(14, min_periods=1).mean()
        atr = atr.replace(0, np.nan).ffill().fillna(1e-10)

        self.df['ls_high'] = ((highest_prev - c) / atr).fillna(0).values
        self.df['ls_low'] = ((c - lowest_prev) / atr).fillna(0).values

        self.df['liq_sweep_high'] = (
            highest_prev.notna() & (h > highest_prev) & (c < highest_prev)
        ).astype(int).values

        self.df['liq_sweep_low'] = (
            lowest_prev.notna() & (l < lowest_prev) & (c > lowest_prev)
        ).astype(int).values

        return self.df

    def identify_breaker_blocks(self, lookback=10):
        """Flag breaker events when structure flips after a fresh OB exists."""
        structure = self.df['structure'].astype(int).values
        demand_recent = pd.Series(self.df.get('ob_demand', 0)).rolling(lookback, min_periods=1).max().values
        supply_recent = pd.Series(self.df.get('ob_supply', 0)).rolling(lookback, min_periods=1).max().values

        breaker = np.zeros(len(self.df))
        prev_structure = np.roll(structure, 1)
        prev_structure[0] = 0
        bullish_flip = (structure == 1) & (prev_structure == -1) & (supply_recent > 0)
        bearish_flip = (structure == -1) & (prev_structure == 1) & (demand_recent > 0)
        breaker[bullish_flip] = 1
        breaker[bearish_flip] = -1

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
        """Detect OTE after confirmed market structure and swing anchors."""
        c = pd.Series(self.df['c'].astype(float).values)
        last_high = pd.Series(self.df.get('last_swing_high', np.nan)).ffill()
        last_low = pd.Series(self.df.get('last_swing_low', np.nan)).ffill()
        structure = pd.Series(self.df.get('structure', 0)).astype(int)
        fib_range = last_high - last_low
        valid_range = fib_range.abs() > 1e-10

        buy_fib = (last_high - c) / (fib_range + 1e-10)
        sell_fib = (c - last_low) / (fib_range + 1e-10)

        self.df['in_ote_buy_zone'] = ((structure == 1) & valid_range & (buy_fib >= 0.618) & (buy_fib <= 0.786)).astype(int)
        self.df['in_ote_sell_zone'] = ((structure == -1) & valid_range & (sell_fib >= 0.618) & (sell_fib <= 0.786)).astype(int)
        self.df['fib_level'] = np.where(structure == -1, sell_fib, buy_fib)

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