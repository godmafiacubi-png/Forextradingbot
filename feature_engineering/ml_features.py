import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MLFeatures:
    """Technical indicators + statistical features for ML"""

    def __init__(self, df):
        self.df = df.copy()

    def add_moving_averages(self):
        c = self.df['c']
        self.df['sma_20'] = c.rolling(20).mean()
        self.df['sma_50'] = c.rolling(50).mean()
        self.df['sma_200'] = c.rolling(200, min_periods=50).mean()
        self.df['ema_12'] = c.ewm(span=12).mean()
        self.df['ema_26'] = c.ewm(span=26).mean()
        self.df['ema_9'] = c.ewm(span=9).mean()

        self.df['price_vs_sma20'] = (c - self.df['sma_20']) / (self.df['sma_20'] + 1e-10)
        self.df['price_vs_sma50'] = (c - self.df['sma_50']) / (self.df['sma_50'] + 1e-10)
        self.df['price_vs_sma200'] = (c - self.df['sma_200']) / (self.df['sma_200'] + 1e-10)

        self.df['ema_cross'] = np.sign(self.df['ema_12'] - self.df['ema_26'])
        self.df['sma_cross_20_50'] = np.sign(self.df['sma_20'] - self.df['sma_50'])

        # Pullback detection: price pulling back toward EMA
        self.df['pullback_to_ema'] = (
            (c - self.df['ema_26']).abs() / (self.df['ema_26'] + 1e-10)
        )

        return self.df

    def add_rsi(self, period=14):
        delta = self.df['c'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        self.df['rsi'] = 100 - (100 / (1 + rs))

        self.df['rsi_oversold'] = (self.df['rsi'] < 30).astype(int)
        self.df['rsi_overbought'] = (self.df['rsi'] > 70).astype(int)
        self.df['rsi_slope'] = self.df['rsi'].diff(5)
        self.df['price_slope'] = self.df['c'].pct_change(5) * 100

        # RSI divergence: price makes new high but RSI doesn't
        rsi = self.df['rsi']
        c = self.df['c']
        self.df['rsi_bull_div'] = (
            (c < c.shift(5)) & (rsi > rsi.shift(5))
        ).astype(int)
        self.df['rsi_bear_div'] = (
            (c > c.shift(5)) & (rsi < rsi.shift(5))
        ).astype(int)

        return self.df

    def add_macd(self):
        ema12 = self.df['c'].ewm(span=12).mean()
        ema26 = self.df['c'].ewm(span=26).mean()
        self.df['macd'] = ema12 - ema26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        self.df['macd_hist_change'] = np.sign(self.df['macd_hist']) - np.sign(self.df['macd_hist'].shift(1))

        # MACD zero line cross
        self.df['macd_above_zero'] = (self.df['macd'] > 0).astype(int)

        return self.df

    def add_bollinger_bands(self, period=20, std_dev=2):
        sma = self.df['c'].rolling(period).mean()
        std = self.df['c'].rolling(period).std()
        self.df['bb_upper'] = sma + (std * std_dev)
        self.df['bb_middle'] = sma
        self.df['bb_lower'] = sma - (std * std_dev)

        bb_width = self.df['bb_upper'] - self.df['bb_lower']
        self.df['bb_percent_b'] = (self.df['c'] - self.df['bb_lower']) / (bb_width + 1e-10)
        self.df['bb_width'] = bb_width / (sma + 1e-10)

        return self.df

    def add_atr(self, period=14):
        h = self.df['h']
        l = self.df['l']
        c = self.df['c']
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(period).mean()
        self.df['atr_pct'] = self.df['atr'] / (c + 1e-10)
        self.df['atr_change'] = self.df['atr'].pct_change(5)

        return self.df

    def add_adx(self, period=14):
        h = self.df['h']
        l = self.df['l']
        c = self.df['c']

        plus_dm = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)

        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        self.df['adx'] = dx.rolling(period).mean()
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
        self.df['strong_trend'] = (self.df['adx'] > 25).astype(int)
        self.df['di_cross'] = np.sign(plus_di - minus_di)

        return self.df

    def add_volume_features(self):
        v = self.df['v'] if 'v' in self.df.columns else self.df.get('tick_volume', pd.Series(0, index=self.df.index))
        c = self.df['c']

        self.df['obv'] = (np.sign(c.diff()) * v).cumsum()
        self.df['vpt'] = (c.pct_change() * v).cumsum()

        vol_ma = v.rolling(20).mean()
        self.df['vol_ratio'] = v / (vol_ma + 1e-10)

        vol_std = v.rolling(20).std()
        self.df['vol_spike'] = (v > vol_ma + 2 * vol_std).astype(int)

        return self.df

    def add_stochastic(self, k_period=14, d_period=3):
        h = self.df['h'].rolling(k_period).max()
        l = self.df['l'].rolling(k_period).min()
        self.df['stoch_k'] = 100 * (self.df['c'] - l) / (h - l + 1e-10)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(d_period).mean()
        self.df['stoch_cross'] = np.sign(self.df['stoch_k'] - self.df['stoch_d'])

        return self.df

    def add_price_action_features(self):
        o = self.df['o']
        h = self.df['h']
        l = self.df['l']
        c = self.df['c']

        body = (c - o).abs()
        total_range = h - l + 1e-10

        self.df['returns'] = c.pct_change()
        self.df['hl_range'] = (h - l) / (c + 1e-10)
        self.df['co_range'] = (c - o) / (c + 1e-10)

        self.df['highest_20'] = h.rolling(20).max()
        self.df['lowest_20'] = l.rolling(20).min()
        range_20 = self.df['highest_20'] - self.df['lowest_20']
        self.df['price_norm'] = (c - self.df['lowest_20']) / (range_20 + 1e-10)

        self.df['momentum'] = c - c.shift(10)
        self.df['roc'] = c.pct_change(10)
        self.df['hvol'] = self.df['returns'].rolling(20).std()

        self.df['body_size'] = body / total_range
        self.df['upper_wick'] = (h - pd.concat([o, c], axis=1).max(axis=1)) / total_range
        self.df['lower_wick'] = (pd.concat([o, c], axis=1).min(axis=1) - l) / total_range

        self.df['candle_type'] = np.sign(c - o)
        self.df['is_doji'] = (body / total_range < 0.1).astype(int)
        self.df['is_hammer'] = ((self.df['lower_wick'] > 0.6) & (self.df['upper_wick'] < 0.1)).astype(int)
        self.df['is_shooting_star'] = ((self.df['upper_wick'] > 0.6) & (self.df['lower_wick'] < 0.1)).astype(int)
        self.df['is_engulfing_bull'] = ((c > o) & (c.shift(1) < o.shift(1)) & (body > body.shift(1))).astype(int)
        self.df['is_engulfing_bear'] = ((c < o) & (c.shift(1) > o.shift(1)) & (body > body.shift(1))).astype(int)

        self.df['consecutive_bull'] = self._count_consecutive(c > o)
        self.df['consecutive_bear'] = self._count_consecutive(c < o)

        # Pin bar (strong rejection)
        self.df['is_pin_bar_bull'] = (
            (self.df['lower_wick'] > 0.65) &
            (self.df['body_size'] < 0.25) &
            (c > o)
        ).astype(int)
        self.df['is_pin_bar_bear'] = (
            (self.df['upper_wick'] > 0.65) &
            (self.df['body_size'] < 0.25) &
            (c < o)
        ).astype(int)

        return self.df

    def _count_consecutive(self, condition):
        result = np.zeros(len(condition))
        count = 0
        for i in range(len(condition)):
            if condition.iloc[i]:
                count += 1
            else:
                count = 0
            result[i] = count
        return result

    def add_keltner_channels(self, period=20, multiplier=2):
        c = self.df['c']
        h = self.df['h']
        l = self.df['l']

        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        self.df['kc_mid'] = c.ewm(span=period).mean()
        self.df['kc_upper'] = self.df['kc_mid'] + multiplier * atr
        self.df['kc_lower'] = self.df['kc_mid'] - multiplier * atr

        if 'bb_upper' in self.df.columns:
            self.df['squeeze'] = (
                (self.df['bb_upper'] < self.df['kc_upper']) &
                (self.df['bb_lower'] > self.df['kc_lower'])
            ).astype(int)

        return self.df

    def add_lag_features(self, lags=[1, 2, 3, 5]):
        for lag in lags:
            self.df[f'return_lag_{lag}'] = self.df['c'].pct_change(lag)
        return self.df

    def get_ml_features(self):
        """Generate all ML features"""
        try:
            logger.debug("Generating ML features...")

            self.add_moving_averages()
            self.add_rsi()
            self.add_macd()
            self.add_bollinger_bands()
            self.add_atr()
            self.add_adx()
            self.add_volume_features()
            self.add_stochastic()
            self.add_price_action_features()
            self.add_keltner_channels()
            self.add_lag_features()

            self.df = self.df.fillna(0)
            self.df = self.df.replace([np.inf, -np.inf], 0)

            logger.debug(f"ML features generated: {self.df.shape[1]} columns")
            return self.df

        except Exception as e:
            logger.error(f"Error generating ML features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self.df