import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Built-in technical indicators without external dependencies"""
    
    @staticmethod
    def sma(data, period=20):
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period=12):
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index"""
        # Plus DM
        up = high.diff()
        down = low.diff() * -1
        
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
        
        tr = TechnicalIndicators.atr(high, low, close, period=1)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    @staticmethod
    def obv(close, volume):
        """On-Balance Volume"""
        obv = pd.Series(np.where(close > close.shift(), volume, 
                                 np.where(close < close.shift(), -volume, 0)), 
                       index=close.index).cumsum()
        return obv
    
    @staticmethod
    def stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=smooth_d).mean()
        
        return k_smooth, d
    
    @staticmethod
    def vpt(close, volume):
        """Volume Price Trend"""
        vpt = pd.Series(np.where(close.diff() == 0, 0, 
                                (close.diff() / close.shift()) * volume),
                       index=close.index).cumsum()
        return vpt
    
    @staticmethod
    def add_all_indicators(df):
        """Add all indicators to dataframe"""
        # Moving Averages
        df['sma_20'] = TechnicalIndicators.sma(df['c'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['c'], 50)
        df['sma_200'] = TechnicalIndicators.sma(df['c'], 200)
        df['ema_12'] = TechnicalIndicators.ema(df['c'], 12)
        df['ema_26'] = TechnicalIndicators.ema(df['c'], 26)
        
        # Momentum
        df['rsi'] = TechnicalIndicators.rsi(df['c'], 14)
        macd, signal, hist = TechnicalIndicators.macd(df['c'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Volatility
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df['c'], 20, 2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        df['atr'] = TechnicalIndicators.atr(df['h'], df['l'], df['c'], 14)
        df['adx'] = TechnicalIndicators.adx(df['h'], df['l'], df['c'], 14)
        
        # Volume
        df['obv'] = TechnicalIndicators.obv(df['c'], df['v'])
        df['vpt'] = TechnicalIndicators.vpt(df['c'], df['v'])
        
        # Stochastic
        k, d = TechnicalIndicators.stochastic(df['h'], df['l'], df['c'])
        df['stoch_k'] = k
        df['stoch_d'] = d
        
        return df