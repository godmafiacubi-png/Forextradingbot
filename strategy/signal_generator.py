import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate signals with ML + ICT confluence scoring"""

    def __init__(self, ml_model):
        self.ml_model = ml_model

    def generate_signals(self, df, ml_threshold_buy=0.54, ml_threshold_sell=0.46):
        """
        Generate signals with ICT confluence scoring.

        ICT Score (0-6):
        +1 for each: OB, FVG, BOS/CHoCH, OTE zone, Structure, Liquidity sweep

        v7.1 Changes:
        - ICT >= 1 ก็ให้ signal ได้ (เดิม >= 2 เข้มเกิน)
        - ML only mode: ML prob สูงมาก (>0.60) ไม่ต้องมี ICT ก็ได้
        - Confidence formula ปรับให้สมดุลกว่าเดิม
        """
        df = df.copy()
        df['signal'] = 0
        df['confidence'] = 0.0
        df['ict_score'] = 0

        try:
            ml_probs = self.ml_model.predict(df)
        except Exception:
            ml_probs = np.full(len(df), 0.5)

        for i in range(len(df)):
            ml_prob = float(ml_probs[i]) if i < len(ml_probs) else 0.5

            # ===== ICT CONFLUENCE SCORE =====
            ict_buy = 0
            ict_sell = 0

            # Order Block
            if df.iloc[i].get('ob_demand', 0):
                ict_buy += 1
            if df.iloc[i].get('ob_supply', 0):
                ict_sell += 1

            # FVG
            if df.iloc[i].get('fvg_bullish', 0):
                ict_buy += 1
            if df.iloc[i].get('fvg_bearish', 0):
                ict_sell += 1

            # BOS / CHoCH
            if df.iloc[i].get('bos_bullish', 0) or df.iloc[i].get('choch_bullish', 0):
                ict_buy += 1
            if df.iloc[i].get('bos_bearish', 0) or df.iloc[i].get('choch_bearish', 0):
                ict_sell += 1

            # OTE Zone
            if df.iloc[i].get('in_ote_buy_zone', 0):
                ict_buy += 1
            if df.iloc[i].get('in_ote_sell_zone', 0):
                ict_sell += 1

            # Structure
            structure = int(df.iloc[i].get('structure', 0))
            if structure == 1:
                ict_buy += 1
            elif structure == -1:
                ict_sell += 1

            # Liquidity sweep (counter-trend signal)
            if df.iloc[i].get('liq_sweep_low', 0):
                ict_buy += 1
            if df.iloc[i].get('liq_sweep_high', 0):
                ict_sell += 1

            # ===== TECHNICAL INDICATORS =====
            rsi = float(df.iloc[i].get('rsi', 50))
            macd_hist = float(df.iloc[i].get('macd_hist', 0))
            stoch_k = float(df.iloc[i].get('stoch_k', 50))
            adx = float(df.iloc[i].get('adx', 0))
            ema_cross = float(df.iloc[i].get('ema_cross', 0))

            # ===== DETERMINE SIGNAL =====
            signal = 0
            confidence = 0.0

            # ----- BUY CONDITIONS -----
            if ml_prob > ml_threshold_buy:
                if ict_buy >= 1:
                    # ML + ICT confluence
                    signal = 1
                    ml_conf = (ml_prob - 0.5) * 2.5
                    ict_conf = min(ict_buy / 3.0, 1.0)
                    confidence = ml_conf * 0.55 + ict_conf * 0.45

                elif ml_prob > 0.60:
                    # Strong ML only — ไม่ต้อง ICT
                    signal = 1
                    ml_conf = (ml_prob - 0.5) * 2.5
                    confidence = ml_conf * 0.70

                elif ml_prob > ml_threshold_buy and adx > 25:
                    # ML + strong trend
                    signal = 1
                    ml_conf = (ml_prob - 0.5) * 2.5
                    confidence = ml_conf * 0.60

            # ----- SELL CONDITIONS -----
            elif ml_prob < ml_threshold_sell:
                if ict_sell >= 1:
                    signal = -1
                    ml_conf = (0.5 - ml_prob) * 2.5
                    ict_conf = min(ict_sell / 3.0, 1.0)
                    confidence = ml_conf * 0.55 + ict_conf * 0.45

                elif ml_prob < 0.40:
                    # Strong ML only
                    signal = -1
                    ml_conf = (0.5 - ml_prob) * 2.5
                    confidence = ml_conf * 0.70

                elif ml_prob < ml_threshold_sell and adx > 25:
                    signal = -1
                    ml_conf = (0.5 - ml_prob) * 2.5
                    confidence = ml_conf * 0.60

            # ===== TECH INDICATOR BONUS =====
            if signal == 1:
                if rsi < 50:
                    confidence += 0.05
                if macd_hist > 0:
                    confidence += 0.04
                if stoch_k < 40:
                    confidence += 0.05
                if ema_cross > 0:
                    confidence += 0.03
                if adx > 25:
                    confidence += 0.04
                if adx > 35:
                    confidence += 0.03
            elif signal == -1:
                if rsi > 50:
                    confidence += 0.05
                if macd_hist < 0:
                    confidence += 0.04
                if stoch_k > 60:
                    confidence += 0.05
                if ema_cross < 0:
                    confidence += 0.03
                if adx > 25:
                    confidence += 0.04
                if adx > 35:
                    confidence += 0.03

            # ===== ICT BONUS =====
            if signal == 1:
                if ict_buy >= 3:
                    confidence += 0.10
                elif ict_buy >= 2:
                    confidence += 0.05
            elif signal == -1:
                if ict_sell >= 3:
                    confidence += 0.10
                elif ict_sell >= 2:
                    confidence += 0.05

            # ===== PENALTY for conflicting signals =====
            if signal == 1 and ict_sell >= 2:
                confidence *= 0.6
            if signal == -1 and ict_buy >= 2:
                confidence *= 0.6

            # Pin bar bonus
            if signal == 1 and df.iloc[i].get('is_pin_bar_bull', 0):
                confidence += 0.08
            if signal == -1 and df.iloc[i].get('is_pin_bar_bear', 0):
                confidence += 0.08

            # Engulfing bonus
            if signal == 1 and df.iloc[i].get('is_engulfing_bull', 0):
                confidence += 0.05
            if signal == -1 and df.iloc[i].get('is_engulfing_bear', 0):
                confidence += 0.05

            # Volume spike bonus
            if signal != 0 and df.iloc[i].get('vol_spike', 0):
                confidence += 0.05

            confidence = np.clip(confidence, 0.0, 1.0)

            df.iloc[i, df.columns.get_loc('signal')] = signal
            df.iloc[i, df.columns.get_loc('confidence')] = confidence
            df.iloc[i, df.columns.get_loc('ict_score')] = max(ict_buy, ict_sell)

        return df