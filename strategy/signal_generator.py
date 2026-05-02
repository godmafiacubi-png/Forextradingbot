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
        Generate signals with ICT-First decision tree (v8.0).

        ICT Score (0-6):
        +1 for each: OB, FVG, BOS/CHoCH, OTE zone, Structure, Liquidity sweep

        v8.0 Changes (Prop Firm / Hedge Fund mindset):
        - GATE 1: ICT >= 2 is a hard gate — no ICT, no signal (no exceptions)
        - GATE 2: ML must confirm direction (ml_prob > threshold)
        - Exception: ICT >= 3 + borderline ML (buy: >0.52, sell: <0.48) allowed with confidence penalty
        - Removed: ML-only mode (ml_prob > 0.60 without ICT) — too risky
        - Removed: ML + ADX mode (without ICT) — too risky
        - Confidence = ICT weight 60% + ML weight 40%
        - ICT bonus weights increased (ICT is now core, not bonus)
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

            # ===== NEW: ICT-FIRST DECISION TREE (v8.0) =====
            signal = 0
            confidence = 0.0

            # GATE 1: ICT must be >= 2 (hard gate, no exceptions)
            if ict_buy >= 2:
                # GATE 2: ML must confirm direction
                if ml_prob > ml_threshold_buy:
                    signal = 1
                    # Confidence = ICT weight 60% + ML weight 40%
                    ict_conf = min(ict_buy / 4.0, 1.0)   # max at 4 ICT signals
                    ml_conf  = (ml_prob - 0.5) * 2.0
                    confidence = ict_conf * 0.60 + np.clip(ml_conf, 0, 1) * 0.40
                # ICT strong (>=3) but ML borderline — allow with penalty
                elif ict_buy >= 3 and ml_prob > 0.52:
                    signal = 1
                    ict_conf = min(ict_buy / 4.0, 1.0)
                    confidence = ict_conf * 0.70   # lower weight, no ML boost

            elif ict_sell >= 2:
                # GATE 2: ML must confirm direction
                if ml_prob < ml_threshold_sell:
                    signal = -1
                    ict_conf = min(ict_sell / 4.0, 1.0)
                    ml_conf  = (0.5 - ml_prob) * 2.0
                    confidence = ict_conf * 0.60 + np.clip(ml_conf, 0, 1) * 0.40
                # ICT strong (>=3) but ML borderline — allow with penalty
                elif ict_sell >= 3 and ml_prob < 0.48:
                    signal = -1
                    ict_conf = min(ict_sell / 4.0, 1.0)
                    confidence = ict_conf * 0.70

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

            # ===== ICT BONUS (increased weight — ICT is now core) =====
            if signal == 1:
                if ict_buy >= 4:
                    confidence += 0.15
                elif ict_buy >= 3:
                    confidence += 0.10
                elif ict_buy >= 2:
                    confidence += 0.05
            elif signal == -1:
                if ict_sell >= 4:
                    confidence += 0.15
                elif ict_sell >= 3:
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