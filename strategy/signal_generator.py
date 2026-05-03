import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Maximum additive bonus that can be applied to base confidence per signal bar
_MAX_TOTAL_BONUS = 0.25


class SignalGenerator:
    """Generate signals with ML + ICT confluence scoring"""

    def __init__(self, ml_model):
        self.ml_model = ml_model

    def generate_signals(self, df, ml_threshold_buy=0.54, ml_threshold_sell=0.46):
        """
        Generate signals with ICT confluence scoring.

        ICT Score (0-6):
        +1 for each: OB, FVG, BOS/CHoCH, OTE zone, Structure, Liquidity sweep

        v8.0 Prop Firm Changes:
        - ICT >= 2 is HARD GATE (mandatory, no ML-only exceptions)
        - ML is CONFIRMATION layer (not trigger)
        - ICT weight: 60%, ML weight: 40% in base confidence
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

            # ===== NEW v8.0: ICT-FIRST DECISION TREE =====
            # GATE 1: ICT >= 2 required (hard gate)
            signal = 0
            confidence = 0.0

            if ict_buy >= 2:
                # GATE 2: ML must confirm direction
                if ml_prob > ml_threshold_buy:
                    signal = 1
                    ict_conf = min(ict_buy / 4.0, 1.0)
                    ml_conf  = np.clip((ml_prob - 0.5) * 2.0, 0, 1)
                    confidence = ict_conf * 0.60 + ml_conf * 0.40
                elif ict_buy >= 3 and ml_prob > 0.52:
                    # Very strong ICT, borderline ML
                    signal = 1
                    ict_conf = min(ict_buy / 4.0, 1.0)
                    confidence = ict_conf * 0.65

            elif ict_sell >= 2:
                if ml_prob < ml_threshold_sell:
                    signal = -1
                    ict_conf = min(ict_sell / 4.0, 1.0)
                    ml_conf  = np.clip((0.5 - ml_prob) * 2.0, 0, 1)
                    confidence = ict_conf * 0.60 + ml_conf * 0.40
                elif ict_sell >= 3 and ml_prob < 0.48:
                    signal = -1
                    ict_conf = min(ict_sell / 4.0, 1.0)
                    confidence = ict_conf * 0.65

            # ===== TECH INDICATOR BONUS (cap at 0.15) =====
            tech_bonus = 0.0
            if signal == 1:
                if rsi < 50:
                    tech_bonus += 0.05
                if macd_hist > 0:
                    tech_bonus += 0.04
                if stoch_k < 40:
                    tech_bonus += 0.05
                if ema_cross > 0:
                    tech_bonus += 0.03
                if adx > 25:
                    tech_bonus += 0.04
                if adx > 35:
                    tech_bonus += 0.03
            elif signal == -1:
                if rsi > 50:
                    tech_bonus += 0.05
                if macd_hist < 0:
                    tech_bonus += 0.04
                if stoch_k > 60:
                    tech_bonus += 0.05
                if ema_cross < 0:
                    tech_bonus += 0.03
                if adx > 25:
                    tech_bonus += 0.04
                if adx > 35:
                    tech_bonus += 0.03
            tech_bonus = min(tech_bonus, 0.15)

            # ===== ICT BONUS =====
            ict_bonus = 0.0
            if signal == 1:
                if ict_buy >= 4:   ict_bonus = 0.15
                elif ict_buy >= 3: ict_bonus = 0.10
                elif ict_buy >= 2: ict_bonus = 0.05
            elif signal == -1:
                if ict_sell >= 4:   ict_bonus = 0.15
                elif ict_sell >= 3: ict_bonus = 0.10
                elif ict_sell >= 2: ict_bonus = 0.05

            # ===== ZIGZAG DIRECTION ALIGNMENT FILTER =====
            zz_bonus = 0.0
            zz_penalty = False
            zz_dir = int(df.iloc[i].get('zz_direction', 0))
            if zz_dir != 0:
                if signal == 1 and zz_dir == -1:
                    zz_penalty = True    # Counter-trend buy in bearish swing — penalise
                elif signal == -1 and zz_dir == 1:
                    zz_penalty = True    # Counter-trend sell in bullish swing — penalise
                elif signal == 1 and zz_dir == 1:
                    zz_bonus = 0.04      # Aligned with bullish zigzag
                elif signal == -1 and zz_dir == -1:
                    zz_bonus = 0.04      # Aligned with bearish zigzag

            # ===== PENALTY for conflicting signals =====
            if signal == 1 and ict_sell >= 2:
                confidence *= 0.6
            if signal == -1 and ict_buy >= 2:
                confidence *= 0.6

            # ===== PATTERN BONUS (cap at 0.10) =====
            pattern_bonus = 0.0
            if signal == 1 and df.iloc[i].get('is_pin_bar_bull', 0):
                pattern_bonus += 0.08
            if signal == -1 and df.iloc[i].get('is_pin_bar_bear', 0):
                pattern_bonus += 0.08
            if signal == 1 and df.iloc[i].get('is_engulfing_bull', 0):
                pattern_bonus += 0.05
            if signal == -1 and df.iloc[i].get('is_engulfing_bear', 0):
                pattern_bonus += 0.05
            if signal != 0 and df.iloc[i].get('vol_spike', 0):
                pattern_bonus += 0.05
            pattern_bonus = min(pattern_bonus, 0.10)

            # ===== CAP TOTAL BONUS AT _MAX_TOTAL_BONUS =====
            total_bonus = min(tech_bonus + ict_bonus + pattern_bonus + zz_bonus, _MAX_TOTAL_BONUS)
            confidence = confidence + total_bonus

            # Apply zigzag counter-trend penalty (after bonuses)
            if zz_penalty:
                confidence *= 0.50

            confidence = np.clip(confidence, 0.0, 1.0)

            df.iloc[i, df.columns.get_loc('signal')] = signal
            df.iloc[i, df.columns.get_loc('confidence')] = confidence
            df.iloc[i, df.columns.get_loc('ict_score')] = max(ict_buy, ict_sell)

        return df