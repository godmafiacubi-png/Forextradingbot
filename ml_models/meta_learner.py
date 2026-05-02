"""
Meta-Learning Module v1.0
"เรียนรู้ว่าควร trade ช่วงไหน"

แทนที่จะ predict ทิศทาง — meta-learner predict ว่า:
  "ตอนนี้ควร active / conservative / stay-out?"

Context features:
  - Market regime + confidence
  - Trading session (London/NY/Asian/Off)
  - Recent win rate (last 20 trades)
  - Recent Sharpe
  - Day-of-week, hour-of-day
  - Consecutive wins / losses
  - ATR percentile (volatility context)
  - Symbol-specific recent performance

Output:
  - activity_score: 0.0–1.0 (trade aggression multiplier)
  - should_trade: bool
  - confidence_scale: multiplier applied to base signal confidence
"""

import os
import logging
import pickle
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# 1. Meta Context Builder
# ============================================================
def _deque50():
    return deque(maxlen=50)

def _deque200():
    return deque(maxlen=200)


class MetaContextBuilder:
    """
    Builds the meta-feature vector from recent trade history +
    current market conditions.
    """

    META_FEATURE_SIZE = 24

    def __init__(self):
        # Rolling windows per symbol
        self._recent_trades: Dict[str, deque] = defaultdict(_deque50)
        self._recent_pnl_pct: Dict[str, deque] = defaultdict(_deque50)
        self._global_trades: deque = deque(maxlen=100)
        self._atr_history: Dict[str, deque] = defaultdict(_deque200)
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._total_trades = 0

    def record_trade(self, symbol: str, pnl: float, pnl_pct: float):
        won = 1 if pnl > 0 else 0
        self._recent_trades[symbol].append(won)
        self._recent_pnl_pct[symbol].append(pnl_pct)
        self._global_trades.append(won)
        self._total_trades += 1
        if won:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    def record_atr(self, symbol: str, atr_pct: float):
        self._atr_history[symbol].append(atr_pct)

    def build(
        self,
        symbol: str,
        regime: str,
        regime_confidence: float,
        atr_pct: float,
        adx: float,
        session: str,
        ml_prob: float,
        base_confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """Returns meta-feature vector of size META_FEATURE_SIZE"""
        ts = timestamp or datetime.now()

        # ---- Session encoding (4 binary) ----
        is_london = 1.0 if "LONDON" in session else 0.0
        is_ny = 1.0 if "NY" in session or "NEW_YORK" in session else 0.0
        is_overlap = 1.0 if "OVERLAP" in session else 0.0
        is_asian = 1.0 if "ASIAN" in session else 0.0

        # ---- Time features (cyclical) ----
        hour = ts.hour
        dow = ts.weekday()  # 0=Mon … 4=Fri
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dow / 5)
        dow_cos = np.cos(2 * np.pi * dow / 5)
        is_weekend = 1.0 if dow >= 5 else 0.0

        # ---- Regime encoding (4 one-hot) ----
        regime_map = {"TRENDING": 0, "RANGING": 1, "VOLATILE": 2, "QUIET": 3}
        regime_vec = [0.0, 0.0, 0.0, 0.0]
        regime_vec[regime_map.get(regime, 3)] = 1.0

        # ---- Recent performance — symbol-specific ----
        sym_trades = list(self._recent_trades.get(symbol, []))
        sym_pnls = list(self._recent_pnl_pct.get(symbol, []))
        sym_wr_20 = np.mean(sym_trades[-20:]) if len(sym_trades) >= 5 else 0.5
        sym_wr_5 = np.mean(sym_trades[-5:]) if len(sym_trades) >= 5 else 0.5
        sym_sharpe = (
            np.mean(sym_pnls[-20:]) / (np.std(sym_pnls[-20:]) + 1e-10)
            if len(sym_pnls) >= 10 else 0.0
        )

        # ---- Recent performance — global ----
        global_trades = list(self._global_trades)
        global_wr = np.mean(global_trades[-20:]) if len(global_trades) >= 5 else 0.5

        # ---- Consecutive streaks (normalised) ----
        consec_wins_n = np.clip(self._consecutive_wins / 10, 0, 1)
        consec_loss_n = np.clip(self._consecutive_losses / 10, 0, 1)

        # ---- ATR percentile (volatility context) ----
        atr_hist = list(self._atr_history.get(symbol, []))
        if len(atr_hist) >= 20:
            atr_pct_rank = np.mean(np.array(atr_hist) <= atr_pct)
        else:
            atr_pct_rank = 0.5

        # ---- Market signals ----
        adx_n = np.clip(adx / 50, 0, 1)
        ml_prob_n = ml_prob
        base_conf_n = base_confidence
        regime_conf_n = regime_confidence

        # ---- Assemble (28 features) ----
        features = np.array([
            # Session (4)
            is_london, is_ny, is_overlap, is_asian,
            # Time (5)
            hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
            # Regime (4 + 1 confidence)
            *regime_vec, regime_conf_n,
            # Symbol performance (3)
            sym_wr_20, sym_wr_5, np.clip(sym_sharpe, -3, 3) / 3,
            # Global performance (1)
            global_wr,
            # Streaks (2)
            consec_wins_n, consec_loss_n,
            # Volatility (2)
            atr_pct_rank, adx_n,
            # Signals (2)
            ml_prob_n, base_conf_n,
        ], dtype=np.float32)

        assert len(features) == self.META_FEATURE_SIZE, \
            f"Meta feature size mismatch: {len(features)} != {self.META_FEATURE_SIZE}"
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


# ============================================================
# 2. Neural Meta-Learner (light MLP)
# ============================================================
if TORCH_AVAILABLE:
    class MetaNet(nn.Module):
        """
        Small MLP that takes meta-context and outputs:
          - activity_score: sigmoid → 0..1
          - confidence_scale: sigmoid → 0.5..1.5
        """
        def __init__(self, input_size: int, hidden: int = 64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_size, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )
            self.activity_head = nn.Linear(hidden // 2, 1)
            self.conf_scale_head = nn.Linear(hidden // 2, 1)

        def forward(self, x):
            h = self.shared(x)
            activity = torch.sigmoid(self.activity_head(h))
            conf_scale = 0.5 + torch.sigmoid(self.conf_scale_head(h))  # 0.5–1.5
            return activity, conf_scale


# ============================================================
# 3. Meta Experience Buffer
# ============================================================
class MetaExperience:
    """Stores (meta_context, outcome) pairs for training the meta-learner"""
    def __init__(self, capacity: int = 5000):
        self._contexts: deque = deque(maxlen=capacity)
        self._outcomes: deque = deque(maxlen=capacity)  # 1=good_trade, 0=bad_trade

    def push(self, context: np.ndarray, won: bool):
        self._contexts.append(context.copy())
        self._outcomes.append(1.0 if won else 0.0)

    def sample(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(self._contexts) < batch_size:
            return None
        import random
        indices = random.sample(range(len(self._contexts)), batch_size)
        X = np.array([list(self._contexts)[i] for i in indices])
        y = np.array([list(self._outcomes)[i] for i in indices])
        return X, y

    def __len__(self):
        return len(self._contexts)


# ============================================================
# 4. Meta-Learner (main class)
# ============================================================
class MetaLearner:
    """
    Decides WHEN to trade based on context.

    Usage:
        meta = MetaLearner()

        # Every bar:
        meta_ctx = meta.context.build(symbol, regime, ...)
        meta.push_context(meta_ctx, symbol)

        # Before entry:
        result = meta.should_trade(meta_ctx, base_confidence)
        if result.trade:
            adjusted_conf = base_confidence * result.confidence_scale

        # After close:
        meta.record_outcome(meta_ctx, pnl > 0)
    """

    def __init__(
        self,
        min_train_samples: int = 200,
        train_every: int = 50,
        activity_threshold: float = 0.40,
        device: Optional[str] = None,
    ):
        self.context = MetaContextBuilder()
        self.experience = MetaExperience()
        self.min_train_samples = min_train_samples
        self.train_every = train_every
        self.activity_threshold = activity_threshold
        self._train_calls = 0
        self._is_trained = False

        self.device = torch.device(
            device if device else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        )

        if TORCH_AVAILABLE:
            self._model = MetaNet(
                input_size=MetaContextBuilder.META_FEATURE_SIZE
            ).to(self.device)
            self._model.eval()
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        else:
            self._model = None

        # Fallback GBM for when neural net not yet trained
        if SKLEARN_AVAILABLE:
            self._gbm = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            )
            self._gbm_scaler = StandardScaler()
            self._gbm_trained = False
        else:
            self._gbm = None

        # Stats
        self._total_filtered = 0
        self._total_passed = 0
        self._activity_history: deque = deque(maxlen=200)

        logger.info(
            f"[MetaLearner] Init: threshold={activity_threshold} "
            f"min_samples={min_train_samples} device={self.device}"
        )

    # ----------------------------------------------------------
    # Core API
    # ----------------------------------------------------------

    class MetaResult:
        __slots__ = ["trade", "activity_score", "confidence_scale", "source"]
        def __init__(self, trade, activity_score, confidence_scale, source):
            self.trade = trade
            self.activity_score = activity_score
            self.confidence_scale = confidence_scale
            self.source = source

    def evaluate(
        self,
        meta_ctx: np.ndarray,
        base_confidence: float,
    ) -> "MetaLearner.MetaResult":
        """
        Returns MetaResult with:
          .trade            — bool: proceed with trade?
          .activity_score   — 0..1 raw score
          .confidence_scale — multiplier for base_confidence
          .source           — which model made the decision
        """
        activity, conf_scale, source = self._predict(meta_ctx)
        self._activity_history.append(activity)

        should_trade = activity >= self.activity_threshold
        if should_trade:
            self._total_passed += 1
        else:
            self._total_filtered += 1

        return self.MetaResult(
            trade=should_trade,
            activity_score=float(activity),
            confidence_scale=float(conf_scale),
            source=source,
        )

    def record_trade(self, symbol: str, pnl: float, pnl_pct: float):
        """Call after every trade close"""
        self.context.record_trade(symbol, pnl, pnl_pct)

    def record_atr(self, symbol: str, atr_pct: float):
        self.context.record_atr(symbol, atr_pct)

    def record_outcome(self, meta_ctx: np.ndarray, won: bool):
        """Push (context, outcome) to experience buffer, trigger training"""
        self.experience.push(meta_ctx, won)
        self._train_calls += 1
        if self._train_calls % self.train_every == 0:
            self._train()

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def _predict(self, meta_ctx: np.ndarray) -> Tuple[float, float, str]:
        """Returns (activity_score, conf_scale, source)"""
        # Neural net (preferred when trained)
        if TORCH_AVAILABLE and self._model is not None and self._is_trained:
            try:
                t = torch.FloatTensor(meta_ctx).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    act, cs = self._model(t)
                return float(act.item()), float(cs.item()), "meta_nn"
            except Exception as e:
                logger.debug(f"[MetaLearner] NN predict error: {e}")

        # GBM fallback
        if SKLEARN_AVAILABLE and self._gbm is not None and self._gbm_trained:
            try:
                ctx_scaled = self._gbm_scaler.transform(meta_ctx.reshape(1, -1))
                prob = self._gbm.predict_proba(ctx_scaled)[0, 1]
                conf_scale = 0.8 + prob * 0.4  # 0.8–1.2
                return float(prob), float(conf_scale), "meta_gbm"
            except Exception:
                pass

        # Default: pass through with neutral score
        return 0.6, 1.0, "meta_default"

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def _train(self):
        """Train both neural net and GBM fallback"""
        if len(self.experience) < self.min_train_samples:
            return

        batch = self.experience.sample(min(len(self.experience), 2000))
        if batch is None:
            return
        X, y = batch

        # ---- Neural net ----
        if TORCH_AVAILABLE and self._model is not None:
            try:
                self._model.train()
                X_t = torch.FloatTensor(X).to(self.device)
                y_t = torch.FloatTensor(y).to(self.device)

                for _ in range(5):  # small inner loop
                    act, _ = self._model(X_t)
                    loss = F.binary_cross_entropy(act.squeeze(-1), y_t)
                    self._optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    self._optimizer.step()

                self._model.eval()
                self._is_trained = True
                logger.info(
                    f"[MetaLearner] NN trained: {len(X)} samples, loss={loss.item():.4f}"
                )
            except Exception as e:
                logger.warning(f"[MetaLearner] NN train error: {e}")

        # ---- GBM ----
        if SKLEARN_AVAILABLE and self._gbm is not None and len(y) >= 50:
            try:
                X_scaled = self._gbm_scaler.fit_transform(X)
                self._gbm.fit(X_scaled, y.astype(int))
                self._gbm_trained = True
                logger.info(f"[MetaLearner] GBM trained: {len(X)} samples")
            except Exception as e:
                logger.warning(f"[MetaLearner] GBM train error: {e}")

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        try:
            state = {
                "is_trained": self._is_trained,
                "gbm_trained": getattr(self, "_gbm_trained", False),
                "total_filtered": self._total_filtered,
                "total_passed": self._total_passed,
                "train_calls": self._train_calls,
                "activity_threshold": self.activity_threshold,
            }
            if TORCH_AVAILABLE and self._model is not None:
                state["model_state"] = self._model.state_dict()
            if SKLEARN_AVAILABLE and getattr(self, "_gbm_trained", False):
                state["gbm"] = self._gbm
                state["gbm_scaler"] = self._gbm_scaler
            with open(path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"[MetaLearner] Saved to {path}")
        except Exception as e:
            logger.error(f"[MetaLearner] Save error: {e}")

    def load(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._is_trained = state.get("is_trained", False)
            self._total_filtered = state.get("total_filtered", 0)
            self._total_passed = state.get("total_passed", 0)
            self._train_calls = state.get("train_calls", 0)
            if TORCH_AVAILABLE and self._model and "model_state" in state:
                self._model.load_state_dict(state["model_state"])
                self._model.eval()
            if SKLEARN_AVAILABLE and "gbm" in state:
                self._gbm = state["gbm"]
                self._gbm_scaler = state["gbm_scaler"]
                self._gbm_trained = True
            logger.info(f"[MetaLearner] Loaded from {path}")
        except Exception as e:
            logger.warning(f"[MetaLearner] Load error: {e}")

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def get_stats(self) -> Dict:
        total = self._total_passed + self._total_filtered
        pass_rate = self._total_passed / max(total, 1)
        avg_activity = float(np.mean(self._activity_history)) if self._activity_history else 0.5
        return {
            "is_trained": self._is_trained,
            "gbm_trained": getattr(self, "_gbm_trained", False),
            "experience_size": len(self.experience),
            "total_evaluated": total,
            "pass_rate": round(pass_rate, 3),
            "filtered_out": self._total_filtered,
            "avg_activity_score": round(avg_activity, 3),
            "activity_threshold": self.activity_threshold,
            "device": str(self.device),
        }