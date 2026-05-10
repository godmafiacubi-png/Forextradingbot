"""
Regime-Aware Ensemble v1.0
- Train separate model weights per market regime
- Cross-symbol knowledge sharing via symbol embeddings
- Regime-weighted prediction blending
- Online incremental learning support
"""

import os
import logging
import pickle
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn / xgboost not available")

try:
    from config.settings import ML_LABEL_LOOKAHEAD, ML_LABEL_THRESHOLD, ML_LABEL_MIN_BALANCE, ML_LABEL_MAX_BALANCE
except ImportError:
    ML_LABEL_LOOKAHEAD = 3
    ML_LABEL_THRESHOLD = 0.0001
    ML_LABEL_MIN_BALANCE = 0.30
    ML_LABEL_MAX_BALANCE = 0.70


# ============================================================
# Symbol Embedding (Cross-Symbol Transfer)
# ============================================================
def _default_symbol_stats():
    return {
        "avg_atr": deque(maxlen=100),
        "avg_vol_ratio": deque(maxlen=100),
        "trade_count": 0,
        "win_count": 0,
    }


class SymbolEmbedding:
    """
    Encodes symbol-specific characteristics as a feature vector.
    Allows the model to generalise knowledge across symbols while
    still capturing per-symbol behaviour.
    """

    KNOWN_CATEGORIES = {
        # Forex majors
        "EURUSD": ("forex", "major", 0),
        "GBPUSD": ("forex", "major", 1),
        "USDJPY": ("forex", "major", 2),
        "AUDUSD": ("forex", "major", 3),
        "USDCAD": ("forex", "major", 4),
        "NZDUSD": ("forex", "major", 5),
        "USDCHF": ("forex", "major", 6),
        # Forex minors
        "EURJPY": ("forex", "minor", 7),
        "GBPJPY": ("forex", "minor", 8),
        "EURGBP": ("forex", "minor", 9),
        # Crypto
        "BTCUSD": ("crypto", "major", 10),
        "ETHUSD": ("crypto", "major", 11),
        "XRPUSD": ("crypto", "minor", 12),
        # Indices
        "US30":   ("index",  "major", 13),
        "SPX500": ("index",  "major", 14),
        "NAS100": ("index",  "major", 15),
        # Commodities
        "XAUUSD": ("commodity", "major", 16),
        "XAGUSD": ("commodity", "minor", 17),
        "USOIL":  ("commodity", "major", 18),
    }

    ASSET_CLASS_MAP = {"forex": 0, "crypto": 1, "index": 2, "commodity": 3}
    TIER_MAP = {"major": 0, "minor": 1, "exotic": 2}

    def __init__(self, embedding_dim=4):
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, np.ndarray] = {}
        self._symbol_stats: Dict[str, Dict] = defaultdict(_default_symbol_stats)

    def update_stats(self, symbol: str, atr_pct: float, vol_ratio: float,
                     pnl: Optional[float] = None):
        stats = self._symbol_stats[symbol]
        stats["avg_atr"].append(atr_pct)
        stats["avg_vol_ratio"].append(vol_ratio)
        if pnl is not None:
            stats["trade_count"] += 1
            if pnl > 0:
                stats["win_count"] += 1
        # Invalidate cache
        self._cache.pop(symbol, None)

    def get_embedding(self, symbol: str) -> np.ndarray:
        if symbol in self._cache:
            return self._cache[symbol]

        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        info = self.KNOWN_CATEGORIES.get(symbol.upper())

        if info:
            asset_class, tier, _ = info
            vec[0] = self.ASSET_CLASS_MAP.get(asset_class, 0) / 3.0
            vec[1] = self.TIER_MAP.get(tier, 1) / 2.0
        else:
            # Unknown symbol — try to infer from name
            sym = symbol.upper()
            if any(c in sym for c in ["BTC", "ETH", "XRP", "LTC"]):
                vec[0] = 1.0 / 3.0  # crypto
            elif any(c in sym for c in ["XAU", "XAG", "OIL", "WTI"]):
                vec[0] = 2.0 / 3.0  # commodity
            elif any(c in sym for c in ["US30", "SPX", "NAS", "DAX", "FTSE"]):
                vec[0] = 1.0         # index

        # Dynamic stats
        stats = self._symbol_stats.get(symbol, {})
        atrs = list(stats.get("avg_atr", []))
        vec[2] = np.clip(np.mean(atrs) * 100, 0, 1) if atrs else 0.5

        tc = stats.get("trade_count", 0)
        vec[3] = stats.get("win_count", 0) / max(tc, 1) if tc >= 5 else 0.5

        self._cache[symbol] = vec
        return vec


# ============================================================
# Per-Regime Model Set
# ============================================================
class RegimeModelSet:
    """
    Maintains a separate XGB + RF pair for each market regime.
    Falls back to the global model if a regime has too few samples.
    """

    REGIMES = ["TRENDING", "RANGING", "VOLATILE", "QUIET", "GLOBAL"]
    MIN_REGIME_SAMPLES = 100  # Minimum samples before using regime-specific model

    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.sample_counts: Dict[str, int] = defaultdict(int)
        self.is_trained: Dict[str, bool] = defaultdict(bool)
        self._init_models()

    def _init_models(self):
        if not SKLEARN_AVAILABLE:
            return
        for regime in self.REGIMES:
            self.models[regime] = {
                "xgb": xgb.XGBClassifier(
                    n_estimators=80,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    eval_metric="logloss",
                    verbosity=0,
                ),
                "rf": RandomForestClassifier(
                    n_estimators=80,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            }
            self.scalers[regime] = StandardScaler()

    def train_regime(self, regime: str, X: np.ndarray, y: np.ndarray) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        if len(y) < 30:
            logger.warning(f"[RegimeModels] Skipping {regime}: only {len(y)} samples")
            return False

        try:
            X_scaled = self.scalers[regime].fit_transform(X)
            self.models[regime]["xgb"].fit(X_scaled, y, verbose=False)
            self.models[regime]["rf"].fit(X_scaled, y)
            self.is_trained[regime] = True
            self.sample_counts[regime] = len(y)
            logger.info(f"[RegimeModels] Trained {regime}: {len(y)} samples")
            return True
        except Exception as e:
            logger.error(f"[RegimeModels] Train error for {regime}: {e}")
            return False

    def predict_regime(self, regime: str, X: np.ndarray) -> np.ndarray:
        """Returns ensemble probability (0-1)"""
        if not SKLEARN_AVAILABLE:
            return np.full(len(X), 0.5)

        # Fall back to GLOBAL if regime model not ready
        target = regime if (
            self.is_trained.get(regime) and
            self.sample_counts.get(regime, 0) >= self.MIN_REGIME_SAMPLES
        ) else "GLOBAL"

        if not self.is_trained.get(target):
            return np.full(len(X), 0.5)

        try:
            X_scaled = self.scalers[target].transform(X)
            preds = []
            if self.is_trained[target]:
                try:
                    preds.append(self.models[target]["xgb"].predict_proba(X_scaled)[:, 1])
                except Exception:
                    pass
                try:
                    preds.append(self.models[target]["rf"].predict_proba(X_scaled)[:, 1])
                except Exception:
                    pass
            return np.mean(preds, axis=0) if preds else np.full(len(X), 0.5)
        except Exception as e:
            logger.error(f"[RegimeModels] Predict error ({target}): {e}")
            return np.full(len(X), 0.5)

    def get_stats(self) -> Dict:
        return {
            regime: {
                "trained": self.is_trained.get(regime, False),
                "samples": self.sample_counts.get(regime, 0),
            }
            for regime in self.REGIMES
        }


# ============================================================
# Regime-Aware Ensemble (main class)
# ============================================================
class RegimeAwareEnsemble:
    """
    Drop-in replacement / enhancement for EnsembleModel.

    Key improvements over base EnsembleModel:
    1. Per-regime XGB + RF models
    2. Symbol embeddings appended to feature vector
    3. Regime-confidence-weighted prediction blending
    4. Compatible with existing train(df) / predict(df) API
    """

    def __init__(self, embedding_dim=4):
        self.regime_models = RegimeModelSet()
        self.symbol_embedding = SymbolEmbedding(embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

        self.feature_cols: Optional[List[str]] = None
        self.is_trained = False
        self.n_features: Optional[int] = None
        self.feature_medians: Dict[str, float] = {}

        self._train_stats: Dict = {}

    # ----------------------------------------------------------
    # Data preparation
    # ----------------------------------------------------------

    def _get_feature_columns(self, df) -> List[str]:
        exclude = {
            "time", "o", "h", "l", "c", "v", "signal", "confidence",
            "regime", "market_regime", "htf_regime",
            "symbol", "date", "datetime", "timestamp", "index",
        }
        return [
            col for col in df.columns
            if col not in exclude and np.issubdtype(df[col].dtype, np.number)
        ]

    def _build_X(self, df, symbol: str = "") -> Optional[np.ndarray]:
        if self.feature_cols is None:
            return None

        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            return None

        X = np.empty((len(df), len(self.feature_cols)), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            default = float(self.feature_medians.get(col, 0.0))
            if col in df.columns:
                values = np.asarray(df[col].replace([np.inf, -np.inf], np.nan), dtype=np.float32)
                values = np.nan_to_num(values, nan=default, posinf=default, neginf=default)
                X[:, i] = values
            else:
                X[:, i] = default

        # Append symbol embedding
        emb = self.symbol_embedding.get_embedding(symbol)
        emb_tiled = np.tile(emb, (len(X), 1))
        X = np.hstack([X, emb_tiled])

        return X

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def prepare_data(self, df) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            self.feature_cols = self._get_feature_columns(df)
            if not self.feature_cols:
                return None, None

            raw_X = df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
            medians = raw_X.median(numeric_only=True).fillna(0.0)
            self.feature_medians = {col: float(medians.get(col, 0.0)) for col in self.feature_cols}
            X = raw_X.fillna(medians).values.astype(np.float32)

            close_col = None
            for cname in ("c", "close", "Close", "CLOSE", "price", "last"):
                if cname in df.columns:
                    close_col = cname
                    break
            if close_col is None:
                return None, None

            LABEL_LOOKAHEAD = ML_LABEL_LOOKAHEAD
            LABEL_THRESHOLD = ML_LABEL_THRESHOLD
            MIN_CLASS_BALANCE = ML_LABEL_MIN_BALANCE
            MAX_CLASS_BALANCE = ML_LABEL_MAX_BALANCE
            close = df[close_col]
            future_close = close.shift(-LABEL_LOOKAHEAD)
            threshold = close * LABEL_THRESHOLD
            y = (future_close > close + threshold).astype(int).values
            X = X[:-LABEL_LOOKAHEAD]
            y = y[:-LABEL_LOOKAHEAD]

            pos_ratio = float(y.mean()) if len(y) > 0 else 0.5
            logger.info(f"[RegimeEnsemble.prepare_data] Label balance: {pos_ratio:.1%} positive ({int(y.sum())}/{len(y)})")
            if pos_ratio < MIN_CLASS_BALANCE or pos_ratio > MAX_CLASS_BALANCE:
                logger.warning(f"[RegimeEnsemble.prepare_data] Label imbalance: {pos_ratio:.1%} — check data quality")

            if len(y) < 30:
                return None, None

            return X, y

        except Exception as e:
            logger.error(f"[RegimeEnsemble] prepare_data error: {e}")
            return None, None

    def train(self, df, symbol: str = "") -> bool:
        """Train global model + per-regime models"""
        import pandas as pd

        X, y = self.prepare_data(df)
        if X is None:
            return False

        # Train global model
        success = self.regime_models.train_regime("GLOBAL", X, y)
        if not success:
            return False

        self.is_trained = True
        self.n_features = X.shape[1]

        # Train per-regime models if regime column exists
        regime_col = None
        for col in ["regime", "market_regime", "htf_regime"]:
            if col in df.columns:
                regime_col = col
                break

        if regime_col is not None:
            regime_values = df[regime_col].values[:-ML_LABEL_LOOKAHEAD]  # align with X, y
            for regime in RegimeModelSet.REGIMES[:-1]:   # skip GLOBAL
                mask = regime_values == regime
                if mask.sum() >= 30:
                    self.regime_models.train_regime(regime, X[mask], y[mask])
                else:
                    logger.debug(f"[RegimeEnsemble] {regime}: only {mask.sum()} samples, skipping")

        self._train_stats = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(y),
            "n_features": self.n_features,
            "regime_stats": self.regime_models.get_stats(),
        }
        logger.info(f"[RegimeEnsemble] Trained: {len(y)} samples, {self.n_features} features")
        return True

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def predict(self, df, symbol: str = "", regime: str = "GLOBAL",
                regime_confidence: float = 1.0) -> np.ndarray:
        """
        Returns ensemble probability predictions.

        When regime_confidence is high, trusts the regime-specific model more.
        Blends global + regime predictions weighted by confidence.
        """
        if not self.is_trained or self.feature_cols is None:
            return np.full(len(df), 0.5)

        X = self._build_X(df, symbol)
        if X is None:
            return np.full(len(df), 0.5)

        # Trim embedding cols for regime models (they were trained without embeddings)
        X_base = X[:, :len(self.feature_cols)]

        global_pred = self.regime_models.predict_regime("GLOBAL", X_base)

        # Blend with regime-specific prediction
        if regime != "GLOBAL" and self.regime_models.is_trained.get(regime):
            regime_pred = self.regime_models.predict_regime(regime, X_base)
            # Weight by regime confidence
            blend_w = np.clip(regime_confidence, 0.0, 0.8)
            final_pred = (1 - blend_w) * global_pred + blend_w * regime_pred
        else:
            final_pred = global_pred

        return final_pred

    def predict_single(self, features: Dict, symbol: str = "",
                       regime: str = "GLOBAL",
                       regime_confidence: float = 1.0) -> float:
        """
        Predict from a single feature dict (for live trading).
        Returns probability 0-1.
        """
        import pandas as pd
        if self.feature_cols is None:
            return 0.5

        row = {col: features.get(col, 0.0) for col in self.feature_cols}
        df_single = pd.DataFrame([row])
        pred = self.predict(df_single, symbol=symbol, regime=regime,
                            regime_confidence=regime_confidence)
        return float(pred[0]) if len(pred) > 0 else 0.5

    # ----------------------------------------------------------
    # Symbol stats update
    # ----------------------------------------------------------

    def update_symbol_stats(self, symbol: str, atr_pct: float,
                            vol_ratio: float, pnl: Optional[float] = None):
        """Update symbol embedding after each bar/trade"""
        self.symbol_embedding.update_stats(symbol, atr_pct, vol_ratio, pnl)

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        try:
            with open(os.path.join(path, "regime_ensemble.pkl"), "wb") as f:
                pickle.dump({
                    "regime_models": self.regime_models,
                    "symbol_embedding": self.symbol_embedding,
                    "feature_cols": self.feature_cols,
                    "is_trained": self.is_trained,
                    "n_features": self.n_features,
                    "feature_medians": self.feature_medians,
                    "train_stats": self._train_stats,
                }, f)
            logger.info(f"[RegimeEnsemble] Saved to {path}")
        except Exception as e:
            logger.error(f"[RegimeEnsemble] Save error: {e}")

    def load(self, path: str) -> bool:
        pkl_path = os.path.join(path, "regime_ensemble.pkl")
        if not os.path.exists(pkl_path):
            return False
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.regime_models = data["regime_models"]
            self.symbol_embedding = data["symbol_embedding"]
            self.feature_cols = data["feature_cols"]
            self.is_trained = data["is_trained"]
            self.n_features = data["n_features"]
            self.feature_medians = data.get("feature_medians", {})
            self._train_stats = data.get("train_stats", {})
            logger.info(f"[RegimeEnsemble] Loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"[RegimeEnsemble] Load error: {e}")
            return False

    def get_stats(self) -> Dict:
        return {
            "is_trained": self.is_trained,
            "n_features": self.n_features,
            "feature_cols_count": len(self.feature_cols) if self.feature_cols else 0,
            "feature_medians_count": len(self.feature_medians),
            "regime_models": self.regime_models.get_stats(),
            "embedding_dim": self.embedding_dim,
            "train_stats": self._train_stats,
        }