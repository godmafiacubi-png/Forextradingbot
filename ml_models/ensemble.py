"""
Ensemble Model v2.0
- XGBoost + RandomForest (global, backward-compatible)
- RegimeAwareEnsemble: per-regime models + symbol embeddings
- RetrainEngine integration (walk-forward + triggered retrain)
- Online incremental update via partial_fit where supported
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import xgboost as xgb

from .regime_aware_ensemble import RegimeAwareEnsemble
from .retrain_engine import RetrainEngine

try:
    from config.settings import ML_LABEL_LOOKAHEAD, ML_LABEL_THRESHOLD, ML_LABEL_MIN_BALANCE, ML_LABEL_MAX_BALANCE
except ImportError:
    ML_LABEL_LOOKAHEAD = 3
    ML_LABEL_THRESHOLD = 0.0001
    ML_LABEL_MIN_BALANCE = 0.30
    ML_LABEL_MAX_BALANCE = 0.70

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble v2.0 — backward-compatible with original API.

    New capabilities (opt-in):
      - use_regime_models=True  → per-regime XGB+RF + symbol embeddings
      - use_retrain_engine=True → walk-forward + triggered background retrain
    """

    def __init__(self, use_regime_models=True, use_retrain_engine=True,
                 retrain_every_n_trades=100, train_window=500,
                 min_bars=200, min_new_bars_for_wf=50):

        # ---- Base models (always present for compatibility) ----
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.is_trained = False
        self.feature_cols = None
        self.n_features = None
        self.model_weights = {"xgb": 0.5, "rf": 0.5}
        self.validation_metrics = {}
        self.feature_medians = {}

        # ---- Regime-aware ensemble ----
        self.use_regime_models = use_regime_models
        self.regime_ensemble: Optional[RegimeAwareEnsemble] = (
            RegimeAwareEnsemble(embedding_dim=4) if use_regime_models else None
        )

        # ---- Retrain engine — disabled เพราะ bar dict มี features ไม่ครบ
        # walk-forward จะทำให้ model เสื่อมจาก 108 → 14 features
        # ใช้ periodic retrain จาก _auto_train() ใน main.py แทน ----
        self.use_retrain_engine = False
        self.retrain_engine = None

        # ---- Regime blending state ----
        self._current_regime = "GLOBAL"
        self._current_regime_confidence = 1.0
        self._current_symbol = ""

    # ----------------------------------------------------------
    # Regime + symbol context setters (call before predict)
    # ----------------------------------------------------------

    def set_context(self, symbol: str, regime: str, regime_confidence: float):
        """Set current symbol/regime context for prediction blending"""
        self._current_symbol = symbol
        self._current_regime = regime
        self._current_regime_confidence = regime_confidence

    def update_symbol_stats(self, symbol: str, atr_pct: float,
                            vol_ratio: float, pnl: Optional[float] = None):
        """Pass symbol stats to embedding layer"""
        if self.regime_ensemble:
            self.regime_ensemble.update_symbol_stats(symbol, atr_pct, vol_ratio, pnl)

    # ----------------------------------------------------------
    # Bar / trade hooks for retrain engine
    # ----------------------------------------------------------

    def on_bar(self, bar: Dict):
        """Feed new bar to walk-forward buffer"""
        if self.retrain_engine:
            self.retrain_engine.add_bar(bar)

    def on_bars(self, bars: List[Dict]):
        """Bulk feed bars"""
        if self.retrain_engine:
            self.retrain_engine.add_bars(bars)

    def on_trade_closed(self, pnl: float, pnl_pct: float,
                        symbol: str = "", atr_pct: float = 0.0,
                        vol_ratio: float = 1.0):
        """Call after each trade close to update retrain + symbol stats"""
        if self.retrain_engine:
            self.retrain_engine.record_trade(pnl, pnl_pct)
        if self.regime_ensemble:
            self.regime_ensemble.update_symbol_stats(symbol, atr_pct, vol_ratio, pnl)

    # ----------------------------------------------------------
    # Internal retrain callback (used by RetrainEngine)
    # ----------------------------------------------------------

    def _retrain_callback(self, train_data: List[Dict]) -> bool:
        """Convert buffered bar dicts back to DataFrame and retrain"""
        try:
            df = pd.DataFrame(train_data)
            if len(df) < 30:
                return False

            logger.debug(f"[Retrain] Raw columns ({len(df.columns)}): {list(df.columns)[:30]}")

            # Normalize column names — รองรับทุก naming convention ที่ on_bar() ส่งมา
            col_map = {
                # close variants
                'close': 'c', 'Close': 'c', 'CLOSE': 'c',
                'price': 'c', 'last_price': 'c', 'bid': 'c',
                # open variants
                'open': 'o', 'Open': 'o', 'OPEN': 'o',
                # high variants
                'high': 'h', 'High': 'h', 'HIGH': 'h',
                # low variants
                'low': 'l', 'Low': 'l', 'LOW': 'l',
                # volume variants
                'volume': 'v', 'Volume': 'v', 'VOLUME': 'v',
                'tick_volume': 'v', 'real_volume': 'v',
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # ถ้ายังไม่มี 'c' — ลอง detect จาก market_data keys ที่น่าจะเป็น price
            if 'c' not in df.columns:
                # หา column ที่ชื่อมี 'price' หรือ 'close' (case-insensitive)
                for col in df.columns:
                    if any(kw in col.lower() for kw in ('price', 'close', 'last')):
                        df = df.rename(columns={col: 'c'})
                        logger.info(f"[Retrain] Auto-mapped '{col}' → 'c'")
                        break

            # ถ้ายังไม่มีจริงๆ — log columns ทั้งหมดให้เห็นชัด
            if 'c' not in df.columns:
                logger.error(
                    f"[Retrain] Cannot find close column. "
                    f"All columns: {list(df.columns)}"
                )
                return False

            return self.train(df)

        except Exception as e:
            logger.error(f"[Ensemble] Retrain callback error: {e}", exc_info=True)
            return False

    # ----------------------------------------------------------
    # Feature selection
    # ----------------------------------------------------------

    def _get_feature_columns(self, df) -> List[str]:
        exclude = {
            "time", "o", "h", "l", "c", "v", "signal", "confidence",
            "regime", "market_regime", "htf_regime",
            # columns ที่ live data ไม่มีแต่อาจอยู่ใน training df
            "symbol", "date", "datetime", "timestamp", "index",
        }
        return [
            col for col in df.columns
            if col not in exclude and np.issubdtype(df[col].dtype, np.number)
        ]

    def prepare_data(self, df):
        try:
            logger.info(f"Preparing data: df.shape={df.shape}")
            feature_cols = self._get_feature_columns(df)
            self.feature_cols = feature_cols
            self.n_features = len(feature_cols)

            if not feature_cols:
                logger.warning("No feature columns found")
                return None, None

            raw_X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            medians = raw_X.median(numeric_only=True).fillna(0.0)
            self.feature_medians = {col: float(medians.get(col, 0.0)) for col in feature_cols}
            X = raw_X.fillna(medians).values.astype(np.float32)
            X = self.scaler.fit_transform(X)

            # หา close column — รองรับทุก naming convention
            close_col = None
            for cname in ('c', 'close', 'Close', 'CLOSE', 'price', 'last'):
                if cname in df.columns:
                    close_col = cname
                    break
            if close_col is None:
                logger.error(
                    f"DataFrame missing close column. "
                    f"Available columns: {list(df.columns)}"
                )
                return None, None

            LABEL_LOOKAHEAD = ML_LABEL_LOOKAHEAD
            LABEL_THRESHOLD = ML_LABEL_THRESHOLD
            MIN_CLASS_BALANCE = ML_LABEL_MIN_BALANCE
            MAX_CLASS_BALANCE = ML_LABEL_MAX_BALANCE
            future_close = df[close_col].shift(-LABEL_LOOKAHEAD)
            threshold = df[close_col] * LABEL_THRESHOLD
            y = (future_close > df[close_col] + threshold).astype(int).values
            X = X[:-LABEL_LOOKAHEAD]
            y = y[:-LABEL_LOOKAHEAD]

            pos_ratio = float(y.mean()) if len(y) > 0 else 0.5
            logger.info(f"[prepare_data] Label balance: {pos_ratio:.1%} positive ({int(y.sum())}/{len(y)})")
            if pos_ratio < MIN_CLASS_BALANCE or pos_ratio > MAX_CLASS_BALANCE:
                logger.warning(f"[prepare_data] Label imbalance detected: {pos_ratio:.1%} positive — check data quality")

            if len(y) < 30:
                logger.warning(f"Only {len(y)} samples. Skipping training.")
                return None, None

            return X, y

        except Exception as e:
            logger.error(f"Error preparing data: {e}", exc_info=True)
            return None, None

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def train(self, df) -> bool:
        """Train base models + regime models"""
        logger.info("Training models...")
        X, y = self.prepare_data(df)
        if X is None or len(y) < 30:
            logger.warning("Not enough data to train")
            return False

        if X is not None and y is not None and len(y) > 0:
            pos_count = int(y.sum())
            neg_count = len(y) - pos_count
            logger.info(f"[Train] {len(y)} samples | BUY:{pos_count} ({pos_count/len(y):.1%}) SELL:{neg_count} ({neg_count/len(y):.1%})")

        # ---- Base models ----
        try:
            self._fit_base_models_with_validation(X, y)
            self.is_trained = True
            logger.info(
                f"Base models trained: {len(y)} samples | "
                f"weights={self.model_weights} | metrics={self.validation_metrics}"
            )
        except Exception as e:
            logger.error(f"Base model training error: {e}", exc_info=True)
            return False

        # ---- Regime-aware models ----
        if self.use_regime_models and self.regime_ensemble is not None:
            try:
                self.regime_ensemble.train(df, symbol=self._current_symbol)
                logger.info("Regime-aware ensemble trained")
            except Exception as e:
                logger.warning(f"Regime ensemble training error (non-fatal): {e}")

        return True


    def _fit_base_models_with_validation(self, X: np.ndarray, y: np.ndarray):
        """Fit base learners and derive validation-aware blend weights.

        The live ensemble used to average XGBoost and RandomForest equally. That
        can over-trust a weak learner after market-regime drift. We keep the
        training API unchanged, reserve a chronological validation slice when
        enough data is available, convert validation log-loss into robust model
        weights, then refit both learners on all available samples.
        """
        self.model_weights = {"xgb": 0.5, "rf": 0.5}
        self.validation_metrics = {}

        if len(y) < 80 or len(np.unique(y)) < 2:
            self.xgb_model.fit(X, y, verbose=False)
            self.rf_model.fit(X, y)
            return

        split_idx = max(int(len(y) * 0.8), len(y) - 250)
        split_idx = min(max(split_idx, 40), len(y) - 20)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            self.xgb_model.fit(X, y, verbose=False)
            self.rf_model.fit(X, y)
            return

        self.xgb_model.fit(X_train, y_train, verbose=False)
        self.rf_model.fit(X_train, y_train)

        model_losses = {}
        for name, model in (("xgb", self.xgb_model), ("rf", self.rf_model)):
            try:
                preds = np.clip(model.predict_proba(X_val)[:, 1], 0.02, 0.98)
                loss = float(log_loss(y_val, preds, labels=[0, 1]))
                model_losses[name] = loss
                self.validation_metrics[f"{name}_log_loss"] = round(loss, 6)
            except Exception as exc:
                logger.warning(f"[Train] Validation scoring failed for {name}: {exc}")

        if len(model_losses) == 2:
            inv = {name: 1.0 / max(loss, 1e-6) for name, loss in model_losses.items()}
            total = sum(inv.values())
            raw_weights = {name: score / total for name, score in inv.items()}
            # Keep both learners represented to avoid overfitting one validation slice.
            self.model_weights = {
                name: round(float(np.clip(weight, 0.2, 0.8)), 4)
                for name, weight in raw_weights.items()
            }
            weight_total = sum(self.model_weights.values())
            self.model_weights = {
                name: round(weight / weight_total, 4)
                for name, weight in self.model_weights.items()
            }
            self.validation_metrics["validation_samples"] = int(len(y_val))

        # Refit on the full data for production inference once weights are set.
        self.xgb_model.fit(X, y, verbose=False)
        self.rf_model.fit(X, y)


    def _build_feature_matrix(self, df, min_coverage: float = 0.5):
        """Build inference features in training order with median imputation.

        Live bars can omit expensive/slow indicators. Filling absent columns with
        the training median keeps scaled values near neutral instead of turning a
        raw zero into a misleading outlier after StandardScaler.transform().
        """
        if not self.feature_cols:
            return None, 0.0

        n_rows = len(df)
        matched_cols = [c for c in self.feature_cols if c in df.columns]
        coverage = len(matched_cols) / len(self.feature_cols)

        if coverage < min_coverage:
            logger.warning(
                f"[Ensemble] Feature coverage {coverage:.0%} "
                f"({len(matched_cols)}/{len(self.feature_cols)}) — returning 0.5"
            )
            return None, coverage

        X = np.empty((n_rows, len(self.feature_cols)), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            default = float(self.feature_medians.get(col, 0.0))
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                X[:, i] = values.fillna(default).to_numpy(dtype=np.float32)
            else:
                X[:, i] = default
        return X, coverage

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def predict(self, df) -> np.ndarray:
        """
        Returns ensemble probability predictions.
        Blends base model + regime-aware model when both are available.
        """
        try:
            if not self.is_trained:
                logger.info("Model not trained; returning neutral.")
                return np.full(len(df), 0.5)

            if self.feature_cols is None:
                logger.error("feature_cols is not set.")
                return np.full(len(df), 0.5)

            # ---- Base model predictions ----
            n_rows = len(df)
            X, coverage = self._build_feature_matrix(df, min_coverage=0.5)
            if X is None:
                return np.full(n_rows, 0.5)

            try:
                X = self.scaler.transform(X)
            except Exception:
                logger.exception("Scaler transform failed")
                return np.full(n_rows, 0.5)

            base_preds = []
            try:
                base_preds.append(self.xgb_model.predict_proba(X)[:, 1])
            except Exception:
                base_preds.append(np.full(len(X), 0.5))
            try:
                base_preds.append(self.rf_model.predict_proba(X)[:, 1])
            except Exception:
                base_preds.append(np.full(len(X), 0.5))

            weights = np.array([
                self.model_weights.get("xgb", 0.5),
                self.model_weights.get("rf", 0.5),
            ], dtype=np.float32)
            weights = weights / max(float(weights.sum()), 1e-9)
            base_ensemble = np.average(np.vstack(base_preds), axis=0, weights=weights)
            base_ensemble = np.clip(base_ensemble, 0.02, 0.98)

            # ---- Regime-aware predictions (blend in) ----
            if (self.use_regime_models and
                    self.regime_ensemble is not None and
                    self.regime_ensemble.is_trained):
                regime_pred = self.regime_ensemble.predict(
                    df,
                    symbol=self._current_symbol,
                    regime=self._current_regime,
                    regime_confidence=self._current_regime_confidence,
                )
                # Blend with regime model only when live features have reasonable
                # coverage; low coverage should lean on the robust global model.
                regime_weight = np.clip(
                    self._current_regime_confidence * 0.5 * coverage, 0.05, 0.5
                )
                final = (1 - regime_weight) * base_ensemble + regime_weight * regime_pred
            else:
                final = base_ensemble

            return final

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return np.full(len(df), 0.5)

    def predict_single(self, features: Dict, symbol: str = "",
                       regime: str = "GLOBAL",
                       regime_confidence: float = 1.0) -> float:
        """Convenience method for live single-bar prediction"""
        if self.regime_ensemble and self.regime_ensemble.is_trained:
            return self.regime_ensemble.predict_single(
                features, symbol=symbol, regime=regime,
                regime_confidence=regime_confidence,
            )
        # Fallback: build single-row DataFrame
        if self.feature_cols:
            row = {col: features.get(col, 0.0) for col in self.feature_cols}
            df_single = pd.DataFrame([row])
            pred = self.predict(df_single)
            return float(pred[0]) if len(pred) > 0 else 0.5
        return 0.5

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        try:
            pickle.dump(self.xgb_model, open(os.path.join(path, "xgb.pkl"), "wb"))
            pickle.dump(self.rf_model, open(os.path.join(path, "rf.pkl"), "wb"))
            pickle.dump(self.scaler, open(os.path.join(path, "scaler.pkl"), "wb"))
            if self.feature_cols:
                with open(os.path.join(path, "features.pkl"), "wb") as f:
                    pickle.dump(self.feature_cols, f)

            import json
            metadata = {
                "name": os.path.basename(path),
                "n_features": self.n_features,
                "is_trained": self.is_trained,
                "version": "2.2",
                "model_weights": self.model_weights,
                "feature_medians": self.feature_medians,
                "validation_metrics": self.validation_metrics,
                "feature_medians_count": len(self.feature_medians),
            }
            with open(os.path.join(path, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            # Save regime ensemble
            if self.regime_ensemble:
                self.regime_ensemble.save(os.path.join(path, "regime"))

            logger.info(f"EnsembleModel v2.2 saved to {path}")
        except Exception as e:
            logger.error(f"Save error: {e}", exc_info=True)

    def load(self, path: str) -> bool:
        try:
            self.xgb_model = pickle.load(open(os.path.join(path, "xgb.pkl"), "rb"))
            self.rf_model = pickle.load(open(os.path.join(path, "rf.pkl"), "rb"))
            try:
                self.scaler = pickle.load(open(os.path.join(path, "scaler.pkl"), "rb"))
            except Exception:
                self.scaler = StandardScaler()

            fc_path = os.path.join(path, "features.pkl")
            if os.path.exists(fc_path):
                with open(fc_path, "rb") as f:
                    self.feature_cols = pickle.load(f)

            meta_path = os.path.join(path, "metadata.pkl")
            if os.path.exists(meta_path):
                meta = pickle.load(open(meta_path, "rb"))
                self.n_features = meta.get("n_features")
                self.is_trained = meta.get("is_trained", True)
                self.model_weights = meta.get("model_weights", self.model_weights)
                self.validation_metrics = meta.get("validation_metrics", self.validation_metrics)
                self.feature_medians = meta.get("feature_medians", self.feature_medians)

            # Load regime ensemble
            regime_path = os.path.join(path, "regime")
            if self.regime_ensemble and os.path.exists(regime_path):
                self.regime_ensemble.load(regime_path)

            logger.info(f"EnsembleModel v2.2 loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Load error: {e}", exc_info=True)
            return False

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def get_stats(self) -> Dict:
        stats = {
            "version": "2.2",
            "is_trained": self.is_trained,
            "n_features": self.n_features,
            "current_symbol": self._current_symbol,
            "current_regime": self._current_regime,
            "model_weights": self.model_weights,
            "validation_metrics": self.validation_metrics,
            "feature_medians_count": len(self.feature_medians),
        }
        if self.regime_ensemble:
            stats["regime_ensemble"] = self.regime_ensemble.get_stats()
        if self.retrain_engine:
            stats["retrain_engine"] = self.retrain_engine.get_stats()
        return stats