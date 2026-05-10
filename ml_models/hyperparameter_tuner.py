import logging
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb
import numpy as np

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Tune ML model hyperparameters with chronological CV for trading data."""

    @staticmethod
    def _time_series_cv(n_samples, cv):
        """Use TimeSeriesSplit to avoid future data leaking into validation folds."""
        n_splits = max(2, min(cv, max(2, n_samples // 50)))
        return TimeSeriesSplit(n_splits=n_splits)

    @staticmethod
    def _class_balance(y_train):
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        if pos == 0 or neg == 0:
            return 1.0, None
        scale_pos_weight = float(np.clip(neg / pos, 0.25, 4.0))
        total = len(y_train)
        class_weight = {
            0: float(np.clip(total / (2.0 * neg), 0.25, 4.0)),
            1: float(np.clip(total / (2.0 * pos), 0.25, 4.0)),
        }
        return scale_pos_weight, class_weight

    @staticmethod
    def tune_xgboost(X_train, y_train, cv=3):
        """Tune XGBoost parameters with leakage-safe time-series folds."""
        scale_pos_weight, _ = HyperparameterTuner._class_balance(y_train)

        param_grid = {
            'n_estimators': [80, 120, 180, 240],
            'max_depth': [3, 4, 5, 7],
            'learning_rate': [0.02, 0.05, 0.08, 0.12],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.85],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0.0, 0.05, 0.1, 0.3],
            'reg_lambda': [0.8, 1.0, 1.5, 2.0],
        }

        base_model = xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            verbosity=0,
        )

        logger.info("Starting XGBoost hyperparameter tuning with TimeSeriesSplit...")

        grid_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=HyperparameterTuner._time_series_cv(len(y_train), cv),
            scoring=make_scorer(f1_score, zero_division=0),
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    @staticmethod
    def tune_random_forest(X_train, y_train, cv=3):
        """Tune RandomForest parameters with leakage-safe time-series folds."""
        _, class_weight = HyperparameterTuner._class_balance(y_train)

        param_grid = {
            'n_estimators': [100, 160, 240, 320],
            'max_depth': [6, 10, 14, 18, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.5],
            'bootstrap': [True],
        }

        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight,
        )

        logger.info("Starting RandomForest hyperparameter tuning with TimeSeriesSplit...")

        grid_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=HyperparameterTuner._time_series_cv(len(y_train), cv),
            scoring=make_scorer(f1_score, zero_division=0),
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best RF params: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_
