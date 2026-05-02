import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Tune ML model hyperparameters"""
    
    @staticmethod
    def tune_xgboost(X_train, y_train, cv=3):
        """Tune XGBoost parameters"""
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        base_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        logger.info("Starting XGBoost hyperparameter tuning...")
        
        grid_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=10,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    @staticmethod
    def tune_random_forest(X_train, y_train, cv=3):
        """Tune RandomForest parameters"""
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        logger.info("Starting RandomForest hyperparameter tuning...")
        
        grid_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=10,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best RF params: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_