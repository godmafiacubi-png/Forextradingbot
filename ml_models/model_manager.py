import os
import json
import pickle
import logging
from datetime import datetime

try:
    import joblib
except ImportError:
    joblib = None

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_dir='./models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _save_obj(self, obj, path):
        if joblib:
            joblib.dump(obj, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)

    def _load_obj(self, path):
        if joblib:
            return joblib.load(path)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def save_model(self, model, name='trading_model'):
        save_dir = os.path.join(self.model_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            if hasattr(model, 'xgb_model') and model.xgb_model is not None:
                self._save_obj(model.xgb_model, os.path.join(save_dir, 'xgb_model.pkl'))
                logger.info("Saved XGBoost")

            if hasattr(model, 'rf_model') and model.rf_model is not None:
                self._save_obj(model.rf_model, os.path.join(save_dir, 'rf_model.pkl'))
                logger.info("Saved RandomForest")

            if hasattr(model, 'scaler') and model.scaler is not None:
                self._save_obj(model.scaler, os.path.join(save_dir, 'scaler.pkl'))
                logger.info("Saved Scaler")

            feature_cols = getattr(model, 'feature_cols', None)
            if feature_cols is not None:
                self._save_obj(feature_cols, os.path.join(save_dir, 'feature_cols.pkl'))
                self._save_obj(feature_cols, os.path.join(save_dir, 'features.pkl'))
                logger.info(f"Saved feature_cols ({len(feature_cols)} features)")

            metadata = {
                'name': name,
                'saved_at': datetime.now().isoformat(),
                'is_trained': getattr(model, 'is_trained', False),
                'n_features': len(feature_cols) if feature_cols else 0,
                'feature_cols': list(feature_cols) if feature_cols else [],
            }

            with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model '{name}' saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, model, name='trading_model'):
        load_dir = os.path.join(self.model_dir, name)

        if not os.path.exists(load_dir):
            logger.warning(f"Model directory not found: {load_dir}")
            return False

        try:
            logger.info(f"Loading model from {load_dir}...")

            xgb_path = os.path.join(load_dir, 'xgb_model.pkl')
            if os.path.exists(xgb_path):
                model.xgb_model = self._load_obj(xgb_path)
                logger.info("Loaded XGBoost")

            rf_path = os.path.join(load_dir, 'rf_model.pkl')
            if os.path.exists(rf_path):
                model.rf_model = self._load_obj(rf_path)
                logger.info("Loaded RandomForest")

            scaler_path = os.path.join(load_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                model.scaler = self._load_obj(scaler_path)
                logger.info("Loaded Scaler")

            model.feature_cols = None

            fc_path = os.path.join(load_dir, 'feature_cols.pkl')
            if os.path.exists(fc_path):
                try:
                    model.feature_cols = self._load_obj(fc_path)
                    logger.info(f"Loaded feature_cols ({len(model.feature_cols)} features)")
                except Exception:
                    logger.debug("feature_cols.pkl load failed", exc_info=True)

            if model.feature_cols is None:
                feat_path = os.path.join(load_dir, 'features.pkl')
                if os.path.exists(feat_path):
                    try:
                        model.feature_cols = self._load_obj(feat_path)
                        logger.info(f"Loaded feature_cols from features.pkl ({len(model.feature_cols)} features)")
                    except Exception:
                        logger.debug("features.pkl load failed", exc_info=True)

            meta_path = os.path.join(load_dir, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata: {metadata}")

                model.is_trained = metadata.get('is_trained', False)
                model.n_features = metadata.get('n_features', None)

                if model.feature_cols is None:
                    saved_cols = metadata.get('feature_cols', None)
                    if saved_cols and len(saved_cols) > 0:
                        model.feature_cols = saved_cols
                        logger.info(f"Loaded feature_cols from metadata ({len(saved_cols)} features)")

            meta_pkl_path = os.path.join(load_dir, 'metadata.pkl')
            if os.path.exists(meta_pkl_path):
                try:
                    meta_pkl = self._load_obj(meta_pkl_path)
                    if not getattr(model, 'is_trained', False):
                        model.is_trained = meta_pkl.get('is_trained', False)
                    if model.n_features is None:
                        model.n_features = meta_pkl.get('n_features', None)
                except Exception:
                    logger.debug("metadata.pkl load failed", exc_info=True)

            logger.info(f"Model '{name}' loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False