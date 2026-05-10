import numpy as np
import pandas as pd

from ml_models.ensemble import EnsembleModel
from ml_models.hyperparameter_tuner import HyperparameterTuner
from ml_models.regime_aware_ensemble import RegimeModelSet


def _market_frame(n=140):
    rng = np.random.default_rng(42)
    close = 1.0 + np.cumsum(rng.normal(0.0002, 0.001, n))
    # A predictive feature: recent momentum with small noise.
    momentum = np.r_[0.0, np.diff(close)] + rng.normal(0, 0.0001, n)
    return pd.DataFrame(
        {
            "c": close,
            "momentum": momentum,
            "vol_ratio": rng.uniform(0.8, 1.2, n),
            "regime": np.where(np.arange(n) % 2 == 0, "TRENDING", "RANGING"),
        }
    )


def test_ensemble_fits_production_scaler_and_validation_metrics():
    model = EnsembleModel(use_regime_models=False)

    assert model.train(_market_frame()) is True

    stats = model.get_stats()
    assert stats["is_trained"] is True
    assert stats["model_weights"]["xgb"] + stats["model_weights"]["rf"] == 1.0
    assert "validation_samples" in stats["validation_metrics"]
    assert "class_balance" in stats["validation_metrics"]

    preds = model.predict(_market_frame(10))
    assert preds.shape == (10,)
    assert np.all((preds >= 0.02) & (preds <= 0.98))


def test_hyperparameter_tuner_uses_time_series_cv_and_class_balance():
    y = np.array([0] * 90 + [1] * 10)
    cv = HyperparameterTuner._time_series_cv(len(y), cv=5)
    folds = list(cv.split(np.zeros((len(y), 2))))

    assert len(folds) == 2
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()

    scale_pos_weight, class_weight = HyperparameterTuner._class_balance(y)
    assert scale_pos_weight == 4.0  # clipped from 9.0 for stability
    assert class_weight[1] > class_weight[0]


def test_regime_model_set_applies_class_balanced_sample_weights():
    regime_models = RegimeModelSet()
    y = np.array([0] * 36 + [1] * 4)
    weights = regime_models._sample_weights(y)

    assert weights[y == 1].mean() > weights[y == 0].mean()
    assert regime_models._class_balance_params(y)["scale_pos_weight"] == 4.0
