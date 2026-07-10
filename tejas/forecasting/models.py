"""
Phase 7 — Model factories.

Three algorithms:
  * ``logreg`` — StandardScaler + LogisticRegression, ``class_weight="balanced"``
  * ``xgb``    — XGBoost, ``scale_pos_weight`` computed from train
  * ``lgbm``   — LightGBM, ``scale_pos_weight`` computed from train

All factories return sklearn-compatible estimators that expose ``fit`` and
``predict_proba``.  Hyperparameters come from :mod:`tejas.config`.
"""

from __future__ import annotations

import numpy as np

from tejas.config import (
    FORECAST_LOGREG_PARAMS,
    FORECAST_XGB_PARAMS,
    FORECAST_LGBM_PARAMS,
)


ALGORITHMS = ("logreg", "xgb", "lgbm")


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Return ``n_neg / n_pos`` for imbalanced binary training.

    Falls back to 1.0 when there are no positives (edge case for a val
    fold that happens to contain no flares within the horizon).
    """
    y = np.asarray(y).astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


# ── Factories ──────────────────────────────────────────────────────────────


def make_logreg(**overrides):
    """Build a scaled logistic-regression Pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    params = {**FORECAST_LOGREG_PARAMS, **overrides}
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(**params)),
        ]
    )


def make_xgb(scale_pos_weight: float = 1.0, **overrides):
    """Build an XGBoost classifier with the configured hyperparameters.

    XGBoost accepts NaN natively, so no imputation step is required.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xgboost is required for make_xgb()") from exc

    params = {**FORECAST_XGB_PARAMS, **overrides}
    params["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**params)


def make_lgbm(scale_pos_weight: float = 1.0, **overrides):
    """Build a LightGBM classifier with the configured hyperparameters.

    LightGBM accepts NaN natively.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lightgbm is required for make_lgbm()") from exc

    params = {**FORECAST_LGBM_PARAMS, **overrides}
    params["scale_pos_weight"] = float(scale_pos_weight)
    params.setdefault("verbose", -1)
    return LGBMClassifier(**params)


def make_model(algorithm: str, y_train: np.ndarray | None = None):
    """Build a model by algorithm name.

    Args:
        algorithm: One of ``logreg`` / ``xgb`` / ``lgbm``.
        y_train: Training labels — only needed for tree models to compute
            ``scale_pos_weight``.  Ignored for LogReg (uses class_weight).

    Returns:
        A fresh, unfitted estimator.
    """
    if algorithm == "logreg":
        return make_logreg()
    if algorithm == "xgb":
        spw = compute_scale_pos_weight(y_train) if y_train is not None else 1.0
        return make_xgb(scale_pos_weight=spw)
    if algorithm == "lgbm":
        spw = compute_scale_pos_weight(y_train) if y_train is not None else 1.0
        return make_lgbm(scale_pos_weight=spw)
    raise ValueError(f"Unknown algorithm: {algorithm}")
