"""
Phase 7 — Training and artifact persistence.

For each ``(horizon, algorithm)`` pair:
  1. Build train / val / test arrays via :mod:`tejas.forecasting.splits`.
  2. Fit the model on train.
  3. Tune the decision threshold on val (max F1).
  4. Compute metrics on val + test.
  5. Persist the fitted estimator to ``MODELS_DIR`` via ``joblib``.

A single manifest ``feature_columns.json`` is written once per training
run, listing the frozen feature order.  Inference reads this manifest and
refuses to run against a row with mismatched columns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tejas.config import (
    MODELS_DIR,
    FORECAST_LABEL_HORIZONS_MIN,
    FORECAST_THRESHOLD_METRIC,
    FORECAST_CALIBRATION_METHOD,
    FORECAST_EARLY_STOPPING_ROUNDS,
    FORECAST_USE_SQRT_SPW,
    FORECAST_SPW_CAP,
)
from tejas.forecasting.evaluation import (
    compute_metrics,
    tune_threshold,
)
from tejas.forecasting.models import (
    ALGORITHMS,
    compute_scale_pos_weight,
    make_model,
)
from tejas.forecasting.splits import build_split_arrays


# ── Single-horizon training ─────────────────────────────────────────────────


def train_one_horizon(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_labels: pd.Series,
    horizon_min: int,
    algorithm: str,
    label_prefix: str = "label_flare_within",
) -> dict:
    """Train one model for one horizon and return a summary dict.

    Args:
        df: Prepared training frame (already downsampled + purged).
        feature_cols: Frozen feature list.
        split_labels: Row-wise split membership.
        horizon_min: One of 5, 15, 30, 60.
        algorithm: One of ``logreg``, ``xgb``, ``lgbm``.
        label_prefix: Prefix of the label column
            (``label_flare_within`` or ``label_sig_flare_within``).

    Returns:
        Dict with keys ``algorithm``, ``horizon_min``, ``model``,
        ``feature_columns``, ``decision_threshold``, ``val_metrics``,
        ``test_metrics``, ``n_train``, ``n_val``, ``n_test``,
        ``n_positive_train``, ``scale_pos_weight``, ``training_seconds``.
    """
    label_col = f"{label_prefix}_{horizon_min}m"
    arrays = build_split_arrays(df, feature_cols, label_col, split_labels)

    X_tr, y_tr = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_te, y_te = arrays["X_test"], arrays["y_test"]

    t0 = time.time()
    model = make_model(algorithm, y_train=y_tr)
    _fit_with_optional_early_stopping(model, algorithm, X_tr, y_tr, X_val, y_val)
    training_seconds = time.time() - t0

    # Raw predictions from the fitted estimator.
    val_prob_raw = _predict_proba_positive(model, X_val)
    test_prob_raw = _predict_proba_positive(model, X_te)

    # Probability calibration on val (isotonic / sigmoid / none).
    # Calibrator is fit on val ↔ val labels; downstream inference
    # applies the same transform to keep test evaluation consistent.
    calibrator = _fit_calibrator(val_prob_raw, y_val, FORECAST_CALIBRATION_METHOD)
    val_prob = _apply_calibrator(calibrator, val_prob_raw)
    test_prob = _apply_calibrator(calibrator, test_prob_raw)

    # Threshold optimisation (config-selected metric — see
    # FORECAST_THRESHOLD_METRIC).
    threshold = tune_threshold(y_val, val_prob, metric=FORECAST_THRESHOLD_METRIC)
    val_metrics = compute_metrics(y_val, val_prob, threshold=threshold)
    test_metrics = compute_metrics(y_te, test_prob, threshold=threshold)

    # scale_pos_weight for the record (0/1 imbalance)
    n_pos_tr = int((y_tr == 1).sum())
    n_neg_tr = int((y_tr == 0).sum())
    spw_raw = (n_neg_tr / n_pos_tr) if n_pos_tr > 0 else 1.0
    spw_effective = (
        compute_scale_pos_weight(
            y_tr, use_sqrt=FORECAST_USE_SQRT_SPW, cap=FORECAST_SPW_CAP
        )
        if n_pos_tr > 0
        else 1.0
    )

    return {
        "algorithm": algorithm,
        "horizon_min": int(horizon_min),
        "model": model,
        "calibrator": calibrator,
        "calibration_method": FORECAST_CALIBRATION_METHOD if calibrator is not None else "none",
        "feature_columns": list(feature_cols),
        "decision_threshold": float(threshold),
        "threshold_metric": FORECAST_THRESHOLD_METRIC,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_te)),
        "n_positive_train": n_pos_tr,
        "scale_pos_weight": float(spw_raw),
        "scale_pos_weight_effective": float(spw_effective),
        "training_seconds": round(training_seconds, 2),
    }


def _fit_with_optional_early_stopping(
    model, algorithm: str, X_tr, y_tr, X_val, y_val
) -> None:
    """Fit ``model`` with early stopping when supported.

    Applies to tree ensembles only; LogReg is fit normally.  When val is
    empty or contains a single class, early stopping is silently skipped
    and the model is fit on train alone (still deterministic, but no
    plateau detection).
    """
    rounds = int(FORECAST_EARLY_STOPPING_ROUNDS)
    val_usable = (
        rounds > 0
        and len(y_val) > 0
        and int((np.asarray(y_val) == 1).sum()) > 0
        and int((np.asarray(y_val) == 0).sum()) > 0
    )

    if algorithm == "xgb" and val_usable:
        try:
            model.set_params(early_stopping_rounds=rounds)
        except (ValueError, AttributeError):
            pass
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            return
        except TypeError:
            # Older XGB signatures — fall through to plain fit.
            pass

    if algorithm == "lgbm" and val_usable:
        try:
            from lightgbm import early_stopping, log_evaluation

            callbacks = [early_stopping(rounds, verbose=False), log_evaluation(0)]
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="average_precision",
                callbacks=callbacks,
            )
            return
        except (ImportError, TypeError):
            pass

    model.fit(X_tr, y_tr)


# ── Probability calibration ────────────────────────────────────────────────


def _fit_calibrator(prob: np.ndarray, y: np.ndarray, method: str):
    """Fit a probability calibrator on (val probs, val labels).

    Returns the fitted calibrator or ``None`` when calibration is
    disabled / not learnable for this fold.  Isotonic requires both
    classes present in ``y``; if only one class is present, we quietly
    fall back to no calibration for that horizon.
    """
    method = str(method).lower()
    if method in ("", "none", "off"):
        return None

    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    if len(prob) < 20 or int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return None

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(prob, y)
        return iso

    if method == "sigmoid":
        # Platt scaling: fit a 1-D logistic regression on the raw probs.
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(prob.reshape(-1, 1), y)
        return lr

    raise ValueError(f"Unknown calibration method: {method!r}")


def _apply_calibrator(calibrator, prob: np.ndarray) -> np.ndarray:
    """Transform raw probabilities through a fitted calibrator."""
    prob = np.asarray(prob, dtype=float)
    if calibrator is None or len(prob) == 0:
        return prob

    from sklearn.isotonic import IsotonicRegression

    if isinstance(calibrator, IsotonicRegression):
        out = calibrator.predict(prob)
    else:
        # Sigmoid (LogisticRegression)
        out = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def _predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) for a fitted binary classifier."""
    if len(X) == 0:
        return np.zeros(0, dtype=float)
    probs = model.predict_proba(X)
    # sklearn/xgboost/lightgbm all order classes as [0, 1] for binary
    # targets containing both classes.  Handle the degenerate case where
    # only one class appeared in training.
    if probs.shape[1] == 1:
        # If the single class was 1, probs is P(y=1); else P(y=0).
        try:
            class_label = int(model.classes_[0])
        except (AttributeError, TypeError):
            class_label = 1
        if class_label == 1:
            return probs[:, 0]
        return 1.0 - probs[:, 0]
    return probs[:, 1]


# ── All horizons × algorithms ───────────────────────────────────────────────


def train_all(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_labels: pd.Series,
    horizons: Iterable[int] | None = None,
    algorithms: Iterable[str] | None = None,
    models_dir: Path | None = None,
    save: bool = True,
    verbose: bool = True,
    label_prefix: str = "label_flare_within",
    artifact_suffix: str = "",
) -> pd.DataFrame:
    """Fit every (horizon, algorithm) combo and persist the artifacts.

    Args:
        df: Prepared training frame.
        feature_cols: Frozen feature list.
        split_labels: Row-wise split membership.
        horizons: Iterable of horizons in minutes.  Defaults to
            :data:`FORECAST_LABEL_HORIZONS_MIN`.
        algorithms: Iterable of algorithm names.  Defaults to all three.
        models_dir: Output directory for artifacts.  Defaults to
            :data:`MODELS_DIR`.
        save: If True, ``joblib.dump`` each fitted model.
        verbose: Print a progress line per fit.

    Returns:
        DataFrame with one row per fit, columns: algorithm, horizon_min,
        n_train, n_val, n_test, n_positive_train, scale_pos_weight,
        decision_threshold, training_seconds, and val_ / test_ prefixed
        metrics.
    """
    horizons = list(horizons) if horizons is not None else list(FORECAST_LABEL_HORIZONS_MIN)
    algorithms = list(algorithms) if algorithms is not None else list(ALGORITHMS)
    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)

    if save:
        models_dir.mkdir(parents=True, exist_ok=True)
        save_feature_columns_manifest(feature_cols, models_dir=models_dir)

    records: list[dict] = []
    for algo in algorithms:
        for h in horizons:
            if verbose:
                print(f"  Training {algo}{artifact_suffix} horizon={h}m ...", flush=True)
            try:
                result = train_one_horizon(
                    df, feature_cols, split_labels, h, algo,
                    label_prefix=label_prefix,
                )
            except ValueError as exc:
                # Most commonly: the horizon's positive class is absent
                # from the train fold, so LogReg/XGBoost refuse to fit.
                # Record a skipped-row and continue.
                if verbose:
                    print(f"    [SKIP] {algo} horizon={h}m: {exc}")
                records.append({
                    "algorithm": algo,
                    "horizon_min": int(h),
                    "n_train": 0, "n_val": 0, "n_test": 0,
                    "n_positive_train": 0, "scale_pos_weight": 1.0,
                    "decision_threshold": float("nan"),
                    "training_seconds": 0.0,
                    "skipped_reason": str(exc),
                })
                continue

            if save:
                import joblib

                artifact_path = models_dir / f"{algo}{artifact_suffix}_horizon_{h}m.joblib"
                joblib.dump(
                    {
                        "model": result["model"],
                        "calibrator": result.get("calibrator"),
                        "calibration_method": result.get("calibration_method", "none"),
                        "algorithm": result["algorithm"],
                        "horizon_min": result["horizon_min"],
                        "feature_columns": result["feature_columns"],
                        "decision_threshold": result["decision_threshold"],
                        "threshold_metric": result.get(
                            "threshold_metric", FORECAST_THRESHOLD_METRIC
                        ),
                        "scale_pos_weight": result["scale_pos_weight"],
                        "scale_pos_weight_effective": result.get(
                            "scale_pos_weight_effective", result["scale_pos_weight"]
                        ),
                    },
                    artifact_path,
                )

            records.append(_flatten_result(result))

            if verbose:
                v = result["val_metrics"]
                t = result["test_metrics"]
                print(
                    f"    val:  pr_auc={v['pr_auc']:.3f}  f1={v['f1']:.3f}  "
                    f"recall={v['recall']:.3f}  fpr={v['false_alarm_rate']:.4f}"
                )
                print(
                    f"    test: pr_auc={t['pr_auc']:.3f}  f1={t['f1']:.3f}  "
                    f"recall={t['recall']:.3f}  fpr={t['false_alarm_rate']:.4f}"
                )

    return pd.DataFrame(records)


def _flatten_result(result: dict) -> dict:
    """Flatten a train_one_horizon result for the summary CSV."""
    flat = {
        "algorithm": result["algorithm"],
        "horizon_min": result["horizon_min"],
        "n_train": result["n_train"],
        "n_val": result["n_val"],
        "n_test": result["n_test"],
        "n_positive_train": result["n_positive_train"],
        "scale_pos_weight": result["scale_pos_weight"],
        "decision_threshold": result["decision_threshold"],
        "training_seconds": result["training_seconds"],
    }
    for split_name in ("val", "test"):
        m = result[f"{split_name}_metrics"]
        for k, v in m.items():
            flat[f"{split_name}_{k}"] = v
    return flat


# ── Manifest ───────────────────────────────────────────────────────────────


def save_feature_columns_manifest(
    feature_cols: list[str],
    models_dir: Path | None = None,
) -> Path:
    """Persist the frozen feature list as JSON alongside model artifacts."""
    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "feature_columns.json"
    payload = {
        "feature_columns": list(feature_cols),
        "n_features": len(feature_cols),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_feature_columns_manifest(models_dir: Path | None = None) -> list[str]:
    """Read the frozen feature list back from disk."""
    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)
    path = models_dir / "feature_columns.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["feature_columns"])
