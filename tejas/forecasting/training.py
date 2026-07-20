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
    tune_threshold_event_level,
)
from tejas.forecasting.models import (
    ALGORITHMS,
    compute_scale_pos_weight,
    make_model,
)
from tejas.forecasting.splits import build_split_arrays


# ── v3 P1: training-fit row exclusions (label surgery §2.2–§2.4) ──────────


def load_uncovered_goes_windows(path: Path | None = None) -> pd.DataFrame:
    """GOES flare windows with no/degraded Aditya coverage (§2.3).

    Written by ``scripts/run_credible_relabel.py``.  Missing file → empty
    frame (no coverage masking), so pre-P1 pipelines keep working.
    """
    if path is None:
        from tejas.config import CLASSIFICATION_DIR, GOES_UNCOVERED_WINDOWS_NAME

        path = Path(CLASSIFICATION_DIR) / GOES_UNCOVERED_WINDOWS_NAME
    if not Path(path).exists():
        return pd.DataFrame(columns=["begin_time", "end_time"])
    out = pd.read_csv(path)
    for col in ("begin_time", "end_time"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def build_train_exclusion_mask(
    df: pd.DataFrame,
    split_labels: pd.Series,
    horizon_min: int,
    buffer_min: float | None = None,
    uncovered_windows: pd.DataFrame | None = None,
    exclude_decaying: bool | None = None,
) -> tuple[np.ndarray, dict]:
    """Boolean mask of rows to EXCLUDE from the horizon-h training fit.

    v3 P1 label surgery — three train-only exclusions (val/test evaluation
    always keeps every row and its strict label):

      §2.2 ambiguity buffer — ``label_minutes_to_next_flare`` in
          ``(h, h + buffer]``: physically near-positive rows labeled 0.
      §2.3 coverage mask — rows inside a GOES-confirmed flare window that
          Aditya did not observe: label unknowable, currently a wrong
          negative.
      §2.4 decaying in-flare rows — behind ``exclude_decaying`` (A/B flag
          ``FORECAST_EXCLUDE_DECAYING_FROM_TRAIN``).

    Returns:
        (mask aligned to ``df`` — True = drop from fit, counts dict).
        The mask is always False outside the train split.
    """
    from tejas.config import (
        FORECAST_LABEL_AMBIGUITY_BUFFER_MIN,
        FORECAST_EXCLUDE_DECAYING_FROM_TRAIN,
    )

    if buffer_min is None:
        buffer_min = FORECAST_LABEL_AMBIGUITY_BUFFER_MIN.get(int(horizon_min), 0.0)
    if exclude_decaying is None:
        exclude_decaying = FORECAST_EXCLUDE_DECAYING_FROM_TRAIN
    if uncovered_windows is None:
        uncovered_windows = load_uncovered_goes_windows()

    train_mask = (split_labels == "train").to_numpy()
    n = len(df)
    buffer_excl = np.zeros(n, dtype=bool)
    coverage_excl = np.zeros(n, dtype=bool)
    decay_excl = np.zeros(n, dtype=bool)

    # §2.2 — don't-care band (h, h + buffer]
    if buffer_min > 0 and "label_minutes_to_next_flare" in df.columns:
        minutes = pd.to_numeric(
            df["label_minutes_to_next_flare"], errors="coerce"
        ).to_numpy()
        h = float(horizon_min)
        with np.errstate(invalid="ignore"):
            buffer_excl = (minutes > h) & (minutes <= h + float(buffer_min))
        buffer_excl &= train_mask

    # §2.3 — unobserved GOES flare windows
    if len(uncovered_windows) and "datetime" in df.columns:
        ts = df["datetime"].to_numpy()
        for _, w in uncovered_windows.iterrows():
            begin = np.datetime64(pd.Timestamp(w["begin_time"]))
            end = np.datetime64(pd.Timestamp(w["end_time"]))
            coverage_excl |= (ts >= begin) & (ts <= end)
        coverage_excl &= train_mask

    # §2.4 — decaying in-flare rows (ablation flag)
    if exclude_decaying and "flare_phase" in df.columns:
        active = (
            pd.to_numeric(df.get("is_flare_active", 0), errors="coerce")
            .fillna(0)
            .to_numpy()
            .astype(int)
            if "is_flare_active" in df.columns
            else np.ones(n, dtype=int)
        )
        decay_excl = (active == 1) & (df["flare_phase"].astype(str) == "decaying").to_numpy()
        decay_excl &= train_mask

    mask = buffer_excl | coverage_excl | decay_excl
    counts = {
        "buffer": int(buffer_excl.sum()),
        "coverage": int(coverage_excl.sum()),
        "decaying": int(decay_excl.sum()),
        "total": int(mask.sum()),
    }
    return mask, counts


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

    # v3 P1 label surgery: drop ambiguity-buffer / unobserved-GOES-window /
    # (optionally) decaying rows from the FIT ONLY.  Val/test keep every row
    # and its strict label — the excluded rows simply leave the train split.
    excl_mask, excl_counts = build_train_exclusion_mask(
        df, split_labels, horizon_min
    )
    fit_split_labels = split_labels.copy()
    if excl_counts["total"]:
        fit_split_labels[excl_mask] = "train_excluded"
        print(
            f"    train-excl[{horizon_min}m]: buffer={excl_counts['buffer']}  "
            f"coverage={excl_counts['coverage']}  "
            f"decaying={excl_counts['decaying']}  "
            f"total={excl_counts['total']}"
        )

    arrays = build_split_arrays(df, feature_cols, label_col, fit_split_labels)

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

    # v3 P0b: Event-level threshold tuning with duty-cycle caps.
    # Extract val timestamps and flare starts for episode-based optimization.
    from tejas.config import (
        FORECAST_MIN_RECALL_PER_HORIZON,
        FORECAST_MAX_DUTY_CYCLE_PER_HORIZON,
        CLASSIFICATION_DIR,
        CREDIBLE_CATALOGUE_NAME,
    )

    val_mask = split_labels == "val"
    val_timestamps = df.loc[val_mask, "datetime"].reset_index(drop=True)

    # Credible flare starts *inside the val window* — compute_event_metrics
    # expects pre-windowed starts; out-of-window flares would dilute recall.
    credible_path = Path(CLASSIFICATION_DIR) / CREDIBLE_CATALOGUE_NAME
    flare_starts = pd.Series([], dtype="datetime64[ns]")
    if credible_path.exists() and len(val_timestamps) > 0:
        credible = pd.read_csv(credible_path)
        starts = pd.to_datetime(credible["start_time"])
        t0, t1 = val_timestamps.min(), val_timestamps.max()
        flare_starts = (
            starts[(starts >= t0) & (starts <= t1)]
            .sort_values()
            .reset_index(drop=True)
        )
    if len(flare_starts) == 0:
        # Loud fallback — an empty flare list silently degrades the tuner
        # to the default threshold.
        print(
            f"    [WARN] horizon={horizon_min}m: no credible flares in val "
            f"window ({credible_path}) — event-level tuner falls back to "
            f"default threshold"
        )

    threshold = tune_threshold_event_level(
        y_prob=val_prob,
        timestamps=val_timestamps,
        flare_starts=flare_starts,
        horizon_min=int(horizon_min),
        min_recall=FORECAST_MIN_RECALL_PER_HORIZON.get(int(horizon_min)),
        max_duty_cycle=FORECAST_MAX_DUTY_CYCLE_PER_HORIZON.get(int(horizon_min)),
    )

    # Log the tuner's operating point on val — FEASIBLE means the recall
    # floor was met under the duty cap; FALLBACK means the floor was
    # unreachable and the max-recall duty-feasible threshold was used.
    if len(flare_starts) > 0:
        from tejas.forecasting.event_metrics import compute_event_metrics

        _m = compute_event_metrics(
            val_timestamps, val_prob, threshold, flare_starts, int(horizon_min)
        )
        _duty = float((np.asarray(val_prob) >= threshold).mean())
        _floor = FORECAST_MIN_RECALL_PER_HORIZON.get(int(horizon_min), 0.0)
        _verdict = "FEASIBLE" if _m["event_recall"] >= _floor else "FALLBACK"
        print(
            f"    tuner[{_verdict}] thr={threshold:.4f}  duty={_duty:.3f}  "
            f"ev_recall={_m['event_recall']:.3f} (floor {_floor})  "
            f"ep_prec={_m['episode_precision']:.3f}"
        )
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
        "n_train_excluded": int(excl_counts["total"]),
        "n_train_excluded_buffer": int(excl_counts["buffer"]),
        "n_train_excluded_coverage": int(excl_counts["coverage"]),
        "n_train_excluded_decaying": int(excl_counts["decaying"]),
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
        "n_train_excluded": result.get("n_train_excluded", 0),
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
