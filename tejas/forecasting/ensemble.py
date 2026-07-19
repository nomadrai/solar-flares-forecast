"""
TEJAS Phase 7 — Stacked Ensemble (Fix 5)
=========================================
Meta-learner combining LogReg + XGBoost + LightGBM predictions.

Architecture:
    Base models: LogisticRegression + XGBoost + LightGBM (already trained
        on the train split; artifacts loaded from MODELS_DIR).
    Meta-learner: LogisticRegression on the (calibrated) val-split
        probabilities from all three base models.

No-future-data guarantee:
    * Base models trained on train split only.
    * Meta-learner trained on val split (chronologically after train;
      base models never fit on val, so val predictions are out-of-fold).
    * Meta threshold tuned on val.
    * Test split only consumed in final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    FORECAST_DIR,
    FORECAST_LABEL_HORIZONS_MIN,
    FORECAST_TARGET_FAR,
    MODELS_DIR,
)
from tejas.forecasting.evaluation import compute_metrics, tune_threshold
from tejas.forecasting.inference import (
    _apply_artifact_calibrator,
    _predict_positive,
    load_forecaster,
)
from tejas.forecasting.sampling import select_feature_columns
from tejas.forecasting.splits import apply_purge, assign_split


BASE_ALGORITHMS = ("logreg", "xgb", "lgbm")


# ── Stacked forecaster (inference-time) ────────────────────────────────────


@dataclass
class StackedForecaster:
    """Meta-learner combining LogReg + XGBoost + LightGBM predictions."""

    base_forecasters: dict
    meta_models: dict
    meta_thresholds: dict
    feature_columns: list = field(default_factory=list)

    def predict_row(self, row):
        base_probs = {}
        for algo, f in self.base_forecasters.items():
            base_probs[algo] = f.predict_row(row)
        out = {}
        for h in FORECAST_LABEL_HORIZONS_MIN:
            x = np.array([[
                base_probs["logreg"][f"prob_{h}m"],
                base_probs["xgb"][f"prob_{h}m"],
                base_probs["lgbm"][f"prob_{h}m"],
            ]])
            p = self.meta_models[h].predict_proba(x)[0, 1]
            out[f"prob_{h}m"] = float(p)
        return out

    def predict_dataframe(self, df):
        n = len(df)
        # Get base probs per algorithm as (n_rows, n_horizons)
        base_probs = {
            algo: f.predict_dataframe(df) for algo, f in self.base_forecasters.items()
        }
        out = pd.DataFrame(index=df.index)
        for h in FORECAST_LABEL_HORIZONS_MIN:
            X = np.column_stack([
                base_probs["logreg"][f"prob_{h}m"].to_numpy(dtype=float),
                base_probs["xgb"][f"prob_{h}m"].to_numpy(dtype=float),
                base_probs["lgbm"][f"prob_{h}m"].to_numpy(dtype=float),
            ])
            out[f"prob_{h}m"] = self.meta_models[h].predict_proba(X)[:, 1]
        return out

    def decision_thresholds(self):
        return {int(h): float(t) for h, t in self.meta_thresholds.items()}


# ── Training ───────────────────────────────────────────────────────────────


def _predict_all_horizons(model_artifact, X, feature_cols=None):
    """Apply a single-horizon artifact (dict) to a matrix."""
    raw = _predict_positive(model_artifact["model"], X)
    return _apply_artifact_calibrator(model_artifact, raw)


def train_stacked_ensemble(
    training_frame=None,
    cadence_s=None,
    output_dir=None,
    models_dir=None,
    verbose=True,
):
    """Fit a stacking meta-learner over the LogReg / XGB / LGBM outputs.

    Steps:
        1. Load base Forecasters (must have been trained already).
        2. Build the val-split feature matrix ``X_val``.
        3. Compute calibrated base probs on val for each horizon.
        4. Fit a LogisticRegression per horizon on the (val_probs, y_val).
        5. Tune the meta decision threshold (FAR-constrained) on val.
        6. Persist artifacts as ``stacked_horizon_{h}m.joblib``.

    Returns:
        DataFrame with val + test metrics per horizon.
    """
    from tejas.config import FORECAST_CADENCE_S

    if cadence_s is None:
        cadence_s = FORECAST_CADENCE_S
    output_dir = Path(output_dir) if output_dir else Path(FORECAST_DIR)
    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)

    if training_frame is None:
        path = output_dir / f"training_frame_{cadence_s}s.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Training frame not found: {path}. "
                f"Run prepare_training_frame first."
            )
        training_frame = pd.read_parquet(path)

    feature_cols = select_feature_columns(training_frame)
    split_labels = assign_split(training_frame)
    training_frame, split_labels = apply_purge(training_frame, split_labels)

    # Load base forecasters.
    if verbose:
        print("  Loading base forecasters (logreg, xgb, lgbm)...")
    bases = {}
    for algo in BASE_ALGORITHMS:
        try:
            bases[algo] = load_forecaster(algo, models_dir=models_dir)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing base model for '{algo}': {exc}.  "
                f"Train the base models first with `--train --algorithm all`."
            ) from exc

    val_mask = (split_labels == "val").to_numpy()
    test_mask = (split_labels == "test").to_numpy()
    val_df = training_frame.loc[val_mask].reset_index(drop=True)
    test_df = training_frame.loc[test_mask].reset_index(drop=True)

    X_val = val_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

    metrics_rows = []
    for h in FORECAST_LABEL_HORIZONS_MIN:
        label_col = f"label_flare_within_{h}m"
        y_val = val_df[label_col].fillna(0).astype(np.int8).to_numpy()
        y_test = test_df[label_col].fillna(0).astype(np.int8).to_numpy()

        # Base probs (calibrated) on val and test for each algorithm.
        base_val_probs = np.column_stack([
            _predict_all_horizons(bases[algo].horizon_models[h], X_val)
            for algo in BASE_ALGORITHMS
        ])
        base_test_probs = np.column_stack([
            _predict_all_horizons(bases[algo].horizon_models[h], X_test)
            for algo in BASE_ALGORITHMS
        ])

        # Meta-learner: LR without regularisation (well-conditioned; 3 feats).
        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1e6, solver="lbfgs", max_iter=500,
            )),
        ])
        if int((y_val == 1).sum()) == 0 or int((y_val == 0).sum()) == 0:
            if verbose:
                print(f"  [SKIP] horizon={h}m: val has a single class.")
            continue
        meta.fit(base_val_probs, y_val)

        val_stack = meta.predict_proba(base_val_probs)[:, 1]
        test_stack = meta.predict_proba(base_test_probs)[:, 1]

        # Threshold: config-selected metric on val (v2: precision at a
        # per-horizon recall floor; falls back to the scalar default).
        from tejas.config import (
            FORECAST_THRESHOLD_METRIC,
            FORECAST_MIN_RECALL_PER_HORIZON,
        )
        threshold = tune_threshold(
            y_val, val_stack,
            metric=FORECAST_THRESHOLD_METRIC,
            target_far=FORECAST_TARGET_FAR,
            min_recall=FORECAST_MIN_RECALL_PER_HORIZON.get(int(h)),
        )
        val_metrics = compute_metrics(y_val, val_stack, threshold=threshold)
        test_metrics = compute_metrics(y_test, test_stack, threshold=threshold)

        art_path = models_dir / f"stacked_horizon_{h}m.joblib"
        joblib.dump({
            "algorithm": "stacked",
            "horizon_min": int(h),
            "meta_model": meta,
            "meta_threshold": float(threshold),
            "base_algorithms": list(BASE_ALGORITHMS),
        }, art_path)

        row = {
            "algorithm": "stacked",
            "horizon_min": int(h),
            "n_train": int(val_mask.sum()),  # meta trains on val
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "decision_threshold": float(threshold),
        }
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        for k, v in test_metrics.items():
            row[f"test_{k}"] = v
        metrics_rows.append(row)

        if verbose:
            print(
                f"  stacked h={h}m  thr={threshold:.2f}  "
                f"val PR-AUC={val_metrics['pr_auc']:.3f} "
                f"test PR-AUC={test_metrics['pr_auc']:.3f} "
                f"FAR={test_metrics['false_alarm_rate']:.3f} "
                f"recall={test_metrics['recall']:.3f}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics_summary_stacked.csv", index=False)
    if verbose and len(metrics_df):
        print(f"  Saved: {output_dir / 'metrics_summary_stacked.csv'}")

    return metrics_df


def load_stacked_forecaster(models_dir=None):
    """Load base forecasters + stacked meta artifacts and return a
    :class:`StackedForecaster` ready for inference."""
    import joblib

    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)
    bases = {algo: load_forecaster(algo, models_dir=models_dir)
             for algo in BASE_ALGORITHMS}

    meta_models = {}
    meta_thresholds = {}
    for h in FORECAST_LABEL_HORIZONS_MIN:
        art_path = models_dir / f"stacked_horizon_{h}m.joblib"
        if not art_path.exists():
            raise FileNotFoundError(f"Missing stacked artifact: {art_path}")
        art = joblib.load(art_path)
        meta_models[int(h)] = art["meta_model"]
        meta_thresholds[int(h)] = float(art["meta_threshold"])

    return StackedForecaster(
        base_forecasters=bases,
        meta_models=meta_models,
        meta_thresholds=meta_thresholds,
        feature_columns=list(bases["xgb"].feature_columns),
    )
