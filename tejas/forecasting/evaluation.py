"""
Phase 7 — Evaluation metrics.

Precision / recall / F1 / PR-AUC / ROC-AUC / false-alarm rate / Brier
score, plus per-event lead time.  Lead time is measured against the Phase
5 master catalogue: for each labelled flare start in the test set, the
first prior timestamp where the horizon probability crosses the tuned
threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tejas.config import (
    FORECAST_DECISION_THRESHOLD_DEFAULT,
    FORECAST_LEAD_TIME_MIN_PROB,
)


# ── Metrics ────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Compute the standard classification metric bundle.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_prob: Predicted probabilities in [0, 1].
        threshold: Decision threshold.  Defaults to
            :data:`FORECAST_DECISION_THRESHOLD_DEFAULT`.

    Returns:
        Dict with keys ``pr_auc``, ``roc_auc``, ``precision``, ``recall``,
        ``f1``, ``false_alarm_rate``, ``brier_score``, ``threshold``,
        ``n_positive``, ``n_negative``, ``n_predicted_positive``.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        brier_score_loss,
    )

    if threshold is None:
        threshold = FORECAST_DECISION_THRESHOLD_DEFAULT

    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(np.int64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    n_pred_pos = int((y_pred == 1).sum())

    def _safe(fn, default=float("nan")):
        try:
            return float(fn())
        except (ValueError, ZeroDivisionError):
            return default

    pr_auc = _safe(lambda: average_precision_score(y_true, y_prob)) if n_pos > 0 else float("nan")
    roc_auc = _safe(lambda: roc_auc_score(y_true, y_prob)) if (n_pos > 0 and n_neg > 0) else float("nan")

    precision = _safe(lambda: precision_score(y_true, y_pred, zero_division=0))
    recall = _safe(lambda: recall_score(y_true, y_pred, zero_division=0))
    f1 = _safe(lambda: f1_score(y_true, y_pred, zero_division=0))

    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_alarm_rate = false_positives / n_neg if n_neg > 0 else float("nan")
    brier = _safe(lambda: brier_score_loss(y_true, y_prob))

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "brier_score": brier,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_predicted_positive": n_pred_pos,
    }


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Return the threshold in ``grid`` that maximises F1 on ``y_prob``.

    Falls back to :data:`FORECAST_DECISION_THRESHOLD_DEFAULT` when ``y_true``
    contains no positives (F1 is undefined).
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    if (y_true == 1).sum() == 0:
        return FORECAST_DECISION_THRESHOLD_DEFAULT

    if grid is None:
        grid = np.arange(0.05, 0.951, 0.05)

    best_thr = FORECAST_DECISION_THRESHOLD_DEFAULT
    best_f1 = -1.0
    for thr in grid:
        y_pred = (y_prob >= thr).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


# ── Lead time ──────────────────────────────────────────────────────────────


def compute_lead_time(
    timestamps: pd.Series,
    y_prob: np.ndarray,
    flare_start_times: pd.Series,
    threshold: float | None = None,
    horizon_min: int | None = None,
) -> pd.DataFrame:
    """For each flare start, find the earliest prior probability crossing.

    A row of the returned DataFrame per flare start.  The crossing search
    is bounded to the ``horizon_min`` prior seconds (default: no bound).

    Args:
        timestamps: Row-aligned timestamps of ``y_prob`` (tz-naive).
        y_prob: Predicted probabilities.
        flare_start_times: Ground-truth flare start times.
        threshold: Probability crossing threshold.  Defaults to
            :data:`FORECAST_LEAD_TIME_MIN_PROB`.
        horizon_min: Search window in minutes before each flare.  ``None``
            means unbounded (whole series before the start).

    Returns:
        DataFrame with columns ``flare_start``, ``first_crossing``,
        ``lead_time_s``.  ``lead_time_s`` is NaN if no crossing occurred
        inside the window.
    """
    if threshold is None:
        threshold = FORECAST_LEAD_TIME_MIN_PROB

    ts = pd.to_datetime(timestamps.to_numpy() if hasattr(timestamps, "to_numpy") else timestamps)
    ts = pd.DatetimeIndex(ts)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(ts) != len(y_prob):
        raise ValueError("timestamps and y_prob length mismatch")

    crossings = ts[y_prob >= threshold]

    records: list[dict] = []
    horizon_delta = (
        pd.Timedelta(minutes=int(horizon_min)) if horizon_min is not None else None
    )

    for start in pd.DatetimeIndex(flare_start_times):
        low = start - horizon_delta if horizon_delta is not None else pd.Timestamp.min
        # crossings strictly before the flare, inside the window
        window_hits = crossings[(crossings >= low) & (crossings < start)]
        if len(window_hits):
            first = window_hits[0]
            lead = (start - first).total_seconds()
        else:
            first = pd.NaT
            lead = float("nan")
        records.append(
            {
                "flare_start": start,
                "first_crossing": first,
                "lead_time_s": lead,
            }
        )

    return pd.DataFrame(records)


# ── Feature importance ──────────────────────────────────────────────────────


def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Return a feature importance DataFrame (tree models only).

    Falls back to an empty DataFrame for models without a native importance
    attribute (LogReg is handled via absolute coefficient magnitudes).
    """
    # sklearn Pipeline (LogReg)
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coefs = np.ravel(clf.coef_)
            return (
                pd.DataFrame({"feature": feature_cols, "importance": np.abs(coefs)})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        return pd.DataFrame(columns=["feature", "importance"])

    # XGBoost / LightGBM
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        n = min(len(importances), len(feature_cols))
        return (
            pd.DataFrame(
                {"feature": feature_cols[:n], "importance": importances[:n]}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return pd.DataFrame(columns=["feature", "importance"])
