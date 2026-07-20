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
    FORECAST_THRESHOLD_METRIC,
    FORECAST_THRESHOLD_GRID_STEP,
    FORECAST_TARGET_FAR,
    FORECAST_FBETA,
    FORECAST_MIN_RECALL,
    FORECAST_MIN_RECALL_PER_HORIZON,
    FORECAST_MAX_DUTY_CYCLE_PER_HORIZON,
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
    metric: str = "f1",
    beta: float | None = None,
    target_far: float | None = None,
    min_recall: float | None = None,
) -> float:
    """Return the threshold in ``grid`` that maximises the chosen metric.

    Supported metrics:

    * ``"f1"`` — classical F1 (default; back-compat).
    * ``"fbeta"`` — F-beta.  ``beta > 1`` weights recall more heavily
      (e.g. F2 for recall-oriented operations).  Uses
      :data:`FORECAST_FBETA` if ``beta`` is ``None``.
    * ``"youden"`` — Youden's J = TPR - FPR.  Directly rewards high
      recall AND low false-alarm rate; scientifically defensible and
      does NOT collapse to the trivial "always-positive" solution when
      the class is imbalanced (unlike raw F1 with a tuned threshold).
    * ``"far_constrained"`` — maximise recall subject to
      ``false_alarm_rate <= target_far``.  Uses
      :data:`FORECAST_TARGET_FAR` if ``target_far`` is ``None``.  If no
      threshold meets the constraint, returns the one with the lowest
      FAR (soft fallback so downstream code still gets a usable value).
    * ``"precision_at_min_recall"`` — maximise precision subject to
      ``recall >= min_recall``.  Uses :data:`FORECAST_MIN_RECALL` if
      ``min_recall`` is ``None``.

    Falls back to :data:`FORECAST_DECISION_THRESHOLD_DEFAULT` when
    ``y_true`` contains no positives (all metrics become degenerate).
    """
    from sklearn.metrics import fbeta_score, f1_score

    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    if (y_true == 1).sum() == 0:
        return FORECAST_DECISION_THRESHOLD_DEFAULT

    if grid is None:
        step = float(FORECAST_THRESHOLD_GRID_STEP)
        grid = np.arange(step, 1.0 - step / 2.0, step)

    metric = str(metric).lower()

    if metric == "f1":
        best_thr = FORECAST_DECISION_THRESHOLD_DEFAULT
        best = -1.0
        for thr in grid:
            y_pred = (y_prob >= thr).astype(np.int64)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best:
                best = score
                best_thr = float(thr)
        return best_thr

    if metric == "fbeta":
        b = float(FORECAST_FBETA if beta is None else beta)
        best_thr = FORECAST_DECISION_THRESHOLD_DEFAULT
        best = -1.0
        for thr in grid:
            y_pred = (y_prob >= thr).astype(np.int64)
            score = fbeta_score(y_true, y_pred, beta=b, zero_division=0)
            if score > best:
                best = score
                best_thr = float(thr)
        return best_thr

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if metric == "youden":
        # Youden's J = TPR - FPR.  Argmax over the grid.  When the
        # positive class dominates (long horizons), a trivially high
        # recall paired with a saturated FPR pushes J to ~0 and any
        # non-trivial threshold that pushes FPR down wins — which is
        # exactly what we want for "high TPR AND low FAR".
        best_thr = FORECAST_DECISION_THRESHOLD_DEFAULT
        best = -1.0
        for thr in grid:
            y_pred = (y_prob >= thr).astype(np.int64)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tpr = tp / n_pos if n_pos > 0 else 0.0
            fpr = fp / n_neg if n_neg > 0 else 0.0
            j = tpr - fpr
            if j > best:
                best = j
                best_thr = float(thr)
        return best_thr

    if metric == "far_constrained":
        cap = float(FORECAST_TARGET_FAR if target_far is None else target_far)
        best_thr = None
        best_recall = -1.0
        # Fallback: track the threshold with the lowest FAR overall.
        fallback_thr = FORECAST_DECISION_THRESHOLD_DEFAULT
        fallback_far = float("inf")
        for thr in grid:
            y_pred = (y_prob >= thr).astype(np.int64)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tpr = tp / n_pos if n_pos > 0 else 0.0
            far = fp / n_neg if n_neg > 0 else 0.0
            if far < fallback_far:
                fallback_far = far
                fallback_thr = float(thr)
            if far <= cap and tpr > best_recall:
                best_recall = tpr
                best_thr = float(thr)
        return best_thr if best_thr is not None else fallback_thr

    if metric == "precision_at_min_recall":
        floor = float(FORECAST_MIN_RECALL if min_recall is None else min_recall)
        best_thr = None
        best_precision = -1.0
        fallback_thr = FORECAST_DECISION_THRESHOLD_DEFAULT
        fallback_recall = -1.0
        for thr in grid:
            y_pred = (y_prob >= thr).astype(np.int64)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            recall = tp / n_pos if n_pos > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if recall > fallback_recall:
                fallback_recall = recall
                fallback_thr = float(thr)
            # ``>=`` so precision ties resolve to the HIGHEST threshold —
            # same precision with fewer fired alerts.  Strict ``>`` made a
            # flat precision curve collapse to the always-fire threshold.
            if recall >= floor and precision >= best_precision:
                best_precision = precision
                best_thr = float(thr)
        return best_thr if best_thr is not None else fallback_thr

    raise ValueError(f"Unknown threshold metric: {metric!r}")


def tune_threshold_default(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tune using the metric configured in :data:`FORECAST_THRESHOLD_METRIC`.

    Thin wrapper so callers get the configured strategy without having
    to pass the metric name explicitly.
    """
    return tune_threshold(y_true, y_prob, metric=FORECAST_THRESHOLD_METRIC)


def tune_threshold_event_level(
    y_prob: np.ndarray,
    timestamps: pd.Series,
    flare_starts: pd.Series,
    horizon_min: int,
    grid: np.ndarray | None = None,
    min_recall: float | None = None,
    max_duty_cycle: float | None = None,
    merge_gap_s: int = 300,
) -> float:
    """Tune threshold on event-level metrics with a duty-cycle cap.

    v3 P0 — replaces the row-level precision_at_min_recall objective with
    an event-level one: maximize episode precision s.t. event recall ≥ floor
    AND alarm duty cycle ≤ cap.  The duty cap makes the always-on corner
    infeasible.

    Args:
        y_prob: Predicted probabilities (val split, calibrated).
        timestamps: Row-aligned datetimes (val split).
        flare_starts: Ground-truth flare start times inside val window.
        horizon_min: Horizon in minutes (5/15/30/60).
        grid: Threshold grid to sweep.  Defaults to a quantile-spaced grid
            over the calibrated val probabilities (better resolution than
            a linear 0.01 grid when probs concentrate near 0).
        min_recall: Event recall floor.  Defaults to
            FORECAST_MIN_RECALL_PER_HORIZON[horizon_min] or
            FORECAST_MIN_RECALL if absent.
        max_duty_cycle: Duty-cycle cap (fraction of val rows above
            threshold).  Defaults to
            FORECAST_MAX_DUTY_CYCLE_PER_HORIZON[horizon_min].
        merge_gap_s: Episode merge gap for computing episode precision
            (default 300 s).

    Returns:
        The threshold maximizing episode precision subject to the two
        constraints.  Falls back to FORECAST_DECISION_THRESHOLD_DEFAULT
        if no threshold is feasible (e.g. val has zero flares).
    """
    from tejas.forecasting.event_metrics import compute_event_metrics

    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) == 0 or len(flare_starts) == 0:
        return FORECAST_DECISION_THRESHOLD_DEFAULT

    if min_recall is None:
        min_recall = float(
            FORECAST_MIN_RECALL_PER_HORIZON.get(
                int(horizon_min), FORECAST_MIN_RECALL
            )
        )
    if max_duty_cycle is None:
        max_duty_cycle = float(
            FORECAST_MAX_DUTY_CYCLE_PER_HORIZON.get(int(horizon_min), 1.0)
        )

    if grid is None:
        # Exhaustive grid: every unique calibrated probability IS a distinct
        # operating point (thresholds between two unique values yield the
        # same alert set).  Quantile grids degenerate here — isotonic
        # calibration collapses probs to a step function with few unique
        # values, so 50 quantiles can dedup to as few as 3 thresholds,
        # leaving most of the duty-cap headroom unexplored.
        uniq = np.unique(y_prob)
        if len(uniq) > 512:
            idx = np.unique(np.linspace(0, len(uniq) - 1, 512).round().astype(int))
            uniq = uniq[idx]
        grid = uniq

    best_thr = None
    best_precision = -1.0
    fallback_thr = FORECAST_DECISION_THRESHOLD_DEFAULT
    fallback_recall = -1.0
    fallback_precision = -1.0

    for thr in grid:
        duty = float((y_prob >= thr).mean())
        if duty > max_duty_cycle:
            continue  # violates duty cap

        m = compute_event_metrics(
            timestamps, y_prob, thr, flare_starts, horizon_min, merge_gap_s
        )
        recall = m["event_recall"]
        precision = m["episode_precision"]

        # Track best duty-feasible fallback: max recall, ties → precision.
        if recall > fallback_recall or (
            recall == fallback_recall and precision > fallback_precision
        ):
            fallback_recall = recall
            fallback_precision = precision
            fallback_thr = float(thr)

        # Feasible: meets recall floor and duty cap
        if recall >= min_recall and precision >= best_precision:
            best_precision = precision
            best_thr = float(thr)

    return best_thr if best_thr is not None else fallback_thr


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
