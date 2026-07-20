"""Diagnostic: replicate the P0b tuner's view of the val split.

For each horizon: load the trained model + calibrator, rebuild the exact
val probabilities, and sweep the quantile grid reporting
(threshold, duty, event_recall, episode_precision) so we can see the
feasible region the tuner was choosing from — and whether the returned
threshold was feasible or a fallback.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from tejas.config import (
    MODELS_DIR,
    FORECAST_DIR,
    FORECAST_CADENCE_S,
    CLASSIFICATION_DIR,
    CREDIBLE_CATALOGUE_NAME,
    FORECAST_MIN_RECALL_PER_HORIZON,
    FORECAST_MAX_DUTY_CYCLE_PER_HORIZON,
)
from tejas.forecasting.sampling import select_feature_columns
from tejas.forecasting.splits import assign_split, apply_purge, build_split_arrays
from tejas.forecasting.training import _predict_proba_positive, _apply_calibrator
from tejas.forecasting.event_metrics import compute_event_metrics

# ── Rebuild the training-time frame ────────────────────────────────────
path = Path(FORECAST_DIR) / f"training_frame_{FORECAST_CADENCE_S}s.parquet"
df = pd.read_parquet(path)
feature_cols = select_feature_columns(df)
split_labels = assign_split(df)
df, split_labels = apply_purge(df, split_labels)

val_mask = split_labels == "val"
val_timestamps = df.loc[val_mask, "datetime"].reset_index(drop=True)

credible = pd.read_csv(Path(CLASSIFICATION_DIR) / CREDIBLE_CATALOGUE_NAME)
starts = pd.to_datetime(credible["start_time"])
t0, t1 = val_timestamps.min(), val_timestamps.max()
flare_starts = starts[(starts >= t0) & (starts <= t1)].sort_values().reset_index(drop=True)

print(f"Val window: {t0} .. {t1}  rows={val_mask.sum():,}")
print(f"Credible flares in val window: {len(flare_starts)}")
print()

for h in [5, 15, 30, 60]:
    art = joblib.load(Path(MODELS_DIR) / f"xgb_horizon_{h}m.joblib")
    model, calibrator = art["model"], art.get("calibrator")
    tuned_thr = art["decision_threshold"]

    label_col = f"label_flare_within_{h}m"
    arrays = build_split_arrays(df, feature_cols, label_col, split_labels)
    X_val = arrays["X_val"]

    val_prob = _apply_calibrator(calibrator, _predict_proba_positive(model, X_val))

    min_recall = FORECAST_MIN_RECALL_PER_HORIZON.get(h)
    max_duty = FORECAST_MAX_DUTY_CYCLE_PER_HORIZON.get(h)

    quantiles = np.linspace(0.01, 0.99, 50)
    grid = np.unique(np.quantile(val_prob, quantiles))

    rows = []
    for thr in grid:
        duty = float((val_prob >= thr).mean())
        m = compute_event_metrics(val_timestamps, val_prob, thr, flare_starts, h)
        rows.append((thr, duty, m["event_recall"], m["episode_precision"], m["n_episodes"]))

    sweep = pd.DataFrame(rows, columns=["thr", "duty", "ev_recall", "ep_precision", "n_episodes"])
    duty_ok = sweep[sweep.duty <= max_duty]
    feasible = duty_ok[duty_ok.ev_recall >= min_recall]

    print(f"=== horizon {h}m  (floor={min_recall}, duty_cap={max_duty}, tuned_thr={tuned_thr:.4f}) ===")
    print(f"  grid size: {len(sweep)}  duty-feasible: {len(duty_ok)}  fully-feasible: {len(feasible)}")
    if len(duty_ok):
        best = duty_ok.loc[duty_ok.ev_recall.idxmax()]
        print(f"  max event recall under duty cap: {best.ev_recall:.3f} "
              f"(thr={best.thr:.4f}, duty={best.duty:.3f}, ep_prec={best.ep_precision:.3f})")
    # What the tuned threshold actually achieves on val:
    m = compute_event_metrics(val_timestamps, val_prob, tuned_thr, flare_starts, h)
    duty_at = float((val_prob >= tuned_thr).mean())
    print(f"  AT TUNED THR: duty={duty_at:.3f}  ev_recall={m['event_recall']:.3f}  "
          f"ep_prec={m['episode_precision']:.3f}  episodes={m['n_episodes']}  "
          f"median_lead={m['median_lead_min']}")
    status = "FEASIBLE (met floor)" if m["event_recall"] >= min_recall and duty_at <= max_duty else "FALLBACK (max-recall under duty cap)"
    print(f"  verdict: {status}")
    # Show the recall/duty frontier (every 5th grid point)
    print("  frontier (thr, duty, ev_recall, ep_prec):")
    for _, r in duty_ok.iloc[::5].iterrows():
        print(f"    {r.thr:8.4f}  {r.duty:6.3f}  {r.ev_recall:6.3f}  {r.ep_precision:6.3f}")
    print()
