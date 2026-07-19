"""
v2 Evaluation — event-level scoreboard + GOES-confirmed subset
==============================================================
Produces data/forecasting/YYYY/MM/forecast_report_v2.md with three views:

  1. Row-level test metrics (threshold from the precision_at_min_recall
     tuner) — comparable to the v1 snapshot.
  2. Event-level metrics vs credible TEJAS events on the test days:
     event recall / episode precision / median lead per horizon.
  3. The same event-level metrics vs GOES-confirmed flares only — the
     externally-validated headline numbers.

Run after scripts/run_forecasting.py --train --evaluate:
    python -m scripts.run_v2_evaluation [--algorithm xgb]
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tejas.config import (
    FORECAST_DIR, MODELS_DIR, CLASSIFICATION_DIR, FORECAST_CADENCE_S,
    FORECAST_LABEL_HORIZONS_MIN, CREDIBLE_CATALOGUE_NAME,
    FORECAST_THRESHOLD_METRIC, FORECAST_MIN_RECALL_PER_HORIZON,
)
from tejas.forecasting.pipeline import _predict_positive, _apply_artifact_calibrator
from tejas.forecasting.splits import assign_split, apply_purge
from tejas.forecasting.training import load_feature_columns_manifest
from tejas.forecasting.event_metrics import compute_event_metrics
from tejas.validation.goes_match import load_goes_flare_list


def _load_test_frame(cadence_s):
    path = Path(FORECAST_DIR) / f"training_frame_{cadence_s}s.parquet"
    frame = pd.read_parquet(path)
    split = assign_split(frame)
    frame, split = apply_purge(frame, split)
    test_df = frame.loc[(split == "test").to_numpy()].reset_index(drop=True)
    return test_df


def _window(series_like, t0, t1):
    s = pd.to_datetime(pd.Series(series_like), utc=True).dt.tz_localize(None)
    return s[(s >= t0) & (s <= t1)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="xgb")
    ap.add_argument("--cadence", type=int, default=FORECAST_CADENCE_S)
    args = ap.parse_args()

    test_df = _load_test_frame(args.cadence)
    feature_cols = load_feature_columns_manifest(models_dir=Path(MODELS_DIR))
    X = test_df[feature_cols].to_numpy(dtype=float)
    ts = pd.to_datetime(test_df["datetime"], utc=True).dt.tz_localize(None)
    t0, t1 = ts.min(), ts.max()
    print(f"Test window: {t0} .. {t1}  ({len(test_df):,} rows)")

    # ── Ground-truth event sets ────────────────────────────────────────
    credible = pd.read_csv(Path(CLASSIFICATION_DIR) / CREDIBLE_CATALOGUE_NAME)
    cred_starts = _window(credible["start_time"], t0, t1)

    goes = load_goes_flare_list()
    goes_starts = _window(goes["begin_time"], t0, t1) if len(goes) else pd.Series(dtype="datetime64[ns]")
    print(f"Credible events in window: {len(cred_starts)}; "
          f"GOES flares in window: {len(goes_starts)}")

    # ── Per-horizon predictions + metrics ──────────────────────────────
    row_rows, ev_rows, goes_rows = [], [], []
    for h in FORECAST_LABEL_HORIZONS_MIN:
        art = joblib.load(Path(MODELS_DIR) / f"{args.algorithm}_horizon_{h}m.joblib")
        thr = float(art["decision_threshold"])
        prob = _apply_artifact_calibrator(art, _predict_positive(art["model"], X))

        y = test_df[f"label_flare_within_{h}m"].fillna(0).astype(int).to_numpy()
        pred = (prob >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        row_rows.append({
            "horizon_min": h, "threshold": thr,
            "row_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "row_recall": tp / (tp + fn) if tp + fn else float("nan"),
        })

        m = compute_event_metrics(ts, prob, thr, cred_starts, h)
        m.pop("lead_times_min")
        ev_rows.append(m)

        if len(goes_starts):
            g = compute_event_metrics(ts, prob, thr, goes_starts, h)
            g.pop("lead_times_min")
            goes_rows.append(g)

    row_df = pd.DataFrame(row_rows)
    ev_df = pd.DataFrame(ev_rows)
    goes_df = pd.DataFrame(goes_rows) if goes_rows else pd.DataFrame()

    ev_df.to_csv(Path(FORECAST_DIR) / "event_metrics_v2.csv", index=False)
    if not goes_df.empty:
        goes_df.to_csv(Path(FORECAST_DIR) / "event_metrics_goes_v2.csv", index=False)

    # ── Report ─────────────────────────────────────────────────────────
    L = []
    L.append("# TEJAS Forecast Report v2 — Precision + Recall Optimization\n")
    L.append(f"- Algorithm: `{args.algorithm}` · cadence {args.cadence}s · "
             f"threshold metric `{FORECAST_THRESHOLD_METRIC}` "
             f"(recall floors {FORECAST_MIN_RECALL_PER_HORIZON})")
    L.append(f"- Labels: credible events only "
             f"(soft ≥ 25 c/s ∪ GOES-matched); external truth: NOAA GOES flare list")
    L.append(f"- Test window: {t0} → {t1} · credible events: {len(cred_starts)} · "
             f"GOES flares: {len(goes_starts)}\n")

    L.append("## Row-level (10 s rows, timing-strict)\n")
    L.append("| Horizon | Threshold | Precision | Recall |")
    L.append("|---|---|---|---|")
    for _, r in row_df.iterrows():
        L.append(f"| {int(r.horizon_min)} min | {r.threshold:.2f} | "
                 f"{r.row_precision:.3f} | {r.row_recall:.3f} |")

    L.append("\n## Event-level vs credible TEJAS events (operator view)\n")
    L.append("| Horizon | Events | Caught | Event recall | Episodes | True | "
             "Episode precision | Median lead (min) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in ev_df.iterrows():
        L.append(f"| {int(r.horizon_min)} min | {int(r.n_events)} | "
                 f"{int(r.n_events_caught)} | **{r.event_recall:.1%}** | "
                 f"{int(r.n_episodes)} | {int(r.n_episodes_true)} | "
                 f"**{r.episode_precision:.1%}** | {r.median_lead_min:.1f} |")

    if not goes_df.empty:
        L.append("\n## Event-level vs GOES-confirmed flares (external truth)\n")
        L.append("| Horizon | GOES flares | Caught | Event recall | "
                 "Episode precision | Median lead (min) |")
        L.append("|---|---|---|---|---|---|")
        for _, r in goes_df.iterrows():
            L.append(f"| {int(r.horizon_min)} min | {int(r.n_events)} | "
                     f"{int(r.n_events_caught)} | **{r.event_recall:.1%}** | "
                     f"**{r.episode_precision:.1%}** | {r.median_lead_min:.1f} |")

    out = Path(FORECAST_DIR) / "forecast_report_v2.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out}")
    print(ev_df.round(3).to_string(index=False))
    if not goes_df.empty:
        print(goes_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
