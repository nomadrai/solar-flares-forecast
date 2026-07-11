"""
Phase 7 — Pipeline orchestrator.

Three public entry points, mirroring the ``*_day`` / ``*_all_days``
convention of Phase 4/5/6:

  * :func:`prepare_training_frame`   — load all Phase 5 Parquets,
    downsample, drop rows with too many NaNs, concatenate, save.
  * :func:`train_all_horizons`       — fit the 12 models and persist
    artifacts + ``metrics_summary.csv`` + ``feature_importance.csv``.
  * :func:`evaluate_all`             — reload models, evaluate on the test
    split, compute lead-time distribution, emit ``forecast_report.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tejas.config import (
    DETECTION_DIR,
    FORECAST_DIR,
    MODELS_DIR,
    FORECAST_CADENCE_S,
    FORECAST_LABEL_HORIZONS_MIN,
    FORECAST_TRAIN_DAYS,
    FORECAST_VAL_DAYS,
    FORECAST_TEST_DAYS,
    FORECAST_THRESHOLD_METRIC,
    FORECAST_CALIBRATION_METHOD,
    YEAR,
    MONTH,
)
from tejas.forecasting.evaluation import (
    compute_lead_time,
    compute_metrics,
    feature_importance,
    tune_threshold,
)
from tejas.forecasting.inference import _apply_artifact_calibrator, _predict_positive
from tejas.forecasting.models import ALGORITHMS
from tejas.forecasting.sampling import (
    downsample,
    drop_incomplete_rows,
    load_detection_day,
    select_feature_columns,
)
from tejas.forecasting.splits import (
    apply_purge,
    assign_split,
)
from tejas.forecasting.training import (
    load_feature_columns_manifest,
    train_all,
)


# ── Frame preparation ──────────────────────────────────────────────────────


def prepare_training_frame(
    cadence_s: int | None = None,
    detection_dir: Path | None = None,
    output_dir: Path | None = None,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assemble a downsampled, purged training frame across all days.

    Args:
        cadence_s: Downsample cadence in seconds.  Defaults to
            :data:`FORECAST_CADENCE_S`.
        detection_dir: Directory with Phase 5 Parquets.
        output_dir: Directory to write ``training_frame_{c}s.parquet``.
        save: If True, persist the resulting frame.
        verbose: Print per-day progress.

    Returns:
        The prepared DataFrame.
    """
    if cadence_s is None:
        cadence_s = FORECAST_CADENCE_S
    detection_dir = Path(detection_dir) if detection_dir else Path(DETECTION_DIR)
    output_dir = Path(output_dir) if output_dir else Path(FORECAST_DIR)

    all_days = list(FORECAST_TRAIN_DAYS) + list(FORECAST_VAL_DAYS) + list(FORECAST_TEST_DAYS)
    date_strs = [f"{YEAR}-{MONTH}-{d}" for d in sorted(all_days)]

    frames = []
    t0 = time.time()
    for date_str in date_strs:
        day_df = load_detection_day(date_str, detection_dir=detection_dir)
        if day_df.empty:
            if verbose:
                print(f"  [SKIP] {date_str}: no Parquet")
            continue
        ds = downsample(day_df, cadence_s)
        frames.append(ds)
        if verbose:
            print(f"  {date_str}: {len(day_df):>7,} -> {len(ds):>6,} rows")

    if not frames:
        raise RuntimeError("No detection Parquets were found.")

    df = pd.concat(frames, ignore_index=True)
    feature_cols = select_feature_columns(df)
    df = drop_incomplete_rows(df, feature_cols)

    elapsed = time.time() - t0
    if verbose:
        print(
            f"  Combined: {len(df):,} rows × {len(df.columns)} cols "
            f"(features={len(feature_cols)})  [{elapsed:.1f}s]"
        )

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"training_frame_{cadence_s}s.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        if verbose:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  Saved: {path.name} ({size_mb:.1f} MB)")

    return df


# ── Full training run ─────────────────────────────────────────────────────


def train_all_horizons(
    cadence_s: int | None = None,
    training_frame: pd.DataFrame | None = None,
    algorithms: Iterable[str] | None = None,
    output_dir: Path | None = None,
    models_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit every ``(horizon, algorithm)`` model and write summary CSVs.

    Args:
        cadence_s: Cadence used to select the training frame Parquet.
            Ignored when ``training_frame`` is provided directly.
        training_frame: Optional in-memory training frame (for tests).
        algorithms: Iterable of algorithm names.  Defaults to all three.
        output_dir: Directory for ``metrics_summary.csv`` +
            ``feature_importance.csv``.
        models_dir: Directory for the joblib artifacts + manifest.
        verbose: Progress logging.

    Returns:
        Metrics summary DataFrame.
    """
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

    if verbose:
        cnt = split_labels.value_counts()
        print(
            f"  Split sizes: train={int(cnt.get('train', 0)):,}  "
            f"val={int(cnt.get('val', 0)):,}  "
            f"test={int(cnt.get('test', 0)):,}"
        )

    metrics = train_all(
        training_frame,
        feature_cols,
        split_labels,
        algorithms=algorithms,
        models_dir=models_dir,
        save=True,
        verbose=verbose,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    if verbose:
        print(f"  Saved: {metrics_path}")

    # Feature importances (tree models only produce useful gains)
    fi_frames = []
    import joblib

    for algo in (algorithms if algorithms else ALGORITHMS):
        for h in FORECAST_LABEL_HORIZONS_MIN:
            art_path = models_dir / f"{algo}_horizon_{h}m.joblib"
            if not art_path.exists():
                continue
            art = joblib.load(art_path)
            fi = feature_importance(art["model"], art["feature_columns"])
            if fi.empty:
                continue
            fi["algorithm"] = algo
            fi["horizon_min"] = h
            fi_frames.append(fi)

    if fi_frames:
        fi_all = pd.concat(fi_frames, ignore_index=True)
        fi_path = output_dir / "feature_importance.csv"
        fi_all.to_csv(fi_path, index=False)
        if verbose:
            print(f"  Saved: {fi_path}")

    return metrics


# ── Fix 4: Significant-flare training ──────────────────────────────────────


def train_significant_flare_models(
    cadence_s: int | None = None,
    training_frame: pd.DataFrame | None = None,
    algorithms: Iterable[str] | None = None,
    output_dir: Path | None = None,
    models_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit class-specific (moderate/strong/severe) models per horizon.

    Uses ``label_sig_flare_within_{h}m`` as the target.  Artifacts are
    written as ``{algo}_sig_horizon_{h}m.joblib``.  A separate metrics
    file ``metrics_summary_sig.csv`` is written next to the standard
    ``metrics_summary.csv``.
    """
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

    # Check that sig-flare labels exist
    required = [f"label_sig_flare_within_{h}m"
                for h in FORECAST_LABEL_HORIZONS_MIN]
    missing = [c for c in required if c not in training_frame.columns]
    if missing:
        raise RuntimeError(
            "Significant-flare labels missing from training frame: "
            f"{missing}.  Run scripts/run_sig_labels.py --all first, then "
            f"re-run scripts/run_forecasting.py --prepare."
        )

    feature_cols = select_feature_columns(training_frame)
    split_labels = assign_split(training_frame)
    training_frame, split_labels = apply_purge(training_frame, split_labels)

    if verbose:
        cnt = split_labels.value_counts()
        pos = {
            h: int(training_frame[f"label_sig_flare_within_{h}m"].sum())
            for h in FORECAST_LABEL_HORIZONS_MIN
        }
        print(
            f"  Sig-flare split sizes: train={int(cnt.get('train', 0)):,}  "
            f"val={int(cnt.get('val', 0)):,}  test={int(cnt.get('test', 0)):,}"
        )
        print(f"  Sig-flare positives per horizon (total): {pos}")

    metrics = train_all(
        training_frame,
        feature_cols,
        split_labels,
        algorithms=algorithms,
        models_dir=models_dir,
        save=True,
        verbose=verbose,
        label_prefix="label_sig_flare_within",
        artifact_suffix="_sig",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_summary_sig.csv"
    metrics.to_csv(metrics_path, index=False)
    if verbose:
        print(f"  Saved: {metrics_path}")

    return metrics


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate_all(
    cadence_s: int | None = None,
    training_frame: pd.DataFrame | None = None,
    algorithm: str = "xgb",
    output_dir: Path | None = None,
    models_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Re-evaluate persisted models on the test split and emit a report.

    Also computes lead-time distributions against the Phase 5 master
    catalogue where available.

    Returns:
        Dict with keys ``metrics``, ``lead_times``.
    """
    if cadence_s is None:
        cadence_s = FORECAST_CADENCE_S
    output_dir = Path(output_dir) if output_dir else Path(FORECAST_DIR)
    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)

    if training_frame is None:
        path = output_dir / f"training_frame_{cadence_s}s.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Training frame not found: {path}. Run prepare_training_frame first."
            )
        training_frame = pd.read_parquet(path)

    feature_cols = load_feature_columns_manifest(models_dir=models_dir)
    split_labels = assign_split(training_frame)
    training_frame, split_labels = apply_purge(training_frame, split_labels)

    test_mask = (split_labels == "test").to_numpy()
    test_df = training_frame.loc[test_mask].reset_index(drop=True)
    if verbose:
        print(f"  Test rows: {len(test_df):,}")

    if len(test_df) == 0:
        return {"metrics": pd.DataFrame(), "lead_times": pd.DataFrame()}

    X_test = test_df[feature_cols].to_numpy(dtype=float)

    # Load flare start times from the Phase 5 master catalogue.  Fall back
    # to inferring them from the per-timestep `flare_id` transitions if the
    # master catalogue is missing (e.g., in the test harness).
    flare_starts = _load_test_flare_starts(test_df)

    import joblib

    metrics_records = []
    lead_records = []
    for h in FORECAST_LABEL_HORIZONS_MIN:
        art_path = models_dir / f"{algorithm}_horizon_{h}m.joblib"
        if not art_path.exists():
            if verbose:
                print(f"  [SKIP] Missing artifact: {art_path.name}")
            continue
        art = joblib.load(art_path)
        model = art["model"]
        threshold = float(art["decision_threshold"])

        raw_prob = _predict_positive(model, X_test)
        prob = _apply_artifact_calibrator(art, raw_prob)
        y_true = test_df[f"label_flare_within_{h}m"].fillna(0).astype(np.int8).to_numpy()
        m = compute_metrics(y_true, prob, threshold=threshold)
        m["algorithm"] = algorithm
        m["horizon_min"] = h
        metrics_records.append(m)

        if len(flare_starts):
            lt = compute_lead_time(
                test_df["datetime"], prob, flare_starts,
                threshold=threshold, horizon_min=h,
            )
            lt["horizon_min"] = h
            lt["algorithm"] = algorithm
            lead_records.append(lt)

    metrics_df = pd.DataFrame(metrics_records)
    lead_df = pd.concat(lead_records, ignore_index=True) if lead_records else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    if not lead_df.empty:
        lead_df.to_csv(output_dir / "lead_time_distribution.csv", index=False)

    _write_report(metrics_df, lead_df, output_dir, algorithm)

    return {"metrics": metrics_df, "lead_times": lead_df}


def _load_test_flare_starts(test_df: pd.DataFrame) -> pd.Series:
    """Return the set of ground-truth flare start times inside the test df.

    Prefers the Phase 5 master catalogue when available; otherwise infers
    from transitions in ``flare_id`` (from ``0`` → non-zero).
    """
    master_path = Path(DETECTION_DIR) / "master_catalogue.csv"
    if master_path.exists():
        cat = pd.read_csv(master_path)
        starts = pd.to_datetime(cat["start_time"]).dt.tz_localize(None)
        # Constrain to the test window
        t0, t1 = test_df["datetime"].min(), test_df["datetime"].max()
        starts = starts[(starts >= t0) & (starts <= t1)]
        return starts.reset_index(drop=True)

    if "flare_id" in test_df.columns:
        ids = test_df["flare_id"].to_numpy()
        transitions = (ids[1:] > 0) & (ids[:-1] == 0)
        starts = test_df["datetime"].to_numpy()[1:][transitions]
        return pd.Series(pd.DatetimeIndex(starts))

    return pd.Series(dtype="datetime64[ns]")


def _write_report(metrics_df: pd.DataFrame, lead_df: pd.DataFrame, output_dir: Path, algorithm: str) -> Path:
    """Emit ``forecast_report.md`` summarising test-set performance."""
    lines = [
        "# TEJAS Phase 7 — Forecast Evaluation Report",
        "",
        f"- **Algorithm evaluated**: `{algorithm}`",
        f"- **Threshold metric**: `{FORECAST_THRESHOLD_METRIC}`  "
        f"(tuned on the val split; higher = fewer positives)",
        f"- **Probability calibration**: `{FORECAST_CALIBRATION_METHOD}`  "
        f"(fit on val, applied to test)",
        f"- **Test window**: computed from the training frame test split.",
        "",
        "## Test-set metrics",
        "",
        "| Horizon | PR-AUC | ROC-AUC | Precision | Recall | F1 | FAR | Threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {int(r['horizon_min'])} min | {r['pr_auc']:.3f} | {r['roc_auc']:.3f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['false_alarm_rate']:.4f} | {r['threshold']:.2f} |"
        )
    lines.append("")

    if not lead_df.empty:
        lines.append("## Lead-time distribution (seconds)")
        lines.append("")
        lines.append("| Horizon | N events | Median | P25 | P75 | % detected |")
        lines.append("|---|---|---|---|---|---|")
        for h, group in lead_df.groupby("horizon_min"):
            n = len(group)
            leads = group["lead_time_s"].dropna()
            det = len(leads) / n if n else 0.0
            median = leads.median() if len(leads) else float("nan")
            q25 = leads.quantile(0.25) if len(leads) else float("nan")
            q75 = leads.quantile(0.75) if len(leads) else float("nan")
            lines.append(
                f"| {int(h)} min | {n} | {median:.0f} | {q25:.0f} | {q75:.0f} | {det:.1%} |"
            )
        lines.append("")

    lines.append(
        f"_Metrics use the val-tuned decision threshold (metric = "
        f"`{FORECAST_THRESHOLD_METRIC}`) applied to calibrated probabilities._"
    )

    path = output_dir / "forecast_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
