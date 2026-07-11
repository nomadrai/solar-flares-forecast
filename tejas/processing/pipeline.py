"""
TEJAS Processing Pipeline
==========================
Orchestrates the full Phase 3 pipeline for a single day or all days.

Pipeline order:
  1. Load unified Parquet (Phase 2 output)
  2. Spike removal → *_clean columns
  3. Signal smoothing → *_smooth columns
  4. CZT background subtraction → czt_background, czt_net_rate, czt_above_bg
  5. Normalization → *_log, *_pctrank columns
  6. Save to data/processed/clean/{date}.parquet
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from tejas.config import UNIFIED_DIR, CLEAN_DIR, YEAR, MONTH, MISSING_DAYS
from tejas.processing.spike_removal import remove_spikes_day
from tejas.processing.smoothing import smooth_day
from tejas.processing.czt_background import subtract_czt_background
from tejas.processing.normalization import normalize_day
from tejas.processing.kalman import smooth_day_kalman   # Fix 6


# ── Single Day Pipeline ────────────────────────────────────────────────────


def process_day(date_str, input_dir=None, output_dir=None, save=True):
    """Run the full Phase 3 processing pipeline for a single day.

    Pipeline:
      Load → Spike removal → Smoothing → CZT background → Normalization → Save

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        input_dir: Directory containing unified Parquet files.
            Defaults to config.UNIFIED_DIR.
        output_dir: Directory for clean output Parquet files.
            Defaults to config.CLEAN_DIR.
        save: If True, save the processed DataFrame to Parquet.

    Returns:
        dict with keys:
          'df': DataFrame — fully processed clean dataset.
          'report': dict — processing statistics for this day.
    """
    if input_dir is None:
        input_dir = UNIFIED_DIR
    if output_dir is None:
        output_dir = CLEAN_DIR

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"Processing: {date_str}")
    print(f"{'='*60}")

    # ── Step 0: Load unified Parquet ───────────────────────────────────
    input_path = input_dir / f"{date_str}.parquet"
    if not input_path.exists():
        print(f"  [SKIP] Input file not found: {input_path}")
        return None

    df = pd.read_parquet(input_path)
    n_rows = len(df)
    n_cols_initial = len(df.columns)
    print(f"  Loaded: {n_rows} rows, {n_cols_initial} columns")

    # ── Step 1: Spike removal ─────────────────────────────────────────
    print(f"  [1/4] Spike removal...")
    spike_result = remove_spikes_day(df)
    df = spike_result["df"]
    spike_report = spike_result["report"]

    total_spikes = sum(r["n_spikes"] for r in spike_report.values())
    print(f"    Removed {total_spikes} spikes across {len(spike_report)} channels")
    for col, info in spike_report.items():
        if info["n_spikes"] > 0:
            print(f"      {col}: {info['n_spikes']} ({info['pct_spikes']:.4f}%)")

    # ── Step 2: Smoothing ─────────────────────────────────────────────
    print(f"  [2/4] Signal smoothing...")
    smooth_result = smooth_day(df)
    df = smooth_result["df"]
    smooth_report = smooth_result["report"]
    print(f"    Smoothed {len(smooth_report)} channels")

    # ── Step 3: CZT background subtraction ────────────────────────────
    print(f"  [3/4] CZT background estimation & subtraction...")
    czt_result = subtract_czt_background(df)
    df = czt_result["df"]
    czt_report = czt_result["report"]
    print(f"    Background median: {czt_report['bg_median']:.1f} c/s, "
          f"std: {czt_report['bg_std']:.1f}")
    print(f"    Quiet fraction: {czt_report['quiet_fraction']*100:.1f}%, "
          f"above-bg fraction: {czt_report['above_bg_fraction']*100:.1f}%")

    # ── Step 4: Normalization ─────────────────────────────────────────
    print(f"  [4/4] Normalization (log1p + percentile rank)...")
    norm_result = normalize_day(df)
    df = norm_result["df"]
    norm_report = norm_result["report"]
    print(f"    Added {norm_report['n_columns']} normalization columns")

    # ── Step 5 (Fix 6): Kalman smoothing (adds *_kalman alongside *_smooth) ─
    print(f"  [5/6] Kalman filtering (CdTe & CZT)...")
    kalman_result = smooth_day_kalman(df)
    df = kalman_result["df"]
    print(f"    Added {len(kalman_result['columns_added'])} Kalman columns: "
          f"{kalman_result['columns_added']}")

    # ── Step 6 (Fix 3): tag GTI boundaries for downstream particle disc. ────
    # We use segment_id transitions as proxies for GTI boundaries — the
    # ingestion module segments the data at every non-trivial data gap
    # (≥ GAP_MEDIUM_S), which matches the physical meaning of a new GTI
    # interval starting.
    print(f"  [6/6] Tagging GTI boundaries (gti_start)...")
    df["gti_start"] = False
    # Union of both HEL1OS segment transitions and any long time-jump.
    if "cdte_segment_id" in df.columns:
        seg = df["cdte_segment_id"].to_numpy()
        # First row + every row where segment_id changes
        transitions = np.concatenate(([True], seg[1:] != seg[:-1]))
        df.loc[transitions, "gti_start"] = True
    if "czt_segment_id" in df.columns:
        seg = df["czt_segment_id"].to_numpy()
        transitions = np.concatenate(([True], seg[1:] != seg[:-1]))
        df.loc[transitions, "gti_start"] = True
    n_gti_starts = int(df["gti_start"].sum())
    print(f"    Marked {n_gti_starts} GTI-start rows")

    # ── Summary ───────────────────────────────────────────────────────
    n_cols_final = len(df.columns)
    elapsed = time.time() - t_start

    print(f"  Result: {n_rows} rows, {n_cols_initial} -> {n_cols_final} columns "
          f"(+{n_cols_final - n_cols_initial}), {elapsed:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{date_str}.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        size_kb = output_path.stat().st_size / 1024
        print(f"  Saved: {output_path} ({size_kb:.0f} KB)")

    # ── Build report ──────────────────────────────────────────────────
    report = {
        "date": date_str,
        "n_rows": n_rows,
        "n_cols_input": n_cols_initial,
        "n_cols_output": n_cols_final,
        "elapsed_s": round(elapsed, 1),
        "total_spikes_removed": total_spikes,
        "czt_bg_median": czt_report["bg_median"],
        "czt_bg_std": czt_report["bg_std"],
        "czt_quiet_fraction": czt_report["quiet_fraction"],
        "czt_above_bg_fraction": czt_report["above_bg_fraction"],
    }

    # Add per-channel spike counts
    for col, info in spike_report.items():
        report[f"spikes_{col}"] = info["n_spikes"]

    return {"df": df, "report": report}


# ── Batch Processing ───────────────────────────────────────────────────────


def process_all_days(input_dir=None, output_dir=None):
    """Process all available days through the Phase 3 pipeline.

    Discovers daily Parquet files in the input directory and processes
    each one.  Generates a processing summary CSV.

    Args:
        input_dir: Directory containing unified Parquet files.
        output_dir: Directory for clean output Parquet files.

    Returns:
        pd.DataFrame — processing summary with one row per day.
    """
    if input_dir is None:
        input_dir = UNIFIED_DIR
    if output_dir is None:
        output_dir = CLEAN_DIR

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Discover available days
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {input_dir}")
        return pd.DataFrame()

    dates = [f.stem for f in parquet_files]
    print(f"\nFound {len(dates)} days to process")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Process each day
    reports = []
    for date_str in dates:
        result = process_day(date_str, input_dir=input_dir,
                             output_dir=output_dir)
        if result is not None:
            reports.append(result["report"])

    # Build summary DataFrame
    if not reports:
        print("\nNo days processed successfully.")
        return pd.DataFrame()

    summary = pd.DataFrame(reports)

    # Save summary CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "processing_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Processing complete: {len(reports)}/{len(dates)} days")
    print(f"Summary saved: {summary_path}")
    print(f"{'='*60}")

    return summary
