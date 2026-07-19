"""
TEJAS Feature Engineering Pipeline
=====================================
Orchestrates all Phase 4 feature groups for a single day or all days.

Pipeline order:
  1. Load clean Parquet (Phase 3 output)
  2. Compute temporal features (rolling stats, derivatives, gradients)
  3. Compute spectral features (hardness, band ratios, spectral slope)
  4. Compute cross-instrument features (ratios, lag, cross-correlation)
  5. Compute statistical features (z-score, percentile, CUSUM)
  6. Save to data/features/{date}.parquet
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from tejas.config import CLEAN_DIR, FEATURES_DIR, YEAR, MONTH, MISSING_DAYS
from tejas.features.temporal import compute_temporal_features
from tejas.features.spectral import compute_spectral_features
from tejas.features.cross_instrument import compute_cross_instrument_features
from tejas.features.statistical import compute_statistical_features
from tejas.features.particle_discriminator import (
    compute_particle_discriminator_features,   # Fix 3
)


# ── Single Day Pipeline ────────────────────────────────────────────────────


def engineer_day(date_str, input_dir=None, output_dir=None, save=True):
    """Run the full Phase 4 feature engineering pipeline for a single day.

    Pipeline:
      Load → Temporal → Spectral → Cross-instrument → Statistical → Save

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        input_dir: Directory containing clean Parquet files (Phase 3 output).
            Defaults to config.CLEAN_DIR.
        output_dir: Directory for feature output Parquet files.
            Defaults to config.FEATURES_DIR.
        save: If True, save the feature DataFrame to Parquet.

    Returns:
        dict with keys:
          'df': DataFrame — fully featured dataset.
          'report': dict — feature engineering statistics for this day.
    """
    if input_dir is None:
        input_dir = CLEAN_DIR
    if output_dir is None:
        output_dir = FEATURES_DIR

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"Feature Engineering: {date_str}")
    print(f"{'='*60}")

    # ── Step 0: Load clean Parquet ─────────────────────────────────────
    input_path = input_dir / f"{date_str}.parquet"
    if not input_path.exists():
        print(f"  [SKIP] Input file not found: {input_path}")
        return None

    df = pd.read_parquet(input_path)
    n_rows = len(df)
    n_cols_initial = len(df.columns)
    print(f"  Loaded: {n_rows} rows, {n_cols_initial} columns")

    # ── Step 1: Temporal features ──────────────────────────────────────
    print(f"  [1/4] Temporal features...")
    temporal_result = compute_temporal_features(df)
    df = temporal_result["df"]
    n_temporal = len(temporal_result["columns_added"])
    print(f"    Added {n_temporal} temporal features")
    for sig, info in temporal_result["report"].items():
        src = info.get("source", "N/A")
        n = info.get("n_features", 0)
        if n > 0:
            print(f"      {sig}: {n} features (source: {src})")

    # ── Step 2: Spectral features ──────────────────────────────────────
    print(f"  [2/4] Spectral features...")
    spectral_result = compute_spectral_features(df)
    df = spectral_result["df"]
    n_spectral = len(spectral_result["columns_added"])
    print(f"    Added {n_spectral} spectral features")

    # ── Step 3: Cross-instrument features ──────────────────────────────
    print(f"  [3/4] Cross-instrument features...")
    cross_result = compute_cross_instrument_features(df)
    df = cross_result["df"]
    n_cross = len(cross_result["columns_added"])
    print(f"    Added {n_cross} cross-instrument features")

    # ── Step 4: Statistical features ───────────────────────────────────
    print(f"  [4/5] Statistical features...")
    stat_result = compute_statistical_features(df)
    df = stat_result["df"]
    n_stat = len(stat_result["columns_added"])
    print(f"    Added {n_stat} statistical features")

    # ── Step 5 (Fix 3): Particle / SEP discriminator features ──────────
    print(f"  [5/6] Particle discriminator features...")
    particle_result = compute_particle_discriminator_features(df)
    df = particle_result["df"]
    n_particle = len(particle_result["columns_added"])
    print(f"    Added {n_particle} particle discriminator features")

    # ── Step 6 (v2): Long-window context features ──────────────────────
    print(f"  [6/6] Long-window context features...")
    from tejas.features.long_context import compute_long_context_features
    long_result = compute_long_context_features(df)
    df = long_result["df"]
    n_long = len(long_result["columns_added"])
    print(f"    Added {n_long} long-context features")

    # ── Summary ────────────────────────────────────────────────────────
    n_cols_final = len(df.columns)
    n_new = n_cols_final - n_cols_initial
    elapsed = time.time() - t_start

    print(f"  Result: {n_rows} rows, {n_cols_initial} -> {n_cols_final} "
          f"columns (+{n_new}), {elapsed:.1f}s")

    # ── Save ───────────────────────────────────────────────────────────
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{date_str}.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # ── Build report ───────────────────────────────────────────────────
    report = {
        "date": date_str,
        "n_rows": n_rows,
        "n_cols_input": n_cols_initial,
        "n_cols_output": n_cols_final,
        "n_features_added": n_new,
        "n_temporal": n_temporal,
        "n_spectral": n_spectral,
        "n_cross_instrument": n_cross,
        "n_statistical": n_stat,
        "n_particle": n_particle,
        "n_long_context": n_long,
        "elapsed_s": round(elapsed, 1),
    }

    return {"df": df, "report": report}


# ── Batch Processing ───────────────────────────────────────────────────────


def engineer_all_days(input_dir=None, output_dir=None):
    """Engineer features for all available days.

    Discovers daily clean Parquet files in the input directory and
    processes each one.  Generates a feature engineering summary CSV.

    Args:
        input_dir: Directory containing clean Parquet files.
        output_dir: Directory for feature output Parquet files.

    Returns:
        pd.DataFrame — feature engineering summary with one row per day.
    """
    if input_dir is None:
        input_dir = CLEAN_DIR
    if output_dir is None:
        output_dir = FEATURES_DIR

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
        result = engineer_day(date_str, input_dir=input_dir,
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
    summary_path = output_dir / "feature_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Feature engineering complete: {len(reports)}/{len(dates)} days")
    print(f"Summary saved: {summary_path}")
    print(f"{'='*60}")

    return summary
