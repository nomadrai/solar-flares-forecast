"""
TEJAS Unified Dataset Builder
==============================
Constructs a dual-cadence unified daily dataset by combining SoLEXS
(1-second native) and HEL1OS (10-second resampled) data onto a common
1-second UTC grid.

Architecture:
  - SoLEXS maps 1:1 onto the grid (native 1s cadence).
  - HEL1OS (resampled to 10s) is forward-filled between successive bins.
  - Forward-fill respects segment boundaries (does not fill across gaps).
  - Metadata columns track data validity and provenance.

Output: one Parquet file per day in data/processed/unified/.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from tejas.config import (
    UNIFIED_DIR,
    UNIFIED_GRID_CADENCE_S,
    HEL1OS_RESAMPLE_CADENCE_S,
)
from tejas.loaders import solexs, hel1os
from tejas.validation.checks import validate_day


# ── Unified Grid Construction ────────────────────────────────────────────────


def build_unified_day(date_str, hel1os_cadence_s=None):
    """Build a dual-cadence unified dataset for a single day.

    Pipeline:
      1. Load SoLEXS data (1s cadence, GTI-filtered)
      2. Load HEL1OS data (resampled to 10s, GTI-filtered, gap-handled)
      3. Create a 1-second UTC grid spanning the day
      4. Merge SoLEXS onto the grid (1:1)
      5. Merge HEL1OS onto the grid with forward-fill (respects segments)
      6. Add metadata columns (validity flags, segment ID, is_filled)
      7. Run validation checks

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        hel1os_cadence_s: HEL1OS resampling cadence. Defaults to config (10s).

    Returns:
        dict with keys:
          'unified': DataFrame — the unified dataset
          'solexs_data': dict — raw SoLEXS loader output
          'hel1os_data': dict — raw HEL1OS loader output
          'validation': ValidationReport — validation results
    """
    if hel1os_cadence_s is None:
        hel1os_cadence_s = HEL1OS_RESAMPLE_CADENCE_S

    print(f"\n{'='*60}")
    print(f"Building unified dataset: {date_str}")
    print(f"{'='*60}")

    # ── Step 1: Load SoLEXS ────────────────────────────────────────────
    print(f"  Loading SoLEXS data...")
    solexs_data = solexs.load_day(date_str, validate=False)
    lc_df = solexs_data["lc"]
    pi_df = solexs_data["pi"]
    gti_df = solexs_data["gti"]

    if lc_df is not None:
        print(f"    LC: {len(lc_df)} rows")
    else:
        print(f"    LC: not found")

    if pi_df is not None:
        print(f"    PI: {len(pi_df)} rows")
    else:
        print(f"    PI: not found")

    # ── Step 2: Load HEL1OS ────────────────────────────────────────────
    print(f"  Loading HEL1OS data (resampling to {hel1os_cadence_s}s)...")
    hel1os_data = hel1os.load_day(date_str, resample_cadence_s=hel1os_cadence_s)
    cdte_df = hel1os_data["cdte"]
    czt_df = hel1os_data["czt"]

    if cdte_df is not None:
        n_real = len(cdte_df[~cdte_df.get("gap_mask", False)])
        print(f"    CdTe: {len(cdte_df)} rows ({n_real} data + "
              f"{len(cdte_df) - n_real} gap markers)")
    else:
        print(f"    CdTe: not found")

    if czt_df is not None:
        n_real = len(czt_df[~czt_df.get("gap_mask", False)])
        print(f"    CZT: {len(czt_df)} rows ({n_real} data + "
              f"{len(czt_df) - n_real} gap markers)")
    else:
        print(f"    CZT: not found")

    # ── Step 3: Create the 1-second UTC grid ───────────────────────────
    print(f"  Building 1-second grid...")
    grid = _create_day_grid(date_str)
    print(f"    Grid: {len(grid)} rows")

    # ── Step 4: Merge SoLEXS onto grid ─────────────────────────────────
    print(f"  Merging SoLEXS...")
    grid = _merge_solexs(grid, lc_df, pi_df, gti_df)

    # ── Step 5: Merge HEL1OS onto grid ─────────────────────────────────
    print(f"  Merging HEL1OS (forward-fill within segments)...")
    grid = _merge_hel1os(grid, cdte_df, czt_df, hel1os_data)

    # ── Step 6: Run validation ─────────────────────────────────────────
    print(f"  Running validation checks...")
    validation = validate_day(date_str, solexs_data, hel1os_data)
    validation.print_summary()

    # Summary stats
    n_total = len(grid)
    n_solexs_valid = int(grid["solexs_valid"].sum()) if "solexs_valid" in grid.columns else 0
    n_hel1os_valid = int(grid["hel1os_valid"].sum()) if "hel1os_valid" in grid.columns else 0
    print(f"  Result: {n_total} rows, "
          f"SoLEXS valid={n_solexs_valid} ({n_solexs_valid/n_total*100:.1f}%), "
          f"HEL1OS valid={n_hel1os_valid} ({n_hel1os_valid/n_total*100:.1f}%)")

    return {
        "unified": grid,
        "solexs_data": solexs_data,
        "hel1os_data": hel1os_data,
        "validation": validation,
    }


def _create_day_grid(date_str):
    """Create a 1-second UTC grid spanning an entire day.

    Generates 86400 rows (00:00:00 to 23:59:59 UTC) with time_unix
    and datetime columns.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with time_unix and datetime columns.
    """
    day_start = pd.Timestamp(date_str, tz="UTC")
    day_start_unix = day_start.timestamp()

    # Build the grid using arithmetic on Unix timestamps (avoids
    # pandas DatetimeIndex.astype(int64) portability issues)
    time_unix = day_start_unix + np.arange(86400, dtype=np.float64)
    datetimes = pd.to_datetime(time_unix, unit="s", utc=True)

    return pd.DataFrame({
        "time_unix": time_unix,
        "datetime": datetimes,
    })


def _merge_solexs(grid, lc_df, pi_df, gti_df):
    """Merge SoLEXS data onto the 1-second grid.

    SoLEXS is at native 1s cadence, so this is a left join on rounded
    time_unix. Also adds the solexs_valid flag based on GTI.

    Args:
        grid: The 1-second grid DataFrame.
        lc_df: SoLEXS LC DataFrame (GTI-filtered, NaN outside GTI).
        pi_df: SoLEXS PI DataFrame (GTI-filtered).
        gti_df: SoLEXS GTI DataFrame.

    Returns:
        Grid with SoLEXS columns added.
    """
    # Merge LC data
    if lc_df is not None and len(lc_df) > 0:
        lc_merge = lc_df[["time_unix", "counts"]].copy()
        lc_merge["time_unix"] = np.round(lc_merge["time_unix"])
        lc_merge = lc_merge.rename(columns={"counts": "solexs_counts"})
        # Drop duplicates in case of rounding collisions (keep first)
        lc_merge = lc_merge.drop_duplicates(subset=["time_unix"], keep="first")
        grid = grid.merge(lc_merge, on="time_unix", how="left")
    else:
        grid["solexs_counts"] = np.nan

    # Merge PI data
    if pi_df is not None and len(pi_df) > 0:
        pi_cols = ["time_unix", "counts_low", "counts_mid", "counts_high",
                    "hardness_ratio"]
        pi_merge = pi_df[pi_cols].copy()
        pi_merge["time_unix"] = np.round(pi_merge["time_unix"])
        pi_merge = pi_merge.rename(columns={
            "counts_low": "solexs_counts_low",
            "counts_mid": "solexs_counts_mid",
            "counts_high": "solexs_counts_high",
            "hardness_ratio": "solexs_hardness_ratio",
        })
        pi_merge = pi_merge.drop_duplicates(subset=["time_unix"], keep="first")
        grid = grid.merge(pi_merge, on="time_unix", how="left")
    else:
        for col in ["solexs_counts_low", "solexs_counts_mid",
                     "solexs_counts_high", "solexs_hardness_ratio"]:
            grid[col] = np.nan

    # Add SoLEXS validity flag based on GTI
    grid["solexs_valid"] = False
    if gti_df is not None and len(gti_df) > 0:
        times = grid["time_unix"].values
        in_gti = np.zeros(len(grid), dtype=bool)
        for _, row in gti_df.iterrows():
            in_gti |= (times >= row["start"]) & (times <= row["stop"])
        grid["solexs_valid"] = in_gti

    return grid


def _merge_hel1os(grid, cdte_df, czt_df, hel1os_data):
    """Merge HEL1OS data onto the 1-second grid with forward-fill.

    HEL1OS data is at 10s cadence (after resampling). Each 10s bin value
    is forward-filled into the next 10 1s-slots on the grid. Forward-fill
    does NOT cross segment boundaries.

    Args:
        grid: The grid DataFrame (already has SoLEXS columns).
        cdte_df: HEL1OS CdTe DataFrame (resampled, with segment_id/gap_mask).
        czt_df: HEL1OS CZT DataFrame (resampled, with segment_id/gap_mask).
        hel1os_data: Full HEL1OS loader output (for GTI info).

    Returns:
        Grid with HEL1OS columns added.
    """
    # ── Merge CdTe ─────────────────────────────────────────────────────
    grid = _merge_single_hel1os(
        grid, cdte_df, prefix="cdte",
        value_cols=["count_rate", "stat_err"],
    )

    # ── Merge CZT ──────────────────────────────────────────────────────
    grid = _merge_single_hel1os(
        grid, czt_df, prefix="czt",
        value_cols=["count_rate", "stat_err"],
    )

    # ── HEL1OS validity flag ───────────────────────────────────────────
    # A grid row is HEL1OS-valid if it has non-NaN CdTe data
    # (CdTe is the primary hard X-ray channel)
    grid["hel1os_valid"] = grid["cdte_count_rate"].notna()

    return grid


def _merge_single_hel1os(grid, hel1os_df, prefix, value_cols):
    """Merge a single HEL1OS detector's data onto the grid.

    Performs a left merge, then forward-fills within segment boundaries.

    Args:
        grid: Grid DataFrame.
        hel1os_df: HEL1OS detector DataFrame (resampled).
        prefix: Column name prefix ('cdte' or 'czt').
        value_cols: List of value columns to merge (e.g., ['count_rate', 'stat_err']).

    Returns:
        Grid with prefixed HEL1OS columns added.
    """
    renamed_cols = {col: f"{prefix}_{col}" for col in value_cols}
    segment_col = f"{prefix}_segment_id"
    filled_col = f"{prefix}_is_filled"

    if hel1os_df is None or len(hel1os_df) == 0:
        for new_col in renamed_cols.values():
            grid[new_col] = np.nan
        grid[segment_col] = np.nan
        grid[filled_col] = False
        return grid

    # Filter out gap marker rows for the merge
    real_data = hel1os_df[
        ~hel1os_df["gap_mask"]
    ].copy() if "gap_mask" in hel1os_df.columns else hel1os_df.copy()

    # Prepare merge columns
    merge_cols = ["time_unix"] + value_cols
    if "segment_id" in real_data.columns:
        merge_cols.append("segment_id")

    merge_df = real_data[merge_cols].copy()
    merge_df["time_unix"] = np.round(merge_df["time_unix"])

    # Rename value columns with prefix
    merge_df = merge_df.rename(columns=renamed_cols)
    if "segment_id" in merge_df.columns:
        merge_df = merge_df.rename(columns={"segment_id": segment_col})

    # Drop duplicates (keep first in case of rounding collisions)
    merge_df = merge_df.drop_duplicates(subset=["time_unix"], keep="first")

    # Left merge onto grid
    grid = grid.merge(merge_df, on="time_unix", how="left")

    # Track which rows are native HEL1OS bins vs forward-filled
    grid[filled_col] = False
    native_mask = grid[list(renamed_cols.values())[0]].notna()

    # Forward-fill within segments
    if segment_col in grid.columns:
        # First, propagate segment_id forward so we know segment boundaries
        grid[segment_col] = grid[segment_col].ffill()

        # Forward-fill value columns within each segment
        for col in renamed_cols.values():
            grid[col] = grid.groupby(segment_col)[col].ffill()

        # Mark filled rows (non-native, non-NaN after fill)
        filled_mask = (~native_mask) & grid[list(renamed_cols.values())[0]].notna()
        grid.loc[filled_mask, filled_col] = True
    else:
        # No segment info — simple forward-fill (no boundary awareness)
        for col in renamed_cols.values():
            grid[col] = grid[col].ffill()

        filled_mask = (~native_mask) & grid[list(renamed_cols.values())[0]].notna()
        grid.loc[filled_mask, filled_col] = True

    return grid


# ── Save to Parquet ──────────────────────────────────────────────────────────


def save_unified_day(date_str, output_dir=None, hel1os_cadence_s=None):
    """Build and save the unified dataset for a single day.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        output_dir: Output directory. Defaults to config.UNIFIED_DIR.
        hel1os_cadence_s: HEL1OS resampling cadence. Defaults to config (10s).

    Returns:
        dict from build_unified_day() (includes unified DataFrame,
        raw data, and validation report).
    """
    if output_dir is None:
        output_dir = UNIFIED_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = build_unified_day(date_str, hel1os_cadence_s=hel1os_cadence_s)
    unified_df = result["unified"]

    # Save to Parquet
    output_path = output_dir / f"{date_str}.parquet"
    unified_df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")

    return result
