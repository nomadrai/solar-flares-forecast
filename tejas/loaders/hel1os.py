"""
TEJAS HEL1OS Loader
====================
Production-grade loaders for HEL1OS data (CdTe and CZT detectors):
  - Light curves: full-range and per-band energy channels
  - GTI files: per-detector good time intervals
  - Multi-dump merging with overlap resolution and tiered gap handling
  - Resampling to configurable cadence (default 10s)

HEL1OS data characteristics (from EDA):
  - Time format: MJD (Modified Julian Date)
  - Cadence: 1 second
  - ~87% zero-count bins (CdTe), ~84% (CZT) at 1s
  - Data arrives in ~12h dumps with 22-650s inter-dump gaps
  - Multiple extensions per file (one per energy band) with different row counts
"""

import re
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

from tejas.config import (
    HEL1OS_DIR,
    HEL1OS_RESAMPLE_CADENCE_S,
)
from tejas.loaders.utils import (
    unix_to_datetime,
    mjd_to_unix,
    apply_gti_mask,
    detect_gaps,
    insert_gap_nans,
    add_segment_ids,
)


# ── Path Helpers ─────────────────────────────────────────────────────────────


def _day_dir(date_str):
    """Get the HEL1OS day directory path."""
    year, month, day = date_str.split("-")
    return HEL1OS_DIR / year / month / day


def _dump_dirs(date_str):
    """Get sorted list of dump directories for a given date."""
    day_dir = _day_dir(date_str)
    if not day_dir.exists():
        return []
    return sorted([d for d in day_dir.iterdir() if d.is_dir()])


def _band_to_colname(band_name):
    """Convert FITS extension band name to a clean column name.

    Handles both short-form and actual FITS extension names:
        'CDTE1_LC_BAND_5.00KEV_TO_20.00KEV'  → 'ctr_5_20'
        'CZT1_LC_BAND_20.00KEV_TO_40.00KEV'  → 'ctr_20_40'
        '5-20 keV'                             → 'ctr_5_20'
    """
    name = band_name.strip().upper()

    # Try to parse actual FITS extension name: *_BAND_X.XXkEV_TO_Y.YYkEV
    m = re.search(r'(\d+(?:\.\d+)?)KEV_TO_(\d+(?:\.\d+)?)KEV', name)
    if m:
        lo = m.group(1).rstrip("0").rstrip(".")
        hi = m.group(2).rstrip("0").rstrip(".")
        lo = lo.replace(".", "p")
        hi = hi.replace(".", "p")
        return f"ctr_{lo}_{hi}"

    # Fallback: simple format like "5-20 keV"
    clean = (name
             .replace(" KEV", "").replace("KEV", "")
             .replace(".", "p")
             .replace("-", "_"))
    return f"ctr_{clean}"


# ── GTI Loader ───────────────────────────────────────────────────────────────


def load_gti(date_str, detector_type):
    """Load HEL1OS GTI files for a given date and detector type.

    Reads gti{cdte|czt}{1|2}.fits from all dump directories. HEL1OS GTI
    values may be in MJD format (values < 100000) and are automatically
    converted to Unix timestamps.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        detector_type: 'cdte' or 'czt'.

    Returns:
        DataFrame with columns: start, stop, start_dt, stop_dt, duration_s,
        detector, dump. Returns None if no GTI data found.
    """
    dump_dirs = _dump_dirs(date_str)
    if not dump_dirs:
        return None

    all_gtis = []

    for dump_dir in dump_dirs:
        for det_num in [1, 2]:
            gti_file = dump_dir / f"gti{detector_type}{det_num}.fits"
            if not gti_file.exists():
                continue

            try:
                with fits.open(gti_file) as hdul:
                    # Find the GTI extension by looking for START/STOP columns
                    gti_ext = None
                    for ext in hdul:
                        if hasattr(ext, "columns") and ext.columns is not None:
                            col_names = [c.name.upper() for c in ext.columns]
                            if "START" in col_names and "STOP" in col_names:
                                gti_ext = ext
                                break

                    if gti_ext is None:
                        continue

                    data = gti_ext.data
                    starts = data["START"].astype(np.float64)
                    stops = data["STOP"].astype(np.float64)

                    # HEL1OS GTI may be in MJD (values < 100000)
                    if len(starts) > 0 and np.median(starts) < 100000:
                        starts = mjd_to_unix(starts)
                        stops = mjd_to_unix(stops)

                    for s, e in zip(starts, stops):
                        all_gtis.append({
                            "start": s,
                            "stop": e,
                            "detector": f"{detector_type}{det_num}",
                            "dump": dump_dir.name,
                        })

            except Exception as e:
                print(f"  Warning: Failed to read {gti_file.name}: {e}")

    if not all_gtis:
        return None

    df = pd.DataFrame(all_gtis)
    df["start_dt"] = unix_to_datetime(df["start"])
    df["stop_dt"] = unix_to_datetime(df["stop"])
    df["duration_s"] = df["stop"] - df["start"]
    return df


# ── Full-Range Light Curve Loader ────────────────────────────────────────────


def load_detector_lightcurves(date_str, detector_type, apply_gti=True,
                               combine=True, resample_cadence_s=None):
    """Load HEL1OS full-range light curves for a given date.

    Reads the last extension (full energy range) from each detector FITS
    file across all dumps. Optionally combines detector pairs (e.g.,
    CdTe1 + CdTe2) by averaging, handles inter-dump gaps with the tiered
    strategy, and resamples to a coarser cadence.

    Pipeline: read → combine detectors → merge dumps → gap handling → resample

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        detector_type: 'cdte' or 'czt'.
        apply_gti: If True, mask values outside GTI with NaN.
        combine: If True, average detector 1 + 2 at each timestamp.
        resample_cadence_s: Target cadence for resampling. If None,
            uses config.HEL1OS_RESAMPLE_CADENCE_S. Set to 1 to skip.

    Returns:
        DataFrame with columns: time_unix, count_rate, stat_err,
        segment_id, gap_mask, datetime.
        Returns None if no data found.
    """
    if resample_cadence_s is None:
        resample_cadence_s = HEL1OS_RESAMPLE_CADENCE_S

    dump_dirs = _dump_dirs(date_str)
    if not dump_dirs:
        return None

    # ── Step 1: Read all raw data ──────────────────────────────────────
    all_dfs = []
    for dump_dir in dump_dirs:
        for det_num in [1, 2]:
            lc_file = dump_dir / f"lightcurve_{detector_type}{det_num}.fits"
            if not lc_file.exists():
                continue

            try:
                with fits.open(lc_file) as hdul:
                    ext = hdul[-1]  # last extension = full range
                    data = ext.data

                    mjd = data["MJD"].astype(np.float64)
                    time_unix = mjd_to_unix(mjd)

                    df = pd.DataFrame({
                        "time_unix": time_unix,
                        "count_rate": data["CTR"].astype(np.float64),
                        "stat_err": data["STAT_ERR"].astype(np.float64),
                        "detector": f"{detector_type}{det_num}",
                        "dump": dump_dir.name,
                    })
                    all_dfs.append(df)

            except Exception as e:
                print(f"  Warning: Failed to read {lc_file.name}: {e}")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("time_unix").reset_index(drop=True)
    combined["datetime"] = unix_to_datetime(combined["time_unix"])

    # ── Step 2: Apply GTI mask ─────────────────────────────────────────
    if apply_gti:
        gti_df = load_gti(date_str, detector_type)
        if gti_df is not None:
            combined = apply_gti_mask(
                combined, gti_df,
                value_cols=["count_rate", "stat_err"]
            )

    # ── Step 3: Combine detectors ──────────────────────────────────────
    if combine:
        combined = _combine_detectors(combined)

    # ── Step 4: Merge dumps (handle overlaps + gaps) ───────────────────
    combined = _merge_dumps(combined)

    # ── Step 5: Resample ───────────────────────────────────────────────
    if resample_cadence_s > 1:
        combined = _resample(combined, resample_cadence_s)

    return combined


# ── Per-Band Light Curve Loader ──────────────────────────────────────────────


def load_detector_bands(date_str, detector_type, apply_gti=True):
    """Load all energy bands from HEL1OS light curves (wide format).

    Each FITS file contains multiple extensions, one per energy band.
    Bands may have different row counts (different time arrays), so each
    band is read independently and merged on rounded timestamps.

    CdTe bands: 5-20, 20-30, 30-40, 40-60, 1.8-90 keV
    CZT bands:  20-40, 40-60, 60-80, 80-150, 18-160 keV

    This function returns raw per-band data without resampling. Resampling
    and gap handling are applied in load_day() after the full-range
    light curve pipeline establishes segment boundaries.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        detector_type: 'cdte' or 'czt'.
        apply_gti: If True, mask values outside GTI with NaN.

    Returns:
        DataFrame with per-band columns (ctr_X_Y, ctr_X_Y_err),
        plus time_unix, detector, dump, datetime.
        Returns None if no data found.
    """
    dump_dirs = _dump_dirs(date_str)
    if not dump_dirs:
        return None

    all_dfs = []

    for dump_dir in dump_dirs:
        for det_num in [1, 2]:
            lc_file = dump_dir / f"lightcurve_{detector_type}{det_num}.fits"
            if not lc_file.exists():
                continue

            try:
                with fits.open(lc_file) as hdul:
                    n_ext = len(hdul)

                    # Read each band independently then merge on rounded time
                    band_frames = []
                    for ext_idx in range(1, n_ext):
                        ext = hdul[ext_idx]
                        col = _band_to_colname(ext.name)
                        mjd = ext.data["MJD"].astype(np.float64)
                        t_unix = mjd_to_unix(mjd)

                        bdf = pd.DataFrame({
                            "time_unix": np.round(t_unix),
                            col: ext.data["CTR"].astype(np.float64),
                            f"{col}_err": ext.data["STAT_ERR"].astype(np.float64),
                        })
                        band_frames.append(bdf)

                    if not band_frames:
                        continue

                    # Merge all bands on rounded time (outer join)
                    merged = band_frames[0]
                    for bdf in band_frames[1:]:
                        merged = pd.merge(merged, bdf, on="time_unix", how="outer")

                    merged = merged.sort_values("time_unix").reset_index(drop=True)
                    merged["detector"] = f"{detector_type}{det_num}"
                    merged["dump"] = dump_dir.name
                    all_dfs.append(merged)

            except Exception as e:
                print(f"  Warning: Failed to read {lc_file.name}: {e}")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["datetime"] = unix_to_datetime(combined["time_unix"])
    combined = combined.sort_values("time_unix").reset_index(drop=True)

    if apply_gti:
        gti_df = load_gti(date_str, detector_type)
        if gti_df is not None:
            # NaN out all numeric band columns outside GTI
            combined = apply_gti_mask(combined, gti_df)

    return combined


# ── Day Loader (Convenience) ────────────────────────────────────────────────


def load_day(date_str, resample_cadence_s=None):
    """Load all HEL1OS data for a single day.

    Full pipeline: load → GTI filter → combine detectors → merge dumps
    → gap handling → resample.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        resample_cadence_s: Target cadence. Defaults to config value (10s).

    Returns:
        dict with keys: 'cdte', 'czt', 'cdte_bands', 'czt_bands',
        'cdte_gti', 'czt_gti'. Each value is a DataFrame or None.
    """
    return {
        "cdte": load_detector_lightcurves(
            date_str, "cdte", resample_cadence_s=resample_cadence_s
        ),
        "czt": load_detector_lightcurves(
            date_str, "czt", resample_cadence_s=resample_cadence_s
        ),
        "cdte_bands": load_detector_bands(date_str, "cdte"),
        "czt_bands": load_detector_bands(date_str, "czt"),
        "cdte_gti": load_gti(date_str, "cdte"),
        "czt_gti": load_gti(date_str, "czt"),
    }


# ── Internal: Detector Combining ────────────────────────────────────────────


def _combine_detectors(df):
    """Average count rates from detector 1 and detector 2.

    Groups by rounded time_unix (1-second resolution) and averages
    count_rate across detectors. Statistical errors are propagated
    in quadrature: err_combined = sqrt(sum(err²)) / N.

    Args:
        df: DataFrame with columns time_unix, count_rate, stat_err,
            detector, dump, datetime.

    Returns:
        DataFrame with detector column removed, averaged values.
    """
    df = df.copy()
    df["time_rounded"] = np.round(df["time_unix"])

    # Track dump identity (use the dump of the first detector for each time)
    dump_map = df.groupby("time_rounded")["dump"].first()

    grouped = df.groupby("time_rounded").agg(
        count_rate=("count_rate", "mean"),
        stat_err=("stat_err", lambda x: np.sqrt(np.sum(x ** 2)) / len(x)),
    ).reset_index()

    grouped = grouped.rename(columns={"time_rounded": "time_unix"})
    grouped["dump"] = grouped["time_unix"].map(dump_map)
    grouped["datetime"] = unix_to_datetime(grouped["time_unix"])

    return grouped


# ── Internal: Dump Merging ───────────────────────────────────────────────────


def _merge_dumps(df):
    """Merge data from multiple dumps with overlap and gap handling.

    Overlap handling (EDA found 3 overlap cases: May 3, 4, 7):
      - When two dumps have overlapping time ranges, keep the first
        dump's data in the overlap region and discard the second dump's
        overlapping rows.

    Gap handling (tiered strategy from EDA recommendations):
      - Short (≤30s): leave as-is, eligible for forward-fill downstream
      - Medium (30-120s): insert NaN row at gap boundary
      - Long (>120s): insert NaN row at gap boundary

    Adds segment_id column that increments at medium and long gaps.

    Args:
        df: DataFrame with time_unix, count_rate, stat_err, dump columns.
            Must be sorted by time_unix.

    Returns:
        DataFrame with overlaps removed, NaN rows at gap boundaries,
        segment_id and gap_mask columns added.
    """
    if df is None or len(df) == 0:
        return df

    df = df.sort_values("time_unix").reset_index(drop=True)

    # ── Handle overlaps ────────────────────────────────────────────────
    # If the 'dump' column exists, resolve overlaps by keeping the first
    # dump's data when time ranges overlap.
    if "dump" in df.columns:
        df = _resolve_overlaps(df)

    # ── Detect and handle gaps ─────────────────────────────────────────
    gap_info = detect_gaps(df["time_unix"].values)

    # Add segment IDs (before NaN insertion, so lengths match)
    df = add_segment_ids(df, gap_info)

    # Insert NaN rows at medium and long gap boundaries
    df = insert_gap_nans(df, gap_info)

    return df


def _resolve_overlaps(df):
    """Remove overlapping data between consecutive dumps.

    When dump B's start time is earlier than dump A's end time, the
    overlapping rows from dump B are removed (first dump wins).

    Args:
        df: DataFrame sorted by time_unix with a 'dump' column.

    Returns:
        DataFrame with overlapping rows removed.
    """
    if "dump" not in df.columns:
        return df

    dumps = df["dump"].unique()
    if len(dumps) <= 1:
        return df

    # Group by dump, find each dump's time range
    dump_ranges = []
    for dump_name in dumps:
        dump_data = df[df["dump"] == dump_name]
        t_min = dump_data["time_unix"].min()
        t_max = dump_data["time_unix"].max()
        dump_ranges.append((dump_name, t_min, t_max))

    # Sort dumps by their start time
    dump_ranges.sort(key=lambda x: x[1])

    # For each consecutive pair, check for overlap
    rows_to_drop = []
    for i in range(len(dump_ranges) - 1):
        dump_a_name, _, t_a_max = dump_ranges[i]
        dump_b_name, t_b_min, _ = dump_ranges[i + 1]

        if t_b_min < t_a_max:
            # Overlap detected: drop dump B's rows that are within dump A's range
            overlap_mask = (
                (df["dump"] == dump_b_name) &
                (df["time_unix"] <= t_a_max)
            )
            rows_to_drop.extend(df.index[overlap_mask].tolist())

    if rows_to_drop:
        df = df.drop(index=rows_to_drop).reset_index(drop=True)

    return df


# ── Internal: Resampling ────────────────────────────────────────────────────


def _resample(df, cadence_s):
    """Resample light curve to a coarser cadence.

    Uses mean aggregation within time bins. Respects segment boundaries:
    does not aggregate across different segments.

    Statistical errors are propagated in quadrature:
    err_resampled = sqrt(sum(err²)) / N

    NaN rows (gap markers) and NaN values (from GTI masking) are
    preserved through the resampling: bins containing only NaN values
    remain NaN.

    Args:
        df: DataFrame with time_unix, count_rate, stat_err, segment_id,
            gap_mask columns.
        cadence_s: Target cadence in seconds.

    Returns:
        Resampled DataFrame with same columns.
    """
    if cadence_s <= 1:
        return df

    df = df.copy()

    # Separate gap-marker rows (inserted NaN rows) from real data
    if "gap_mask" in df.columns:
        gap_rows = df[df["gap_mask"] == True].copy()
        data_rows = df[df["gap_mask"] == False].copy()
    else:
        gap_rows = pd.DataFrame()
        data_rows = df.copy()

    if len(data_rows) == 0:
        return df

    # Create time bins aligned to exact multiples of cadence_s
    data_rows["time_bin"] = np.floor(data_rows["time_unix"] / cadence_s) * cadence_s

    # Group by time_bin AND segment_id to avoid cross-segment aggregation
    group_cols = ["time_bin"]
    if "segment_id" in data_rows.columns:
        group_cols.append("segment_id")

    agg_dict = {
        "count_rate": "mean",
        "stat_err": lambda x: np.sqrt(np.sum(x ** 2)) / len(x),
    }

    grouped = data_rows.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={"time_bin": "time_unix"})

    # Restore gap_mask = False for all resampled data rows
    grouped["gap_mask"] = False

    # Preserve dump column if present (use the dump of the first row in bin)
    if "dump" in data_rows.columns:
        dump_map = data_rows.groupby(group_cols)["dump"].first().reset_index()
        dump_map = dump_map.rename(columns={"time_bin": "time_unix"})
        # Build merge keys using renamed column names
        merge_keys = ["time_unix"] + [c for c in group_cols if c != "time_bin"]
        grouped = grouped.merge(dump_map, on=merge_keys, how="left")

    # Re-add gap marker rows
    if len(gap_rows) > 0:
        grouped = pd.concat([grouped, gap_rows], ignore_index=True)

    grouped = grouped.sort_values("time_unix").reset_index(drop=True)
    grouped["datetime"] = unix_to_datetime(grouped["time_unix"])

    return grouped
