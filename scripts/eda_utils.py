"""
TEJAS EDA Utilities
===================
Shared reader functions and constants for the EDA scripts.
Refactored from view_solexs_hel1os_data.py with enhancements for
batch processing across the full month of data.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Users\aadit\Documents\Projects\TEJAS")
SOLEXS_DIR = PROJECT_DIR / "data" / "raw" / "solexs"
HEL1OS_DIR = PROJECT_DIR / "data" / "raw" / "hel1os"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"

YEAR = "2026"
MONTH = "05"


# ── General Utilities ────────────────────────────────────────────────────────

def ensure_output_dir():
    """Create the EDA output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def unix_to_datetime(unix_times):
    """Convert Unix timestamps to UTC-aware datetime."""
    return pd.to_datetime(unix_times, unit="s", origin="unix", utc=True)


def mjd_to_unix(mjd):
    """Convert Modified Julian Date to Unix timestamp.

    Formula: Unix = (MJD - 40587) × 86400
    Reference: CONTEXT.md HEL1OS section.
    """
    return (np.asarray(mjd, dtype=np.float64) - 40587.0) * 86400.0


def get_available_days():
    """Return sorted lists of available day strings ('01', '02', ...).

    Returns:
        (solexs_days, hel1os_days): each a sorted list of 2-digit strings.
    """
    solexs_base = SOLEXS_DIR / YEAR / MONTH
    hel1os_base = HEL1OS_DIR / YEAR / MONTH

    solexs_days = (
        sorted([d.name for d in solexs_base.iterdir() if d.is_dir()])
        if solexs_base.exists() else []
    )
    hel1os_days = (
        sorted([d.name for d in hel1os_base.iterdir() if d.is_dir()])
        if hel1os_base.exists() else []
    )
    return solexs_days, hel1os_days


def day_to_date_str(day):
    """Convert day string '01' to date string '2026-05-01'."""
    return f"{YEAR}-{MONTH}-{day}"


# ── SoLEXS Readers ──────────────────────────────────────────────────────────

def read_solexs_lc(date_str):
    """Read SoLEXS light curve for a given date.

    Args:
        date_str: 'YYYY-MM-DD' format.

    Returns:
        DataFrame[time_unix, counts, datetime] or None if file not found.
    """
    year, month, day = date_str.split("-")
    compact = f"{year}{month}{day}"
    path = SOLEXS_DIR / year / month / day / f"AL1_SOLEXS_{compact}_SDD2_L1.lc.gz"

    if not path.exists():
        return None

    with fits.open(path) as hdul:
        data = hdul["RATE"].data
        df = pd.DataFrame({
            "time_unix": data["TIME"].astype(np.float64),
            "counts": data["COUNTS"].astype(np.float64),
        })
        df["datetime"] = unix_to_datetime(df["time_unix"])

    return df


def read_solexs_pi(date_str):
    """Read SoLEXS PI (spectral) file and extract multi-band light curves.

    Multi-band extraction follows the channel ranges defined in data_used.md:
      - Low:  channels 59–105 → ~2.8–5 keV
      - Mid:  channels 105–189 → ~5–10 keV
      - High: channels 189–340 → ~10–22 keV

    Returns:
        DataFrame[time_unix, counts_low, counts_mid, counts_high,
                  counts_total_pi, hardness_ratio, datetime]
        or None if file not found.
    """
    year, month, day = date_str.split("-")
    compact = f"{year}{month}{day}"
    path = SOLEXS_DIR / year / month / day / f"AL1_SOLEXS_{compact}_SDD2_L1.pi.gz"

    if not path.exists():
        return None

    with fits.open(path) as hdul:
        spec = hdul["SPECTRUM"].data
        tstart = spec["TSTART"].astype(np.float64)
        counts = spec["COUNTS"]  # shape: (N_seconds, 340)

    # Channel ranges (0-indexed numpy slices; channels are 1-indexed in manual)
    ch_low = slice(58, 105)     # ~2.8–5 keV
    ch_mid = slice(105, 189)    # ~5–10 keV
    ch_high = slice(189, 340)   # ~10–22 keV

    df = pd.DataFrame({
        "time_unix": tstart,
        "counts_low": np.sum(counts[:, ch_low], axis=1).astype(np.float64),
        "counts_mid": np.sum(counts[:, ch_mid], axis=1).astype(np.float64),
        "counts_high": np.sum(counts[:, ch_high], axis=1).astype(np.float64),
    })
    df["counts_total_pi"] = df["counts_low"] + df["counts_mid"] + df["counts_high"]
    df["hardness_ratio"] = df["counts_high"] / df["counts_low"].replace(0, np.nan)
    df["datetime"] = unix_to_datetime(df["time_unix"])

    return df


def read_solexs_gti(date_str):
    """Read SoLEXS Good Time Interval file.

    Returns:
        DataFrame[start, stop, start_dt, stop_dt, duration_s]
        or None if file not found.
    """
    year, month, day = date_str.split("-")
    compact = f"{year}{month}{day}"
    path = SOLEXS_DIR / year / month / day / f"AL1_SOLEXS_{compact}_SDD2_L1.gti.gz"

    if not path.exists():
        return None

    with fits.open(path) as hdul:
        data = hdul["GTI"].data
        df = pd.DataFrame({
            "start": data["START"].astype(np.float64),
            "stop": data["STOP"].astype(np.float64),
        })
        df["start_dt"] = unix_to_datetime(df["start"])
        df["stop_dt"] = unix_to_datetime(df["stop"])
        df["duration_s"] = df["stop"] - df["start"]

    return df


def apply_gti(df, gti_df, time_col="time_unix"):
    """Filter DataFrame to keep only rows within GTI intervals.

    Args:
        df: DataFrame with a time column.
        gti_df: GTI DataFrame with 'start' and 'stop' columns.
        time_col: Name of the time column in df.

    Returns:
        Filtered DataFrame (reset index).
    """
    if gti_df is None or len(gti_df) == 0:
        return df

    mask = np.zeros(len(df), dtype=bool)
    times = df[time_col].values

    for _, row in gti_df.iterrows():
        mask |= (times >= row["start"]) & (times <= row["stop"])

    return df[mask].reset_index(drop=True)


# ── HEL1OS Readers ──────────────────────────────────────────────────────────

def _band_to_colname(band_name):
    """Convert FITS extension band name to a clean column name.

    Handles both short-form and actual FITS extension names:
        '5-20 keV'                              -> 'ctr_5_20'
        '1.8-90 keV'                            -> 'ctr_1p8_90'
        'CDTE1_LC_BAND_5.00KEV_TO_20.00KEV'    -> 'ctr_5_20'
        'CZT1_LC_BAND_20.00KEV_TO_40.00KEV'    -> 'ctr_20_40'
    """
    import re

    name = band_name.strip().upper()

    # Try to parse actual FITS extension name: *_BAND_X.XXkEV_TO_Y.YYkEV
    m = re.search(r'(\d+(?:\.\d+)?)KEV_TO_(\d+(?:\.\d+)?)KEV', name)
    if m:
        lo = m.group(1).rstrip("0").rstrip(".")   # "5.00" -> "5"
        hi = m.group(2).rstrip("0").rstrip(".")   # "20.00" -> "20"
        lo = lo.replace(".", "p")
        hi = hi.replace(".", "p")
        return f"ctr_{lo}_{hi}"

    # Fallback: simple format like "5-20 keV"
    clean = (name
             .replace(" KEV", "").replace("KEV", "")
             .replace(".", "p")
             .replace("-", "_"))
    return f"ctr_{clean}"


def read_hel1os_full_range(date_str, detector_type):
    """Read HEL1OS full-range light curves for a given date.

    Reads only the last extension (full energy range) for each detector
    file found in each dump directory.

    Args:
        date_str: 'YYYY-MM-DD' format.
        detector_type: 'cdte' or 'czt'.

    Returns:
        DataFrame[time_unix, count_rate, stat_err, detector, dump, datetime]
        or None if no data found.
    """
    year, month, day = date_str.split("-")
    day_dir = HEL1OS_DIR / year / month / day

    if not day_dir.exists():
        return None

    dump_dirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])
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
    combined["datetime"] = unix_to_datetime(combined["time_unix"])
    combined = combined.sort_values("time_unix").reset_index(drop=True)
    return combined


def read_hel1os_all_bands(date_str, detector_type):
    """Read all energy bands from HEL1OS light curves (wide format).

    Each FITS file contains multiple extensions, one per energy band.
    **Bands may have different row counts** (different time arrays),
    so each band is read independently and merged on rounded timestamps.

    CdTe columns: ctr_5_20, ctr_20_30, ctr_30_40, ctr_40_60, ctr_1p8_90
    CZT columns:  ctr_20_40, ctr_40_60, ctr_60_80, ctr_80_150, ctr_18_160

    Args:
        date_str: 'YYYY-MM-DD' format.
        detector_type: 'cdte' or 'czt'.

    Returns:
        DataFrame with per-band columns, or None if no data found.
    """
    year, month, day = date_str.split("-")
    day_dir = HEL1OS_DIR / year / month / day

    if not day_dir.exists():
        return None

    dump_dirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])
    all_dfs = []

    for dump_dir in dump_dirs:
        for det_num in [1, 2]:
            lc_file = dump_dir / f"lightcurve_{detector_type}{det_num}.fits"
            if not lc_file.exists():
                continue

            try:
                with fits.open(lc_file) as hdul:
                    n_ext = len(hdul)

                    # Read each band independently (they may have different
                    # row counts / time arrays), then merge on rounded time.
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

                    # Merge all bands on rounded time (outer join keeps all timestamps)
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
    return combined


def read_hel1os_gti(date_str, detector_type):
    """Read HEL1OS GTI files for a given date and detector type.

    Returns:
        DataFrame[start, stop, start_dt, stop_dt, duration_s, detector, dump]
        or None if no data found.
    """
    year, month, day = date_str.split("-")
    day_dir = HEL1OS_DIR / year / month / day

    if not day_dir.exists():
        return None

    dump_dirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])
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


# ── Combining & Resampling ──────────────────────────────────────────────────

def combine_detectors_mean(df, group_resolution_s=1.0):
    """Average count rates from multiple detectors at each timestamp.

    Groups by time_unix rounded to ``group_resolution_s`` and averages
    count_rate across detectors. Errors are propagated in quadrature.

    Args:
        df: DataFrame with columns time_unix, count_rate, stat_err.
        group_resolution_s: Rounding resolution for time grouping.

    Returns:
        DataFrame[time_unix, count_rate, stat_err, datetime].
    """
    df = df.copy()
    df["time_rounded"] = (
        np.round(df["time_unix"] / group_resolution_s) * group_resolution_s
    )

    grouped = df.groupby("time_rounded").agg(
        count_rate=("count_rate", "mean"),
        stat_err=("stat_err", lambda x: np.sqrt(np.sum(x ** 2)) / len(x)),
    ).reset_index()

    grouped = grouped.rename(columns={"time_rounded": "time_unix"})
    grouped["datetime"] = unix_to_datetime(grouped["time_unix"])
    return grouped


def resample_to_cadence(df, cadence_s, time_col="time_unix", value_cols=None):
    """Resample a time series to a coarser cadence using mean aggregation.

    Bins are aligned to exact multiples of ``cadence_s`` from Unix epoch.
    Empty bins are naturally skipped (no NaN rows).

    Args:
        df: Input DataFrame.
        cadence_s: Target cadence in seconds.
        time_col: Name of the Unix-time column.
        value_cols: Column name(s) to aggregate. If None, aggregates all
                    numeric columns except time_col.

    Returns:
        Resampled DataFrame with time_unix and datetime columns.
    """
    df = df.copy()
    df["time_bin"] = np.floor(df[time_col] / cadence_s) * cadence_s

    if value_cols is None:
        value_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in (time_col, "time_bin")
        ]
    elif isinstance(value_cols, str):
        value_cols = [value_cols]

    agg_dict = {col: "mean" for col in value_cols}
    grouped = df.groupby("time_bin").agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={"time_bin": "time_unix"})
    grouped["datetime"] = unix_to_datetime(grouped["time_unix"])
    return grouped
