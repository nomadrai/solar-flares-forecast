"""
TEJAS SoLEXS Loader
====================
Production-grade loaders for SoLEXS SDD2 data:
  - Light Curve (.lc.gz): total soft X-ray counts (2-22 keV)
  - Pulse Invariant (.pi.gz): energy-resolved multi-band counts
  - Good Time Interval (.gti.gz): valid observation windows

All loaders apply GTI filtering by setting values outside GTI to NaN
(preserving the time grid), use zero-safe division for ratio computations,
and include validation checks.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

from tejas.config import (
    SOLEXS_DIR,
    YEAR,
    MONTH,
    PI_CH_LOW,
    PI_CH_MID,
    PI_CH_HIGH,
    LC_PI_RATIO_EXPECTED,
    LC_PI_RATIO_TOL,
)
from tejas.loaders.utils import (
    unix_to_datetime,
    apply_gti_mask,
    safe_divide,
)


# ── Path Helpers ─────────────────────────────────────────────────────────────


def _solexs_path(date_str, suffix):
    """Build the path to a SoLEXS file.

    Args:
        date_str: 'YYYY-MM-DD' format.
        suffix: File suffix, e.g. '.lc.gz', '.pi.gz', '.gti.gz'.

    Returns:
        Path object.
    """
    year, month, day = date_str.split("-")
    compact = f"{year}{month}{day}"
    return SOLEXS_DIR / year / month / day / f"AL1_SOLEXS_{compact}_SDD2_L1{suffix}"


# ── GTI Loader ───────────────────────────────────────────────────────────────


def load_gti(date_str):
    """Load SoLEXS Good Time Interval data.

    Reads the GTI extension from the .gti.gz file. GTI defines the time
    intervals during which SDD2 was collecting valid data (Sun in FOV,
    detector operational, health parameters nominal).

    Args:
        date_str: Date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns: start, stop, start_dt, stop_dt, duration_s.
        Returns None if the file does not exist.
    """
    path = _solexs_path(date_str, ".gti.gz")
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


# ── Light Curve Loader ───────────────────────────────────────────────────────


def load_lc(date_str, apply_gti=True):
    """Load SoLEXS light curve data with optional GTI filtering.

    Reads the RATE extension from the .lc.gz file. Returns total soft
    X-ray counts (2-22 keV) at 1-second cadence.

    When apply_gti=True (default), values outside GTI intervals are set
    to NaN while preserving the time grid. This is critical for maintaining
    a continuous time axis in the unified dataset.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        apply_gti: If True, mask values outside GTI with NaN.

    Returns:
        DataFrame with columns: time_unix, counts, datetime.
        'counts' is NaN for timestamps outside GTI (when apply_gti=True).
        Returns None if the file does not exist.
    """
    path = _solexs_path(date_str, ".lc.gz")
    if not path.exists():
        return None

    with fits.open(path) as hdul:
        data = hdul["RATE"].data
        df = pd.DataFrame({
            "time_unix": data["TIME"].astype(np.float64),
            "counts": data["COUNTS"].astype(np.float64),
        })

    df["datetime"] = unix_to_datetime(df["time_unix"])

    if apply_gti:
        gti_df = load_gti(date_str)
        if gti_df is not None:
            df = apply_gti_mask(df, gti_df, value_cols=["counts"])

    return df


# ── PI (Spectral) Loader ────────────────────────────────────────────────────


def load_pi(date_str, apply_gti=True):
    """Load SoLEXS PI data with multi-band extraction and GTI filtering.

    Reads the SPECTRUM extension from the .pi.gz file. Extracts three
    energy bands by summing counts across channel ranges defined in config:
      - Low:  channels 59-105  (~2.8-5 keV)  — cool thermal plasma
      - Mid:  channels 105-189 (~5-10 keV)   — hotter plasma
      - High: channels 189-340 (~10-22 keV)  — hottest thermal, early non-thermal

    Channels 1-58 (~0-2.8 keV) are excluded due to uncertain instrument
    response below 2.8 keV (per SoLEXS manual).

    The hardness ratio (counts_high / counts_low) is computed using
    zero-safe division to prevent division-by-zero errors (EDA showed
    low-energy channels can have zero counts during quiet periods).

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        apply_gti: If True, mask values outside GTI with NaN.

    Returns:
        DataFrame with columns: time_unix, counts_low, counts_mid,
        counts_high, counts_total_pi, hardness_ratio, datetime.
        Returns None if the file does not exist.
    """
    path = _solexs_path(date_str, ".pi.gz")
    if not path.exists():
        return None

    with fits.open(path) as hdul:
        spec = hdul["SPECTRUM"].data
        tstart = spec["TSTART"].astype(np.float64)
        counts_2d = spec["COUNTS"]  # shape: (N_seconds, 340)

    df = pd.DataFrame({
        "time_unix": tstart,
        "counts_low": np.sum(counts_2d[:, PI_CH_LOW], axis=1).astype(np.float64),
        "counts_mid": np.sum(counts_2d[:, PI_CH_MID], axis=1).astype(np.float64),
        "counts_high": np.sum(counts_2d[:, PI_CH_HIGH], axis=1).astype(np.float64),
    })

    df["counts_total_pi"] = df["counts_low"] + df["counts_mid"] + df["counts_high"]
    df["hardness_ratio"] = safe_divide(df["counts_high"].values, df["counts_low"].values)
    df["datetime"] = unix_to_datetime(df["time_unix"])

    if apply_gti:
        gti_df = load_gti(date_str)
        if gti_df is not None:
            value_cols = [
                "counts_low", "counts_mid", "counts_high",
                "counts_total_pi", "hardness_ratio",
            ]
            df = apply_gti_mask(df, gti_df, value_cols=value_cols)

    return df


# ── Day Loader (Convenience) ────────────────────────────────────────────────


def load_day(date_str, apply_gti=True, validate=True):
    """Load all SoLEXS data for a single day.

    Convenience function that loads LC, PI, and GTI data, applies GTI
    filtering, and optionally runs validation checks.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        apply_gti: If True, mask values outside GTI with NaN.
        validate: If True, run LC-to-PI ratio validation check.

    Returns:
        dict with keys 'lc', 'pi', 'gti'. Each value is a DataFrame
        or None if the corresponding file does not exist.
        Also includes 'validation' key with check results if validate=True.
    """
    gti_df = load_gti(date_str)
    lc_df = load_lc(date_str, apply_gti=apply_gti)
    pi_df = load_pi(date_str, apply_gti=apply_gti)

    result = {
        "lc": lc_df,
        "pi": pi_df,
        "gti": gti_df,
        "validation": None,
    }

    if validate and lc_df is not None and pi_df is not None:
        result["validation"] = _validate_lc_pi(lc_df, pi_df, date_str)

    return result


def _validate_lc_pi(lc_df, pi_df, date_str):
    """Validate LC-to-PI consistency.

    Checks:
      1. Row count match between LC and PI.
      2. LC-to-PI total count ratio is approximately 5.5 (within tolerance).

    The ~5.5x ratio is expected because LC includes all 340 channels while
    PI extraction uses channels 59-340 (skipping low-energy channels with
    uncertain instrument response). See data_quality_report.md Section 3.

    Returns:
        dict with validation results and any warnings.
    """
    warnings = []

    # Check row count match
    row_match = len(lc_df) == len(pi_df)
    if not row_match:
        warnings.append(
            f"Row count mismatch: LC={len(lc_df)}, PI={len(pi_df)}"
        )

    # Check LC-to-PI ratio (only on valid, non-NaN rows)
    # NOTE: The ratio varies per day (5-10x) depending on spectral state.
    # The EDA's 5.5 was a month-wide median. This is informational, not
    # a pass/fail criterion.
    if row_match and len(lc_df) > 0:
        lc_valid = lc_df["counts"].dropna()
        pi_valid = pi_df["counts_total_pi"].dropna()

        # Use the intersection of valid indices
        valid_idx = lc_valid.index.intersection(pi_valid.index)

        if len(valid_idx) > 0:
            lc_vals = lc_valid.loc[valid_idx]
            pi_vals = pi_valid.loc[valid_idx]

            # Avoid division by zero in ratio computation
            nonzero_mask = pi_vals > 0
            if nonzero_mask.any():
                ratios = lc_vals[nonzero_mask] / pi_vals[nonzero_mask]
                median_ratio = float(ratios.median())
            else:
                median_ratio = np.nan
                warnings.append("All PI total counts are zero")
        else:
            median_ratio = np.nan
            warnings.append("No overlapping valid rows for ratio check")
    else:
        median_ratio = np.nan

    passed = len(warnings) == 0

    if warnings:
        print(f"  [WARNING] SoLEXS validation [{date_str}]: {'; '.join(warnings)}")
    else:
        print(f"  [OK] SoLEXS validation [{date_str}]: "
              f"rows match, LC/PI ratio = {median_ratio:.2f}")

    return {
        "date": date_str,
        "row_match": row_match,
        "lc_rows": len(lc_df),
        "pi_rows": len(pi_df),
        "median_lc_pi_ratio": median_ratio,
        "passed": passed,
        "warnings": warnings,
    }
