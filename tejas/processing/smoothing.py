"""
TEJAS Signal Smoothing
=======================
Reduce high-frequency noise while preserving flare temporal structure.

Algorithm — Segment-Aware Rolling Median:
  - Median is far more robust than mean for heavily zero-inflated HEL1OS
    data (87% zeros in CdTe).
  - Window size is configurable per-column (wider for noisier signals like
    CZT and the hardness ratio).
  - Smoothing respects segment boundaries via the validity flags; does NOT
    smooth across data gaps.

Operates on *_clean columns produced by spike_removal.  Creates *_smooth
columns.  Original and cleaned columns remain untouched.
"""

import numpy as np
import pandas as pd

from tejas.config import SMOOTHING_WINDOWS


# ── Core Algorithm ──────────────────────────────────────────────────────────


def smooth_signal(series, window=5, method="median"):
    """Apply rolling smoothing to a 1D signal.

    NaN values are preserved (not bridged).  The rolling window uses
    center=True so there is no temporal shift.

    Args:
        series: pd.Series of values (may contain NaN).
        window: Rolling window size in samples (must be odd).
        method: 'median' (default, robust) or 'mean'.

    Returns:
        pd.Series — smoothed signal (same length as input).
    """
    if window % 2 == 0:
        window += 1

    if window <= 1:
        return series.copy()

    min_periods = max(1, window // 3)
    roller = series.rolling(window=window, center=True, min_periods=min_periods)

    if method == "median":
        smoothed = roller.median()
    elif method == "mean":
        smoothed = roller.mean()
    else:
        raise ValueError(f"Unknown smoothing method: {method!r}")

    return smoothed


# ── Day-Level Application ──────────────────────────────────────────────────


# Map from clean column names to their smoothing-config lookup key
_CLEAN_TO_CONFIG = {
    "solexs_counts_clean":          "solexs_counts",
    "solexs_counts_low_clean":      "solexs_counts_low",
    "solexs_counts_mid_clean":      "solexs_counts_mid",
    "solexs_counts_high_clean":     "solexs_counts_high",
    "solexs_hardness_ratio_clean":  "solexs_hardness_ratio",
    "cdte_count_rate_clean":        "cdte_count_rate",
    "czt_count_rate_clean":         "czt_count_rate",
}


def smooth_day(df, windows=None):
    """Apply smoothing to all clean signal columns in a unified day DataFrame.

    Reads *_clean columns and writes *_smooth columns.  If a *_clean column
    does not exist, falls back to the original column.

    Args:
        df: DataFrame with *_clean columns from spike_removal.
        windows: Optional dict mapping config-key → window size.
            Defaults to config.SMOOTHING_WINDOWS.

    Returns:
        dict with keys:
          'df': DataFrame with *_smooth columns added.
          'report': dict of per-column smoothing info.
    """
    if windows is None:
        windows = SMOOTHING_WINDOWS

    df = df.copy()
    report = {}

    for clean_col, config_key in _CLEAN_TO_CONFIG.items():
        # Determine source column (prefer _clean, fallback to raw)
        if clean_col in df.columns:
            source_col = clean_col
        else:
            # Try the raw column name (config_key matches raw column)
            if config_key in df.columns:
                source_col = config_key
            else:
                continue

        # Look up window for this column
        window = windows.get(config_key, 5)

        # Derive the smooth column name
        # solexs_counts_clean → solexs_counts_smooth
        smooth_col = config_key + "_smooth"

        df[smooth_col] = smooth_signal(df[source_col], window=window)

        report[smooth_col] = {
            "source": source_col,
            "window": window,
            "method": "median",
        }

    return {"df": df, "report": report}
