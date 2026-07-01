"""
TEJAS Loader Utilities
======================
Shared functions used by both SoLEXS and HEL1OS loaders:
  - Time conversions (Unix ↔ datetime, MJD → Unix)
  - GTI mask application (NaN-based, preserves time grid)
  - Zero-safe division
  - Gap detection and NaN insertion (tiered strategy)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from tejas.config import (
    MJD_UNIX_EPOCH,
    SECONDS_PER_DAY,
    ZERO_COUNT_EPS,
    GAP_SHORT_S,
    GAP_MEDIUM_S,
)


# ── Time Conversions ────────────────────────────────────────────────────────


def unix_to_datetime(unix_times):
    """Convert Unix timestamps to UTC-aware datetime.

    Args:
        unix_times: Scalar, array, or Series of Unix timestamps.

    Returns:
        DatetimeIndex or Series with UTC timezone.
    """
    return pd.to_datetime(unix_times, unit="s", origin="unix", utc=True)


def mjd_to_unix(mjd):
    """Convert Modified Julian Date to Unix timestamp.

    Formula: Unix = (MJD - 40587) × 86400
    Reference: CONTEXT.md HEL1OS section.

    Args:
        mjd: Scalar, array, or Series of MJD values.

    Returns:
        numpy array of Unix timestamps (float64).
    """
    return (np.asarray(mjd, dtype=np.float64) - MJD_UNIX_EPOCH) * SECONDS_PER_DAY


# ── GTI Mask Application ────────────────────────────────────────────────────


def apply_gti_mask(df, gti_df, time_col="time_unix", value_cols=None):
    """Mark data outside Good Time Intervals as NaN.

    Unlike the EDA approach (which drops rows), this preserves the time grid
    by setting value columns to NaN for timestamps outside any GTI interval.
    This is important for maintaining a continuous time axis in the unified
    dataset.

    Args:
        df: Input DataFrame with a time column and value columns.
        gti_df: GTI DataFrame with 'start' and 'stop' columns (Unix time).
            If None or empty, returns df unchanged.
        time_col: Name of the Unix-time column in df.
        value_cols: List of column names to set to NaN outside GTI.
            If None, sets all numeric columns except time_col and 'datetime'.

    Returns:
        DataFrame with values outside GTI set to NaN (same shape as input).
    """
    if gti_df is None or len(gti_df) == 0:
        return df

    df = df.copy()
    times = df[time_col].values

    # Build a boolean mask: True if within any GTI interval
    in_gti = np.zeros(len(df), dtype=bool)
    for _, row in gti_df.iterrows():
        in_gti |= (times >= row["start"]) & (times <= row["stop"])

    # Determine which columns to NaN-out
    if value_cols is None:
        value_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in (time_col,)
        ]
        # Also exclude 'datetime' if it somehow got included
        value_cols = [c for c in value_cols if c != "datetime"]

    # Set values outside GTI to NaN
    outside_gti = ~in_gti
    if outside_gti.any():
        for col in value_cols:
            if col in df.columns:
                df.loc[outside_gti, col] = np.nan

    return df


# ── Zero-Safe Division ──────────────────────────────────────────────────────


def safe_divide(numerator, denominator, eps=None):
    """Divide with zero-count protection.

    Replaces zero (or near-zero) values in the denominator with eps
    before dividing, preventing division-by-zero in ratio computations
    like hardness_ratio = counts_high / counts_low.

    Args:
        numerator: Array-like numerator values.
        denominator: Array-like denominator values.
        eps: Minimum denominator value. Defaults to config.ZERO_COUNT_EPS.

    Returns:
        numpy array of quotients (float64).
    """
    if eps is None:
        eps = ZERO_COUNT_EPS
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return num / np.maximum(den, eps)


# ── Gap Detection & Handling ────────────────────────────────────────────────


@dataclass
class GapInfo:
    """Results of gap detection analysis.

    Attributes:
        gap_indices: Array indices where gaps start (the index BEFORE the gap).
        gap_durations: Duration of each gap in seconds.
        gap_types: Classification of each gap: 'short', 'medium', or 'long'.
        segment_ids: Array of segment IDs (same length as input time array).
            Increments at medium and long gaps.
        n_short: Count of short gaps (≤ GAP_SHORT_S).
        n_medium: Count of medium gaps (GAP_SHORT_S < gap ≤ GAP_MEDIUM_S).
        n_long: Count of long gaps (> GAP_MEDIUM_S).
    """
    gap_indices: np.ndarray
    gap_durations: np.ndarray
    gap_types: np.ndarray
    segment_ids: np.ndarray
    n_short: int
    n_medium: int
    n_long: int


def detect_gaps(time_unix, short_s=None, medium_s=None):
    """Detect and classify gaps in a time series using tiered thresholds.

    Tiered gap strategy (from EDA recommendations):
      - Short (≤ short_s):  routine telemetry gaps, eligible for forward-fill
      - Medium (short_s < gap ≤ medium_s): NaN insertion (configurable)
      - Long (> medium_s):  always NaN, never interpolate

    Segment IDs increment at medium and long gaps, so downstream code
    can process contiguous segments independently without crossing gaps
    where data was absent.

    Args:
        time_unix: 1D array of Unix timestamps (must be sorted).
        short_s: Short gap threshold in seconds. Defaults to config.GAP_SHORT_S.
        medium_s: Medium gap threshold in seconds. Defaults to config.GAP_MEDIUM_S.

    Returns:
        GapInfo dataclass with gap locations, durations, types, and segment IDs.
    """
    if short_s is None:
        short_s = GAP_SHORT_S
    if medium_s is None:
        medium_s = GAP_MEDIUM_S

    times = np.asarray(time_unix, dtype=np.float64)
    n = len(times)

    if n <= 1:
        return GapInfo(
            gap_indices=np.array([], dtype=int),
            gap_durations=np.array([], dtype=np.float64),
            gap_types=np.array([], dtype="U6"),
            segment_ids=np.zeros(n, dtype=int),
            n_short=0, n_medium=0, n_long=0,
        )

    # Compute time differences between consecutive points
    dt = np.diff(times)

    # A "gap" is any time difference significantly larger than the expected
    # cadence. We use 1.5s as the minimum gap threshold (expected cadence is
    # 1s for raw data, so anything > 1.5s is a gap).
    MIN_GAP_S = 1.5
    gap_mask = dt > MIN_GAP_S
    gap_indices = np.where(gap_mask)[0]
    gap_durations = dt[gap_indices]

    # Classify each gap
    gap_types = np.empty(len(gap_indices), dtype="U6")
    gap_types[gap_durations <= short_s] = "short"
    gap_types[(gap_durations > short_s) & (gap_durations <= medium_s)] = "medium"
    gap_types[gap_durations > medium_s] = "long"

    # Build segment IDs: increment at medium and long gaps
    segment_ids = np.zeros(n, dtype=int)
    segment_boundary_mask = gap_durations > short_s  # medium or long
    segment_boundary_indices = gap_indices[segment_boundary_mask]

    for idx in segment_boundary_indices:
        # Everything after this gap index gets a higher segment ID
        segment_ids[idx + 1:] += 1

    n_short = int(np.sum(gap_types == "short"))
    n_medium = int(np.sum(gap_types == "medium"))
    n_long = int(np.sum(gap_types == "long"))

    return GapInfo(
        gap_indices=gap_indices,
        gap_durations=gap_durations,
        gap_types=gap_types,
        segment_ids=segment_ids,
        n_short=n_short,
        n_medium=n_medium,
        n_long=n_long,
    )


def insert_gap_nans(df, gap_info, time_col="time_unix"):
    """Insert NaN rows at medium and long gap boundaries.

    For each medium or long gap, inserts a single NaN-valued row at the
    midpoint of the gap. This ensures that:
      1. Plotting libraries show a visible break (not a connected line).
      2. Rolling-window computations don't accidentally span the gap.
      3. The gap_mask column marks these rows as non-real data.

    Short gaps are left intact (eligible for forward-fill downstream).

    Args:
        df: DataFrame with time and value columns.
        gap_info: GapInfo from detect_gaps().
        time_col: Name of the Unix-time column.

    Returns:
        DataFrame with NaN rows inserted at medium/long gap boundaries,
        sorted by time, with a 'gap_mask' column (True = inserted NaN row).
    """
    if len(gap_info.gap_indices) == 0:
        df = df.copy()
        df["gap_mask"] = False
        return df

    # Mark original rows as real data
    df = df.copy()
    df["gap_mask"] = False

    # Collect NaN rows to insert
    nan_rows = []
    for i, gap_type in enumerate(gap_info.gap_types):
        if gap_type in ("medium", "long"):
            idx = gap_info.gap_indices[i]
            t_before = df.iloc[idx][time_col]
            t_after = df.iloc[idx + 1][time_col]
            t_mid = (t_before + t_after) / 2.0

            # Create a NaN row with only time and gap_mask set
            nan_row = {col: np.nan for col in df.columns}
            nan_row[time_col] = t_mid
            nan_row["gap_mask"] = True

            # Preserve non-numeric columns that shouldn't be NaN
            for col in df.columns:
                if df[col].dtype == object or col in ("detector", "dump"):
                    nan_row[col] = None

            nan_rows.append(nan_row)

    if nan_rows:
        nan_df = pd.DataFrame(nan_rows)
        df = pd.concat([df, nan_df], ignore_index=True)
        df = df.sort_values(time_col).reset_index(drop=True)

    return df


def add_segment_ids(df, gap_info, time_col="time_unix"):
    """Add segment_id column to a DataFrame based on gap analysis.

    The segment_id increments at medium and long gaps. Rows within the
    same segment belong to a contiguous data block that can be safely
    processed together (rolling windows, interpolation, etc.).

    Args:
        df: DataFrame with a time column.
        gap_info: GapInfo from detect_gaps().
        time_col: Name of the Unix-time column.

    Returns:
        DataFrame with 'segment_id' column added.
    """
    df = df.copy()

    if len(gap_info.segment_ids) == len(df):
        # Direct assignment if lengths match (before NaN insertion)
        df["segment_id"] = gap_info.segment_ids
    else:
        # After NaN insertion, we need to recompute segment IDs
        # based on the gap_mask column
        times = df[time_col].values
        gap_info_new = detect_gaps(times)
        df["segment_id"] = gap_info_new.segment_ids

    return df
