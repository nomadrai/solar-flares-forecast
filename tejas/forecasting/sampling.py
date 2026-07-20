"""
Phase 7 — Sampling utilities.

Loads Phase 5 labelled feature Parquets, downsamples them to the chosen
training cadence, and separates features from labels.  Enforces the
:data:`FORECAST_DROP_COLUMNS` blacklist — no label column can ever appear in
the returned feature list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tejas.config import (
    DETECTION_DIR,
    FORECAST_DROP_COLUMNS,
    FORECAST_DROP_COLUMN_PATTERNS,
    FORECAST_MIN_NONNULL_FEATURE_RATIO,
)


# ── Loading ────────────────────────────────────────────────────────────────


def load_detection_day(date_str: str, detection_dir: Path | None = None) -> pd.DataFrame:
    """Load a single Phase 5 labelled feature Parquet.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format.
        detection_dir: Directory containing the Parquet files.  Defaults to
            :data:`config.DETECTION_DIR`.

    Returns:
        DataFrame with a proper ``datetime`` column.  Returns an empty
        DataFrame if the file does not exist.
    """
    detection_dir = Path(detection_dir) if detection_dir else Path(DETECTION_DIR)
    path = detection_dir / f"{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if "datetime" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])
        # Normalise to tz-naive so downstream splitting can compare with the
        # tz-naive day strings from config.
        try:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    return df


# ── Downsampling ────────────────────────────────────────────────────────────


def downsample(df: pd.DataFrame, cadence_s: int) -> pd.DataFrame:
    """Downsample a 1-second Parquet to the requested cadence.

    Keeps every ``cadence_s``-th row aligned to a whole-second grid on
    ``time_unix``.  Segment boundaries (from ``segment_id``, if present) are
    preserved: the first sample of each new segment is always kept.

    Args:
        df: Detection Parquet DataFrame at 1 s cadence.
        cadence_s: Target cadence in seconds (e.g. 10, 30).  Must be ≥ 1.

    Returns:
        Downsampled DataFrame with a reset integer index.
    """
    if cadence_s < 1:
        raise ValueError(f"cadence_s must be ≥ 1 (got {cadence_s})")
    if df.empty:
        return df.copy()
    if cadence_s == 1:
        return df.reset_index(drop=True)

    if "time_unix" not in df.columns:
        # Fall back to positional sampling if time_unix is unavailable.
        keep_mask = (np.arange(len(df)) % cadence_s) == 0
    else:
        # Align to whole-second boundaries.  This ensures runs across days
        # sample the same wall-clock seconds.
        t = df["time_unix"].to_numpy()
        keep_mask = (np.mod(t.astype(np.int64), cadence_s) == 0)

    # Force-keep the first sample of each segment so we don't accidentally
    # drop the row that carries a segment_id transition.
    if "segment_id" in df.columns:
        seg = df["segment_id"].to_numpy()
        # np.diff prepended with True at index 0
        seg_starts = np.concatenate(([True], seg[1:] != seg[:-1]))
        keep_mask = keep_mask | seg_starts

    return df.loc[keep_mask].reset_index(drop=True)


# ── Feature / label selection ───────────────────────────────────────────────


def select_feature_columns(
    df: pd.DataFrame, drop_columns: Iterable[str] | None = None
) -> list[str]:
    """Return the deterministic list of feature columns.

    Everything in ``df.columns`` is kept EXCEPT names in the blacklist
    and names ending with any suffix in :data:`FORECAST_DROP_COLUMN_PATTERNS`.
    Ordering is the original column order — deterministic given the same
    input file layout.

    Args:
        df: Any Phase 5 (or downstream) DataFrame.
        drop_columns: Override for :data:`FORECAST_DROP_COLUMNS`.  Used by
            tests to inject custom blacklists.

    Returns:
        List of column names safe to feed into a model.
    """
    drop = set(drop_columns) if drop_columns is not None else set(FORECAST_DROP_COLUMNS)
    # Pattern guard: also drop any column ending with a forbidden suffix.
    # This catches per-instrument metadata variants (*_segment_id, *_stat_err)
    # that would otherwise bypass the literal blacklist.
    patterns = FORECAST_DROP_COLUMN_PATTERNS
    return [
        c for c in df.columns
        if c not in drop and not any(c.endswith(pat) for pat in patterns)
    ]


def select_label_column(df: pd.DataFrame, horizon_min: int) -> pd.Series:
    """Extract the binary forecasting label for a specific horizon.

    Args:
        df: Phase 5 labelled DataFrame.
        horizon_min: One of 5, 15, 30, 60.

    Returns:
        int8 Series in {0, 1} named ``label_flare_within_{h}m``.

    Raises:
        KeyError: The requested label column is not present.
    """
    col = f"label_flare_within_{horizon_min}m"
    if col not in df.columns:
        raise KeyError(f"Label column missing: {col}")
    # Cast NaN → 0; Phase 5 already fills these, but be defensive.
    return df[col].fillna(0).astype(np.int8)


# ── NaN handling ────────────────────────────────────────────────────────────


def drop_incomplete_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_nonnull_ratio: float | None = None,
) -> pd.DataFrame:
    """Drop rows where too many features are NaN.

    Rolling-window features produce NaN for the first N-1 samples of each
    segment.  Rather than forward-filling (which can inject spurious values
    at boundaries), we drop rows whose non-null fraction across the feature
    columns is below the configured threshold.  This matches the Phase 5
    convention of letting NaN propagate and dropping at boundaries.

    Args:
        df: DataFrame.
        feature_cols: Columns to evaluate for completeness.
        min_nonnull_ratio: Threshold in [0, 1].  Defaults to
            :data:`FORECAST_MIN_NONNULL_FEATURE_RATIO`.

    Returns:
        Filtered DataFrame with a reset index.
    """
    if min_nonnull_ratio is None:
        min_nonnull_ratio = FORECAST_MIN_NONNULL_FEATURE_RATIO
    if df.empty or not feature_cols:
        return df.copy()

    feature_frame = df[feature_cols]
    n_features = len(feature_cols)
    nonnull_count = feature_frame.notna().sum(axis=1)
    keep = nonnull_count >= int(np.ceil(min_nonnull_ratio * n_features))
    return df.loc[keep].reset_index(drop=True)
