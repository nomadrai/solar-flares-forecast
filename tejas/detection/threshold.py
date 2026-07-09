"""
TEJAS Adaptive Threshold Engine
==================================
Computes, at every timestep, an adaptive baseline (trailing mean + std)
and corresponding z-score for a given signal.  The baseline is strictly
causal — it only uses past data — and segment-aware, meaning it never
spans a GTI-gap boundary.

Used by the detector module to determine flare start and end triggers.
"""

import numpy as np
import pandas as pd

from tejas.config import (
    DETECTION_BASELINE_WINDOW_S,
    DETECTION_SIGMA_START,
    DETECTION_SIGMA_END,
    ZSCORE_STD_FLOOR,
)


def compute_adaptive_threshold(
    series,
    baseline_window_s=None,
    sigma_start=None,
    sigma_end=None,
    segment_ids=None,
    std_floor=None,
):
    """Compute adaptive threshold and z-score for a signal.

    The threshold is based on a trailing rolling window of the signal.
    If segment_ids are provided, the rolling window resets at segment
    boundaries so that baselines never span GTI gaps.

    Args:
        series: pd.Series of signal values (e.g. solexs_counts_smooth).
        baseline_window_s: Trailing window size in seconds (= samples at
            1s cadence).  Defaults to config.DETECTION_BASELINE_WINDOW_S.
        sigma_start: Number of std deviations above baseline to trigger
            flare start.  Defaults to config.DETECTION_SIGMA_START.
        sigma_end: Number of std deviations above baseline to declare
            flare end.  Defaults to config.DETECTION_SIGMA_END.
        segment_ids: Optional pd.Series of segment IDs (e.g.
            cdte_segment_id).  If provided, rolling windows do not cross
            segment boundaries.
        std_floor: Minimum value for the trailing std to prevent
            divide-by-zero in quiet periods.  Defaults to
            config.ZSCORE_STD_FLOOR.

    Returns:
        pd.DataFrame with columns:
          - baseline_mean : trailing window mean (causal)
          - baseline_std  : trailing window std  (causal, floored)
          - threshold_hi  : baseline_mean + sigma_start * baseline_std
          - threshold_lo  : baseline_mean + sigma_end   * baseline_std
          - z_score       : (signal - baseline_mean) / baseline_std
    """
    if baseline_window_s is None:
        baseline_window_s = DETECTION_BASELINE_WINDOW_S
    if sigma_start is None:
        sigma_start = DETECTION_SIGMA_START
    if sigma_end is None:
        sigma_end = DETECTION_SIGMA_END
    if std_floor is None:
        std_floor = ZSCORE_STD_FLOOR

    n = len(series)
    min_periods = min(60, baseline_window_s)

    if segment_ids is not None and not segment_ids.isna().all():
        # Segment-aware rolling: compute within each segment independently,
        # then stitch results back in original index order.
        baseline_mean = pd.Series(np.nan, index=series.index)
        baseline_std = pd.Series(np.nan, index=series.index)

        for seg_id, group in series.groupby(segment_ids):
            if pd.isna(seg_id):
                continue
            seg_mean = group.rolling(
                window=baseline_window_s,
                min_periods=min_periods,
                center=False,
            ).mean()
            seg_std = group.rolling(
                window=baseline_window_s,
                min_periods=min_periods,
                center=False,
            ).std()
            baseline_mean.iloc[seg_mean.index] = seg_mean
            baseline_std.iloc[seg_std.index] = seg_std
    else:
        # No segments — simple trailing rolling window.
        baseline_mean = series.rolling(
            window=baseline_window_s,
            min_periods=min_periods,
            center=False,
        ).mean()
        baseline_std = series.rolling(
            window=baseline_window_s,
            min_periods=min_periods,
            center=False,
        ).std()

    # Floor the std to prevent divide-by-zero in very quiet periods.
    baseline_std = baseline_std.clip(lower=std_floor)

    # Compute thresholds and z-score.
    threshold_hi = baseline_mean + sigma_start * baseline_std
    threshold_lo = baseline_mean + sigma_end * baseline_std
    z_score = (series - baseline_mean) / baseline_std

    return pd.DataFrame({
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "threshold_hi": threshold_hi,
        "threshold_lo": threshold_lo,
        "z_score": z_score,
    }, index=series.index)
