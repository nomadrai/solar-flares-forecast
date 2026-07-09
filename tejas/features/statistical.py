"""
TEJAS Statistical Features
============================
Context-aware statistical features that capture how unusual the current
signal is relative to recent history.

These features provide the ML model with "anomaly awareness" — the
ability to detect that something unusual is happening before a clear
flare signature emerges in the raw signal.

Features per source signal:
  - Z-score relative to trailing baseline (multiple windows)
  - Percentile rank within trailing window (multiple windows)
  - CUSUM (Cumulative Sum) for sustained shift detection

All operations are strictly causal (trailing windows only).
"""

import numpy as np
import pandas as pd

from tejas.config import (
    STAT_ZSCORE_WINDOWS,
    STAT_PCTRANK_WINDOWS,
    CUSUM_RESET_INTERVAL_S,
    ZSCORE_STD_FLOOR,
)


# ── Source Column Resolution ────────────────────────────────────────────────

_SOURCE_SIGNALS = [
    {
        "name": "solexs",
        "preferred": "solexs_counts_smooth",
        "fallbacks": ["solexs_counts_clean", "solexs_counts"],
    },
    {
        "name": "cdte",
        "preferred": "cdte_count_rate_smooth",
        "fallbacks": ["cdte_count_rate_clean", "cdte_count_rate"],
    },
    {
        "name": "czt",
        "preferred": "czt_net_rate",
        "fallbacks": ["czt_count_rate_smooth", "czt_count_rate_clean",
                       "czt_count_rate"],
    },
]


def _resolve_source(df, signal_def):
    """Find the best available source column for a signal."""
    if signal_def["preferred"] in df.columns:
        return signal_def["preferred"]
    for fb in signal_def["fallbacks"]:
        if fb in df.columns:
            return fb
    return None


# ── Core Feature Functions ──────────────────────────────────────────────────


def compute_zscore(series, window, std_floor=None):
    """Compute z-score relative to trailing rolling baseline.

    z = (x - trailing_mean) / max(trailing_std, floor)

    The std floor prevents extreme z-scores during very stable quiet
    periods where std → 0.

    Args:
        series: pd.Series of values.
        window: Trailing window size in samples.
        std_floor: Minimum std value. Defaults to config.

    Returns:
        pd.Series — z-score values.
    """
    if std_floor is None:
        std_floor = ZSCORE_STD_FLOOR

    rmean = series.rolling(window=window, min_periods=1).mean()
    rstd = series.rolling(window=window, min_periods=2).std(ddof=1)
    rstd = rstd.clip(lower=std_floor)

    return (series - rmean) / rstd


def compute_trailing_pctrank(series, window):
    """Compute percentile rank within a trailing window.

    For each position t, ranks x[t] relative to the values in
    [t-window+1, t].  Returns a value in [0.0, 1.0].

    Args:
        series: pd.Series of values.
        window: Trailing window size in samples.

    Returns:
        pd.Series — percentile ranks.
    """
    def _rank_in_window(arr):
        """Rank the last element relative to the window."""
        if len(arr) < 2:
            return np.nan
        val = arr[-1]
        if np.isnan(val):
            return np.nan
        # Count values ≤ val (excluding NaN)
        valid = arr[np.isfinite(arr)]
        if len(valid) < 2:
            return np.nan
        return np.sum(valid <= val) / len(valid)

    return series.rolling(window=window, min_periods=2).apply(
        _rank_in_window, raw=True
    )


def compute_cusum(series, window_for_mean, reset_interval=None):
    """Compute CUSUM (Cumulative Sum of deviations from trailing mean).

    CUSUM accumulates small positive deviations that individually are
    not significant but collectively indicate a sustained rising trend.
    It is reset periodically to prevent unbounded growth.

    Algorithm:
      1. Compute trailing mean over `window_for_mean`
      2. Compute deviation = x - trailing_mean
      3. Accumulate max(0, cusum + deviation) (one-sided, positive shifts)
      4. Reset to 0 every `reset_interval` samples

    Args:
        series: pd.Series of values.
        window_for_mean: Window for trailing mean baseline.
        reset_interval: Reset interval in samples. Defaults to config.

    Returns:
        pd.Series — CUSUM values.
    """
    if reset_interval is None:
        reset_interval = CUSUM_RESET_INTERVAL_S

    rmean = series.rolling(window=window_for_mean, min_periods=1).mean()
    deviation = series - rmean

    # Compute CUSUM with periodic resets
    values = deviation.values
    cusum = np.zeros(len(values))

    for i in range(len(values)):
        if np.isnan(values[i]):
            cusum[i] = cusum[i - 1] if i > 0 else 0.0
        elif i % reset_interval == 0:
            cusum[i] = max(0.0, values[i])
        else:
            cusum[i] = max(0.0, cusum[i - 1] + values[i])

    return pd.Series(cusum, index=series.index)


# ── Day-Level Application ──────────────────────────────────────────────────


def compute_statistical_features(df, zscore_windows=None,
                                  pctrank_windows=None,
                                  cusum_reset=None):
    """Compute all statistical features for a unified day DataFrame.

    Creates new columns:
      {signal}_zscore_{w}      — z-score relative to trailing mean
      {signal}_pctrank_{w}     — percentile rank in trailing window
      {signal}_cusum           — cumulative sum shift detector

    Args:
        df: DataFrame with clean/smoothed signal columns.
        zscore_windows: List of z-score window sizes. Defaults to config.
        pctrank_windows: List of percentile rank window sizes.
        cusum_reset: CUSUM reset interval. Defaults to config.

    Returns:
        dict with keys:
          'df': DataFrame with statistical feature columns added.
          'columns_added': list of new column names.
          'report': dict of per-signal summary.
    """
    if zscore_windows is None:
        zscore_windows = STAT_ZSCORE_WINDOWS
    if pctrank_windows is None:
        pctrank_windows = STAT_PCTRANK_WINDOWS
    if cusum_reset is None:
        cusum_reset = CUSUM_RESET_INTERVAL_S

    df = df.copy()
    columns_added = []
    report = {}

    for sig_def in _SOURCE_SIGNALS:
        sig_name = sig_def["name"]
        source_col = _resolve_source(df, sig_def)

        if source_col is None:
            report[sig_name] = {"source": None, "n_features": 0}
            continue

        source = df[source_col]
        sig_columns = []

        # Z-scores
        for w in zscore_windows:
            col = f"{sig_name}_zscore_{w}"
            df[col] = compute_zscore(source, w)
            sig_columns.append(col)

        # Trailing percentile ranks
        for w in pctrank_windows:
            col = f"{sig_name}_pctrank_{w}"
            df[col] = compute_trailing_pctrank(source, w)
            sig_columns.append(col)

        # CUSUM (uses 300s as the baseline mean window)
        col = f"{sig_name}_cusum"
        df[col] = compute_cusum(source, window_for_mean=300,
                                reset_interval=cusum_reset)
        sig_columns.append(col)

        columns_added.extend(sig_columns)
        report[sig_name] = {
            "source": source_col,
            "n_features": len(sig_columns),
        }

    return {"df": df, "columns_added": columns_added, "report": report}
