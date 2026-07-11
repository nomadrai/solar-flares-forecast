"""
TEJAS Temporal Features
========================
Per-instrument temporal features computed from smoothed/cleaned signals.

All rolling windows are **strictly trailing (causal)** — the window at
time t only includes observations at times [t-w+1, t].  No future data
is ever used, preserving scientific integrity during inference/playback.

Features per source signal:
  - Rolling mean (multiple windows)
  - Rolling standard deviation (multiple windows)
  - First derivative (instantaneous rate of change)
  - Second derivative (acceleration)
  - Flux gradient over multiple horizons
  - Rate of rise (positive derivative, zeroed during decay)
  - Rate of decay (negative derivative, zeroed during rise)
"""

import numpy as np
import pandas as pd

from tejas.config import (
    FEATURE_ROLLING_WINDOWS,
    FEATURE_GRADIENT_HORIZONS,
    ZERO_COUNT_EPS,
)


# ── Source Column Resolution ────────────────────────────────────────────────

# Priority order: smoothed → cleaned → raw.
# Smoothed signals are best for derivative/rolling features because they
# remove high-frequency noise that would produce noisy gradients.
_SOURCE_SIGNALS = [
    {
        "name": "solexs",
        "preferred": "solexs_counts_smooth",
        "fallbacks": ["solexs_counts_clean", "solexs_counts"],
    },
    {
        # Fix 6: prefer Kalman-filtered CdTe (STAT_ERR-aware, fully causal).
        "name": "cdte",
        "preferred": "cdte_count_rate_kalman",
        "fallbacks": ["cdte_count_rate_smooth", "cdte_count_rate_clean",
                       "cdte_count_rate"],
    },
    {
        # CZT: net_rate (background-subtracted) is still the physically
        # cleanest input for temporal derivatives; Kalman is exposed as a
        # secondary fallback / feature input.
        "name": "czt",
        "preferred": "czt_net_rate",
        "fallbacks": ["czt_count_rate_kalman", "czt_count_rate_smooth",
                       "czt_count_rate_clean", "czt_count_rate"],
    },
]


def _resolve_source(df, signal_def):
    """Find the best available source column for a signal.

    Args:
        df: DataFrame to search.
        signal_def: Dict with 'preferred' and 'fallbacks' keys.

    Returns:
        Column name (str) or None if no source is available.
    """
    if signal_def["preferred"] in df.columns:
        return signal_def["preferred"]
    for fb in signal_def["fallbacks"]:
        if fb in df.columns:
            return fb
    return None


# ── Core Feature Functions ──────────────────────────────────────────────────


def compute_rolling_mean(series, window):
    """Trailing rolling mean (causal, no future leakage).

    Args:
        series: pd.Series of values.
        window: Window size in samples (seconds at 1s cadence).

    Returns:
        pd.Series — rolling mean.
    """
    return series.rolling(window=window, min_periods=1).mean()


def compute_rolling_std(series, window):
    """Trailing rolling standard deviation (causal).

    Args:
        series: pd.Series of values.
        window: Window size in samples.

    Returns:
        pd.Series — rolling std (ddof=1).
    """
    return series.rolling(window=window, min_periods=2).std(ddof=1)


def compute_derivative(series, order=1):
    """Compute the n-th order finite difference derivative.

    Uses diff(1) which computes x[t] - x[t-1], giving the instantaneous
    rate of change.  This is strictly causal (only uses past data).

    Args:
        series: pd.Series of values.
        order: Derivative order (1 or 2).

    Returns:
        pd.Series — derivative signal.
    """
    result = series
    for _ in range(order):
        result = result.diff(1)
    return result


def compute_gradient(series, horizon):
    """Compute flux gradient over a given time horizon.

    gradient = (x[t] - x[t - horizon]) / horizon

    This measures the average rate of change over the horizon, which is
    less noisy than the instantaneous derivative for longer time scales.

    Args:
        series: pd.Series of values.
        horizon: Lookback distance in samples.

    Returns:
        pd.Series — gradient signal.
    """
    return (series - series.shift(horizon)) / horizon


# ── Day-Level Application ──────────────────────────────────────────────────


def compute_temporal_features(df, windows=None, gradients=None):
    """Compute all temporal features for a unified day DataFrame.

    Creates new columns with naming convention:
      {signal}_rmean_{window}
      {signal}_rstd_{window}
      {signal}_deriv1
      {signal}_deriv2
      {signal}_grad_{horizon}
      {signal}_rise_rate
      {signal}_decay_rate

    Args:
        df: DataFrame with clean/smoothed signal columns.
        windows: List of rolling window sizes. Defaults to config.
        gradients: List of gradient horizons. Defaults to config.

    Returns:
        dict with keys:
          'df': DataFrame with temporal feature columns added.
          'columns_added': list of new column names.
          'report': dict of per-signal summary.
    """
    if windows is None:
        windows = FEATURE_ROLLING_WINDOWS
    if gradients is None:
        gradients = FEATURE_GRADIENT_HORIZONS

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

        # Rolling mean
        for w in windows:
            col = f"{sig_name}_rmean_{w}"
            df[col] = compute_rolling_mean(source, w)
            sig_columns.append(col)

        # Rolling std
        for w in windows:
            col = f"{sig_name}_rstd_{w}"
            df[col] = compute_rolling_std(source, w)
            sig_columns.append(col)

        # First derivative
        col = f"{sig_name}_deriv1"
        deriv1 = compute_derivative(source, order=1)
        df[col] = deriv1
        sig_columns.append(col)

        # Second derivative
        col = f"{sig_name}_deriv2"
        df[col] = compute_derivative(source, order=2)
        sig_columns.append(col)

        # Gradients over multiple horizons
        for h in gradients:
            col = f"{sig_name}_grad_{h}"
            df[col] = compute_gradient(source, h)
            sig_columns.append(col)

        # Rate of rise (positive derivative only)
        col = f"{sig_name}_rise_rate"
        df[col] = deriv1.clip(lower=0.0)
        sig_columns.append(col)

        # Rate of decay (negative derivative only)
        col = f"{sig_name}_decay_rate"
        df[col] = deriv1.clip(upper=0.0)
        sig_columns.append(col)

        columns_added.extend(sig_columns)
        report[sig_name] = {
            "source": source_col,
            "n_features": len(sig_columns),
        }

    return {"df": df, "columns_added": columns_added, "report": report}
