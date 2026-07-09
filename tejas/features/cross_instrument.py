"""
TEJAS Cross-Instrument Features
=================================
Features combining signals across SoLEXS, CdTe, and CZT to exploit
the multi-wavelength coverage (2–150 keV) that is TEJAS's unique
scientific advantage.

Key physical motivation — the Neupert effect:
  Hard X-ray emission (non-thermal electrons) peaks during the impulsive
  phase, while soft X-ray emission (heated plasma) peaks during the
  gradual phase.  The time lag between them is a fundamental diagnostic
  of flare energy transport.  The EDA phase confirmed 30-120s lags in
  several events.

Features:
  - Soft-to-hard X-ray ratios (SoLEXS/CdTe, SoLEXS/CZT, CdTe/CZT)
  - Ratio derivatives and rolling means
  - Cross-correlation lag and peak correlation (Neupert effect)

Cross-correlation is expensive, so it is computed every XCORR_COMPUTE_EVERY_S
seconds and forward-filled between computation points.
"""

import numpy as np
import pandas as pd

from tejas.config import (
    ZERO_COUNT_EPS,
    XCORR_WINDOW_S,
    XCORR_MAX_LAG_S,
    XCORR_COMPUTE_EVERY_S,
)


# ── Source Column Resolution ────────────────────────────────────────────────

_SOLEXS_PRIORITY = ["solexs_counts_smooth", "solexs_counts_clean",
                    "solexs_counts"]
_CDTE_PRIORITY = ["cdte_count_rate_smooth", "cdte_count_rate_clean",
                  "cdte_count_rate"]
_CZT_PRIORITY = ["czt_net_rate", "czt_count_rate_smooth",
                 "czt_count_rate_clean", "czt_count_rate"]


def _resolve(df, priority_list):
    """Return the first available column name from the priority list."""
    for col in priority_list:
        if col in df.columns:
            return col
    return None


def _safe_ratio(numerator, denominator, eps=None):
    """Compute ratio with denominator floor."""
    if eps is None:
        eps = ZERO_COUNT_EPS
    return numerator / denominator.clip(lower=eps)


# ── Cross-Correlation ──────────────────────────────────────────────────────


def _normalized_cross_correlation(x, y, max_lag):
    """Compute normalized cross-correlation between two 1D arrays.

    Returns the lag (in samples) that maximises the cross-correlation
    and the peak correlation value.  Positive lag means y leads x
    (y peaks first).

    Args:
        x: np.ndarray — signal A (e.g., SoLEXS).
        y: np.ndarray — signal B (e.g., CdTe).
        max_lag: Maximum lag to search (samples).

    Returns:
        (best_lag, peak_corr) — lag in samples and correlation [-1, 1].
        Returns (NaN, NaN) if insufficient valid data.
    """
    # Remove NaN and require minimum length
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < max_lag * 2:
        return np.nan, np.nan

    x_clean = x.copy()
    y_clean = y.copy()
    x_clean[~valid] = 0.0
    y_clean[~valid] = 0.0

    # Normalise
    x_std = np.std(x_clean)
    y_std = np.std(y_clean)
    if x_std < 1e-10 or y_std < 1e-10:
        return np.nan, np.nan

    x_norm = (x_clean - np.mean(x_clean)) / x_std
    y_norm = (y_clean - np.mean(y_clean)) / y_std

    n = len(x_norm)
    best_lag = 0
    best_corr = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.dot(x_norm[lag:], y_norm[:n - lag]) / n
        else:
            corr = np.dot(x_norm[:n + lag], y_norm[-lag:]) / n

        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


def compute_xcorr_features(df, solexs_col, cdte_col,
                           window_s=None, max_lag_s=None,
                           compute_every_s=None):
    """Compute sliding cross-correlation between SoLEXS and CdTe.

    The cross-correlation is computed at intervals of compute_every_s
    seconds to reduce computation cost, then forward-filled.

    Args:
        df: DataFrame with time-series data.
        solexs_col: Name of SoLEXS source column.
        cdte_col: Name of CdTe source column.
        window_s: Trailing window for cross-correlation.
        max_lag_s: Maximum lag to search (±).
        compute_every_s: Interval between computations.

    Returns:
        dict with 'lag' and 'xcorr' Series (same index as df).
    """
    if window_s is None:
        window_s = XCORR_WINDOW_S
    if max_lag_s is None:
        max_lag_s = XCORR_MAX_LAG_S
    if compute_every_s is None:
        compute_every_s = XCORR_COMPUTE_EVERY_S

    n = len(df)
    lag_values = np.full(n, np.nan)
    xcorr_values = np.full(n, np.nan)

    solexs = df[solexs_col].values
    cdte = df[cdte_col].values

    # Compute at regular intervals, starting after the first full window
    for i in range(window_s, n, compute_every_s):
        x_window = solexs[i - window_s:i]
        y_window = cdte[i - window_s:i]
        lag, corr = _normalized_cross_correlation(x_window, y_window,
                                                  max_lag_s)
        lag_values[i] = lag
        xcorr_values[i] = corr

    # Forward-fill between computation points
    lag_series = pd.Series(lag_values, index=df.index).ffill()
    xcorr_series = pd.Series(xcorr_values, index=df.index).ffill()

    return {"lag": lag_series, "xcorr": xcorr_series}


# ── Day-Level Application ──────────────────────────────────────────────────


def compute_cross_instrument_features(df):
    """Compute all cross-instrument features for a unified day DataFrame.

    Creates new columns:
      ratio_solexs_cdte             — soft/hard X-ray ratio
      ratio_solexs_czt              — soft/high-energy ratio
      ratio_cdte_czt                — hard X-ray spectral hardness
      ratio_solexs_cdte_deriv1      — temporal change in soft/hard ratio
      ratio_solexs_cdte_rmean_60    — rolling mean (60s)
      ratio_solexs_cdte_rmean_300   — rolling mean (300s)
      lag_solexs_cdte               — Neupert effect lag (seconds)
      xcorr_solexs_cdte             — peak cross-correlation value

    Args:
        df: DataFrame with multi-instrument signal columns.

    Returns:
        dict with keys:
          'df': DataFrame with cross-instrument feature columns added.
          'columns_added': list of new column names.
          'report': summary dict.
    """
    df = df.copy()
    columns_added = []

    solexs_col = _resolve(df, _SOLEXS_PRIORITY)
    cdte_col = _resolve(df, _CDTE_PRIORITY)
    czt_col = _resolve(df, _CZT_PRIORITY)

    has_solexs = solexs_col is not None
    has_cdte = cdte_col is not None
    has_czt = czt_col is not None

    # ── Ratio features ─────────────────────────────────────────────────

    if has_solexs and has_cdte:
        solexs = df[solexs_col]
        cdte = df[cdte_col]

        # Soft-to-hard ratio
        col = "ratio_solexs_cdte"
        ratio_sc = _safe_ratio(solexs, cdte)
        df[col] = ratio_sc
        columns_added.append(col)

        # Derivative of soft/hard ratio
        col = "ratio_solexs_cdte_deriv1"
        df[col] = ratio_sc.diff(1)
        columns_added.append(col)

        # Rolling means of soft/hard ratio
        for w in [60, 300]:
            col = f"ratio_solexs_cdte_rmean_{w}"
            df[col] = ratio_sc.rolling(window=w, min_periods=1).mean()
            columns_added.append(col)

    if has_solexs and has_czt:
        col = "ratio_solexs_czt"
        df[col] = _safe_ratio(df[solexs_col], df[czt_col])
        columns_added.append(col)

    if has_cdte and has_czt:
        col = "ratio_cdte_czt"
        df[col] = _safe_ratio(df[cdte_col], df[czt_col])
        columns_added.append(col)

    # ── Cross-correlation features (Neupert effect) ────────────────────

    if has_solexs and has_cdte:
        xcorr_result = compute_xcorr_features(df, solexs_col, cdte_col)

        col = "lag_solexs_cdte"
        df[col] = xcorr_result["lag"]
        columns_added.append(col)

        col = "xcorr_solexs_cdte"
        df[col] = xcorr_result["xcorr"]
        columns_added.append(col)

    report = {
        "solexs_source": solexs_col,
        "cdte_source": cdte_col,
        "czt_source": czt_col,
        "n_features": len(columns_added),
    }

    return {"df": df, "columns_added": columns_added, "report": report}
