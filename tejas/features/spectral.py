"""
TEJAS Spectral Features
========================
SoLEXS PI-derived spectral features that exploit energy-resolved
information for flare precursor detection.

Spectral hardening — the hardness ratio increasing before flare onset —
is a well-known precursor signature.  The EDA phase confirmed this pattern
in several flare events.  These features give the ML model access to
spectral evolution information that total-counts features alone cannot
provide.

Features:
  - Hardness ratio rolling mean and derivatives
  - Band ratios (mid/low, high/mid) and their derivatives
  - Spectral slope proxy and its derivative

All operations are strictly causal (trailing windows only).
"""

import numpy as np
import pandas as pd

from tejas.config import ZERO_COUNT_EPS


# ── Source Column Resolution ────────────────────────────────────────────────

_SPECTRAL_SOURCES = {
    "counts_low":  ["solexs_counts_low_smooth", "solexs_counts_low_clean",
                    "solexs_counts_low"],
    "counts_mid":  ["solexs_counts_mid_smooth", "solexs_counts_mid_clean",
                    "solexs_counts_mid"],
    "counts_high": ["solexs_counts_high_smooth", "solexs_counts_high_clean",
                    "solexs_counts_high"],
    "hardness":    ["solexs_hardness_ratio_smooth",
                    "solexs_hardness_ratio_clean",
                    "solexs_hardness_ratio"],
}

# Hardness ratio rolling windows (subset of temporal windows; focused
# on precursor-relevant time scales)
_HR_ROLLING_WINDOWS = [30, 60, 300]
_HR_GRADIENT_HORIZONS = [30, 60, 300]


def _resolve(df, priority_list):
    """Return the first available column name from the priority list."""
    for col in priority_list:
        if col in df.columns:
            return col
    return None


# ── Core Feature Functions ──────────────────────────────────────────────────


def _safe_ratio(numerator, denominator, eps=None):
    """Compute ratio with denominator floor to prevent divide-by-zero.

    Args:
        numerator: pd.Series.
        denominator: pd.Series.
        eps: Floor value. Defaults to config.ZERO_COUNT_EPS.

    Returns:
        pd.Series — ratio values.
    """
    if eps is None:
        eps = ZERO_COUNT_EPS
    return numerator / denominator.clip(lower=eps)


def _safe_log(series, eps=None):
    """Compute log with floor to handle zeros safely.

    Args:
        series: pd.Series of non-negative values.
        eps: Floor value. Defaults to config.ZERO_COUNT_EPS.

    Returns:
        pd.Series — log values.
    """
    if eps is None:
        eps = ZERO_COUNT_EPS
    return np.log(series.clip(lower=eps))


# ── Day-Level Application ──────────────────────────────────────────────────


def compute_spectral_features(df):
    """Compute all spectral features for a unified day DataFrame.

    Creates new columns:
      solexs_hr_rmean_{w}          — rolling mean of hardness ratio
      solexs_hr_deriv1             — rate of spectral hardening
      solexs_hr_grad_{h}           — HR gradient over horizon
      solexs_band_ratio_mid_low    — mid/low energy ratio
      solexs_band_ratio_high_mid   — high/mid energy ratio
      solexs_band_ratio_mid_low_deriv1  — temporal change in mid/low
      solexs_band_ratio_high_mid_deriv1 — temporal change in high/mid
      solexs_spectral_slope        — approximate power-law index proxy
      solexs_spectral_slope_deriv1 — rate of spectral slope change

    Args:
        df: DataFrame with SoLEXS spectral columns.

    Returns:
        dict with keys:
          'df': DataFrame with spectral feature columns added.
          'columns_added': list of new column names.
          'report': summary dict.
    """
    df = df.copy()
    columns_added = []

    # Resolve source columns
    hr_col = _resolve(df, _SPECTRAL_SOURCES["hardness"])
    low_col = _resolve(df, _SPECTRAL_SOURCES["counts_low"])
    mid_col = _resolve(df, _SPECTRAL_SOURCES["counts_mid"])
    high_col = _resolve(df, _SPECTRAL_SOURCES["counts_high"])

    has_hr = hr_col is not None
    has_bands = all(c is not None for c in [low_col, mid_col, high_col])

    # ── Hardness ratio features ────────────────────────────────────────

    if has_hr:
        hr = df[hr_col]

        # Rolling mean of hardness ratio
        for w in _HR_ROLLING_WINDOWS:
            col = f"solexs_hr_rmean_{w}"
            df[col] = hr.rolling(window=w, min_periods=1).mean()
            columns_added.append(col)

        # First derivative of hardness ratio (rate of spectral hardening)
        col = "solexs_hr_deriv1"
        df[col] = hr.diff(1)
        columns_added.append(col)

        # Gradient of HR over multiple horizons
        for h in _HR_GRADIENT_HORIZONS:
            col = f"solexs_hr_grad_{h}"
            df[col] = (hr - hr.shift(h)) / h
            columns_added.append(col)

    # ── Band ratio features ────────────────────────────────────────────

    if has_bands:
        low = df[low_col]
        mid = df[mid_col]
        high = df[high_col]

        # Mid-to-low ratio
        col = "solexs_band_ratio_mid_low"
        ratio_ml = _safe_ratio(mid, low)
        df[col] = ratio_ml
        columns_added.append(col)

        # High-to-mid ratio
        col = "solexs_band_ratio_high_mid"
        ratio_hm = _safe_ratio(high, mid)
        df[col] = ratio_hm
        columns_added.append(col)

        # Derivatives of band ratios
        col = "solexs_band_ratio_mid_low_deriv1"
        df[col] = ratio_ml.diff(1)
        columns_added.append(col)

        col = "solexs_band_ratio_high_mid_deriv1"
        df[col] = ratio_hm.diff(1)
        columns_added.append(col)

        # Spectral slope proxy: (log(high) - log(low)) / delta_energy
        # Approximate energy centers: low ~3.9 keV, high ~16 keV
        # delta_E ≈ 12.1 keV (used as a scaling constant)
        delta_e = 12.1
        col = "solexs_spectral_slope"
        slope = (_safe_log(high) - _safe_log(low)) / delta_e
        df[col] = slope
        columns_added.append(col)

        col = "solexs_spectral_slope_deriv1"
        df[col] = slope.diff(1)
        columns_added.append(col)

    report = {
        "hr_source": hr_col,
        "band_sources": {
            "low": low_col, "mid": mid_col, "high": high_col
        } if has_bands else None,
        "n_features": len(columns_added),
    }

    return {"df": df, "columns_added": columns_added, "report": report}
