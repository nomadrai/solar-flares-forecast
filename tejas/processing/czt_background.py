"""
TEJAS CZT Background Estimation & Subtraction
===============================================
Estimate and subtract the non-solar background from CZT data so the ML
model can distinguish solar signal from cosmic/particle background.

CZT's 32 cm² area picks up ~23 c/s of non-solar background (cosmic diffuse
X-ray background, charged particles from GCRs/SEPs, spacecraft-induced
secondaries).  During quiet periods this is 100% background; during strong
flares the solar signal dominates.

Algorithm — Quiet-Period Percentile Baseline:
  1. Identify quiet periods using SoLEXS as reference (SoLEXS below threshold
     → Sun is quiet → CZT signal is pure background).
  2. Estimate CZT background as the 25th percentile of CZT counts within
     a 1-hour sliding window that only includes quiet-period bins.
  3. Interpolate the background estimate across active (non-quiet) periods.
  4. Subtract: czt_net = max(czt − background, 0).

NOT applied to CdTe: its 0.5 cm² area has negligible background (~0.15 c/s).
"""

import numpy as np
import pandas as pd

from tejas.config import (
    CZT_BG_QUIET_THRESHOLD,
    CZT_BG_WINDOW_S,
    CZT_BG_PERCENTILE,
    CZT_BG_SIGNIFICANCE,
)


# ── Core Algorithm ──────────────────────────────────────────────────────────


def estimate_czt_background(
    df,
    quiet_threshold=None,
    window_s=None,
    percentile=None,
):
    """Estimate the non-solar CZT background level.

    Uses SoLEXS as the reference for quiet-period identification.  The
    background is estimated from quiet-period CZT data only, then
    interpolated across active periods.

    Args:
        df: Unified day DataFrame with at least 'solexs_counts_clean' (or
            'solexs_counts') and 'czt_count_rate_clean' (or 'czt_count_rate').
        quiet_threshold: SoLEXS c/s below which the Sun is "quiet".
            Defaults to config.CZT_BG_QUIET_THRESHOLD.
        window_s: Sliding window size in seconds for background estimation.
            Defaults to config.CZT_BG_WINDOW_S.
        percentile: Percentile of quiet-period CZT rates to use as
            background estimate.  Defaults to config.CZT_BG_PERCENTILE.

    Returns:
        dict with keys:
          'background': pd.Series — estimated background (c/s), same index.
          'quiet_mask': pd.Series[bool] — True for quiet-period bins.
          'bg_median': float — overall median background estimate.
          'bg_std': float — estimated background noise level.
    """
    if quiet_threshold is None:
        quiet_threshold = CZT_BG_QUIET_THRESHOLD
    if window_s is None:
        window_s = CZT_BG_WINDOW_S
    if percentile is None:
        percentile = CZT_BG_PERCENTILE

    # ── Select source columns ──────────────────────────────────────────
    solexs_col = _select_column(df, "solexs_counts_clean", "solexs_counts")
    czt_col = _select_column(df, "czt_count_rate_clean", "czt_count_rate")

    if solexs_col is None or czt_col is None:
        # Cannot estimate background without both signals
        return _empty_result(df)

    solexs = df[solexs_col]
    czt = df[czt_col]

    # ── Step 1: Identify quiet periods ─────────────────────────────────
    # Use a smoothed SoLEXS to avoid micro-fluctuations triggering
    # quiet/active boundaries on every bin.
    solexs_smooth = solexs.rolling(
        window=301, center=True, min_periods=100
    ).median()
    quiet_mask = (solexs_smooth < quiet_threshold) & solexs_smooth.notna()

    # Need a minimum fraction of quiet bins to estimate background
    quiet_fraction = quiet_mask.sum() / len(df)
    if quiet_fraction < 0.05:
        # Less than 5% quiet data — cannot reliably estimate background.
        # Fall back to the overall lower percentile of CZT.
        bg_value = czt.quantile(percentile / 100.0)
        if pd.isna(bg_value):
            return _empty_result(df)
        background = pd.Series(bg_value, index=df.index)
        bg_std = czt.std() if czt.notna().any() else 0.0
        return {
            "background": background,
            "quiet_mask": quiet_mask,
            "bg_median": float(bg_value),
            "bg_std": float(bg_std) if pd.notna(bg_std) else 0.0,
        }

    # ── Step 2: Estimate background from quiet-period CZT ─────────────
    # Set CZT to NaN during active periods, then compute rolling percentile
    czt_quiet_only = czt.copy()
    czt_quiet_only[~quiet_mask] = np.nan

    # Rolling percentile over a wide window (quiet bins only)
    background = czt_quiet_only.rolling(
        window=window_s, center=True, min_periods=max(10, window_s // 10)
    ).quantile(percentile / 100.0)

    # ── Step 3: Interpolate across active periods ──────────────────────
    # Fill NaN gaps in the background estimate (active periods, edges)
    background = background.interpolate(method="linear", limit_direction="both")

    # Final fallback: if still NaN at edges, use nearest valid value
    background = background.ffill().bfill()

    # If background is entirely NaN (shouldn't happen), fall back
    if background.isna().all():
        return _empty_result(df)

    # Compute background statistics
    bg_median = float(background.median())
    # Estimate background noise from quiet-period residuals
    quiet_residuals = czt[quiet_mask] - background[quiet_mask]
    bg_std = float(quiet_residuals.std()) if len(quiet_residuals) > 10 else 0.0
    if pd.isna(bg_std):
        bg_std = 0.0

    return {
        "background": background,
        "quiet_mask": quiet_mask,
        "bg_median": bg_median,
        "bg_std": bg_std,
    }


# ── Subtraction & Flagging ──────────────────────────────────────────────────


def subtract_czt_background(df, quiet_threshold=None, window_s=None,
                             percentile=None, significance=None):
    """Estimate CZT background and subtract it from the signal.

    Adds three new columns to the DataFrame:
      - czt_background:  estimated non-solar background (c/s)
      - czt_net_rate:    background-subtracted CZT signal (clamped ≥ 0)
      - czt_above_bg:    bool — True when net signal exceeds significance
                         threshold (likely solar signal present)

    Args:
        df: Unified day DataFrame.
        quiet_threshold: SoLEXS threshold for quiet-period identification.
        window_s: Sliding window for background estimation.
        percentile: Percentile for background estimation.
        significance: Multiplier for background noise to set the
            "above background" threshold.  Defaults to config.

    Returns:
        dict with keys:
          'df': DataFrame with czt_background, czt_net_rate, czt_above_bg.
          'report': dict with background statistics.
    """
    if significance is None:
        significance = CZT_BG_SIGNIFICANCE

    df = df.copy()

    # Estimate background
    bg_result = estimate_czt_background(
        df,
        quiet_threshold=quiet_threshold,
        window_s=window_s,
        percentile=percentile,
    )

    background = bg_result["background"]
    bg_std = bg_result["bg_std"]

    # Select CZT source column
    czt_col = _select_column(df, "czt_count_rate_clean", "czt_count_rate")

    # Add background column
    df["czt_background"] = background

    if czt_col is not None:
        czt = df[czt_col]
        # Background-subtracted signal (clamp at 0)
        df["czt_net_rate"] = (czt - background).clip(lower=0.0)
        # Flag bins where net signal exceeds significance threshold
        above_threshold = significance * max(bg_std, 1.0)
        df["czt_above_bg"] = df["czt_net_rate"] > above_threshold
    else:
        df["czt_net_rate"] = np.nan
        df["czt_above_bg"] = False

    report = {
        "bg_median": bg_result["bg_median"],
        "bg_std": bg_std,
        "quiet_fraction": float(bg_result["quiet_mask"].sum()) / len(df),
        "above_bg_fraction": (
            float(df["czt_above_bg"].sum()) / len(df) if czt_col else 0.0
        ),
    }

    return {"df": df, "report": report}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _select_column(df, preferred, fallback):
    """Return the name of the first available column."""
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    return None


def _empty_result(df):
    """Return a no-op background result when estimation is impossible."""
    return {
        "background": pd.Series(np.nan, index=df.index),
        "quiet_mask": pd.Series(False, index=df.index),
        "bg_median": np.nan,
        "bg_std": 0.0,
    }
