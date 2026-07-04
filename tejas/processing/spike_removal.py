"""
TEJAS Spike Removal
====================
Detect and replace spurious single-bin spikes (cosmic ray hits, telemetry
glitches) while preserving real solar flare signatures.

Algorithm — Isolated Spike Detector:
  1. Compute rolling median (robust local baseline).
  2. Compute residuals = value − local median.
  3. Flag bins where |residual| > n_sigma × MAD (Median Absolute Deviation).
  4. Apply isolation test: only replace spikes that are "isolated" (fewer
     than min_consecutive consecutive flagged bins).  Real flares produce
     sustained elevations; cosmic rays hit 1-2 bins.
  5. Replace isolated spikes with local median.

Design decisions:
  - Median + MAD instead of mean + std — handles heavily skewed Poisson
    distributions (87% zeros in CdTe).
  - Operates per-segment to avoid crossing data gaps.
  - Creates *_clean columns alongside originals; never overwrites raw data.
  - SoLEXS spike mask is propagated to all band columns (Q1 recommendation).
"""

import numpy as np
import pandas as pd

from tejas.config import SPIKE_REMOVAL


# ── Core Algorithm ──────────────────────────────────────────────────────────


def remove_spikes(series, window=61, n_sigma=5, min_consecutive=3):
    """Detect and replace isolated spikes in a 1D signal.

    Uses a rolling-median + MAD approach with an isolation test to
    distinguish cosmic-ray hits (1-2 bins) from real flare rises (≥3 bins).

    Args:
        series: pd.Series of count rates (may contain NaN).
        window: Rolling-median window size in samples (must be odd).
        n_sigma: Threshold multiplier for MAD-based spike detection.
        min_consecutive: Minimum length of a consecutive flagged run
            to be considered a real signal (preserved, not removed).

    Returns:
        dict with keys:
          'cleaned': pd.Series — spike-replaced signal.
          'spike_mask': pd.Series[bool] — True at positions of removed spikes.
          'n_spikes': int — number of individual spike bins removed.
    """
    if window % 2 == 0:
        window += 1

    cleaned = series.copy()
    spike_mask = pd.Series(False, index=series.index)

    # Nothing to do on all-NaN or very short series
    valid_mask = series.notna()
    if valid_mask.sum() < window:
        return {"cleaned": cleaned, "spike_mask": spike_mask, "n_spikes": 0}

    # Step 1: Rolling median baseline (centered, min_periods handles edges)
    local_median = series.rolling(
        window=window, center=True, min_periods=window // 3
    ).median()

    # Step 2: Residuals
    residuals = series - local_median

    # Step 3: Rolling MAD (Median Absolute Deviation)
    abs_residuals = residuals.abs()
    local_mad = abs_residuals.rolling(
        window=window, center=True, min_periods=window // 3
    ).median()

    # Prevent zero MAD (constant signal regions) from flagging everything.
    # For zero-inflated Poisson data (CdTe: 87% zeros), the rolling median
    # is 0 and MAD is 0.  A count of 1 is a normal Poisson fluctuation,
    # not a spike.  We set a meaningful floor based on the signal's
    # interquartile range so only truly extreme values are flagged.
    global_mad = abs_residuals.median()
    global_iqr = series.quantile(0.75) - series.quantile(0.25)
    if pd.isna(global_mad) or global_mad == 0:
        global_mad = max(global_iqr, 1.0)
    mad_floor = max(global_mad * 0.1, 1.0)
    local_mad = local_mad.clip(lower=mad_floor)

    # Flag candidate spikes: residual exceeds threshold
    # Scale MAD to approximate std: std ≈ 1.4826 × MAD for normal data
    threshold = n_sigma * 1.4826 * local_mad

    # Additional absolute-minimum threshold: a residual below this value
    # is NEVER considered a spike regardless of the statistical test.
    # This protects normal Poisson fluctuations in sparse detectors
    # (e.g., CdTe count of 1 when baseline is 0).
    abs_min_threshold = max(series.quantile(0.99) * 0.5, 5.0) if series.notna().any() else 5.0
    threshold = threshold.clip(lower=abs_min_threshold)

    candidates = (residuals.abs() > threshold) & valid_mask

    # Step 4: Isolation test — only remove runs shorter than min_consecutive
    spike_mask = _apply_isolation_test(candidates, min_consecutive)

    # Step 5: Replace isolated spikes with local median
    cleaned[spike_mask] = local_median[spike_mask]

    n_spikes = int(spike_mask.sum())

    return {"cleaned": cleaned, "spike_mask": spike_mask, "n_spikes": n_spikes}


def _apply_isolation_test(candidates, min_consecutive):
    """Keep only isolated flagged runs (shorter than min_consecutive).

    Consecutive runs of flagged bins that are ≥ min_consecutive are
    assumed to be real signal (e.g., flare onset) and are UN-flagged.
    Only short isolated runs (cosmic rays, glitches) remain flagged.

    Args:
        candidates: pd.Series[bool] — candidate spike flags.
        min_consecutive: Minimum run length to preserve as real signal.

    Returns:
        pd.Series[bool] — refined spike mask (True = spike to remove).
    """
    mask = candidates.copy()
    values = candidates.values.astype(bool)

    if not values.any():
        return mask

    # Identify runs of consecutive True values
    # Use diff to find run boundaries
    padded = np.concatenate(([False], values, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for start, end in zip(starts, ends):
        run_length = end - start
        if run_length >= min_consecutive:
            # This is a sustained elevation — likely real signal, UN-flag it
            mask.iloc[start:end] = False

    return mask


# ── Day-Level Application ──────────────────────────────────────────────────


# Mapping from unified DataFrame columns to their instrument config key
_COLUMN_INSTRUMENT_MAP = {
    "solexs_counts": "solexs",
    "cdte_count_rate": "cdte",
    "czt_count_rate": "czt",
}

# SoLEXS band columns that share the spike mask from solexs_counts (Q1)
_SOLEXS_BAND_COLUMNS = [
    "solexs_counts_low",
    "solexs_counts_mid",
    "solexs_counts_high",
    "solexs_hardness_ratio",
]


def remove_spikes_day(df):
    """Apply spike removal to all signal columns in a unified day DataFrame.

    For SoLEXS, the spike mask from solexs_counts is propagated to all
    band columns and hardness ratio (Q1 recommendation: a cosmic ray
    spike in total counts is a spike in all bands).

    Creates new columns with '_clean' suffix alongside originals.

    Args:
        df: Unified day DataFrame (from Phase 2 Parquet).

    Returns:
        dict with keys:
          'df': DataFrame with *_clean columns added.
          'report': dict of per-column spike removal statistics.
    """
    df = df.copy()
    report = {}

    # ── Primary signal columns ──────────────────────────────────────────
    solexs_spike_mask = None

    for col, instrument in _COLUMN_INSTRUMENT_MAP.items():
        if col not in df.columns:
            continue

        params = SPIKE_REMOVAL[instrument]
        result = remove_spikes(
            df[col],
            window=params["window"],
            n_sigma=params["n_sigma"],
            min_consecutive=params["min_consecutive"],
        )

        clean_col = f"{col}_clean"
        df[clean_col] = result["cleaned"]

        report[col] = {
            "n_spikes": result["n_spikes"],
            "pct_spikes": (
                result["n_spikes"] / df[col].notna().sum() * 100
                if df[col].notna().sum() > 0 else 0.0
            ),
        }

        # Save SoLEXS mask for band propagation
        if instrument == "solexs":
            solexs_spike_mask = result["spike_mask"]

    # ── Propagate SoLEXS spike mask to band columns (Q1) ────────────────
    if solexs_spike_mask is not None:
        # For band counts: replace with NaN at spike positions
        # (we don't have a reliable local median for individual bands
        # because they are noisier — NaN is safer)
        solexs_median = df["solexs_counts"].rolling(
            window=SPIKE_REMOVAL["solexs"]["window"],
            center=True,
            min_periods=SPIKE_REMOVAL["solexs"]["window"] // 3,
        ).median()

        for band_col in _SOLEXS_BAND_COLUMNS:
            if band_col not in df.columns:
                continue

            clean_col = f"{band_col}_clean"
            df[clean_col] = df[band_col].copy()

            if band_col == "solexs_hardness_ratio":
                # For ratio: recompute from cleaned high/low if available
                if ("solexs_counts_high_clean" in df.columns and
                        "solexs_counts_low_clean" in df.columns):
                    low_safe = df["solexs_counts_low_clean"].clip(lower=0.5)
                    df[clean_col] = df["solexs_counts_high_clean"] / low_safe
                # Also NaN at spike positions
                df.loc[solexs_spike_mask, clean_col] = np.nan
            else:
                # For band counts: compute band-specific local median and
                # replace at spike positions
                band_median = df[band_col].rolling(
                    window=SPIKE_REMOVAL["solexs"]["window"],
                    center=True,
                    min_periods=SPIKE_REMOVAL["solexs"]["window"] // 3,
                ).median()
                df.loc[solexs_spike_mask, clean_col] = band_median[solexs_spike_mask]

            n_band_spikes = int(solexs_spike_mask.sum())
            report[band_col] = {
                "n_spikes": n_band_spikes,
                "pct_spikes": (
                    n_band_spikes / df[band_col].notna().sum() * 100
                    if df[band_col].notna().sum() > 0 else 0.0
                ),
                "mask_source": "solexs_counts",
            }

    return {"df": df, "report": report}
