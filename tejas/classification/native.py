"""
TEJAS Native Intensity Classifier
====================================
Assigns each soft-detected flare a TEJAS-native intensity class based on
its SoLEXS peak count rate.  Does not depend on GOES data — this is the
"always available" classification layer of Phase 6.

Bins are configurable via tejas.config.INTENSITY_BIN_EDGES and follow the
order-of-magnitude structure of GOES letter classes (each step ~4-5×).
"""

import numpy as np
import pandas as pd

from tejas.config import (
    INTENSITY_BIN_EDGES,
    INTENSITY_BIN_LABELS,
    INTENSITY_SCORE_MAX_COUNTS,
)


def classify_counts(peak_counts):
    """Return the native intensity class label for one or many peak counts.

    Args:
        peak_counts: Scalar or array-like of SoLEXS peak count rates (c/s).
            NaN inputs yield NaN outputs.

    Returns:
        Same shape as input.  For arrays, returns a pd.Series/np.ndarray
        of str labels (NaN where input was NaN).  For scalars, returns
        a str or float('nan').
    """
    scalar_input = np.isscalar(peak_counts)
    arr = np.atleast_1d(peak_counts).astype(float)

    labels = np.empty(arr.shape, dtype=object)
    nan_mask = np.isnan(arr)
    labels[nan_mask] = np.nan

    valid = ~nan_mask
    if valid.any():
        idx = np.digitize(arr[valid], INTENSITY_BIN_EDGES, right=False)
        for i, bin_idx in zip(np.where(valid)[0], idx):
            labels[i] = INTENSITY_BIN_LABELS[bin_idx]

    if scalar_input:
        val = labels[0]
        return val if isinstance(val, str) else float("nan")
    return labels


def intensity_score(peak_counts):
    """Compute a log-scaled score in [0, 1] from peak counts.

    score = log(counts + 1) / log(MAX_COUNTS)

    Values above INTENSITY_SCORE_MAX_COUNTS clip to 1.0.  Useful as a
    single-number severity feature for downstream models.

    Args:
        peak_counts: Scalar or array-like of SoLEXS peak count rates (c/s).

    Returns:
        Same shape as input.  NaN preserved.
    """
    scalar_input = np.isscalar(peak_counts)
    arr = np.atleast_1d(peak_counts).astype(float)

    with np.errstate(invalid="ignore"):
        scores = np.log1p(arr) / np.log(INTENSITY_SCORE_MAX_COUNTS)
        scores = np.clip(scores, 0.0, 1.0)
    scores[np.isnan(arr)] = np.nan

    if scalar_input:
        return float(scores[0])
    return scores


def classify_dataframe(df, peak_col="peak_solexs_counts"):
    """Add intensity_class and intensity_score columns to a catalogue DataFrame.

    Rows where peak_col is NaN (hard-only events) get NaN in both new
    columns — this preserves scientific integrity per the Phase 6 design.

    Args:
        df: DataFrame with a peak-counts column.
        peak_col: Name of the peak count column (default: peak_solexs_counts).

    Returns:
        Copy of df with 'intensity_class' and 'intensity_score' columns.
    """
    out = df.copy()
    out["intensity_class"] = classify_counts(out[peak_col].values)
    out["intensity_score"] = intensity_score(out[peak_col].values)
    return out
