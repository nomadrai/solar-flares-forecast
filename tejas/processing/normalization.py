"""
TEJAS Normalization
====================
Prepare signals for downstream feature engineering by applying light,
non-destructive transformations that compress dynamic range and enable
cross-day comparison.

Transforms applied:
  1. log1p — compresses dynamic range (SoLEXS spans 0–3812 c/s).
     Uses log1p(x) = log(1+x) to handle zeros safely.
  2. Percentile rank — day-relative intensity (0.0 = minimum, 1.0 = max
     observed that day).  Useful for normalizing across days with very
     different activity levels.

Design decisions:
  - Tree-based models (XGBoost, LightGBM) are scale-invariant, so we
    normalize lightly and keep raw values accessible.
  - No min-max scaling (destroyed by flare peaks, meaningless for Poisson).
  - log1p instead of log (handles CdTe's 87% zeros).
  - Creates *_log and *_pctrank columns; raw/clean values untouched.
"""

import numpy as np
import pandas as pd


# ── Core Transforms ─────────────────────────────────────────────────────────


def log1p_transform(series):
    """Apply log(1 + x) transform to a signal.

    Safe for zero and near-zero values.  NaN is preserved.

    Args:
        series: pd.Series of non-negative values.

    Returns:
        pd.Series — log1p-transformed signal.
    """
    return np.log1p(series.clip(lower=0.0))


def percentile_rank(series):
    """Compute day-level percentile rank (0.0 to 1.0) for a signal.

    Each non-NaN value is ranked relative to all other non-NaN values
    in the series.  NaN values remain NaN.

    Args:
        series: pd.Series of values.

    Returns:
        pd.Series — percentile ranks in [0.0, 1.0].
    """
    return series.rank(method="average", pct=True)


# ── Day-Level Application ──────────────────────────────────────────────────


# Columns to normalize and their source (prefer _clean, then _net, then raw)
_NORM_TARGETS = [
    # (output_base, preferred_source, fallback_source)
    ("solexs_counts",     "solexs_counts_clean",     "solexs_counts"),
    ("cdte_count_rate",   "cdte_count_rate_clean",   "cdte_count_rate"),
    ("czt_net_rate",      "czt_net_rate",            "czt_count_rate"),
]


def normalize_day(df):
    """Apply all normalization transforms to a unified day DataFrame.

    Creates *_log and *_pctrank columns for each target signal.

    Args:
        df: DataFrame with clean/processed signal columns.

    Returns:
        dict with keys:
          'df': DataFrame with normalization columns added.
          'stats': dict of per-column normalization statistics.
          'report': summary dict.
    """
    df = df.copy()
    stats = {}
    columns_added = []

    for output_base, preferred, fallback in _NORM_TARGETS:
        # Select source column
        if preferred in df.columns:
            source_col = preferred
        elif fallback in df.columns:
            source_col = fallback
        else:
            continue

        source = df[source_col]

        # log1p transform
        log_col = f"{output_base}_log"
        df[log_col] = log1p_transform(source)
        columns_added.append(log_col)

        # Percentile rank
        pctrank_col = f"{output_base}_pctrank"
        df[pctrank_col] = percentile_rank(source)
        columns_added.append(pctrank_col)

        # Compute and store statistics for this column
        valid = source.dropna()
        col_stats = {}
        if len(valid) > 0:
            col_stats = {
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "median": float(valid.median()),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "n_valid": int(len(valid)),
                "n_nan": int(source.isna().sum()),
            }
        else:
            col_stats = {
                "mean": np.nan, "std": np.nan, "median": np.nan,
                "min": np.nan, "max": np.nan, "n_valid": 0,
                "n_nan": int(len(source)),
            }
        stats[output_base] = col_stats

    report = {
        "columns_added": columns_added,
        "n_columns": len(columns_added),
    }

    return {"df": df, "stats": stats, "report": report}


def compute_normalization_stats(df):
    """Compute day-level normalization statistics without modifying df.

    Useful for inspecting the data distribution before deciding on
    normalization strategy.

    Args:
        df: Unified day DataFrame.

    Returns:
        dict mapping column_base → stats dict (mean, std, median, etc.).
    """
    result = normalize_day(df)
    return result["stats"]
