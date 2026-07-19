"""
TEJAS Long-Window Context Features (v2 optimization)
====================================================
The longest existing trailing window is 600 s, so the model cannot tell a
quiet-Sun morning from an active-region afternoon.  These features give it
hour-scale context — the discriminative signal the v2 plan targets:

  bg_pctrank_1h / _3h / _6h    percentile of current SoLEXS flux within the
                               trailing 1/3/6-hour distribution
  rise_consistency_300s        fraction of positive 1-s derivatives in the
                               last 5 minutes (sustained rise vs spike)
  time_elevated_s              seconds since flux was last at/below its
                               trailing 1-hour median (event-phase context)

All windows are strictly trailing (causal).  Rows are a regular 1-s grid,
so positional windows equal time windows.
"""

import numpy as np
import pandas as pd

from tejas.config import (
    LONG_CONTEXT_PCTRANK_WINDOWS_S,
    LONG_CONTEXT_RISE_WINDOW_S,
    LONG_CONTEXT_MEDIAN_WINDOW_S,
)


def _resolve_solexs(df):
    """Best available SoLEXS source column (same priority as statistical)."""
    for col in ("solexs_counts_smooth", "solexs_counts_clean", "solexs_counts"):
        if col in df.columns:
            return col
    raise KeyError("No SoLEXS counts column found for long-context features.")


def compute_long_context_features(df):
    """Add the 5 long-window context features.

    Args:
        df: DataFrame on the regular 1-s grid with a SoLEXS counts column.

    Returns:
        dict with keys 'df', 'columns_added', 'report'.
    """
    df = df.copy()
    added = []
    src = _resolve_solexs(df)
    flux = df[src].astype(float)

    # ── Background percentile ranks (trailing 1/3/6 h) ────────────────
    for w in LONG_CONTEXT_PCTRANK_WINDOWS_S:
        col = f"bg_pctrank_{w // 3600}h"
        df[col] = (
            flux.rolling(w, min_periods=max(60, w // 10))
            .rank(pct=True)
            .astype(np.float32)
        )
        added.append(col)

    # ── Rise consistency (fraction of positive derivatives, 5 min) ────
    deriv = flux.diff()
    pos = (deriv > 0).astype(float)
    pos[deriv.isna()] = np.nan
    w = LONG_CONTEXT_RISE_WINDOW_S
    df["rise_consistency_300s"] = (
        pos.rolling(w, min_periods=max(30, w // 5)).mean().astype(np.float32)
    )
    added.append("rise_consistency_300s")

    # ── Time since flux last at/below trailing 1-h median ─────────────
    w = LONG_CONTEXT_MEDIAN_WINDOW_S
    med = flux.rolling(w, min_periods=max(300, w // 10)).median()
    at_or_below = (flux <= med) | flux.isna() | med.isna()
    idx = np.arange(len(df), dtype=float)
    last_below = pd.Series(np.where(at_or_below.values, idx, np.nan))
    last_below = last_below.ffill()
    elapsed = idx - last_below.values          # NaN until first at/below row
    elapsed = np.nan_to_num(elapsed, nan=0.0)  # start-of-day: not elevated
    df["time_elevated_s"] = elapsed.astype(np.float32)
    added.append("time_elevated_s")

    report = {
        "source_column": src,
        "columns_added": added,
        "mean_bg_pctrank_6h": float(df["bg_pctrank_6h"].mean()),
        "mean_time_elevated_s": float(df["time_elevated_s"].mean()),
    }
    return {"df": df, "columns_added": added, "report": report}
