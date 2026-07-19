"""
TEJAS Credible-Event Filter (v2 forecasting relabelling)
========================================================
The raw master catalogue holds 1,751 events, but ~86% are hard-only noise
blips (NULL intensity) or sub-noise-floor "faint" soft events.  Training and
scoring the forecaster on those caps discrimination near random.

A *credible* event is one worth alerting on:

    credible = (has_soft AND peak_solexs_counts >= FORECAST_EVENT_MIN_PEAK_COUNTS)
               OR goes_matched              # externally confirmed by NOAA GOES

The GOES-match arm rescues genuine flares that TEJAS detected only faintly or
in the hard channel, so external truth widens recall rather than narrowing it.

Only the forecasting *targets* use this subset; the full catalogue still feeds
the AR-proxy activity-context features.
"""

import numpy as np
import pandas as pd

from tejas.config import (
    FORECAST_EVENT_MIN_PEAK_COUNTS,
    FORECAST_EVENT_REQUIRE_SOFT,
)


def credible_mask(catalogue):
    """Boolean Series: which catalogue rows are credible forecasting targets.

    Args:
        catalogue: master catalogue DataFrame.  Uses 'has_soft',
            'peak_solexs_counts', and (if present) 'goes_matched'.

    Returns:
        Boolean Series aligned to catalogue.index.
    """
    counts = pd.to_numeric(catalogue.get("peak_solexs_counts"), errors="coerce")
    strong_soft = counts >= FORECAST_EVENT_MIN_PEAK_COUNTS

    if FORECAST_EVENT_REQUIRE_SOFT and "has_soft" in catalogue.columns:
        has_soft = catalogue["has_soft"].astype(bool)
        strong_soft = strong_soft & has_soft

    strong_soft = strong_soft.fillna(False)

    if "goes_matched" in catalogue.columns:
        goes = catalogue["goes_matched"].fillna(False).astype(bool)
        return (strong_soft | goes)
    return strong_soft


def build_credible_catalogue(catalogue):
    """Return the credible subset of the catalogue (copy, index reset).

    Adds an ``is_credible`` column to the returned frame for transparency.
    """
    mask = credible_mask(catalogue)
    out = catalogue.loc[mask].copy().reset_index(drop=True)
    out["is_credible"] = True
    return out


def relabel_frame(df, credible_starts, horizons):
    """Recompute forecasting labels on one day's frame from credible starts.

    Uses a month-wide sorted array of credible flare start times so that a
    flare early on day N+1 correctly labels the tail of day N (the original
    per-day attachment left those rows as false negatives).  Vectorised via
    searchsorted — O(n log m) rather than the original O(n·m) loop.

    Overwrites ``label_minutes_to_next_flare`` and ``label_flare_within_{h}m``.
    Leaves flare_id / is_flare_active / flare_phase / AR-proxy / sig-labels
    untouched.

    Args:
        df: day frame with a 'datetime' column.
        credible_starts: array-like of credible flare start timestamps (UTC).
        horizons: list of horizon minutes, e.g. [5, 15, 30, 60].

    Returns:
        df (modified in place and returned).
    """
    ts = pd.to_datetime(df["datetime"])
    tz = getattr(ts.dt, "tz", None)
    ts_ns = ts.values.astype("datetime64[ns]")

    starts = pd.to_datetime(pd.Series(list(credible_starts)))
    # Align tz with the frame (frame may be tz-aware UTC; starts tz-naive).
    if tz is not None:
        starts = starts.dt.tz_localize(tz) if starts.dt.tz is None else starts.dt.tz_convert(tz)
    else:
        starts = starts.dt.tz_localize(None) if starts.dt.tz is not None else starts
    starts_ns = np.sort(starts.values.astype("datetime64[ns]"))

    n = len(df)
    minutes_to_next = np.full(n, np.nan)
    if len(starts_ns) > 0:
        # First start strictly greater than each timestamp (side='right').
        idx = np.searchsorted(starts_ns, ts_ns, side="right")
        has_future = idx < len(starts_ns)
        nxt = starts_ns[np.clip(idx, 0, len(starts_ns) - 1)]
        delta_min = (nxt - ts_ns) / np.timedelta64(1, "m")
        minutes_to_next = np.where(has_future, delta_min.astype(float), np.nan)

    df["label_minutes_to_next_flare"] = minutes_to_next
    for h in horizons:
        col = f"label_flare_within_{h}m"
        lab = (minutes_to_next <= h).astype(float)
        lab[np.isnan(minutes_to_next)] = 0.0
        df[col] = lab.astype(int)
    return df


def summarise(catalogue):
    """Human-readable dict of credible-filter counts (for logging/report)."""
    mask = credible_mask(catalogue)
    n = len(catalogue)
    n_cred = int(mask.sum())
    counts = pd.to_numeric(catalogue.get("peak_solexs_counts"), errors="coerce")
    has_soft = catalogue.get("has_soft", pd.Series(False, index=catalogue.index)).astype(bool)
    n_soft_strong = int(((counts >= FORECAST_EVENT_MIN_PEAK_COUNTS) & has_soft).fillna(False).sum())
    n_goes = int(catalogue.get("goes_matched", pd.Series(False, index=catalogue.index)).fillna(False).astype(bool).sum())
    n_goes_only = 0
    if "goes_matched" in catalogue.columns:
        goes = catalogue["goes_matched"].fillna(False).astype(bool)
        soft_strong = ((counts >= FORECAST_EVENT_MIN_PEAK_COUNTS) & has_soft).fillna(False)
        n_goes_only = int((goes & ~soft_strong).sum())
    return {
        "total_events": n,
        "credible_events": n_cred,
        "credible_fraction": n_cred / n if n else 0.0,
        "soft_strong": n_soft_strong,
        "goes_matched": n_goes,
        "goes_rescued_only": n_goes_only,
        "min_peak_counts": FORECAST_EVENT_MIN_PEAK_COUNTS,
    }
