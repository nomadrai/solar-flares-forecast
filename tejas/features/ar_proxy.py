"""
TEJAS Active Region (AR) Proxy Features (Fix 7)
================================================
Derived from the TEJAS flare history catalogue (Phase 5 master catalogue).

Since Aditya-L1 has no magnetograph, these features use the recent flare
history as a proxy for how "energized" the AR is:
  * Short inter-flare intervals   → active AR → true alerts more likely
  * Long quiet periods            → inactive AR → precursor-like signals
                                    are likely noise

Features added (all strictly causal — only look at flare events that have
already completed):
  * ar_flares_in_last_6h   — number of past flares in trailing 6h
  * ar_flares_in_last_12h  — number of past flares in trailing 12h
  * ar_time_since_last_s   — seconds since the last completed flare
  * ar_flare_rate_6h       — flares per hour in trailing 6h
  * ar_max_peak_last_6h    — peak SoLEXS counts of the strongest recent flare
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    AR_PROXY_SHORT_WINDOW_H,
    AR_PROXY_LONG_WINDOW_H,
    DETECTION_DIR,
)


AR_FEATURE_NAMES = [
    "ar_flares_in_last_6h",
    "ar_flares_in_last_12h",
    "ar_time_since_last_s",
    "ar_flare_rate_6h",
    "ar_max_peak_last_6h",
]


def _fill_neutral(df):
    for col in AR_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = np.float32(0.0)
    df["ar_time_since_last_s"] = df["ar_time_since_last_s"].fillna(86400.0)
    return df


def _load_catalogue(path):
    """Load the Phase 5 master catalogue.  Returns None if missing."""
    if path is None:
        path = Path(DETECTION_DIR) / "master_catalogue.csv"
    else:
        path = Path(path)
    if not path.exists():
        return None
    try:
        cat = pd.read_csv(
            path,
            parse_dates=["start_time", "end_time", "peak_time"],
        )
    except (ValueError, KeyError):
        cat = pd.read_csv(path)
        for col in ("start_time", "end_time", "peak_time"):
            if col in cat.columns:
                cat[col] = pd.to_datetime(cat[col], errors="coerce")
    return cat


def compute_ar_proxy_features(df, catalogue_path=None):
    """Compute AR-proxy features from the master flare catalogue.

    Args:
        df: Phase 5 labelled feature Parquet (must have 'datetime' column).
        catalogue_path: Path to master_catalogue.csv.  Defaults to the
            Phase 5 default location.

    Returns:
        Dict with keys df, columns_added, report.
    """
    df = df.copy()
    added = list(AR_FEATURE_NAMES)

    cat = _load_catalogue(catalogue_path)
    if cat is None or len(cat) == 0:
        # No catalogue → neutral values (quiet Sun).
        for col in AR_FEATURE_NAMES:
            df[col] = np.float32(0.0)
        df["ar_time_since_last_s"] = np.float32(86400.0)  # 24h if unknown
        return {"df": df, "columns_added": added,
                "report": {"warning": "no catalogue"}}

    # Sort by end_time so we can do causal lookback.
    cat = cat.dropna(subset=["end_time"]).copy()
    cat = cat.sort_values("end_time").reset_index(drop=True)
    end_times = cat["end_time"].values.astype("datetime64[ns]")
    peak_counts = (
        cat["peak_solexs_counts"].fillna(0).values
        if "peak_solexs_counts" in cat.columns
        else np.zeros(len(cat))
    )

    # Timestamps in ns for fast comparison
    df_times = pd.to_datetime(df["datetime"])
    # Drop tz — end_times is naive already
    try:
        df_times = df_times.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    time_ns = df_times.values.astype("datetime64[ns]").astype("int64")
    end_ns = end_times.astype("int64")

    w_short_ns = int(AR_PROXY_SHORT_WINDOW_H * 3600 * 1e9)
    w_long_ns = int(AR_PROXY_LONG_WINDOW_H * 3600 * 1e9)

    # Compute at minute resolution then forward-fill to full grid.
    if len(time_ns) == 0:
        for col in AR_FEATURE_NAMES:
            df[col] = np.float32(0.0)
        return {"df": df, "columns_added": added,
                "report": {"warning": "empty df"}}

    t_min = df_times.min()
    t_max = df_times.max()
    minute_times = pd.date_range(t_min, t_max, freq="1min")
    if len(minute_times) == 0:
        minute_times = pd.DatetimeIndex([t_min])
    minute_ns = minute_times.values.astype("datetime64[ns]").astype("int64")

    n_min = len(minute_ns)
    n6 = np.zeros(n_min, dtype=np.int32)
    n12 = np.zeros(n_min, dtype=np.int32)
    since = np.full(n_min, 86400.0, dtype=np.float32)  # default: 24h
    rate = np.zeros(n_min, dtype=np.float32)
    peak6 = np.zeros(n_min, dtype=np.float32)

    # searchsorted approach for efficiency
    end_sorted = np.sort(end_ns)
    # index of last event with end_time <= t
    last_idx = np.searchsorted(end_sorted, minute_ns, side="right") - 1

    for i in range(n_min):
        t = minute_ns[i]
        li = last_idx[i]
        if li < 0:
            continue

        # Time since last completed flare (seconds)
        since[i] = float((t - end_sorted[li]) / 1e9)

        # Range for 6h window
        lo6 = np.searchsorted(end_sorted, t - w_short_ns, side="left")
        c6 = max(0, li - lo6 + 1)
        n6[i] = c6
        rate[i] = c6 / AR_PROXY_SHORT_WINDOW_H if c6 > 0 else 0.0

        # Range for 12h window
        lo12 = np.searchsorted(end_sorted, t - w_long_ns, side="left")
        c12 = max(0, li - lo12 + 1)
        n12[i] = c12

        # Peak-in-6h — need to map from the sorted-end-index range back to
        # peak counts.  Because we sorted end_times once, build a mapping.
        # For speed, look at cat rows whose end_time is in [t-6h, t].
        if c6 > 0:
            # The sorted permutation ordering
            # We already have end_sorted matching cat sorted by end_time,
            # and peak_counts is aligned to cat's sort order (already sorted).
            peak6[i] = float(peak_counts[lo6: li + 1].max()) if c6 > 0 else 0.0

    # Build minute-indexed Series and reindex to df timestamps (ffill).
    minute_idx = pd.DatetimeIndex(minute_times)
    def _reindex(vals):
        s = pd.Series(vals, index=minute_idx)
        return s.reindex(df_times, method="ffill").values

    df["ar_flares_in_last_6h"] = _reindex(n6.astype(np.float32))
    df["ar_flares_in_last_12h"] = _reindex(n12.astype(np.float32))
    df["ar_time_since_last_s"] = _reindex(since)
    df["ar_flare_rate_6h"] = _reindex(rate)
    df["ar_max_peak_last_6h"] = _reindex(peak6)

    df = _fill_neutral(df)

    report = {
        "columns_added": added,
        "catalogue_path": str(_default_path_used(catalogue_path)),
        "n_events_in_catalogue": int(len(cat)),
        "mean_flares_in_last_6h": float(df["ar_flares_in_last_6h"].mean()),
        "max_flares_in_last_6h": float(df["ar_flares_in_last_6h"].max()),
        "mean_time_since_last_s": float(df["ar_time_since_last_s"].mean()),
    }
    return {"df": df, "columns_added": added, "report": report}


def _default_path_used(path):
    if path is not None:
        return Path(path)
    return Path(DETECTION_DIR) / "master_catalogue.csv"


def augment_day_with_ar_proxy(date_str, detection_dir=None,
                                catalogue_path=None, save=True):
    """Augment a Phase 5 labelled Parquet with AR-proxy features.

    Reads data/detection/YYYY/MM/{date_str}.parquet, adds AR-proxy
    features, and writes back to the same location.

    Args:
        date_str: 'YYYY-MM-DD'
        detection_dir: Directory containing Phase 5 parquets.  Defaults
            to DETECTION_DIR from config.
        catalogue_path: Path to master_catalogue.csv.  Defaults to the
            master_catalogue.csv inside detection_dir.
        save: If True, overwrite the input parquet in place.

    Returns:
        Dict with keys df, report.  None if the parquet doesn't exist.
    """
    detection_dir = Path(detection_dir) if detection_dir else Path(DETECTION_DIR)
    input_path = detection_dir / f"{date_str}.parquet"
    if not input_path.exists():
        return None

    df = pd.read_parquet(input_path)
    if "datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["datetime"]
    ):
        df["datetime"] = pd.to_datetime(df["datetime"])

    result = compute_ar_proxy_features(
        df, catalogue_path=(catalogue_path or (detection_dir / "master_catalogue.csv")),
    )
    df_out = result["df"]

    if save:
        df_out.to_parquet(input_path, index=False, engine="pyarrow")

    return {"df": df_out, "report": result["report"], "path": str(input_path)}


def augment_all_days_with_ar_proxy(detection_dir=None, catalogue_path=None):
    """Augment all Phase 5 parquets with AR-proxy features."""
    detection_dir = Path(detection_dir) if detection_dir else Path(DETECTION_DIR)
    reports = []
    parquets = sorted(detection_dir.glob("2*-*-*.parquet"))
    if not parquets:
        print(f"No detection parquets found in {detection_dir}")
        return reports
    for p in parquets:
        date_str = p.stem
        print(f"  AR-proxy: {date_str}...", end="")
        res = augment_day_with_ar_proxy(
            date_str, detection_dir=detection_dir,
            catalogue_path=catalogue_path, save=True,
        )
        if res is None:
            print(" [SKIP]")
            continue
        r = res["report"]
        print(
            f" ok  mean_flares_6h={r.get('mean_flares_in_last_6h', 0):.2f} "
            f"mean_since={r.get('mean_time_since_last_s', 0):.0f}s"
        )
        r["date"] = date_str
        reports.append(r)
    return reports
