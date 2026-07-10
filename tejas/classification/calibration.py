"""
TEJAS SoLEXS → GOES Calibration
================================
Optional module.  Fits an empirical mapping from SoLEXS peak count rate
to GOES 1-8 Å flux by cross-matching detected flare peaks with fetched
GOES XRS data.

Model: log10(flux) = a * log10(counts) + b
Fit:   HuberRegressor if sklearn is available, else numpy polyfit.

The resulting artifact (JSON) is persisted so that repeat classification
runs do not need to refit or refetch data.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    CALIBRATION_ARTIFACT,
    CALIBRATION_MIN_SAMPLES,
    CALIBRATION_MIN_R2,
    GOES_MATCH_TOLERANCE_S,
    GOES_REFERENCE_DIR,
)


def load_goes_reference(directory=None):
    """Load all GOES daily CSVs into a single time-sorted DataFrame.

    Args:
        directory: Root directory containing YYYY-MM-DD.csv files.
            Defaults to config.GOES_REFERENCE_DIR.

    Returns:
        DataFrame with columns [time_unix, flux_wm2], sorted by time.
        Empty DataFrame if no files found.
    """
    if directory is None:
        directory = GOES_REFERENCE_DIR

    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame(columns=["time_unix", "flux_wm2"])

    files = sorted(directory.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["time_unix", "flux_wm2"])

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["time_unix", "flux_wm2"])
    df = df.sort_values("time_unix").reset_index(drop=True)
    return df


def cross_match(catalogue, goes_df, tolerance_s=None):
    """Attach nearest GOES flux to each catalogue event's peak_time.

    Args:
        catalogue: DataFrame with 'peak_time' (datetime) and
            'peak_solexs_counts' (float) columns.  Rows with NaN counts
            are excluded (hard-only events).
        goes_df: DataFrame from load_goes_reference().
        tolerance_s: Maximum time offset in seconds.  Rows outside
            tolerance are dropped.  Defaults to
            config.GOES_MATCH_TOLERANCE_S.

    Returns:
        DataFrame with columns [peak_counts, goes_flux] for matched
        events only.
    """
    if tolerance_s is None:
        tolerance_s = GOES_MATCH_TOLERANCE_S

    if goes_df.empty:
        return pd.DataFrame(columns=["peak_counts", "goes_flux"])

    cat = catalogue.copy()
    cat["peak_time"] = pd.to_datetime(cat["peak_time"], utc=True).dt.tz_localize(None)
    cat = cat.dropna(subset=["peak_solexs_counts"])
    if cat.empty:
        return pd.DataFrame(columns=["peak_counts", "goes_flux"])

    cat["peak_unix"] = (
        (cat["peak_time"] - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")
    )

    goes_times = goes_df["time_unix"].values
    goes_flux = goes_df["flux_wm2"].values

    matched = []
    for _, row in cat.iterrows():
        t = row["peak_unix"]
        idx = np.searchsorted(goes_times, t)
        candidates = []
        if idx < len(goes_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(goes_times[i] - t))
        if abs(goes_times[best] - t) <= tolerance_s:
            matched.append({
                "peak_counts": row["peak_solexs_counts"],
                "goes_flux": float(goes_flux[best]),
            })

    return pd.DataFrame(matched)


def fit_log_log(pairs):
    """Fit log10(flux) = a * log10(counts) + b to matched pairs.

    Args:
        pairs: DataFrame with 'peak_counts' and 'goes_flux' columns.

    Returns:
        dict with keys {a, b, r2, n_samples} or None if the fit does not
        meet quality thresholds (min samples or min R²).
    """
    if len(pairs) < CALIBRATION_MIN_SAMPLES:
        return None

    p = pairs.dropna()
    p = p[(p["peak_counts"] > 0) & (p["goes_flux"] > 0)]
    if len(p) < CALIBRATION_MIN_SAMPLES:
        return None

    x = np.log10(p["peak_counts"].values)
    y = np.log10(p["goes_flux"].values)

    # Try HuberRegressor for robustness; fall back to polyfit.
    try:
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor().fit(x.reshape(-1, 1), y)
        a = float(model.coef_[0])
        b = float(model.intercept_)
    except Exception:
        a, b = np.polyfit(x, y, 1)
        a, b = float(a), float(b)

    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if r2 < CALIBRATION_MIN_R2:
        return None

    return {"a": a, "b": b, "r2": r2, "n_samples": len(p)}


def calibrate(catalogue, goes_df=None, save_artifact=True):
    """Full calibration pipeline: match, fit, persist.

    Args:
        catalogue: Master catalogue DataFrame.
        goes_df: Pre-loaded GOES reference.  If None, loaded from disk.
        save_artifact: If True, write result to CALIBRATION_ARTIFACT.

    Returns:
        Calibration dict {a, b, r2, n_samples, fit_date, goes_source} or
        None if fit fails.
    """
    if goes_df is None:
        goes_df = load_goes_reference()

    pairs = cross_match(catalogue, goes_df)
    fit = fit_log_log(pairs)
    if fit is None:
        return None

    fit["fit_date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fit["goes_source"] = "NOAA SWPC GOES XRS 1-8A"

    if save_artifact:
        CALIBRATION_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION_ARTIFACT, "w") as f:
            json.dump(fit, f, indent=2)

    return fit


def load_calibration(path=None):
    """Load an existing calibration artifact if present.

    Args:
        path: Path to the JSON artifact.  Defaults to CALIBRATION_ARTIFACT.

    Returns:
        Calibration dict or None if the file does not exist.
    """
    if path is None:
        path = CALIBRATION_ARTIFACT
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
