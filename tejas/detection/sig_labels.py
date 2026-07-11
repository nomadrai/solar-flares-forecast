"""
TEJAS Significant-Flare Label Augmenter (Fix 4)
=================================================
Reads Phase 6 classification output (`intensity_class` column) and adds
per-timestep binary labels `label_sig_flare_within_{h}m` to the Phase 5
parquet files.

A "significant" event is one whose intensity_class is in
FORECAST_SIGNIFICANT_INTENSITY_CLASSES (moderate / strong / severe by
default — matches the physically meaningful C+/M/X regime given that
Phase 6 GOES calibration was not run).

If Phase 6 does populate goes_class (via `run_classification.py --calibrate`),
callers can pass ``use_goes=True`` to switch to the standard NOAA M/X filter.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    CLASSIFICATION_DIR,
    DETECTION_DIR,
    FORECAST_LABEL_HORIZONS_MIN_SIG,
    FORECAST_SIGNIFICANT_INTENSITY_CLASSES,
)


def _load_classification_catalogue(path=None):
    """Load the Phase 6 augmented catalogue."""
    if path is None:
        path = Path(CLASSIFICATION_DIR) / "master_catalogue.csv"
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


def _sig_event_mask(cat, use_goes=False):
    """Return a boolean mask over `cat` rows marking significant events."""
    if use_goes and "goes_class" in cat.columns and cat["goes_class"].notna().any():
        return cat["goes_class"].isin(["M", "X"])
    if "intensity_class" in cat.columns:
        return cat["intensity_class"].isin(FORECAST_SIGNIFICANT_INTENSITY_CLASSES)
    # Fallback: use peak_solexs_counts >= threshold at 400 c/s
    if "peak_solexs_counts" in cat.columns:
        return cat["peak_solexs_counts"].fillna(0) >= 400.0
    # No signal to work with — no significant events.
    return pd.Series(False, index=cat.index)


def add_sig_labels_to_dataframe(df, catalogue, use_goes=False):
    """Add label_sig_flare_within_{h}m columns to a Phase 5 DataFrame.

    Args:
        df: DataFrame with 'datetime' column (from Phase 5 parquet).
        catalogue: Phase 6 augmented catalogue with intensity_class /
            goes_class columns.
        use_goes: If True, use goes_class instead of intensity_class.

    Returns:
        DataFrame with new label columns.  If no significant events fall
        into this df's time range, the label columns are all zero.
    """
    df = df.copy()
    for h in FORECAST_LABEL_HORIZONS_MIN_SIG:
        col = f"label_sig_flare_within_{h}m"
        df[col] = np.int8(0)

    if catalogue is None or len(catalogue) == 0:
        return df

    sig_mask = _sig_event_mask(catalogue, use_goes=use_goes)
    sig_events = catalogue[sig_mask].dropna(subset=["start_time"])
    if len(sig_events) == 0:
        return df

    # Ensure tz-naive datetime for comparison.
    times = pd.to_datetime(df["datetime"])
    try:
        times = times.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    for _, evt in sig_events.iterrows():
        start = pd.Timestamp(evt["start_time"])
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        for h in FORECAST_LABEL_HORIZONS_MIN_SIG:
            col = f"label_sig_flare_within_{h}m"
            win_start = start - pd.Timedelta(minutes=h)
            mask = (times >= win_start) & (times < start)
            df.loc[mask.values, col] = np.int8(1)

    return df


def augment_day_with_sig_labels(date_str, detection_dir=None,
                                  catalogue_path=None, use_goes=False,
                                  save=True):
    """Augment a Phase 5 parquet with significant-flare labels."""
    detection_dir = Path(detection_dir) if detection_dir else Path(DETECTION_DIR)
    input_path = detection_dir / f"{date_str}.parquet"
    if not input_path.exists():
        return None

    df = pd.read_parquet(input_path)
    if "datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["datetime"]
    ):
        df["datetime"] = pd.to_datetime(df["datetime"])

    cat = _load_classification_catalogue(catalogue_path)
    df_out = add_sig_labels_to_dataframe(df, cat, use_goes=use_goes)

    if save:
        df_out.to_parquet(input_path, index=False, engine="pyarrow")

    sig_counts = {
        f"label_sig_flare_within_{h}m": int(
            df_out[f"label_sig_flare_within_{h}m"].sum()
        )
        for h in FORECAST_LABEL_HORIZONS_MIN_SIG
    }

    return {
        "df": df_out,
        "path": str(input_path),
        "sig_counts": sig_counts,
    }


def augment_all_days_with_sig_labels(detection_dir=None,
                                       catalogue_path=None,
                                       use_goes=False):
    """Augment every Phase 5 parquet with significant-flare labels."""
    detection_dir = Path(detection_dir) if detection_dir else Path(DETECTION_DIR)
    parquets = sorted(detection_dir.glob("2*-*-*.parquet"))
    reports = []
    if not parquets:
        print(f"No detection parquets in {detection_dir}")
        return reports
    for p in parquets:
        date_str = p.stem
        res = augment_day_with_sig_labels(
            date_str, detection_dir=detection_dir,
            catalogue_path=catalogue_path, use_goes=use_goes, save=True,
        )
        if res is None:
            continue
        counts = res["sig_counts"]
        total = sum(counts.values())
        print(f"  sig-labels: {date_str}  total_positives={total}")
        counts["date"] = date_str
        reports.append(counts)
    return reports
