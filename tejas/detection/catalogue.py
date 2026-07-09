"""
TEJAS Flare Catalogue Builder
=================================
Builds independent per-instrument flare catalogues and cross-matches them
into a unified master catalogue.

Independent catalogues:
  - SoLEXS (soft X-ray, 2–22 keV)
  - HEL1OS CdTe (hard X-ray, 8–70 keV)
  - HEL1OS CZT (high-energy, 20–150 keV)

Master catalogue columns include: event_id, start/peak/end times,
peak counts from each instrument, Neupert lag, confidence level,
and a placeholder classification field for Phase 6.
"""

import numpy as np
import pandas as pd

from tejas.config import (
    DETECTION_BASELINE_WINDOW_S,
    DETECTION_SIGMA_START,
    DETECTION_SIGMA_END,
    DETECTION_MIN_DURATION_S,
    DETECTION_MIN_GAP_S,
    DETECTION_MAX_DURATION_S,
    DETECTION_CZT_SIGNIFICANCE,
    DETECTION_CROSS_MATCH_WINDOW_S,
)
from tejas.detection.threshold import compute_adaptive_threshold
from tejas.detection.detector import detect_flares_in_signal


# ── Independent Catalogue Builders ──────────────────────────────────────────


def detect_solexs_flares(df):
    """Detect flares in the SoLEXS soft X-ray signal.

    Uses solexs_counts_smooth as the primary signal.
    SoLEXS has no segment_id (continuous day-long file).

    Args:
        df: Feature DataFrame with datetime and signal columns.

    Returns:
        list[dict] — detected SoLEXS events.
    """
    signal_col = _resolve_signal_col(
        df, ["solexs_counts_smooth", "solexs_counts_clean", "solexs_counts"]
    )
    if signal_col is None:
        return []

    series = df[signal_col]
    timestamps = df["datetime"]

    threshold_df = compute_adaptive_threshold(series, segment_ids=None)
    events = detect_flares_in_signal(series, timestamps, threshold_df)

    return events


def detect_cdte_flares(df):
    """Detect flares in the HEL1OS CdTe hard X-ray signal.

    Uses cdte_count_rate_smooth as the primary signal.
    Segment-aware via cdte_segment_id.

    Args:
        df: Feature DataFrame with datetime and signal columns.

    Returns:
        list[dict] — detected CdTe events.
    """
    signal_col = _resolve_signal_col(
        df,
        ["cdte_count_rate_smooth", "cdte_count_rate_clean", "cdte_count_rate"],
    )
    if signal_col is None:
        return []

    series = df[signal_col]
    timestamps = df["datetime"]
    segment_ids = df.get("cdte_segment_id")

    threshold_df = compute_adaptive_threshold(
        series, segment_ids=segment_ids
    )
    events = detect_flares_in_signal(series, timestamps, threshold_df)

    return events


def detect_czt_flares(df):
    """Detect flares in the HEL1OS CZT high-energy signal.

    Uses czt_net_rate (background-subtracted) as the signal.
    Only runs detection if there are periods where czt_above_bg is True.
    Segment-aware via czt_segment_id.

    Args:
        df: Feature DataFrame with datetime and signal columns.

    Returns:
        list[dict] — detected CZT events (may be empty).
    """
    # Check if CZT ever rises above background
    if "czt_above_bg" in df.columns:
        if not df["czt_above_bg"].any():
            return []

    signal_col = _resolve_signal_col(df, ["czt_net_rate"])
    if signal_col is None:
        return []

    series = df[signal_col]
    timestamps = df["datetime"]
    segment_ids = df.get("czt_segment_id")

    # Use a higher significance threshold for CZT
    threshold_df = compute_adaptive_threshold(
        series,
        sigma_start=DETECTION_CZT_SIGNIFICANCE,
        sigma_end=DETECTION_SIGMA_END,
        segment_ids=segment_ids,
    )
    events = detect_flares_in_signal(series, timestamps, threshold_df)

    return events


# ── Master Catalogue Builder ───────────────────────────────────────────────


def build_master_catalogue(soft_events, hard_events, czt_events, date_str,
                           start_event_id=1):
    """Cross-match independent catalogues into a unified master catalogue.

    Matching logic:
      1. For each soft event, find overlapping or near-coincident hard event.
      2. Soft events without hard match → soft_only.
      3. Hard events without soft match → hard_only.
      4. CZT events are attached to existing master events if overlapping.

    Args:
        soft_events: list[dict] — SoLEXS detections.
        hard_events: list[dict] — CdTe detections.
        czt_events: list[dict] — CZT detections.
        date_str: Date string for the 'date' column.
        start_event_id: Starting event_id for this day (for global uniqueness).

    Returns:
        pd.DataFrame — master catalogue with one row per event.
    """
    window = pd.Timedelta(seconds=DETECTION_CROSS_MATCH_WINDOW_S)
    hard_matched = set()  # indices of hard events already matched
    master_rows = []

    # ── Step 1: Match soft events to hard events ──────────────────────
    for soft in soft_events:
        best_hard = None
        best_hard_idx = None
        best_overlap = -1

        for j, hard in enumerate(hard_events):
            if j in hard_matched:
                continue

            # Check temporal overlap or proximity
            overlap = _compute_overlap(soft, hard, window)
            if overlap > best_overlap:
                best_overlap = overlap
                best_hard = hard
                best_hard_idx = j

        if best_hard is not None and best_overlap > 0:
            hard_matched.add(best_hard_idx)
            row = _build_master_row(
                soft_event=soft, hard_event=best_hard, date_str=date_str
            )
        else:
            row = _build_master_row(
                soft_event=soft, hard_event=None, date_str=date_str
            )

        master_rows.append(row)

    # ── Step 2: Add unmatched hard events ─────────────────────────────
    for j, hard in enumerate(hard_events):
        if j in hard_matched:
            continue
        row = _build_master_row(
            soft_event=None, hard_event=hard, date_str=date_str
        )
        master_rows.append(row)

    if not master_rows:
        return _empty_master_catalogue()

    # ── Step 3: Sort by start_time and assign event IDs ───────────────
    master_df = pd.DataFrame(master_rows)
    master_df = master_df.sort_values("start_time").reset_index(drop=True)
    master_df["event_id"] = range(
        start_event_id, start_event_id + len(master_df)
    )

    # ── Step 4: Attach CZT matches ───────────────────────────────────
    master_df["has_high_energy"] = False
    master_df["peak_czt_counts"] = np.nan

    for czt in czt_events:
        for idx, row in master_df.iterrows():
            overlap = _compute_overlap_from_row(row, czt, window)
            if overlap > 0:
                master_df.at[idx, "has_high_energy"] = True
                existing = master_df.at[idx, "peak_czt_counts"]
                if pd.isna(existing) or czt["peak_counts"] > existing:
                    master_df.at[idx, "peak_czt_counts"] = czt["peak_counts"]

    # Reorder columns
    col_order = [
        "event_id", "date", "start_time", "peak_time", "end_time",
        "duration_s", "peak_solexs_counts", "peak_solexs_zscore",
        "mean_solexs_counts", "peak_cdte_counts", "peak_cdte_zscore",
        "peak_czt_counts", "has_soft", "has_hard", "has_high_energy",
        "neupert_lag_s", "confidence", "classification",
    ]
    # Only include columns that exist
    col_order = [c for c in col_order if c in master_df.columns]
    master_df = master_df[col_order]

    return master_df


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_signal_col(df, candidates):
    """Return the first available column from the candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _compute_overlap(event_a, event_b, window):
    """Compute temporal overlap score between two events.

    Returns:
        float — overlap in seconds.  > 0 means events overlap or are
        within the cross-match window.  <= 0 means no match.
    """
    # Extend event ranges by the cross-match window
    a_start = event_a["start_time"] - window
    a_end = event_a["end_time"] + window
    b_start = event_b["start_time"]
    b_end = event_b["end_time"]

    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    return (overlap_end - overlap_start).total_seconds()


def _compute_overlap_from_row(row, event, window):
    """Compute overlap between a master catalogue row and an event dict."""
    a_start = row["start_time"] - window
    a_end = row["end_time"] + window
    b_start = event["start_time"]
    b_end = event["end_time"]

    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    return (overlap_end - overlap_start).total_seconds()


def _build_master_row(soft_event=None, hard_event=None, date_str=""):
    """Build a single master catalogue row from soft and/or hard events."""
    has_soft = soft_event is not None
    has_hard = hard_event is not None

    # Determine primary times
    if has_soft and has_hard:
        start = min(soft_event["start_time"], hard_event["start_time"])
        end = max(soft_event["end_time"], hard_event["end_time"])
        peak = soft_event["peak_time"]  # primary = soft for peak reference
        confidence = "confirmed"
    elif has_soft:
        start = soft_event["start_time"]
        end = soft_event["end_time"]
        peak = soft_event["peak_time"]
        confidence = "soft_only"
    else:  # hard only
        start = hard_event["start_time"]
        end = hard_event["end_time"]
        peak = hard_event["peak_time"]
        confidence = "hard_only"

    duration = (end - start).total_seconds()

    # Neupert lag: hard peak leads soft peak (positive = CdTe leads)
    neupert_lag = np.nan
    if has_soft and has_hard:
        neupert_lag = (
            soft_event["peak_time"] - hard_event["peak_time"]
        ).total_seconds()

    return {
        "date": date_str,
        "start_time": start,
        "peak_time": peak,
        "end_time": end,
        "duration_s": duration,
        "peak_solexs_counts": soft_event["peak_counts"] if has_soft else np.nan,
        "peak_solexs_zscore": soft_event["peak_zscore"] if has_soft else np.nan,
        "mean_solexs_counts": soft_event["mean_counts"] if has_soft else np.nan,
        "peak_cdte_counts": hard_event["peak_counts"] if has_hard else np.nan,
        "peak_cdte_zscore": hard_event["peak_zscore"] if has_hard else np.nan,
        "has_soft": has_soft,
        "has_hard": has_hard,
        "has_high_energy": False,  # populated later by CZT matching
        "neupert_lag_s": neupert_lag,
        "confidence": confidence,
        "classification": "",  # placeholder for Phase 6
    }


def _empty_master_catalogue():
    """Return an empty DataFrame with the master catalogue schema."""
    return pd.DataFrame(columns=[
        "event_id", "date", "start_time", "peak_time", "end_time",
        "duration_s", "peak_solexs_counts", "peak_solexs_zscore",
        "mean_solexs_counts", "peak_cdte_counts", "peak_cdte_zscore",
        "peak_czt_counts", "has_soft", "has_hard", "has_high_energy",
        "neupert_lag_s", "confidence", "classification",
    ])


def events_to_dataframe(events, prefix=""):
    """Convert a list of event dicts to a DataFrame.

    Args:
        events: list[dict] from detect_flares_in_signal().
        prefix: Optional column prefix (e.g. 'solexs_').

    Returns:
        pd.DataFrame with event columns.
    """
    if not events:
        cols = [
            "start_time", "end_time", "peak_time", "duration_s",
            "start_counts", "peak_counts", "mean_counts", "peak_zscore",
        ]
        return pd.DataFrame(columns=[f"{prefix}{c}" for c in cols]
                            if prefix else cols)

    df = pd.DataFrame(events)
    if prefix:
        df.columns = [f"{prefix}{c}" for c in df.columns]
    return df
