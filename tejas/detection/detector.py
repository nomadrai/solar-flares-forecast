"""
TEJAS Per-Instrument Flare Detector
=======================================
Runs a two-state state machine (QUIET ↔ ACTIVE) over a signal's z-score
time series to identify flare events: start, peak, and end.

The detector:
  1. Triggers ACTIVE when z_score > sigma_start threshold.
  2. Stays ACTIVE until z_score < sigma_end threshold (with debounce).
  3. Identifies the peak as the global maximum within each event.
  4. Discards events shorter than min_duration.
  5. Merges events separated by less than min_gap.
  6. Discards events longer than max_duration.
"""

import numpy as np
import pandas as pd

from tejas.config import (
    DETECTION_MIN_DURATION_S,
    DETECTION_MIN_GAP_S,
    DETECTION_MAX_DURATION_S,
)


def detect_flares_in_signal(
    series,
    timestamps,
    threshold_df,
    min_duration_s=None,
    min_gap_s=None,
    max_duration_s=None,
):
    """Detect flare events in a single instrument's signal.

    Uses the precomputed threshold DataFrame (from threshold.py) to
    identify periods where the signal is significantly above baseline.

    Args:
        series: pd.Series of signal values (count rates).
        timestamps: pd.Series of pd.Timestamp (datetime column).
        threshold_df: pd.DataFrame from compute_adaptive_threshold(),
            must have columns: z_score, threshold_hi, threshold_lo.
        min_duration_s: Minimum event duration in seconds.
            Defaults to config.DETECTION_MIN_DURATION_S.
        min_gap_s: Merge events separated by less than this (seconds).
            Defaults to config.DETECTION_MIN_GAP_S.
        max_duration_s: Discard events longer than this (seconds).
            Defaults to config.DETECTION_MAX_DURATION_S.

    Returns:
        list[dict] — each dict represents one detected event:
          {
            'start_time':   pd.Timestamp,
            'end_time':     pd.Timestamp,
            'peak_time':    pd.Timestamp,
            'duration_s':   float,
            'start_counts': float,
            'peak_counts':  float,
            'mean_counts':  float,
            'peak_zscore':  float,
          }
    """
    if min_duration_s is None:
        min_duration_s = DETECTION_MIN_DURATION_S
    if min_gap_s is None:
        min_gap_s = DETECTION_MIN_GAP_S
    if max_duration_s is None:
        max_duration_s = DETECTION_MAX_DURATION_S

    z_score = threshold_df["z_score"].values
    threshold_hi = threshold_df["threshold_hi"].values
    signal = series.values
    ts = timestamps.values  # numpy datetime64 array

    n = len(signal)
    if n == 0:
        return []

    # ── Step 1: Identify raw active regions ───────────────────────────
    # A region is "active" when signal > threshold_hi (start trigger).
    # It stays active until signal drops below threshold_lo (end trigger).
    # Use the z-score thresholds from config (sigma_start, sigma_end)
    # which are already baked into threshold_hi and threshold_lo.

    sigma_start_vals = threshold_df["threshold_hi"].values
    sigma_end_vals = threshold_df["threshold_lo"].values

    active = np.zeros(n, dtype=bool)
    in_event = False

    for i in range(n):
        if np.isnan(signal[i]) or np.isnan(sigma_start_vals[i]):
            # NaN — not in an event, break any current event
            in_event = False
            continue

        if not in_event:
            # Check start trigger: signal above high threshold
            if signal[i] > sigma_start_vals[i]:
                in_event = True
                active[i] = True
        else:
            # In event: stay active until signal drops below low threshold
            if signal[i] > sigma_end_vals[i]:
                active[i] = True
            else:
                in_event = False

    # ── Step 2: Extract contiguous active regions as raw events ────────
    raw_events = _extract_contiguous_regions(active, signal, ts, z_score)

    if not raw_events:
        return []

    # ── Step 3: Filter by minimum duration ────────────────────────────
    events = [e for e in raw_events if e["duration_s"] >= min_duration_s]

    if not events:
        return []

    # ── Step 4: Merge events separated by less than min_gap ───────────
    events = _merge_nearby_events(events, min_gap_s, signal, ts, z_score)

    # ── Step 5: Filter by maximum duration ────────────────────────────
    events = [e for e in events if e["duration_s"] <= max_duration_s]

    return events


def _extract_contiguous_regions(active, signal, ts, z_score):
    """Extract contiguous True regions from the active mask.

    Returns a list of event dicts with start/end/peak information.
    """
    events = []
    n = len(active)
    i = 0

    while i < n:
        if not active[i]:
            i += 1
            continue

        # Found start of an active region
        start_idx = i
        while i < n and active[i]:
            i += 1
        end_idx = i - 1  # last active index (inclusive)

        # Extract event info
        event_signal = signal[start_idx:end_idx + 1]
        event_z = z_score[start_idx:end_idx + 1]

        # Peak = global maximum within event window
        peak_offset = np.nanargmax(event_signal)
        peak_idx = start_idx + peak_offset

        start_ts = pd.Timestamp(ts[start_idx])
        end_ts = pd.Timestamp(ts[end_idx])
        peak_ts = pd.Timestamp(ts[peak_idx])
        duration = (end_ts - start_ts).total_seconds()

        events.append({
            "start_time": start_ts,
            "end_time": end_ts,
            "peak_time": peak_ts,
            "duration_s": duration,
            "start_counts": float(signal[start_idx]),
            "peak_counts": float(signal[peak_idx]),
            "mean_counts": float(np.nanmean(event_signal)),
            "peak_zscore": float(event_z[peak_offset])
                          if not np.isnan(event_z[peak_offset]) else 0.0,
            "_start_idx": start_idx,
            "_end_idx": end_idx,
        })

    return events


def _merge_nearby_events(events, min_gap_s, signal, ts, z_score):
    """Merge events separated by less than min_gap_s seconds.

    When two events are merged, the peak is recalculated over the
    combined window (including the gap between them).
    """
    if len(events) <= 1:
        return events

    merged = [events[0].copy()]

    for event in events[1:]:
        prev = merged[-1]
        gap = (event["start_time"] - prev["end_time"]).total_seconds()

        if gap <= min_gap_s:
            # Merge: extend prev to cover both events
            new_start_idx = prev["_start_idx"]
            new_end_idx = event["_end_idx"]

            combined_signal = signal[new_start_idx:new_end_idx + 1]
            combined_z = z_score[new_start_idx:new_end_idx + 1]
            peak_offset = np.nanargmax(combined_signal)
            peak_idx = new_start_idx + peak_offset

            prev["end_time"] = event["end_time"]
            prev["_end_idx"] = new_end_idx
            prev["duration_s"] = (
                prev["end_time"] - prev["start_time"]
            ).total_seconds()
            prev["peak_time"] = pd.Timestamp(ts[peak_idx])
            prev["peak_counts"] = float(signal[peak_idx])
            prev["mean_counts"] = float(np.nanmean(combined_signal))
            prev["peak_zscore"] = float(combined_z[peak_offset]) \
                if not np.isnan(combined_z[peak_offset]) else 0.0
        else:
            merged.append(event.copy())

    # Strip internal indices before returning
    for e in merged:
        e.pop("_start_idx", None)
        e.pop("_end_idx", None)

    return merged
