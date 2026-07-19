"""
TEJAS Event-Level Forecast Metrics
==================================
Row-level precision/recall punishes timing: an alert 7 min before a flare
counts as a false positive under the 5-min label even though the operator
got a perfectly good warning.  This module scores the forecaster the way an
operator experiences it:

  * event recall      — fraction of credible flares with at least one alert
                        inside the h-minute window before their start;
  * episode precision — consecutive alert rows are merged into alert
                        "episodes"; an episode is TRUE if a credible flare
                        starts within h minutes after the episode begins;
  * lead time         — minutes from first in-window alert to flare start.

All computations are retrospective evaluation only (labels may look at the
future); nothing here feeds model inputs.
"""

import numpy as np
import pandas as pd

from tejas.config import EVENT_ALERT_MERGE_GAP_S


def _to_unix(values):
    """datetime-like array/Series → float64 unix seconds (UTC, tz-dropped)."""
    dt = pd.to_datetime(pd.Series(values), utc=True).dt.tz_localize(None)
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().values


def merge_alert_episodes(alert_times_unix, merge_gap_s=None):
    """Group sorted alert timestamps into episodes.

    Args:
        alert_times_unix: 1-D array of unix seconds where the model fired.
        merge_gap_s: gaps <= this merge into one episode
            (default EVENT_ALERT_MERGE_GAP_S).

    Returns:
        List of (episode_start_unix, episode_end_unix) tuples.
    """
    if merge_gap_s is None:
        merge_gap_s = EVENT_ALERT_MERGE_GAP_S
    t = np.sort(np.asarray(alert_times_unix, dtype=float))
    if len(t) == 0:
        return []
    episodes = []
    start = prev = t[0]
    for x in t[1:]:
        if x - prev > merge_gap_s:
            episodes.append((start, prev))
            start = x
        prev = x
    episodes.append((start, prev))
    return episodes


def compute_event_metrics(
    timestamps,
    y_prob,
    threshold,
    flare_starts,
    horizon_min,
    merge_gap_s=None,
):
    """Event-level scoreboard for one horizon.

    Args:
        timestamps: row-aligned datetimes of ``y_prob``.
        y_prob: predicted probabilities (same length).
        threshold: decision threshold for firing.
        flare_starts: credible flare start datetimes falling within the
            evaluated period (e.g. test days only).
        horizon_min: horizon in minutes (5/15/30/60).
        merge_gap_s: episode merge gap (default config).

    Returns:
        dict with event_recall, episode_precision, n_events, n_events_caught,
        n_episodes, n_episodes_true, median_lead_min, lead_times_min (list).
    """
    ts = _to_unix(timestamps)
    order = np.argsort(ts)
    ts = ts[order]
    prob = np.asarray(y_prob, dtype=float)[order]

    fired = ts[prob >= float(threshold)]
    episodes = merge_alert_episodes(fired, merge_gap_s)

    starts = np.sort(_to_unix(flare_starts))
    h_s = float(horizon_min) * 60.0

    # ── Event recall + lead times ─────────────────────────────────────
    n_events = len(starts)
    caught = 0
    lead_times = []
    for s in starts:
        # alerts inside (s - h, s)
        lo = np.searchsorted(fired, s - h_s, side="right")
        hi = np.searchsorted(fired, s, side="left")
        if hi > lo:
            caught += 1
            first_alert = fired[lo]
            lead_times.append((s - first_alert) / 60.0)

    # ── Episode precision ─────────────────────────────────────────────
    # An episode is true if any credible flare starts within h minutes
    # after the episode's first alert (operator reading: "the warning was
    # followed by a real flare in the promised window").
    n_episodes = len(episodes)
    n_true = 0
    for ep_start, ep_end in episodes:
        lo = np.searchsorted(starts, ep_start, side="left")
        hi = np.searchsorted(starts, ep_end + h_s, side="right")
        if hi > lo:
            n_true += 1

    return {
        "horizon_min": int(horizon_min),
        "n_events": int(n_events),
        "n_events_caught": int(caught),
        "event_recall": caught / n_events if n_events else float("nan"),
        "n_episodes": int(n_episodes),
        "n_episodes_true": int(n_true),
        "episode_precision": n_true / n_episodes if n_episodes else float("nan"),
        "median_lead_min": float(np.median(lead_times)) if lead_times else float("nan"),
        "lead_times_min": [float(x) for x in lead_times],
    }


def compute_event_metrics_all_horizons(
    timestamps,
    probs_by_horizon,
    thresholds_by_horizon,
    flare_starts,
    merge_gap_s=None,
):
    """Run compute_event_metrics for every horizon; returns a DataFrame."""
    rows = []
    for h, prob in probs_by_horizon.items():
        thr = thresholds_by_horizon[h]
        m = compute_event_metrics(
            timestamps, prob, thr, flare_starts, h, merge_gap_s
        )
        m.pop("lead_times_min")
        rows.append(m)
    return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)
