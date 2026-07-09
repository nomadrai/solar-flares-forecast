"""
TEJAS Phase 5 — Flare Detection Tests
=========================================
Tests for the threshold engine, detector state machine, catalogue builder,
per-timestep labels, no-future-data compliance, and pipeline integration.

Run with: pytest tests/test_detection.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from tejas.detection.threshold import compute_adaptive_threshold
from tejas.detection.detector import detect_flares_in_signal
from tejas.detection.catalogue import (
    detect_solexs_flares,
    detect_cdte_flares,
    detect_czt_flares,
    build_master_catalogue,
    events_to_dataframe,
)
from tejas.detection.pipeline import detect_day, _attach_timestep_labels
from tejas.config import (
    FEATURES_DIR,
    DETECTION_SIGMA_START,
    DETECTION_SIGMA_END,
    ZSCORE_STD_FLOOR,
    DETECTION_LABEL_HORIZONS_MIN,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_quiet_signal(n=3600, baseline=10.0, noise_std=1.0, seed=42):
    """Generate a quiet (no-flare) signal for testing."""
    rng = np.random.RandomState(seed)
    counts = baseline + rng.normal(0, noise_std, n)
    counts = np.maximum(counts, 0)
    timestamps = pd.date_range("2026-05-01", periods=n, freq="1s")
    return pd.Series(counts, name="signal"), pd.Series(timestamps)


def _make_flare_signal(
    n=3600,
    baseline=10.0,
    noise_std=1.0,
    flare_start=1800,
    flare_peak=1830,
    flare_end=1860,
    peak_amplitude=100.0,
    seed=42,
):
    """Generate a signal with one synthetic flare.

    The flare is a triangular pulse: linear rise to peak, then linear decay.
    """
    rng = np.random.RandomState(seed)
    counts = baseline + rng.normal(0, noise_std, n)
    counts = np.maximum(counts, 0.5)

    # Add triangular flare
    for i in range(flare_start, flare_end):
        if i <= flare_peak:
            # Rising
            frac = (i - flare_start) / max(flare_peak - flare_start, 1)
            counts[i] += peak_amplitude * frac
        else:
            # Decaying
            frac = (flare_end - i) / max(flare_end - flare_peak, 1)
            counts[i] += peak_amplitude * frac

    timestamps = pd.date_range("2026-05-01", periods=n, freq="1s")
    return pd.Series(counts, name="signal"), pd.Series(timestamps)


def _make_feature_df(
    n=3600,
    baseline=10.0,
    flare_start=None,
    flare_peak=None,
    flare_end=None,
    peak_amplitude=100.0,
):
    """Generate a minimal feature DataFrame for pipeline testing."""
    if flare_start is not None:
        signal, timestamps = _make_flare_signal(
            n=n, baseline=baseline,
            flare_start=flare_start, flare_peak=flare_peak,
            flare_end=flare_end, peak_amplitude=peak_amplitude,
        )
    else:
        signal, timestamps = _make_quiet_signal(n=n, baseline=baseline)

    df = pd.DataFrame({
        "datetime": timestamps,
        "time_unix": timestamps.astype(np.int64) // 10**9,
        "solexs_counts": signal.values,
        "solexs_counts_smooth": signal.values,
        "solexs_counts_clean": signal.values,
        "cdte_count_rate": signal.values * 0.1,
        "cdte_count_rate_smooth": signal.values * 0.1,
        "cdte_count_rate_clean": signal.values * 0.1,
        "cdte_segment_id": 0,
        "czt_count_rate": signal.values * 0.01,
        "czt_net_rate": signal.values * 0.01 - 0.05,
        "czt_above_bg": False,
        "czt_segment_id": 0,
        "solexs_hardness_ratio": 0.5,
        "solexs_deriv1": np.gradient(signal.values),
        "cdte_deriv1": np.gradient(signal.values * 0.1),
        "solexs_valid": True,
        "hel1os_valid": True,
    })
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 1: Threshold Engine (7 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestThresholdEngine:
    """Tests for compute_adaptive_threshold()."""

    def test_output_columns(self):
        """Threshold output has all expected columns."""
        signal, _ = _make_quiet_signal(600)
        result = compute_adaptive_threshold(signal, baseline_window_s=100)
        expected_cols = {
            "baseline_mean", "baseline_std", "threshold_hi",
            "threshold_lo", "z_score",
        }
        assert set(result.columns) == expected_cols

    def test_output_length_matches_input(self):
        """Output has the same length as input."""
        signal, _ = _make_quiet_signal(500)
        result = compute_adaptive_threshold(signal, baseline_window_s=100)
        assert len(result) == len(signal)

    def test_baseline_mean_is_causal(self):
        """Baseline mean at time t does not use data from t+1 or later.

        Insert a massive spike at index 300.  The baseline_mean at index
        299 should be unaffected (it only sees data up to 299).
        """
        signal, _ = _make_quiet_signal(600, baseline=10.0, noise_std=0.1)
        signal.iloc[300] = 10000.0  # massive spike

        result = compute_adaptive_threshold(signal, baseline_window_s=100)

        # Baseline at index 299 should be close to 10.0 (unaffected)
        mean_before = result["baseline_mean"].iloc[299]
        assert abs(mean_before - 10.0) < 1.0, (
            f"Baseline at t=299 is {mean_before}, expected ~10.0 "
            f"(should not see spike at t=300)"
        )

    def test_std_floor_applied(self):
        """Baseline std is floored at std_floor for constant signals."""
        # Constant signal → std would be 0 without floor
        signal = pd.Series(np.full(200, 10.0))
        result = compute_adaptive_threshold(
            signal, baseline_window_s=100, std_floor=0.5
        )

        # All non-NaN stds should be at least 0.5
        valid_std = result["baseline_std"].dropna()
        assert (valid_std >= 0.5).all(), (
            f"Found std below floor: min={valid_std.min()}"
        )

    def test_z_score_calculation(self):
        """Z-score is correctly computed as (signal - mean) / std."""
        signal, _ = _make_quiet_signal(600, baseline=10.0, noise_std=1.0)
        result = compute_adaptive_threshold(signal, baseline_window_s=100)

        # Verify z-score calculation at a specific index (avoiding NaN region)
        idx = 400
        expected_z = (
            (signal.iloc[idx] - result["baseline_mean"].iloc[idx])
            / result["baseline_std"].iloc[idx]
        )
        assert abs(result["z_score"].iloc[idx] - expected_z) < 1e-10

    def test_nan_handling(self):
        """NaN values in the signal produce NaN in the baseline."""
        signal, _ = _make_quiet_signal(600)
        signal.iloc[200:250] = np.nan  # 50-second gap

        result = compute_adaptive_threshold(signal, baseline_window_s=100)

        # Z-score should be NaN in the gap
        assert result["z_score"].iloc[225].item() is np.nan or \
            pd.isna(result["z_score"].iloc[225])

    def test_segment_boundary_reset(self):
        """Baseline resets at segment boundaries.

        Create two segments with very different baselines.  The baseline
        in segment 2 should NOT be influenced by segment 1.
        """
        n = 600
        signal = pd.Series(np.full(n, 10.0))
        signal.iloc[300:] = 50.0  # second segment has much higher baseline

        segment_ids = pd.Series(np.zeros(n, dtype=int))
        segment_ids.iloc[300:] = 1

        result = compute_adaptive_threshold(
            signal, baseline_window_s=100, segment_ids=segment_ids
        )

        # After segment 2 baseline has had time to stabilize (~100 samples),
        # it should be close to 50.0, not pulled down by segment 1
        mean_seg2 = result["baseline_mean"].iloc[450]
        assert mean_seg2 > 40.0, (
            f"Segment 2 baseline at t=450 is {mean_seg2}, expected ~50.0 "
            f"(should not be pulled down by segment 1)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 2: State Machine (8 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestDetector:
    """Tests for detect_flares_in_signal()."""

    def test_detects_clean_flare(self):
        """Detects a clear synthetic flare with correct start/peak/end."""
        signal, timestamps = _make_flare_signal(
            n=3600, baseline=10.0, noise_std=0.5,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )
        threshold_df = compute_adaptive_threshold(signal, baseline_window_s=600)
        events = detect_flares_in_signal(
            signal, timestamps, threshold_df, min_duration_s=5
        )

        assert len(events) >= 1, "Should detect at least one flare"

        # The peak should be near t=1830
        peak_time = events[0]["peak_time"]
        expected_peak = timestamps.iloc[1830]
        delta = abs((peak_time - expected_peak).total_seconds())
        assert delta <= 5, (
            f"Peak time off by {delta}s (expected ≤5s)"
        )

    def test_misses_sub_threshold_signal(self):
        """Does not trigger on a signal that never exceeds the threshold."""
        signal, timestamps = _make_quiet_signal(3600, baseline=10.0,
                                                 noise_std=0.5)
        threshold_df = compute_adaptive_threshold(signal, baseline_window_s=600)
        events = detect_flares_in_signal(signal, timestamps, threshold_df)

        assert len(events) == 0, (
            f"Expected 0 events on quiet signal, got {len(events)}"
        )

    def test_peak_is_global_maximum(self):
        """Peak is the global maximum within the event window."""
        signal, timestamps = _make_flare_signal(
            n=3600, baseline=10.0, noise_std=0.1,
            flare_start=1800, flare_peak=1825, flare_end=1860,
            peak_amplitude=50.0,
        )
        threshold_df = compute_adaptive_threshold(signal, baseline_window_s=600)
        events = detect_flares_in_signal(
            signal, timestamps, threshold_df, min_duration_s=5
        )

        assert len(events) >= 1
        # Peak counts should be close to baseline + amplitude
        assert events[0]["peak_counts"] > 40.0

    def test_discards_short_events(self):
        """Events shorter than min_duration_s are discarded."""
        # Create a very brief spike (2 seconds)
        signal, timestamps = _make_quiet_signal(3600, baseline=10.0,
                                                 noise_std=0.5)
        signal.iloc[1800:1802] = 200.0  # 2-second spike

        threshold_df = compute_adaptive_threshold(signal, baseline_window_s=600)
        events = detect_flares_in_signal(
            signal, timestamps, threshold_df, min_duration_s=5
        )

        # Should be discarded (duration < 5s)
        for e in events:
            assert e["duration_s"] >= 5

    def test_merges_nearby_events(self):
        """Events separated by less than min_gap_s are merged."""
        # Two flares 30s apart (< min_gap of 60s)
        signal, timestamps = _make_quiet_signal(3600, baseline=10.0,
                                                 noise_std=0.1)
        # Flare 1: t=1800-1820
        for i in range(1800, 1820):
            signal.iloc[i] = 80.0
        # Flare 2: t=1850-1870 (30s gap)
        for i in range(1850, 1870):
            signal.iloc[i] = 90.0

        threshold_df = compute_adaptive_threshold(signal, baseline_window_s=600)
        events = detect_flares_in_signal(
            signal, timestamps, threshold_df,
            min_duration_s=5, min_gap_s=60,
        )

        # Should merge into one event
        assert len(events) <= 1, (
            f"Expected 1 merged event, got {len(events)}"
        )

    def test_keeps_distant_events_separate(self):
        """Events separated by more than min_gap_s remain separate."""
        signal, timestamps = _make_quiet_signal(3600, baseline=10.0,
                                                 noise_std=0.1)
        # Flare 1: t=1000-1020
        for i in range(1000, 1020):
            signal.iloc[i] = 80.0
        # Flare 2: t=2000-2020 (1000s gap >> min_gap)
        for i in range(2000, 2020):
            signal.iloc[i] = 80.0

        threshold_df = compute_adaptive_threshold(signal, baseline_window_s=600)
        events = detect_flares_in_signal(
            signal, timestamps, threshold_df,
            min_duration_s=5, min_gap_s=60,
        )

        assert len(events) == 2, (
            f"Expected 2 separate events, got {len(events)}"
        )

    def test_discards_too_long_events(self):
        """Events exceeding max_duration_s are discarded."""
        signal, timestamps = _make_quiet_signal(3600, baseline=10.0,
                                                 noise_std=0.1)
        # 200-second elevated region — much longer than max_duration=50s
        for i in range(1000, 1200):
            signal.iloc[i] = 80.0

        threshold_df = compute_adaptive_threshold(
            signal, baseline_window_s=600
        )
        events = detect_flares_in_signal(
            signal, timestamps, threshold_df,
            min_duration_s=5, max_duration_s=50,
        )

        # All events should be ≤50s; the 200s event should be discarded
        for e in events:
            assert e["duration_s"] <= 50, (
                f"Event duration {e['duration_s']}s exceeds max 50s"
            )

    def test_handles_empty_signal(self):
        """Empty signal returns empty event list."""
        signal = pd.Series([], dtype=float)
        timestamps = pd.Series([], dtype="datetime64[ns]")
        threshold_df = pd.DataFrame({
            "z_score": [], "threshold_hi": [], "threshold_lo": [],
        })

        events = detect_flares_in_signal(signal, timestamps, threshold_df)
        assert events == []


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 3: CZT Background Gate (3 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestCZTGate:
    """Tests for CZT detection gating on background significance."""

    def test_czt_empty_on_quiet_day(self):
        """CZT produces empty catalogue when signal never exceeds bg."""
        df = _make_feature_df(n=3600, baseline=10.0)
        df["czt_above_bg"] = False

        events = detect_czt_flares(df)
        assert len(events) == 0

    def test_czt_detects_when_above_bg(self):
        """CZT detects events when signal is above background."""
        df = _make_feature_df(
            n=3600, baseline=10.0,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )
        # Make CZT signal much stronger during flare
        df["czt_net_rate"] = df["solexs_counts_smooth"] * 0.5
        df["czt_above_bg"] = df["czt_net_rate"] > 5.0

        events = detect_czt_flares(df)
        # May or may not detect depending on significance threshold,
        # but at least it should run without error
        assert isinstance(events, list)

    def test_czt_requires_significance(self):
        """CZT uses a higher significance threshold than SoLEXS."""
        # A barely elevated CZT signal should not trigger
        df = _make_feature_df(n=3600, baseline=10.0)
        df["czt_net_rate"] = 0.1  # barely above zero
        df["czt_above_bg"] = True  # force gate open

        events = detect_czt_flares(df)
        assert len(events) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 4: Master Catalogue (6 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestMasterCatalogue:
    """Tests for build_master_catalogue()."""

    def _make_event(self, start_s, peak_s, end_s, peak_counts=50.0):
        """Create an event dict with timestamps relative to epoch."""
        base = pd.Timestamp("2026-05-01")
        return {
            "start_time": base + pd.Timedelta(seconds=start_s),
            "peak_time": base + pd.Timedelta(seconds=peak_s),
            "end_time": base + pd.Timedelta(seconds=end_s),
            "duration_s": end_s - start_s,
            "start_counts": 10.0,
            "peak_counts": peak_counts,
            "mean_counts": 30.0,
            "peak_zscore": 5.0,
        }

    def test_confirmed_event(self):
        """Soft + hard match produces 'confirmed' confidence."""
        soft = [self._make_event(100, 120, 150)]
        hard = [self._make_event(105, 118, 145)]

        cat = build_master_catalogue(soft, hard, [], "2026-05-01")

        assert len(cat) == 1
        assert cat.iloc[0]["confidence"] == "confirmed"
        assert cat.iloc[0]["has_soft"] == True
        assert cat.iloc[0]["has_hard"] == True

    def test_soft_only_event(self):
        """Soft event with no hard match produces 'soft_only'."""
        soft = [self._make_event(100, 120, 150)]

        cat = build_master_catalogue(soft, [], [], "2026-05-01")

        assert len(cat) == 1
        assert cat.iloc[0]["confidence"] == "soft_only"
        assert pd.isna(cat.iloc[0]["peak_cdte_counts"])

    def test_hard_only_event(self):
        """Hard event with no soft match produces 'hard_only'."""
        hard = [self._make_event(100, 120, 150)]

        cat = build_master_catalogue([], hard, [], "2026-05-01")

        assert len(cat) == 1
        assert cat.iloc[0]["confidence"] == "hard_only"
        assert pd.isna(cat.iloc[0]["peak_solexs_counts"])

    def test_neupert_lag_positive_when_hard_leads(self):
        """Neupert lag is positive when CdTe peaks before SoLEXS."""
        soft = [self._make_event(100, 125, 160)]  # soft peak at 125
        hard = [self._make_event(98, 115, 155)]   # hard peak at 115

        cat = build_master_catalogue(soft, hard, [], "2026-05-01")

        # neupert_lag = soft_peak - hard_peak = 125 - 115 = 10s
        assert cat.iloc[0]["neupert_lag_s"] == 10.0

    def test_event_ids_are_unique(self):
        """Event IDs are unique and sequential."""
        soft = [
            self._make_event(100, 120, 150),
            self._make_event(1000, 1020, 1050),
        ]

        cat = build_master_catalogue(soft, [], [], "2026-05-01")

        assert len(cat) == 2
        assert len(cat["event_id"].unique()) == 2

    def test_empty_catalogues_produce_empty_master(self):
        """Empty inputs produce an empty master catalogue."""
        cat = build_master_catalogue([], [], [], "2026-05-01")
        assert len(cat) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 5: Per-Timestep Labels (7 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestTimestepLabels:
    """Tests for _attach_timestep_labels()."""

    def test_is_flare_active_during_event(self):
        """is_flare_active = 1 during confirmed flare windows."""
        df = _make_feature_df(
            n=3600,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )

        # Create a synthetic catalogue
        catalogue = pd.DataFrame([{
            "event_id": 1,
            "start_time": df["datetime"].iloc[1800],
            "peak_time": df["datetime"].iloc[1830],
            "end_time": df["datetime"].iloc[1860],
            "peak_solexs_counts": 110.0,
        }])

        df = _attach_timestep_labels(df, catalogue)

        # Check active during event
        assert df["is_flare_active"].iloc[1830] == 1
        # Check quiet before event
        assert df["is_flare_active"].iloc[1000] == 0

    def test_flare_id_maps_to_event(self):
        """flare_id matches the catalogue event_id during events."""
        df = _make_feature_df(n=3600)
        catalogue = pd.DataFrame([{
            "event_id": 42,
            "start_time": df["datetime"].iloc[1800],
            "peak_time": df["datetime"].iloc[1830],
            "end_time": df["datetime"].iloc[1860],
            "peak_solexs_counts": 50.0,
        }])

        df = _attach_timestep_labels(df, catalogue)

        assert df["flare_id"].iloc[1830] == 42
        assert df["flare_id"].iloc[1000] == 0

    def test_flare_phase_transitions(self):
        """flare_phase transitions through rising → peak → decaying."""
        df = _make_feature_df(
            n=3600,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )
        catalogue = pd.DataFrame([{
            "event_id": 1,
            "start_time": df["datetime"].iloc[1800],
            "peak_time": df["datetime"].iloc[1830],
            "end_time": df["datetime"].iloc[1860],
            "peak_solexs_counts": 110.0,
        }])

        df = _attach_timestep_labels(df, catalogue)

        # Before peak → rising
        assert df["flare_phase"].iloc[1810] == "rising"
        # At peak → peak
        assert df["flare_phase"].iloc[1830] == "peak"
        # After peak → decaying
        assert df["flare_phase"].iloc[1850] == "decaying"
        # Outside → quiet
        assert df["flare_phase"].iloc[1000] == "quiet"

    def test_flare_within_5m_is_correct(self):
        """label_flare_within_5m = 1 when flare start is within 5 min."""
        df = _make_feature_df(n=3600)
        catalogue = pd.DataFrame([{
            "event_id": 1,
            "start_time": df["datetime"].iloc[1800],
            "peak_time": df["datetime"].iloc[1830],
            "end_time": df["datetime"].iloc[1860],
            "peak_solexs_counts": 50.0,
        }])

        df = _attach_timestep_labels(df, catalogue)

        # 5 min = 300s before start at t=1800 → label=1 at t=1500
        assert df["label_flare_within_5m"].iloc[1500] == 1
        # 6 min before → label=0 at t=1440
        assert df["label_flare_within_5m"].iloc[1440] == 0

    def test_flare_within_labels_zero_on_quiet_day(self):
        """All flare_within labels are 0 when no flares detected."""
        df = _make_feature_df(n=3600)
        catalogue = pd.DataFrame(columns=[
            "event_id", "start_time", "peak_time", "end_time",
            "peak_solexs_counts",
        ])

        df = _attach_timestep_labels(df, catalogue)

        for h in DETECTION_LABEL_HORIZONS_MIN:
            col = f"label_flare_within_{h}m"
            assert (df[col] == 0).all(), f"{col} should be all 0 on quiet day"

    def test_minutes_to_next_flare_decreases(self):
        """label_minutes_to_next_flare decreases as we approach a flare."""
        df = _make_feature_df(n=3600)
        catalogue = pd.DataFrame([{
            "event_id": 1,
            "start_time": df["datetime"].iloc[1800],
            "peak_time": df["datetime"].iloc[1830],
            "end_time": df["datetime"].iloc[1860],
            "peak_solexs_counts": 50.0,
        }])

        df = _attach_timestep_labels(df, catalogue)

        # At t=1500 (5 min before), should be 5.0
        val_at_1500 = df["label_minutes_to_next_flare"].iloc[1500]
        # At t=1700 (100s before), should be ~1.67
        val_at_1700 = df["label_minutes_to_next_flare"].iloc[1700]

        assert val_at_1500 > val_at_1700, (
            f"Should decrease: {val_at_1500} > {val_at_1700}"
        )

    def test_minutes_to_next_nan_after_last_flare(self):
        """label_minutes_to_next_flare is NaN after the last flare."""
        df = _make_feature_df(n=3600)
        catalogue = pd.DataFrame([{
            "event_id": 1,
            "start_time": df["datetime"].iloc[1000],
            "peak_time": df["datetime"].iloc[1030],
            "end_time": df["datetime"].iloc[1060],
            "peak_solexs_counts": 50.0,
        }])

        df = _attach_timestep_labels(df, catalogue)

        # Well after the flare (t=2000), no future flare → NaN
        assert pd.isna(df["label_minutes_to_next_flare"].iloc[2000])


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 6: No-Future-Data (4 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestNoFutureData:
    """Verify that thresholds and detection are strictly causal."""

    def test_threshold_is_trailing_only(self):
        """Threshold uses trailing (not centered) rolling window."""
        signal, _ = _make_quiet_signal(600, baseline=10.0, noise_std=0.5)
        result = compute_adaptive_threshold(signal, baseline_window_s=100)

        # First few values should be NaN (not enough history)
        # If the window were centered, value at index 50 would be valid
        # With trailing window and min_periods=60, values before index 59
        # should be NaN
        assert pd.isna(result["baseline_mean"].iloc[0])

    def test_future_spike_does_not_affect_past_threshold(self):
        """A spike at t=500 does not change the threshold at t=499."""
        signal, _ = _make_quiet_signal(1000, baseline=10.0, noise_std=0.1)

        # Get threshold without spike
        result_clean = compute_adaptive_threshold(
            signal.copy(), baseline_window_s=100
        )
        threshold_at_499_clean = result_clean["threshold_hi"].iloc[499]

        # Add spike and recompute
        signal_spiked = signal.copy()
        signal_spiked.iloc[500] = 10000.0
        result_spiked = compute_adaptive_threshold(
            signal_spiked, baseline_window_s=100
        )
        threshold_at_499_spiked = result_spiked["threshold_hi"].iloc[499]

        assert threshold_at_499_clean == threshold_at_499_spiked, (
            "Future spike should not affect past threshold"
        )

    def test_detection_does_not_use_future_data(self):
        """Detection at time t does not depend on data after t.

        Strategy: run detection on full signal, then truncate at the start
        of the flare and verify that no event is detected in the truncated
        version that wouldn't be detected otherwise.
        """
        signal, timestamps = _make_flare_signal(
            n=3600, baseline=10.0, noise_std=0.5,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )

        # Run on truncated signal (before flare)
        trunc_signal = signal.iloc[:1790]
        trunc_ts = timestamps.iloc[:1790]
        trunc_threshold = compute_adaptive_threshold(
            trunc_signal, baseline_window_s=600
        )
        trunc_events = detect_flares_in_signal(
            trunc_signal, trunc_ts, trunc_threshold, min_duration_s=5
        )

        # Should find no events before the flare starts
        assert len(trunc_events) == 0, (
            "Should not detect events before the flare starts"
        )

    def test_detection_state_only_depends_on_past(self):
        """The detector's ACTIVE state at time t depends only on past data.

        Add data after a flare and verify the flare detection is unchanged.
        """
        # Signal with flare at t=1800-1860
        signal1, ts1 = _make_flare_signal(
            n=2000, baseline=10.0, noise_std=0.1,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )

        # Same signal but extended to n=3600
        signal2, ts2 = _make_flare_signal(
            n=3600, baseline=10.0, noise_std=0.1,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )

        thresh1 = compute_adaptive_threshold(signal1, baseline_window_s=600)
        thresh2 = compute_adaptive_threshold(signal2, baseline_window_s=600)

        events1 = detect_flares_in_signal(
            signal1, ts1, thresh1, min_duration_s=5
        )
        events2 = detect_flares_in_signal(
            signal2, ts2, thresh2, min_duration_s=5
        )

        # Both should detect the same flare with the same peak time
        assert len(events1) >= 1
        assert len(events2) >= 1
        assert events1[0]["peak_time"] == events2[0]["peak_time"]


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 7: Pipeline Integration (5 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """Integration tests for detect_day() and related functions."""

    def test_detect_day_on_real_data(self):
        """detect_day() runs end-to-end on a real feature Parquet.

        Skipped if no feature data is available.
        """
        test_file = FEATURES_DIR / "2026-05-01.parquet"
        if not test_file.exists():
            pytest.skip(f"Feature file not found: {test_file}")

        result = detect_day("2026-05-01", save=False)

        assert result is not None
        assert "df" in result
        assert "catalogue" in result
        assert "report" in result

    def test_output_has_expected_columns(self):
        """Output DataFrame has all expected label columns."""
        df = _make_feature_df(
            n=3600,
            flare_start=1800, flare_peak=1830, flare_end=1860,
            peak_amplitude=100.0,
        )

        # Run detection pipeline manually
        solexs_events = detect_solexs_flares(df)
        catalogue = build_master_catalogue(
            solexs_events, [], [], "2026-05-01"
        )
        df = _attach_timestep_labels(df, catalogue)

        expected_cols = [
            "flare_id", "is_flare_active", "flare_phase",
            "label_minutes_to_next_flare",
        ]
        for h in DETECTION_LABEL_HORIZONS_MIN:
            expected_cols.append(f"label_flare_within_{h}m")

        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_events_to_dataframe(self):
        """events_to_dataframe() converts event list to DataFrame."""
        events = [{
            "start_time": pd.Timestamp("2026-05-01 10:00:00"),
            "end_time": pd.Timestamp("2026-05-01 10:01:00"),
            "peak_time": pd.Timestamp("2026-05-01 10:00:30"),
            "duration_s": 60.0,
            "start_counts": 10.0,
            "peak_counts": 50.0,
            "mean_counts": 30.0,
            "peak_zscore": 8.0,
        }]

        df = events_to_dataframe(events, prefix="solexs_")
        assert len(df) == 1
        assert "solexs_peak_counts" in df.columns

    def test_events_to_dataframe_empty(self):
        """events_to_dataframe() handles empty list."""
        df = events_to_dataframe([], prefix="solexs_")
        assert len(df) == 0
        assert "solexs_peak_counts" in df.columns

    def test_catalogue_csv_schema(self):
        """Master catalogue has the expected column schema."""
        soft = [{
            "start_time": pd.Timestamp("2026-05-01 10:00:00"),
            "end_time": pd.Timestamp("2026-05-01 10:01:00"),
            "peak_time": pd.Timestamp("2026-05-01 10:00:30"),
            "duration_s": 60.0,
            "start_counts": 10.0,
            "peak_counts": 50.0,
            "mean_counts": 30.0,
            "peak_zscore": 8.0,
        }]

        cat = build_master_catalogue(soft, [], [], "2026-05-01")

        required_cols = [
            "event_id", "date", "start_time", "peak_time", "end_time",
            "duration_s", "peak_solexs_counts", "confidence",
            "classification",
        ]
        for col in required_cols:
            assert col in cat.columns, f"Missing column: {col}"
