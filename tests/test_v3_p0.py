"""
TEJAS v3 Optimization — Phase P0 Tests
=======================================
Tests for the P0a blacklist hygiene (metadata column removal + pattern guard)
and P0b event-level threshold tuner with duty-cycle caps.

Run with: pytest tests/test_v3_p0.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tejas.config import (
    FORECAST_DROP_COLUMNS,
    FORECAST_DROP_COLUMN_PATTERNS,
    FORECAST_MAX_DUTY_CYCLE_PER_HORIZON,
    FORECAST_MIN_RECALL_PER_HORIZON,
)
from tejas.forecasting.sampling import select_feature_columns
from tejas.forecasting.evaluation import tune_threshold_event_level
from tejas.forecasting.inference import Forecaster


# ══════════════════════════════════════════════════════════════════════════
# P0a: Metadata blacklist + pattern guard
# ══════════════════════════════════════════════════════════════════════════


class TestP0aBlacklist:
    """Test that metadata columns are excluded from the feature matrix."""

    def test_literal_metadata_in_blacklist(self):
        """The four metadata columns must be in FORECAST_DROP_COLUMNS."""
        required = {"cdte_segment_id", "czt_segment_id", "cdte_stat_err", "czt_stat_err"}
        assert required.issubset(FORECAST_DROP_COLUMNS), (
            "v3 P0a: cdte_segment_id, czt_segment_id, cdte_stat_err, "
            "czt_stat_err must be in FORECAST_DROP_COLUMNS"
        )

    def test_pattern_guard_exists(self):
        """FORECAST_DROP_COLUMN_PATTERNS must include the two suffixes."""
        assert "_segment_id" in FORECAST_DROP_COLUMN_PATTERNS
        assert "_stat_err" in FORECAST_DROP_COLUMN_PATTERNS

    def test_select_feature_columns_drops_literal_metadata(self):
        """select_feature_columns must reject the four literal columns."""
        df = pd.DataFrame({
            "solexs_counts": [10, 20],
            "cdte_segment_id": [0, 1],
            "czt_segment_id": [0, 0],
            "cdte_stat_err": [1.0, 1.5],
            "czt_stat_err": [2.0, 2.5],
            "solexs_gradient": [0.1, 0.2],
        })
        cols = select_feature_columns(df)
        assert "cdte_segment_id" not in cols
        assert "czt_segment_id" not in cols
        assert "cdte_stat_err" not in cols
        assert "czt_stat_err" not in cols
        assert "solexs_counts" in cols
        assert "solexs_gradient" in cols

    def test_select_feature_columns_drops_pattern_matches(self):
        """Pattern guard must drop any column ending with _segment_id or _stat_err."""
        df = pd.DataFrame({
            "solexs_counts": [10, 20],
            "future_instrument_segment_id": [0, 1],  # hypothetical future column
            "hel1os_combined_stat_err": [1.0, 1.5],  # not a real column
            "solexs_gradient": [0.1, 0.2],
        })
        cols = select_feature_columns(df)
        assert "future_instrument_segment_id" not in cols
        assert "hel1os_combined_stat_err" not in cols
        assert "solexs_counts" in cols
        assert "solexs_gradient" in cols

    def test_forecaster_rejects_literal_metadata(self):
        """Forecaster._validate_columns must reject literal blacklist names."""
        forecaster = Forecaster(
            algorithm="xgb",
            feature_columns=["solexs_counts", "solexs_gradient"],
            horizon_models={},
            forbidden_columns=FORECAST_DROP_COLUMNS,
        )
        # Valid row: passes
        forecaster._validate_columns(["solexs_counts", "solexs_gradient"])

        # Row with cdte_segment_id: must fail
        with pytest.raises(ValueError, match="Forbidden columns"):
            forecaster._validate_columns(["solexs_counts", "cdte_segment_id"])

    def test_forecaster_rejects_pattern_matches(self):
        """Forecaster._validate_columns must reject columns matching patterns."""
        forecaster = Forecaster(
            algorithm="xgb",
            feature_columns=["solexs_counts", "solexs_gradient"],
            horizon_models={},
            forbidden_columns=FORECAST_DROP_COLUMNS,
        )
        # Row with a future *_segment_id column: must fail
        with pytest.raises(ValueError, match="Forbidden column patterns"):
            forecaster._validate_columns(
                ["solexs_counts", "future_instrument_segment_id"]
            )

        # Row with a *_stat_err column: must fail
        with pytest.raises(ValueError, match="Forbidden column patterns"):
            forecaster._validate_columns(["solexs_counts", "hel1os_combined_stat_err"])


# ══════════════════════════════════════════════════════════════════════════
# P0b: Event-level threshold tuner with duty-cycle caps
# ══════════════════════════════════════════════════════════════════════════


class TestP0bEventLevelTuner:
    """Test the event-level threshold optimizer with duty-cycle caps."""

    def test_duty_cycle_caps_exist(self):
        """FORECAST_MAX_DUTY_CYCLE_PER_HORIZON must define caps for all horizons."""
        for h in [5, 15, 30, 60]:
            assert h in FORECAST_MAX_DUTY_CYCLE_PER_HORIZON, (
                f"v3 P0b: horizon {h} missing from FORECAST_MAX_DUTY_CYCLE_PER_HORIZON"
            )
            cap = FORECAST_MAX_DUTY_CYCLE_PER_HORIZON[h]
            assert 0.0 < cap <= 1.0, f"Duty cap for {h}m must be in (0,1], got {cap}"

    def test_tune_threshold_event_level_returns_feasible_threshold(self):
        """The tuner must return a threshold that respects the duty cap."""
        # Synthetic val split: 100 rows, 10 flares
        n = 1000
        timestamps = pd.date_range("2026-05-21 00:00:00", periods=n, freq="10s")
        y_prob = np.random.RandomState(42).rand(n) * 0.5  # mostly low probs
        # Inject 10 flares at known times
        flare_starts = pd.to_datetime([
            "2026-05-21 00:10:00", "2026-05-21 00:30:00", "2026-05-21 00:50:00",
            "2026-05-21 01:10:00", "2026-05-21 01:30:00", "2026-05-21 01:50:00",
            "2026-05-21 02:10:00", "2026-05-21 02:30:00", "2026-05-21 02:50:00",
            "2026-05-21 03:10:00",
        ])
        # Boost probs in the 5-min windows before flares
        for start in flare_starts:
            mask = (timestamps >= start - pd.Timedelta(minutes=5)) & (timestamps < start)
            y_prob[mask] = 0.8 + 0.1 * np.random.rand(mask.sum())

        horizon_min = 5
        max_duty = FORECAST_MAX_DUTY_CYCLE_PER_HORIZON[horizon_min]  # 0.10
        min_recall = FORECAST_MIN_RECALL_PER_HORIZON[horizon_min]    # 0.80

        thr = tune_threshold_event_level(
            y_prob, timestamps, flare_starts, horizon_min,
            min_recall=min_recall, max_duty_cycle=max_duty,
        )

        # Check duty cycle constraint
        duty = (y_prob >= thr).mean()
        assert duty <= max_duty + 1e-6, (
            f"Tuned threshold {thr:.3f} yields duty {duty:.3f}, exceeds cap {max_duty}"
        )

    def test_tune_threshold_event_level_with_zero_flares_returns_default(self):
        """When val has no flares, the tuner must fall back to default."""
        n = 100
        timestamps = pd.date_range("2026-05-21 00:00:00", periods=n, freq="10s")
        y_prob = np.random.rand(n) * 0.5
        flare_starts = pd.Series([], dtype="datetime64[ns]")  # empty

        from tejas.config import FORECAST_DECISION_THRESHOLD_DEFAULT
        thr = tune_threshold_event_level(y_prob, timestamps, flare_starts, 5)
        assert thr == FORECAST_DECISION_THRESHOLD_DEFAULT

    def test_tune_threshold_event_level_prefers_higher_precision(self):
        """Among feasible thresholds, the tuner must pick the highest precision."""
        # Setup: 200 rows, 5 flares, controllable probs
        n = 2000
        timestamps = pd.date_range("2026-05-21 00:00:00", periods=n, freq="1s")
        y_prob = np.full(n, 0.05)  # baseline low
        flare_starts = pd.to_datetime([
            "2026-05-21 00:10:00", "2026-05-21 00:20:00", "2026-05-21 00:30:00",
            "2026-05-21 00:40:00", "2026-05-21 00:50:00",
        ])
        # Windows before flares: high prob
        for start in flare_starts:
            mask = (timestamps >= start - pd.Timedelta(minutes=5)) & (timestamps < start)
            y_prob[mask] = 0.9

        # Inject some false-positive periods (high prob, no flare follows)
        fp_windows = [
            (pd.Timestamp("2026-05-21 00:05:00"), pd.Timestamp("2026-05-21 00:06:00")),
            (pd.Timestamp("2026-05-21 00:15:00"), pd.Timestamp("2026-05-21 00:16:00")),
        ]
        for t0, t1 in fp_windows:
            mask = (timestamps >= t0) & (timestamps < t1)
            y_prob[mask] = 0.7

        thr = tune_threshold_event_level(
            y_prob, timestamps, flare_starts, 5,
            min_recall=0.8, max_duty_cycle=0.15,
        )

        # A higher threshold should reduce FP episodes → higher precision
        # Check that the tuner didn't collapse to threshold=0.05 (always-on)
        assert thr > 0.10, (
            f"Tuner should prefer precision; threshold {thr:.3f} seems too low"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
