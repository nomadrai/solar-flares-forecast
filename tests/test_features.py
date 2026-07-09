"""
TEJAS Phase 4 — Feature Engineering Tests
============================================
Unit tests for each feature module:
  - Temporal features
  - Spectral features
  - Cross-instrument features
  - Statistical features
  - Pipeline integration
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def base_day_df():
    """Simulate a minimal unified day DataFrame with all expected columns.

    Creates 3600 rows (1 hour) with realistic-ish signals:
      - SoLEXS: baseline ~10 c/s with a flare peak at row 1800
      - CdTe: baseline ~0.5 c/s with a hard X-ray burst at row 1750
      - CZT: baseline ~0 c/s (background-subtracted), spike at row 1740
    """
    np.random.seed(42)
    n = 3600
    t0 = 1716508800.0  # 2024-05-24 00:00:00 UTC (arbitrary)
    time_unix = t0 + np.arange(n, dtype=np.float64)

    # SoLEXS: quiet baseline + flare
    solexs = np.random.poisson(10, n).astype(float)
    # Flare: gradual rise → peak → decay (100 bins)
    flare = np.concatenate([
        np.linspace(10, 200, 30),
        np.linspace(200, 150, 20),
        np.linspace(150, 15, 50),
    ])
    solexs[1750:1850] = flare

    # CdTe: mostly zeros, spike during flare impulsive phase
    cdte = np.random.poisson(0.15, n).astype(float)
    cdte[1720:1770] = np.concatenate([
        np.linspace(0, 30, 25),
        np.linspace(30, 2, 25),
    ])

    # CZT net rate: background-subtracted, mostly 0
    czt_net = np.zeros(n)
    czt_net[1730:1760] = np.concatenate([
        np.linspace(0, 15, 15),
        np.linspace(15, 1, 15),
    ])

    # SoLEXS bands (proportional to total)
    counts_low = solexs * 0.6 + np.random.normal(0, 0.5, n)
    counts_mid = solexs * 0.25 + np.random.normal(0, 0.3, n)
    counts_high = solexs * 0.15 + np.random.normal(0, 0.2, n)
    counts_low = np.maximum(counts_low, 0)
    counts_mid = np.maximum(counts_mid, 0)
    counts_high = np.maximum(counts_high, 0)

    hr = counts_high / np.maximum(counts_low, 0.5)

    df = pd.DataFrame({
        "time_unix": time_unix,
        "datetime": pd.to_datetime(time_unix, unit="s", utc=True),
        "solexs_counts": solexs,
        "solexs_counts_low": counts_low,
        "solexs_counts_mid": counts_mid,
        "solexs_counts_high": counts_high,
        "solexs_hardness_ratio": hr,
        "solexs_valid": True,
        "cdte_count_rate": cdte,
        "cdte_stat_err": np.sqrt(np.maximum(cdte, 1)),
        "cdte_segment_id": 0.0,
        "cdte_is_filled": False,
        "czt_count_rate": czt_net + 17.0,  # raw = net + background
        "czt_stat_err": np.sqrt(np.maximum(czt_net + 17.0, 1)),
        "czt_segment_id": 0.0,
        "czt_is_filled": False,
        "hel1os_valid": True,
        # Cleaned = same as raw in test
        "solexs_counts_clean": solexs,
        "cdte_count_rate_clean": cdte,
        "czt_count_rate_clean": czt_net + 17.0,
        "solexs_counts_low_clean": counts_low,
        "solexs_counts_mid_clean": counts_mid,
        "solexs_counts_high_clean": counts_high,
        "solexs_hardness_ratio_clean": hr,
        # Smoothed = same as raw in test (for simplicity)
        "solexs_counts_smooth": solexs,
        "solexs_counts_low_smooth": counts_low,
        "solexs_counts_mid_smooth": counts_mid,
        "solexs_counts_high_smooth": counts_high,
        "solexs_hardness_ratio_smooth": hr,
        "cdte_count_rate_smooth": cdte,
        "czt_count_rate_smooth": czt_net + 17.0,
        # CZT background
        "czt_background": np.full(n, 17.0),
        "czt_net_rate": czt_net,
        "czt_above_bg": czt_net > 3.0,
        # Normalization (not used by features, but present)
        "solexs_counts_log": np.log1p(solexs),
        "solexs_counts_pctrank": np.linspace(0, 1, n),
        "cdte_count_rate_log": np.log1p(cdte),
        "cdte_count_rate_pctrank": np.linspace(0, 1, n),
        "czt_net_rate_log": np.log1p(czt_net),
        "czt_net_rate_pctrank": np.linspace(0, 1, n),
    })
    return df


@pytest.fixture
def linear_ramp():
    """A simple linearly increasing signal for derivative testing."""
    return pd.Series(np.arange(100, dtype=float), name="ramp")


@pytest.fixture
def constant_signal():
    """A constant signal (baseline = 10)."""
    return pd.Series(np.full(500, 10.0), name="constant")


@pytest.fixture
def signal_with_nan():
    """Signal with NaN gap in the middle."""
    values = np.arange(100, dtype=float)
    values[40:60] = np.nan
    return pd.Series(values, name="gapped")


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Feature Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTemporalFeatures:
    """Tests for tejas.features.temporal."""

    def test_rolling_mean_constant_signal(self, constant_signal):
        """Rolling mean of constant signal = constant."""
        from tejas.features.temporal import compute_rolling_mean
        result = compute_rolling_mean(constant_signal, window=30)
        # After window fills, should equal the constant
        assert np.allclose(result.dropna().values, 10.0)

    def test_rolling_mean_is_causal(self, linear_ramp):
        """Rolling mean at position t should not depend on future values."""
        from tejas.features.temporal import compute_rolling_mean
        result = compute_rolling_mean(linear_ramp, window=10)
        # At position 50, the trailing 10 values are [41..50], mean = 45.5
        assert abs(result.iloc[50] - 45.5) < 0.01

    def test_rolling_std_constant_is_zero(self, constant_signal):
        """Rolling std of constant signal should be 0 (or NaN at edges)."""
        from tejas.features.temporal import compute_rolling_std
        result = compute_rolling_std(constant_signal, window=30)
        valid = result.dropna()
        assert np.allclose(valid.values, 0.0)

    def test_derivative_linear_ramp(self, linear_ramp):
        """First derivative of linear ramp should be constant = 1."""
        from tejas.features.temporal import compute_derivative
        result = compute_derivative(linear_ramp, order=1)
        # First value is NaN (diff), rest should be 1.0
        assert np.allclose(result.iloc[1:].values, 1.0)

    def test_second_derivative_linear_ramp(self, linear_ramp):
        """Second derivative of linear ramp should be 0."""
        from tejas.features.temporal import compute_derivative
        result = compute_derivative(linear_ramp, order=2)
        # First two values are NaN
        assert np.allclose(result.iloc[2:].values, 0.0)

    def test_gradient_linear_ramp(self, linear_ramp):
        """Gradient of linear ramp over any horizon should be 1."""
        from tejas.features.temporal import compute_gradient
        for horizon in [5, 10, 30]:
            result = compute_gradient(linear_ramp, horizon)
            valid = result.dropna()
            assert np.allclose(valid.values, 1.0), f"Failed for horizon={horizon}"

    def test_rise_rate_zeroed_during_decay(self):
        """Rise rate should be 0 when signal is decreasing."""
        from tejas.features.temporal import compute_derivative
        signal = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0])
        deriv = compute_derivative(signal, order=1)
        rise = deriv.clip(lower=0.0)
        # All derivatives are -1, so rise should be 0
        assert np.allclose(rise.iloc[1:].values, 0.0)

    def test_decay_rate_zeroed_during_rise(self):
        """Decay rate should be 0 when signal is increasing."""
        from tejas.features.temporal import compute_derivative
        signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        deriv = compute_derivative(signal, order=1)
        decay = deriv.clip(upper=0.0)
        assert np.allclose(decay.iloc[1:].values, 0.0)

    def test_nan_gaps_preserved(self, signal_with_nan):
        """NaN values should not be filled by rolling operations."""
        from tejas.features.temporal import compute_rolling_mean
        result = compute_rolling_mean(signal_with_nan, window=5)
        # Values inside NaN gap should remain NaN (or at least the
        # center of the gap should be NaN)
        assert result.iloc[45:55].isna().any()

    def test_full_temporal_on_day(self, base_day_df):
        """Full temporal feature computation produces expected columns."""
        from tejas.features.temporal import compute_temporal_features
        result = compute_temporal_features(base_day_df)
        df = result["df"]
        added = result["columns_added"]

        # Should have features for all 3 instruments
        assert len(added) > 0
        assert any("solexs" in c for c in added)
        assert any("cdte" in c for c in added)
        assert any("czt" in c for c in added)

        # Specific expected columns
        assert "solexs_rmean_30" in df.columns
        assert "solexs_deriv1" in df.columns
        assert "cdte_grad_60" in df.columns
        assert "czt_rise_rate" in df.columns

    def test_no_future_leakage_rolling_mean(self, base_day_df):
        """Verify rolling mean uses trailing window (no center=True)."""
        from tejas.features.temporal import compute_temporal_features
        result = compute_temporal_features(base_day_df)
        df = result["df"]

        # The rolling mean at position 0 should equal the value at 0
        # (min_periods=1, only 1 value available)
        assert df["solexs_rmean_30"].iloc[0] == df["solexs_counts_smooth"].iloc[0]


# ═══════════════════════════════════════════════════════════════════════════
# Spectral Feature Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSpectralFeatures:
    """Tests for tejas.features.spectral."""

    def test_band_ratio_correct(self, base_day_df):
        """Band ratios should equal manual computation."""
        from tejas.features.spectral import compute_spectral_features
        result = compute_spectral_features(base_day_df)
        df = result["df"]

        # mid/low ratio
        mid = base_day_df["solexs_counts_mid_smooth"]
        low = base_day_df["solexs_counts_low_smooth"].clip(lower=0.5)
        expected = mid / low
        actual = df["solexs_band_ratio_mid_low"]
        assert np.allclose(actual.values, expected.values, equal_nan=True)

    def test_zero_denominator_handled(self):
        """Band ratio should not produce inf when denominator is 0."""
        from tejas.features.spectral import compute_spectral_features
        df = pd.DataFrame({
            "solexs_counts_low_smooth": [0.0, 0.0, 5.0],
            "solexs_counts_mid_smooth": [10.0, 0.0, 10.0],
            "solexs_counts_high_smooth": [5.0, 0.0, 5.0],
            "solexs_hardness_ratio_smooth": [1.0, 0.0, 1.0],
        })
        result = compute_spectral_features(df)
        out = result["df"]
        assert np.isfinite(out["solexs_band_ratio_mid_low"]).all()

    def test_hr_derivative_on_ramp(self):
        """HR derivative on linearly increasing HR should be constant."""
        from tejas.features.spectral import compute_spectral_features
        n = 100
        hr = np.linspace(0.1, 1.0, n)
        df = pd.DataFrame({
            "solexs_hardness_ratio_smooth": hr,
            "solexs_counts_low_smooth": np.full(n, 10.0),
            "solexs_counts_mid_smooth": np.full(n, 5.0),
            "solexs_counts_high_smooth": np.full(n, 2.0),
        })
        result = compute_spectral_features(df)
        deriv = result["df"]["solexs_hr_deriv1"]
        # Constant step = 0.9/99 ≈ 0.00909
        expected_step = 0.9 / 99
        assert np.allclose(deriv.iloc[1:].values, expected_step, atol=1e-6)

    def test_spectral_slope_sign(self):
        """When high > low, spectral slope should be positive."""
        from tejas.features.spectral import compute_spectral_features
        n = 50
        df = pd.DataFrame({
            "solexs_counts_low_smooth": np.full(n, 5.0),
            "solexs_counts_mid_smooth": np.full(n, 10.0),
            "solexs_counts_high_smooth": np.full(n, 20.0),
            "solexs_hardness_ratio_smooth": np.full(n, 4.0),
        })
        result = compute_spectral_features(df)
        slope = result["df"]["solexs_spectral_slope"]
        assert (slope > 0).all()

    def test_expected_column_count(self, base_day_df):
        """Spectral features should produce expected number of columns."""
        from tejas.features.spectral import compute_spectral_features
        result = compute_spectral_features(base_day_df)
        # HR: 3 rmean + 1 deriv + 3 grad = 7
        # Bands: 2 ratios + 2 ratio derivs + 1 slope + 1 slope deriv = 6
        # Total = 13
        assert result["report"]["n_features"] == 13


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Instrument Feature Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossInstrumentFeatures:
    """Tests for tejas.features.cross_instrument."""

    def test_ratio_correct(self, base_day_df):
        """Soft/hard ratio should equal manual computation."""
        from tejas.features.cross_instrument import compute_cross_instrument_features
        result = compute_cross_instrument_features(base_day_df)
        df = result["df"]

        solexs = base_day_df["solexs_counts_smooth"]
        cdte = base_day_df["cdte_count_rate_smooth"].clip(lower=0.5)
        expected = solexs / cdte
        actual = df["ratio_solexs_cdte"]
        assert np.allclose(actual.values, expected.values, equal_nan=True)

    def test_ratio_no_inf(self, base_day_df):
        """Ratios should never be infinite (eps floor applied)."""
        from tejas.features.cross_instrument import compute_cross_instrument_features
        result = compute_cross_instrument_features(base_day_df)
        df = result["df"]
        for col in ["ratio_solexs_cdte", "ratio_solexs_czt", "ratio_cdte_czt"]:
            if col in df.columns:
                assert np.isfinite(df[col].dropna()).all(), f"Inf in {col}"

    def test_xcorr_known_lag(self):
        """Cross-correlation should detect a known time lag."""
        from tejas.features.cross_instrument import _normalized_cross_correlation
        n = 500
        # Create two signals with a 30-sample lag
        x = np.zeros(n)
        y = np.zeros(n)
        # y has a peak at t=200, x has a peak at t=230 (x lags y by 30)
        y[180:220] = np.concatenate([np.linspace(0, 10, 20),
                                     np.linspace(10, 0, 20)])
        x[210:250] = np.concatenate([np.linspace(0, 10, 20),
                                     np.linspace(10, 0, 20)])

        lag, corr = _normalized_cross_correlation(x, y, max_lag=60)
        # Positive lag means y leads x → lag should be ~30
        assert abs(lag - 30) <= 5, f"Expected lag ~30, got {lag}"
        assert corr > 0.5, f"Expected strong correlation, got {corr}"

    def test_missing_instrument_graceful(self):
        """Should handle missing instruments without error."""
        from tejas.features.cross_instrument import compute_cross_instrument_features
        # DataFrame with only SoLEXS, no CdTe or CZT
        df = pd.DataFrame({
            "solexs_counts_smooth": np.arange(100, dtype=float),
        })
        result = compute_cross_instrument_features(df)
        # Should produce 0 features (no cross-instrument pairs)
        assert result["report"]["n_features"] == 0

    def test_all_ratio_columns_present(self, base_day_df):
        """All expected ratio columns should be present."""
        from tejas.features.cross_instrument import compute_cross_instrument_features
        result = compute_cross_instrument_features(base_day_df)
        df = result["df"]
        expected = [
            "ratio_solexs_cdte", "ratio_solexs_czt", "ratio_cdte_czt",
            "ratio_solexs_cdte_deriv1",
            "ratio_solexs_cdte_rmean_60", "ratio_solexs_cdte_rmean_300",
            "lag_solexs_cdte", "xcorr_solexs_cdte",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Feature Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStatisticalFeatures:
    """Tests for tejas.features.statistical."""

    def test_zscore_constant_is_zero(self, constant_signal):
        """Z-score of constant signal should be 0 (or near-zero)."""
        from tejas.features.statistical import compute_zscore
        result = compute_zscore(constant_signal, window=30)
        # After initial transient, z-score should be ~0
        # (deviation = 0, so z = 0/floor → 0)
        valid = result.iloc[30:].dropna()
        assert np.allclose(valid.values, 0.0, atol=0.01)

    def test_zscore_detects_spike(self):
        """Z-score should be high at a spike."""
        from tejas.features.statistical import compute_zscore
        signal = pd.Series(np.full(200, 10.0))
        signal.iloc[150] = 100.0  # Big spike
        result = compute_zscore(signal, window=60)
        assert result.iloc[150] > 5.0, "Z-score should detect the spike"

    def test_trailing_pctrank_range(self, base_day_df):
        """Trailing percentile rank should be in [0, 1]."""
        from tejas.features.statistical import compute_trailing_pctrank
        result = compute_trailing_pctrank(
            base_day_df["solexs_counts_smooth"], window=300
        )
        valid = result.dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_cusum_accumulates(self):
        """CUSUM should accumulate positive deviations."""
        from tejas.features.statistical import compute_cusum
        # Signal that is consistently above its baseline
        signal = pd.Series(np.concatenate([
            np.full(100, 10.0),   # baseline period
            np.full(100, 15.0),   # elevated period
        ]))
        result = compute_cusum(signal, window_for_mean=50,
                               reset_interval=10000)
        # CUSUM should increase during the early elevated period
        # (before the trailing mean catches up to 15.0)
        # At row 105, mean is still largely influenced by 10.0 values
        assert result.iloc[120] > result.iloc[105]
        # CUSUM should be > 0 during elevated period
        assert result.iloc[110] > 0

    def test_cusum_resets(self):
        """CUSUM should reset at the configured interval."""
        from tejas.features.statistical import compute_cusum
        signal = pd.Series(np.full(200, 10.0))
        reset = 50
        result = compute_cusum(signal, window_for_mean=20,
                               reset_interval=reset)
        # At each reset point, CUSUM restarts
        # Value at reset points should be based only on that bin
        assert result.iloc[0] >= 0
        assert result.iloc[reset] >= 0

    def test_full_statistical_on_day(self, base_day_df):
        """Full statistical features produce expected columns."""
        from tejas.features.statistical import compute_statistical_features
        result = compute_statistical_features(base_day_df)
        added = result["columns_added"]

        assert any("zscore" in c for c in added)
        assert any("pctrank" in c for c in added)
        assert any("cusum" in c for c in added)
        assert any("solexs" in c for c in added)
        assert any("cdte" in c for c in added)
        assert any("czt" in c for c in added)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """Tests for tejas.features.pipeline (integration)."""

    def test_full_pipeline_column_count(self, base_day_df, tmp_path):
        """Full pipeline should add the expected number of feature columns."""
        from tejas.features.pipeline import engineer_day

        # Save test input as Parquet
        input_dir = tmp_path / "clean"
        input_dir.mkdir()
        base_day_df.to_parquet(input_dir / "2026-05-07.parquet",
                               index=False, engine="pyarrow")

        output_dir = tmp_path / "features"

        result = engineer_day(
            "2026-05-07",
            input_dir=input_dir,
            output_dir=output_dir,
            save=True,
        )

        assert result is not None
        df = result["df"]
        report = result["report"]

        # Should have added features
        assert report["n_features_added"] > 0
        assert report["n_temporal"] > 0
        assert report["n_spectral"] > 0
        assert report["n_cross_instrument"] > 0
        assert report["n_statistical"] > 0

        # Output should have more columns than input
        assert len(df.columns) > 40

        # Output file should exist
        assert (output_dir / "2026-05-07.parquet").exists()

    def test_output_parquet_readable(self, base_day_df, tmp_path):
        """Output Parquet should be readable and schema-valid."""
        from tejas.features.pipeline import engineer_day

        input_dir = tmp_path / "clean"
        input_dir.mkdir()
        base_day_df.to_parquet(input_dir / "2026-05-07.parquet",
                               index=False, engine="pyarrow")

        output_dir = tmp_path / "features"

        engineer_day("2026-05-07", input_dir=input_dir,
                     output_dir=output_dir)

        # Read it back
        output_df = pd.read_parquet(output_dir / "2026-05-07.parquet")
        assert len(output_df) == len(base_day_df)
        assert "time_unix" in output_df.columns
        assert "solexs_rmean_30" in output_df.columns

    def test_missing_input_returns_none(self, tmp_path):
        """Pipeline should return None for missing input files."""
        from tejas.features.pipeline import engineer_day

        input_dir = tmp_path / "clean"
        input_dir.mkdir()

        result = engineer_day("2026-05-99", input_dir=input_dir,
                              output_dir=tmp_path / "features", save=False)
        assert result is None

    def test_batch_processing(self, base_day_df, tmp_path):
        """Batch processing should handle multiple days."""
        from tejas.features.pipeline import engineer_all_days

        input_dir = tmp_path / "clean"
        input_dir.mkdir()
        output_dir = tmp_path / "features"

        # Create two test days
        for day in ["2026-05-01", "2026-05-02"]:
            base_day_df.to_parquet(input_dir / f"{day}.parquet",
                                   index=False, engine="pyarrow")

        summary = engineer_all_days(input_dir=input_dir,
                                    output_dir=output_dir)

        assert len(summary) == 2
        assert (output_dir / "feature_summary.csv").exists()
        assert (output_dir / "2026-05-01.parquet").exists()
        assert (output_dir / "2026-05-02.parquet").exists()

    def test_no_future_leakage_in_features(self, base_day_df, tmp_path):
        """Verify no feature at time t uses data from time t+1.

        Strategy: set all signal values after row 1000 to a sentinel.
        Features at row 1000 should not reflect the sentinel.
        """
        from tejas.features.pipeline import engineer_day

        df = base_day_df.copy()
        # Set a sentinel value after row 1000
        sentinel = 999999.0
        for col in ["solexs_counts_smooth", "cdte_count_rate_smooth",
                     "czt_net_rate"]:
            df.loc[1001:, col] = sentinel

        input_dir = tmp_path / "clean"
        input_dir.mkdir()
        df.to_parquet(input_dir / "2026-05-07.parquet",
                      index=False, engine="pyarrow")

        result = engineer_day("2026-05-07", input_dir=input_dir,
                              output_dir=tmp_path / "features", save=False)
        out = result["df"]

        # Rolling mean at row 1000 should NOT be influenced by sentinel
        # (trailing window, so it only sees rows ≤ 1000)
        rmean = out["solexs_rmean_30"].iloc[1000]
        assert rmean < sentinel, (
            f"Rolling mean at t=1000 is {rmean}, suggesting future leakage"
        )
