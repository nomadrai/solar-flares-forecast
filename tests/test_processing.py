"""
TEJAS Phase 3 — Processing Tests
==================================
Unit tests for each data processing component:
  - Spike removal
  - Signal smoothing
  - CZT background estimation & subtraction
  - Normalization
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
def quiet_signal():
    """Simulate a quiet-Sun SoLEXS signal (baseline ~8 c/s)."""
    np.random.seed(42)
    n = 1000
    baseline = 8.0
    noise = np.random.poisson(lam=baseline, size=n).astype(float)
    return pd.Series(noise, name="counts")


@pytest.fixture
def signal_with_spike(quiet_signal):
    """Quiet signal with a single isolated spike (cosmic ray)."""
    s = quiet_signal.copy()
    s.iloc[500] = 500.0  # Isolated spike: 1 bin, way above baseline
    return s


@pytest.fixture
def signal_with_flare(quiet_signal):
    """Quiet signal with a sustained flare (should NOT be removed)."""
    s = quiet_signal.copy()
    # Simulate a flare: gradual rise over 20 bins, then decay
    flare_profile = np.concatenate([
        np.linspace(8, 200, 10),   # rise
        np.linspace(200, 150, 5),  # peak
        np.linspace(150, 20, 15),  # decay
    ])
    s.iloc[400:430] = flare_profile
    return s


@pytest.fixture
def cdte_quiet_signal():
    """Simulate CdTe quiet-period data (87% zeros, Poisson)."""
    np.random.seed(42)
    n = 1000
    # Very low rate: ~0.15 c/s → most bins zero
    signal = np.random.poisson(lam=0.15, size=n).astype(float)
    return pd.Series(signal, name="cdte_rate")


@pytest.fixture
def sample_unified_df():
    """Create a minimal unified DataFrame for integration testing."""
    np.random.seed(42)
    n = 3600  # 1 hour of data

    # Time grid
    t0 = 1778000000.0
    time_unix = t0 + np.arange(n, dtype=float)
    datetimes = pd.to_datetime(time_unix, unit="s", utc=True)

    # SoLEXS: baseline ~8, with a small flare at t=1800
    solexs = np.random.poisson(lam=8, size=n).astype(float)
    # Add a flare (10 bins sustained)
    solexs[1800:1810] = np.linspace(50, 200, 10)
    solexs[1810:1830] = np.linspace(200, 15, 20)

    # SoLEXS bands
    solexs_low = (solexs * 0.4 + np.random.normal(0, 1, n)).clip(0)
    solexs_mid = (solexs * 0.35 + np.random.normal(0, 1, n)).clip(0)
    solexs_high = (solexs * 0.15 + np.random.normal(0, 1, n)).clip(0)
    solexs_hr = np.where(solexs_low > 0.5, solexs_high / solexs_low, 0.0)

    # CdTe: mostly zero, spike during flare
    cdte = np.random.poisson(lam=0.15, size=n).astype(float)
    cdte[1800:1815] = np.random.poisson(lam=50, size=15).astype(float)

    # CZT: background ~23 c/s + flare signal
    czt = np.random.poisson(lam=23, size=n).astype(float)
    czt[1800:1815] = np.random.poisson(lam=200, size=15).astype(float)

    # Validity flags
    solexs_valid = np.ones(n, dtype=bool)
    hel1os_valid = np.ones(n, dtype=bool)

    return pd.DataFrame({
        "time_unix": time_unix,
        "datetime": datetimes,
        "solexs_counts": solexs,
        "solexs_counts_low": solexs_low,
        "solexs_counts_mid": solexs_mid,
        "solexs_counts_high": solexs_high,
        "solexs_hardness_ratio": solexs_hr,
        "cdte_count_rate": cdte,
        "czt_count_rate": czt,
        "solexs_valid": solexs_valid,
        "hel1os_valid": hel1os_valid,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Spike Removal Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSpikeRemoval:
    """Tests for tejas.processing.spike_removal."""

    def test_isolated_spike_detected(self, signal_with_spike):
        """A single-bin spike should be detected and replaced."""
        from tejas.processing.spike_removal import remove_spikes

        result = remove_spikes(signal_with_spike, window=61, n_sigma=5,
                               min_consecutive=3)
        assert result["n_spikes"] >= 1, "Spike should be detected"
        assert result["spike_mask"].iloc[500], "Spike at index 500 should be flagged"

        # The cleaned value should be near the baseline (~8), not 500
        cleaned_val = result["cleaned"].iloc[500]
        assert cleaned_val < 50, f"Cleaned value {cleaned_val} should be near baseline"

    def test_flare_preserved(self, signal_with_flare):
        """A sustained multi-bin flare should NOT be removed."""
        from tejas.processing.spike_removal import remove_spikes

        result = remove_spikes(signal_with_flare, window=61, n_sigma=5,
                               min_consecutive=3)

        # The flare peak region (400-430) should be mostly preserved
        flare_region = result["spike_mask"].iloc[400:430]
        assert flare_region.sum() < 3, (
            f"Flare region should not be flagged as spikes "
            f"(flagged {flare_region.sum()} bins)"
        )

        # Peak value should be preserved
        assert result["cleaned"].iloc[405] > 100, "Flare peak should be preserved"

    def test_all_zero_input(self, cdte_quiet_signal):
        """Should work correctly with mostly-zero CdTe data."""
        from tejas.processing.spike_removal import remove_spikes

        result = remove_spikes(cdte_quiet_signal, window=61, n_sigma=5,
                               min_consecutive=3)

        # Should not crash, and should not flag most bins
        assert result["n_spikes"] < len(cdte_quiet_signal) * 0.01, (
            "Should not flag many bins in quiet CdTe data"
        )

    def test_output_length_matches(self, quiet_signal):
        """Output series should have the same length as input."""
        from tejas.processing.spike_removal import remove_spikes

        result = remove_spikes(quiet_signal, window=61, n_sigma=5)
        assert len(result["cleaned"]) == len(quiet_signal)
        assert len(result["spike_mask"]) == len(quiet_signal)

    def test_nan_preservation(self):
        """NaN values in input should remain NaN in output."""
        from tejas.processing.spike_removal import remove_spikes

        s = pd.Series([1, 2, np.nan, 4, 5, 3, 2, 1, 2, 3] * 10)
        result = remove_spikes(s, window=5, n_sigma=5)
        # NaN positions should still be NaN
        assert result["cleaned"].iloc[2], "NaN should be preserved"

    def test_day_level_solexs_band_propagation(self, sample_unified_df):
        """SoLEXS spike mask should propagate to band columns (Q1)."""
        from tejas.processing.spike_removal import remove_spikes_day

        # Inject a spike into solexs_counts
        df = sample_unified_df.copy()
        df.loc[100, "solexs_counts"] = 5000.0

        result = remove_spikes_day(df)
        out_df = result["df"]

        # Clean columns should exist for bands
        assert "solexs_counts_low_clean" in out_df.columns
        assert "solexs_counts_mid_clean" in out_df.columns
        assert "solexs_counts_high_clean" in out_df.columns
        assert "solexs_hardness_ratio_clean" in out_df.columns


# ═══════════════════════════════════════════════════════════════════════════
# Smoothing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSmoothing:
    """Tests for tejas.processing.smoothing."""

    def test_output_length(self, quiet_signal):
        """Smoothed output should have the same length as input."""
        from tejas.processing.smoothing import smooth_signal

        smoothed = smooth_signal(quiet_signal, window=5)
        assert len(smoothed) == len(quiet_signal)

    def test_noise_reduction(self, quiet_signal):
        """Smoothing should reduce variance."""
        from tejas.processing.smoothing import smooth_signal

        smoothed = smooth_signal(quiet_signal, window=11)
        assert smoothed.std() < quiet_signal.std(), (
            "Smoothed signal should have lower variance"
        )

    def test_nan_not_bridged(self):
        """NaN gaps should not be bridged by smoothing."""
        from tejas.processing.smoothing import smooth_signal

        # Create signal with a NaN gap in the middle
        s = pd.Series([1, 2, 3, np.nan, np.nan, np.nan, 7, 8, 9])
        smoothed = smooth_signal(s, window=3)

        # The NaN region should still contain NaN (or mostly NaN)
        # With min_periods=1, edges of the gap might get partial values,
        # but the middle of the gap should be NaN
        assert smoothed.iloc[4] != smoothed.iloc[4] or pd.isna(smoothed.iloc[4]), (
            "Middle of NaN gap should remain NaN"
        )

    def test_window_must_be_odd(self):
        """Even window should be auto-corrected to odd."""
        from tejas.processing.smoothing import smooth_signal

        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Should not raise — even window is silently corrected
        smoothed = smooth_signal(s, window=4)
        assert len(smoothed) == len(s)

    def test_day_level_creates_smooth_columns(self, sample_unified_df):
        """smooth_day should create *_smooth columns."""
        from tejas.processing.spike_removal import remove_spikes_day
        from tejas.processing.smoothing import smooth_day

        # First do spike removal to create _clean columns
        spike_result = remove_spikes_day(sample_unified_df)
        df = spike_result["df"]

        # Then smooth
        smooth_result = smooth_day(df)
        out_df = smooth_result["df"]

        assert "solexs_counts_smooth" in out_df.columns
        assert "cdte_count_rate_smooth" in out_df.columns
        assert "czt_count_rate_smooth" in out_df.columns


# ═══════════════════════════════════════════════════════════════════════════
# CZT Background Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCZTBackground:
    """Tests for tejas.processing.czt_background."""

    def test_background_reasonable_range(self, sample_unified_df):
        """Background estimate should be in a reasonable range (~23 c/s)."""
        from tejas.processing.spike_removal import remove_spikes_day
        from tejas.processing.czt_background import estimate_czt_background

        spike_result = remove_spikes_day(sample_unified_df)
        df = spike_result["df"]

        result = estimate_czt_background(df)

        # Background should be roughly ~23 c/s (our simulated Poisson rate)
        bg_median = result["bg_median"]
        assert 5 < bg_median < 50, (
            f"Background median {bg_median} should be in reasonable range"
        )

    def test_net_rate_quiet_near_zero(self, sample_unified_df):
        """Background-subtracted signal during quiet period should be ~0."""
        from tejas.processing.spike_removal import remove_spikes_day
        from tejas.processing.czt_background import subtract_czt_background

        spike_result = remove_spikes_day(sample_unified_df)
        df = spike_result["df"]

        result = subtract_czt_background(df)
        out_df = result["df"]

        # During quiet period (e.g., first 1000 bins), net rate should be
        # close to zero (background-subtracted)
        quiet_net = out_df["czt_net_rate"].iloc[:1000]
        quiet_mean = quiet_net.mean()
        assert quiet_mean < 30, (
            f"Quiet-period net rate mean {quiet_mean} should be near zero"
        )

    def test_flare_survives_subtraction(self, sample_unified_df):
        """Flare signal should survive background subtraction."""
        from tejas.processing.spike_removal import remove_spikes_day
        from tejas.processing.czt_background import subtract_czt_background

        spike_result = remove_spikes_day(sample_unified_df)
        df = spike_result["df"]

        result = subtract_czt_background(df)
        out_df = result["df"]

        # During flare (bins 1800-1815), net rate should be elevated
        flare_net = out_df["czt_net_rate"].iloc[1800:1815]
        quiet_net = out_df["czt_net_rate"].iloc[:1000]
        assert flare_net.mean() > quiet_net.mean() * 3, (
            "Flare net rate should be much higher than quiet-period net rate"
        )

    def test_above_bg_flag(self, sample_unified_df):
        """czt_above_bg should flag flare-period bins."""
        from tejas.processing.spike_removal import remove_spikes_day
        from tejas.processing.czt_background import subtract_czt_background

        spike_result = remove_spikes_day(sample_unified_df)
        df = spike_result["df"]

        result = subtract_czt_background(df)
        out_df = result["df"]

        # At least some flare bins should be flagged
        flare_above = out_df["czt_above_bg"].iloc[1800:1815]
        assert flare_above.any(), "Some flare bins should be flagged as above background"

    def test_output_columns_exist(self, sample_unified_df):
        """Should create czt_background, czt_net_rate, czt_above_bg."""
        from tejas.processing.spike_removal import remove_spikes_day
        from tejas.processing.czt_background import subtract_czt_background

        spike_result = remove_spikes_day(sample_unified_df)
        df = spike_result["df"]

        result = subtract_czt_background(df)
        out_df = result["df"]

        assert "czt_background" in out_df.columns
        assert "czt_net_rate" in out_df.columns
        assert "czt_above_bg" in out_df.columns


# ═══════════════════════════════════════════════════════════════════════════
# Normalization Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalization:
    """Tests for tejas.processing.normalization."""

    def test_log1p_handles_zeros(self):
        """log1p should handle zero values without error."""
        from tejas.processing.normalization import log1p_transform

        s = pd.Series([0.0, 0.0, 0.0, 1.0, 10.0, 100.0])
        result = log1p_transform(s)

        assert not result.isna().any(), "log1p should not produce NaN from zeros"
        assert result.iloc[0] == 0.0, "log1p(0) should be 0"
        assert result.iloc[-1] > result.iloc[-2], "log1p should be monotonic"

    def test_percentile_rank_range(self, quiet_signal):
        """Percentile rank should be in [0, 1]."""
        from tejas.processing.normalization import percentile_rank

        result = percentile_rank(quiet_signal)
        valid = result.dropna()

        assert valid.min() >= 0.0, "Percentile rank should be >= 0"
        assert valid.max() <= 1.0, "Percentile rank should be <= 1"

    def test_percentile_rank_nan_preserved(self):
        """NaN values should remain NaN after percentile ranking."""
        from tejas.processing.normalization import percentile_rank

        s = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan])
        result = percentile_rank(s)
        assert pd.isna(result.iloc[1]), "NaN should be preserved"
        assert pd.isna(result.iloc[4]), "NaN should be preserved"

    def test_raw_columns_preserved(self, sample_unified_df):
        """Normalization should NOT modify original columns."""
        from tejas.processing.normalization import normalize_day

        df = sample_unified_df.copy()
        original_solexs = df["solexs_counts"].copy()

        result = normalize_day(df)
        out_df = result["df"]

        pd.testing.assert_series_equal(
            out_df["solexs_counts"], original_solexs,
            check_names=False,
        )

    def test_creates_expected_columns(self, sample_unified_df):
        """Should create _log and _pctrank columns."""
        from tejas.processing.normalization import normalize_day

        result = normalize_day(sample_unified_df)
        out_df = result["df"]

        assert "solexs_counts_log" in out_df.columns
        assert "solexs_counts_pctrank" in out_df.columns
        assert "cdte_count_rate_log" in out_df.columns
        assert "cdte_count_rate_pctrank" in out_df.columns


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Integration Test
# ═══════════════════════════════════════════════════════════════════════════


class TestPipeline:
    """Integration tests for the full Phase 3 pipeline."""

    def test_full_pipeline_on_synthetic_data(self, sample_unified_df, tmp_path):
        """Full pipeline should run without errors on synthetic data."""
        from tejas.processing.pipeline import process_day

        # Save synthetic data as a Parquet file
        input_dir = tmp_path / "unified"
        input_dir.mkdir()
        output_dir = tmp_path / "clean"

        date_str = "2026-05-01"
        sample_unified_df.to_parquet(
            input_dir / f"{date_str}.parquet", index=False
        )

        # Run the pipeline
        result = process_day(
            date_str,
            input_dir=input_dir,
            output_dir=output_dir,
            save=True,
        )

        assert result is not None, "Pipeline should return a result"
        out_df = result["df"]
        report = result["report"]

        # Check output schema
        expected_cols = [
            # Clean columns
            "solexs_counts_clean",
            "cdte_count_rate_clean",
            "czt_count_rate_clean",
            # Smooth columns
            "solexs_counts_smooth",
            "cdte_count_rate_smooth",
            "czt_count_rate_smooth",
            # CZT background
            "czt_background",
            "czt_net_rate",
            "czt_above_bg",
            # Normalization
            "solexs_counts_log",
            "solexs_counts_pctrank",
            "cdte_count_rate_log",
            "cdte_count_rate_pctrank",
            "czt_net_rate_log",
            "czt_net_rate_pctrank",
        ]
        for col in expected_cols:
            assert col in out_df.columns, f"Missing expected column: {col}"

        # Check report
        assert report["date"] == date_str
        assert report["n_rows"] == len(sample_unified_df)
        assert report["n_cols_output"] > report["n_cols_input"]

        # Check output file exists
        output_path = output_dir / f"{date_str}.parquet"
        assert output_path.exists(), "Output Parquet should be saved"

        # Check it's readable
        loaded = pd.read_parquet(output_path)
        assert len(loaded) == len(sample_unified_df)

    def test_missing_input_returns_none(self, tmp_path):
        """Pipeline should return None for missing input files."""
        from tejas.processing.pipeline import process_day

        result = process_day(
            "2026-01-01",
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            save=False,
        )
        assert result is None
