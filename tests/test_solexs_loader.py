"""
Tests for TEJAS SoLEXS Loader
==============================
Tests loading LC, PI, and GTI data for a known day (May 1, 2026).
Validates:
  - Correct column names and types
  - GTI masking (values outside GTI become NaN, time grid preserved)
  - Hardness ratio has no division-by-zero errors
  - LC-to-PI ratio validation passes (~5.5x)
  - Row count match between LC and PI
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tejas.loaders import solexs
from tejas.loaders.utils import safe_divide


# ── Fixtures ─────────────────────────────────────────────────────────────────

TEST_DATE = "2026-05-01"


@pytest.fixture(scope="module")
def lc_data():
    """Load SoLEXS LC for May 1 (with GTI applied)."""
    return solexs.load_lc(TEST_DATE, apply_gti=True)


@pytest.fixture(scope="module")
def lc_data_no_gti():
    """Load SoLEXS LC for May 1 (without GTI)."""
    return solexs.load_lc(TEST_DATE, apply_gti=False)


@pytest.fixture(scope="module")
def pi_data():
    """Load SoLEXS PI for May 1 (with GTI applied)."""
    return solexs.load_pi(TEST_DATE, apply_gti=True)


@pytest.fixture(scope="module")
def gti_data():
    """Load SoLEXS GTI for May 1."""
    return solexs.load_gti(TEST_DATE)


@pytest.fixture(scope="module")
def day_data():
    """Load all SoLEXS data for May 1."""
    return solexs.load_day(TEST_DATE, validate=True)


# ── Test LC Loader ───────────────────────────────────────────────────────────


class TestSoLEXSLC:

    def test_lc_loads_successfully(self, lc_data):
        """LC data should load and return a DataFrame."""
        assert lc_data is not None
        assert isinstance(lc_data, pd.DataFrame)
        assert len(lc_data) > 0

    def test_lc_has_required_columns(self, lc_data):
        """LC DataFrame must have time_unix, counts, datetime columns."""
        required = {"time_unix", "counts", "datetime"}
        assert required.issubset(set(lc_data.columns))

    def test_lc_time_is_float64(self, lc_data):
        """time_unix must be float64 (not int, to preserve sub-second precision)."""
        assert lc_data["time_unix"].dtype == np.float64

    def test_lc_counts_is_float64(self, lc_data):
        """counts must be float64 (allows NaN for GTI masking)."""
        assert lc_data["counts"].dtype == np.float64

    def test_lc_datetime_is_utc(self, lc_data):
        """datetime column must be timezone-aware (UTC)."""
        assert lc_data["datetime"].dt.tz is not None

    def test_lc_time_monotonic(self, lc_data):
        """Timestamps must be monotonically increasing."""
        diffs = np.diff(lc_data["time_unix"].values)
        assert np.all(diffs >= 0), "Timestamps are not monotonically increasing"

    def test_gti_preserves_time_grid(self, lc_data, lc_data_no_gti):
        """GTI masking should preserve the time grid (same row count)."""
        assert len(lc_data) == len(lc_data_no_gti)

    def test_gti_creates_nans(self, lc_data, lc_data_no_gti):
        """GTI masking should set some values to NaN (unless 100% GTI coverage)."""
        n_nan_gti = lc_data["counts"].isna().sum()
        n_nan_no_gti = lc_data_no_gti["counts"].isna().sum()
        # With GTI applied, there should be at least as many NaNs
        assert n_nan_gti >= n_nan_no_gti


# ── Test PI Loader ───────────────────────────────────────────────────────────


class TestSoLEXSPI:

    def test_pi_loads_successfully(self, pi_data):
        """PI data should load and return a DataFrame."""
        assert pi_data is not None
        assert isinstance(pi_data, pd.DataFrame)
        assert len(pi_data) > 0

    def test_pi_has_required_columns(self, pi_data):
        """PI DataFrame must have multi-band columns."""
        required = {
            "time_unix", "counts_low", "counts_mid", "counts_high",
            "counts_total_pi", "hardness_ratio", "datetime",
        }
        assert required.issubset(set(pi_data.columns))

    def test_pi_band_sum_equals_total(self, pi_data):
        """counts_total_pi should equal sum of low + mid + high."""
        valid = pi_data.dropna(subset=["counts_low", "counts_mid", "counts_high"])
        if len(valid) > 0:
            computed = valid["counts_low"] + valid["counts_mid"] + valid["counts_high"]
            np.testing.assert_array_almost_equal(
                valid["counts_total_pi"].values,
                computed.values,
                decimal=5,
            )

    def test_hardness_ratio_no_inf(self, pi_data):
        """Hardness ratio should never be inf (zero-safe division)."""
        valid = pi_data["hardness_ratio"].dropna()
        assert not np.any(np.isinf(valid.values)), "Inf values in hardness_ratio"

    def test_hardness_ratio_non_negative(self, pi_data):
        """Hardness ratio should be non-negative (counts are non-negative)."""
        valid = pi_data["hardness_ratio"].dropna()
        assert np.all(valid.values >= 0), "Negative values in hardness_ratio"


# ── Test GTI Loader ──────────────────────────────────────────────────────────


class TestSoLEXSGTI:

    def test_gti_loads_successfully(self, gti_data):
        """GTI data should load and return a DataFrame."""
        assert gti_data is not None
        assert isinstance(gti_data, pd.DataFrame)

    def test_gti_has_required_columns(self, gti_data):
        """GTI DataFrame must have start, stop, duration_s columns."""
        required = {"start", "stop", "start_dt", "stop_dt", "duration_s"}
        assert required.issubset(set(gti_data.columns))

    def test_gti_start_before_stop(self, gti_data):
        """Every GTI interval must have start < stop."""
        assert np.all(gti_data["start"] < gti_data["stop"])

    def test_gti_durations_positive(self, gti_data):
        """All GTI durations must be positive."""
        assert np.all(gti_data["duration_s"] > 0)


# ── Test Day Loader ──────────────────────────────────────────────────────────


class TestSoLEXSDay:

    def test_day_returns_dict(self, day_data):
        """load_day should return a dict with lc, pi, gti, validation keys."""
        assert isinstance(day_data, dict)
        assert "lc" in day_data
        assert "pi" in day_data
        assert "gti" in day_data
        assert "validation" in day_data

    def test_row_count_match(self, day_data):
        """LC and PI should have the same number of rows."""
        lc = day_data["lc"]
        pi = day_data["pi"]
        if lc is not None and pi is not None:
            assert len(lc) == len(pi)

    def test_lc_pi_ratio_validation(self, day_data):
        """The LC-to-PI ratio validation should pass."""
        validation = day_data["validation"]
        if validation is not None:
            assert validation["passed"], f"Validation failed: {validation['warnings']}"


# ── Test safe_divide ─────────────────────────────────────────────────────────


class TestSafeDivide:

    def test_normal_division(self):
        """Normal division should work as expected."""
        result = safe_divide(np.array([10.0]), np.array([2.0]))
        np.testing.assert_almost_equal(result, [5.0])

    def test_zero_denominator(self):
        """Zero denominator should use eps, not produce inf."""
        result = safe_divide(np.array([10.0]), np.array([0.0]), eps=0.5)
        assert np.isfinite(result[0])
        np.testing.assert_almost_equal(result, [20.0])  # 10 / 0.5

    def test_negative_denominator_below_eps(self):
        """Very small positive denominator below eps should be clamped."""
        result = safe_divide(np.array([10.0]), np.array([0.1]), eps=0.5)
        np.testing.assert_almost_equal(result, [20.0])  # 10 / max(0.1, 0.5) = 10/0.5
