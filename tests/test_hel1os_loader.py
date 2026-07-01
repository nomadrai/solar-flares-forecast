"""
Tests for TEJAS HEL1OS Loader
===============================
Tests loading CdTe and CZT light curves for a known day (May 1, 2026).
Validates:
  - Data loading and column structure
  - Dump merging (overlap handling for May 3, 4, 7)
  - Gap detection (tiered classification)
  - Resampling (10s bins, error propagation)
  - Segment ID assignment
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tejas.loaders import hel1os
from tejas.loaders.utils import detect_gaps, GapInfo


# ── Fixtures ─────────────────────────────────────────────────────────────────

TEST_DATE = "2026-05-01"
OVERLAP_DATE = "2026-05-03"  # Known overlap day from EDA


@pytest.fixture(scope="module")
def cdte_data():
    """Load CdTe full-range light curve for May 1."""
    return hel1os.load_detector_lightcurves(TEST_DATE, "cdte")


@pytest.fixture(scope="module")
def czt_data():
    """Load CZT full-range light curve for May 1."""
    return hel1os.load_detector_lightcurves(TEST_DATE, "czt")


@pytest.fixture(scope="module")
def cdte_gti():
    """Load CdTe GTI for May 1."""
    return hel1os.load_gti(TEST_DATE, "cdte")


@pytest.fixture(scope="module")
def day_data():
    """Load all HEL1OS data for May 1."""
    return hel1os.load_day(TEST_DATE)


# ── Test CdTe Loader ────────────────────────────────────────────────────────


class TestHEL1OSCdTe:

    def test_cdte_loads_successfully(self, cdte_data):
        """CdTe data should load and return a DataFrame."""
        assert cdte_data is not None
        assert isinstance(cdte_data, pd.DataFrame)
        assert len(cdte_data) > 0

    def test_cdte_has_required_columns(self, cdte_data):
        """CdTe DataFrame must have expected columns."""
        required = {"time_unix", "count_rate", "stat_err", "datetime"}
        assert required.issubset(set(cdte_data.columns))

    def test_cdte_count_rate_dtype(self, cdte_data):
        """count_rate must be float64."""
        assert cdte_data["count_rate"].dtype == np.float64

    def test_cdte_has_segment_id(self, cdte_data):
        """Processed CdTe data should have segment_id column."""
        assert "segment_id" in cdte_data.columns

    def test_cdte_has_gap_mask(self, cdte_data):
        """Processed CdTe data should have gap_mask column."""
        assert "gap_mask" in cdte_data.columns

    def test_cdte_time_mostly_monotonic(self, cdte_data):
        """Timestamps should be monotonically increasing after processing."""
        times = cdte_data["time_unix"].dropna().values
        diffs = np.diff(times)
        # Allow a small number of non-positive diffs from gap marker insertion
        n_backwards = np.sum(diffs < 0)
        assert n_backwards == 0, f"{n_backwards} backwards jump(s)"


# ── Test CZT Loader ─────────────────────────────────────────────────────────


class TestHEL1OSCZT:

    def test_czt_loads_successfully(self, czt_data):
        """CZT data should load and return a DataFrame."""
        assert czt_data is not None
        assert isinstance(czt_data, pd.DataFrame)
        assert len(czt_data) > 0

    def test_czt_has_required_columns(self, czt_data):
        """CZT DataFrame must have expected columns."""
        required = {"time_unix", "count_rate", "stat_err", "datetime"}
        assert required.issubset(set(czt_data.columns))


# ── Test GTI Loader ──────────────────────────────────────────────────────────


class TestHEL1OSGTI:

    def test_gti_loads_successfully(self, cdte_gti):
        """CdTe GTI should load and return a DataFrame."""
        # HEL1OS GTI may or may not exist depending on data
        # Just verify it returns a DataFrame or None
        assert cdte_gti is None or isinstance(cdte_gti, pd.DataFrame)

    def test_gti_structure(self, cdte_gti):
        """If GTI exists, it should have start/stop columns."""
        if cdte_gti is not None and len(cdte_gti) > 0:
            assert "start" in cdte_gti.columns
            assert "stop" in cdte_gti.columns


# ── Test Gap Detection ───────────────────────────────────────────────────────


class TestGapDetection:

    def test_detect_gaps_no_gaps(self):
        """Continuous 1s cadence should produce no gaps."""
        times = np.arange(0, 100, 1.0)
        gap_info = detect_gaps(times)
        assert gap_info.n_short == 0
        assert gap_info.n_medium == 0
        assert gap_info.n_long == 0

    def test_detect_gaps_short(self):
        """A 10s gap should be classified as 'short' (≤30s)."""
        times = np.concatenate([
            np.arange(0, 50, 1.0),     # 0-49
            np.arange(60, 110, 1.0),   # 60-109 (10s gap)
        ])
        gap_info = detect_gaps(times)
        assert gap_info.n_short == 1
        assert gap_info.n_medium == 0
        assert gap_info.n_long == 0
        # Short gaps don't increment segment_id
        assert gap_info.segment_ids[-1] == 0

    def test_detect_gaps_medium(self):
        """A 60s gap should be classified as 'medium' (30-120s)."""
        times = np.concatenate([
            np.arange(0, 50, 1.0),
            np.arange(110, 160, 1.0),  # 60s gap
        ])
        gap_info = detect_gaps(times)
        assert gap_info.n_medium == 1
        # Medium gaps increment segment_id
        assert gap_info.segment_ids[-1] == 1

    def test_detect_gaps_long(self):
        """A 200s gap should be classified as 'long' (>120s)."""
        times = np.concatenate([
            np.arange(0, 50, 1.0),
            np.arange(250, 300, 1.0),  # 200s gap
        ])
        gap_info = detect_gaps(times)
        assert gap_info.n_long == 1
        assert gap_info.segment_ids[-1] == 1

    def test_detect_gaps_multiple(self):
        """Multiple gaps of different types should be classified correctly."""
        times = np.concatenate([
            np.arange(0, 50, 1.0),      # Segment 0
            np.arange(60, 110, 1.0),    # Short gap (10s)
            np.arange(200, 250, 1.0),   # Medium gap (90s)
            np.arange(500, 550, 1.0),   # Long gap (250s)
        ])
        gap_info = detect_gaps(times)
        assert gap_info.n_short == 1
        assert gap_info.n_medium == 1
        assert gap_info.n_long == 1
        # Segment IDs: 0, 0, 1, 2
        assert gap_info.segment_ids[0] == 0
        assert gap_info.segment_ids[60] == 0   # after short gap, same segment
        assert gap_info.segment_ids[110] == 1  # after medium gap, new segment
        assert gap_info.segment_ids[160] == 2  # after long gap, new segment

    def test_detect_gaps_empty(self):
        """Empty input should return empty GapInfo."""
        gap_info = detect_gaps(np.array([]))
        assert gap_info.n_short == 0
        assert len(gap_info.segment_ids) == 0


# ── Test Overlap Handling ────────────────────────────────────────────────────


class TestOverlapHandling:

    def test_overlap_resolution_may3(self):
        """May 3 has a known overlap — verify it's resolved."""
        data = hel1os.load_detector_lightcurves(OVERLAP_DATE, "cdte")
        if data is not None:
            times = data["time_unix"].dropna().values
            diffs = np.diff(times)
            # After overlap resolution, no backwards jumps
            assert np.all(diffs >= 0), "Overlap not resolved: backwards timestamps found"


# ── Test Resampling ──────────────────────────────────────────────────────────


class TestResampling:

    def test_resampling_reduces_rows(self):
        """Resampling to 10s should produce ~10x fewer rows than 1s."""
        data_1s = hel1os.load_detector_lightcurves(
            TEST_DATE, "cdte", resample_cadence_s=1
        )
        data_10s = hel1os.load_detector_lightcurves(
            TEST_DATE, "cdte", resample_cadence_s=10
        )
        if data_1s is not None and data_10s is not None:
            # Allow some tolerance for gap markers and edge effects
            ratio = len(data_1s) / max(len(data_10s), 1)
            assert ratio > 5, f"Resampling ratio {ratio:.1f} is too low (expected ~10)"

    def test_resampled_data_has_valid_counts(self):
        """Resampled data should have non-negative count rates."""
        data = hel1os.load_detector_lightcurves(TEST_DATE, "cdte")
        if data is not None:
            valid = data["count_rate"].dropna()
            assert np.all(valid.values >= 0), "Negative count rates after resampling"


# ── Test Day Loader ──────────────────────────────────────────────────────────


class TestHEL1OSDay:

    def test_day_returns_dict(self, day_data):
        """load_day should return a dict with expected keys."""
        assert isinstance(day_data, dict)
        expected_keys = {"cdte", "czt", "cdte_bands", "czt_bands",
                         "cdte_gti", "czt_gti"}
        assert expected_keys.issubset(set(day_data.keys()))
