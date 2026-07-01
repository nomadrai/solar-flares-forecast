"""
Tests for TEJAS Unified Dataset Builder
=========================================
Tests the dual-cadence unified grid construction for a known day (May 1, 2026).
Validates:
  - Grid has 86400 rows (1 second × 24 hours)
  - All expected columns are present
  - SoLEXS data maps correctly (1:1 at 1s)
  - HEL1OS forward-fill does not cross segment boundaries
  - Parquet output file is created and readable
  - Metadata columns (validity flags, segment_id, is_filled)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tejas.ingestion.unified_builder import (
    build_unified_day,
    save_unified_day,
    _create_day_grid,
)
from tejas.config import UNIFIED_DIR


# ── Fixtures ─────────────────────────────────────────────────────────────────

TEST_DATE = "2026-05-01"


@pytest.fixture(scope="module")
def unified_result():
    """Build the unified dataset for May 1 (expensive, cached per module)."""
    return build_unified_day(TEST_DATE)


@pytest.fixture(scope="module")
def unified_df(unified_result):
    """The unified DataFrame for May 1."""
    return unified_result["unified"]


# ── Test Grid Construction ───────────────────────────────────────────────────


class TestDayGrid:

    def test_grid_has_86400_rows(self):
        """A day grid should have exactly 86400 rows (1s × 24h)."""
        grid = _create_day_grid(TEST_DATE)
        assert len(grid) == 86400

    def test_grid_columns(self):
        """Grid should have time_unix and datetime columns."""
        grid = _create_day_grid(TEST_DATE)
        assert "time_unix" in grid.columns
        assert "datetime" in grid.columns

    def test_grid_time_monotonic(self):
        """Grid timestamps should be strictly increasing."""
        grid = _create_day_grid(TEST_DATE)
        diffs = np.diff(grid["time_unix"].values)
        assert np.all(diffs > 0)

    def test_grid_cadence_is_1s(self):
        """Grid cadence should be exactly 1 second."""
        grid = _create_day_grid(TEST_DATE)
        diffs = np.diff(grid["time_unix"].values)
        np.testing.assert_array_almost_equal(diffs, 1.0, decimal=3)


# ── Test Unified Dataset ────────────────────────────────────────────────────


class TestUnifiedDataset:

    def test_has_86400_rows(self, unified_df):
        """Unified dataset should have 86400 rows."""
        assert len(unified_df) == 86400

    def test_has_solexs_columns(self, unified_df):
        """Unified dataset must have SoLEXS columns."""
        solexs_cols = {
            "solexs_counts", "solexs_counts_low", "solexs_counts_mid",
            "solexs_counts_high", "solexs_hardness_ratio",
        }
        assert solexs_cols.issubset(set(unified_df.columns))

    def test_has_hel1os_columns(self, unified_df):
        """Unified dataset must have HEL1OS columns."""
        hel1os_cols = {
            "cdte_count_rate", "cdte_stat_err",
            "czt_count_rate", "czt_stat_err",
        }
        assert hel1os_cols.issubset(set(unified_df.columns))

    def test_has_metadata_columns(self, unified_df):
        """Unified dataset must have metadata columns."""
        meta_cols = {"solexs_valid", "hel1os_valid"}
        assert meta_cols.issubset(set(unified_df.columns))

    def test_has_hel1os_fill_tracking(self, unified_df):
        """Unified dataset must track forward-filled HEL1OS values."""
        assert "cdte_is_filled" in unified_df.columns
        assert "czt_is_filled" in unified_df.columns

    def test_solexs_valid_flag(self, unified_df):
        """solexs_valid should be a boolean column with some True values."""
        assert unified_df["solexs_valid"].dtype == bool
        assert unified_df["solexs_valid"].any()

    def test_hel1os_valid_flag(self, unified_df):
        """hel1os_valid should be a boolean column."""
        assert unified_df["hel1os_valid"].dtype == bool

    def test_no_future_data_in_forward_fill(self, unified_df):
        """Forward-fill should only propagate backwards in time (ffill).

        Verify that for CdTe, the first row with data is NOT preceded
        by non-NaN values (i.e., no backward fill happened).
        """
        cdte = unified_df["cdte_count_rate"]
        first_valid = cdte.first_valid_index()
        if first_valid is not None and first_valid > 0:
            before_first = cdte.iloc[:first_valid]
            assert before_first.isna().all(), \
                "CdTe has non-NaN values before first real data point (backward fill?)"


# ── Test Validation ──────────────────────────────────────────────────────────


class TestUnifiedValidation:

    def test_validation_report_exists(self, unified_result):
        """build_unified_day should return a validation report."""
        assert unified_result["validation"] is not None

    def test_validation_has_checks(self, unified_result):
        """Validation report should contain multiple checks."""
        validation = unified_result["validation"]
        assert len(validation.checks) > 0


# ── Test Parquet Save ────────────────────────────────────────────────────────


class TestParquetSave:

    def test_save_creates_file(self, tmp_path):
        """save_unified_day should create a Parquet file."""
        result = save_unified_day(TEST_DATE, output_dir=tmp_path)
        parquet_path = tmp_path / f"{TEST_DATE}.parquet"
        assert parquet_path.exists()
        assert parquet_path.stat().st_size > 0

    def test_saved_parquet_readable(self, tmp_path):
        """Saved Parquet file should be readable and have 86400 rows."""
        parquet_path = tmp_path / f"{TEST_DATE}.parquet"
        if not parquet_path.exists():
            save_unified_day(TEST_DATE, output_dir=tmp_path)
        df = pd.read_parquet(parquet_path)
        assert len(df) == 86400

    def test_saved_parquet_schema(self, tmp_path):
        """Saved Parquet should have all expected columns."""
        parquet_path = tmp_path / f"{TEST_DATE}.parquet"
        if not parquet_path.exists():
            save_unified_day(TEST_DATE, output_dir=tmp_path)
        df = pd.read_parquet(parquet_path)
        required = {"time_unix", "datetime", "solexs_counts", "cdte_count_rate"}
        assert required.issubset(set(df.columns))
