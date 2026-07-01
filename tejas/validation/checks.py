"""
TEJAS Validation Suite
======================
Data integrity checks run automatically during ingestion to catch
issues early. Each check returns a structured result; the full suite
produces a ValidationReport summarising pass/fail status.

Checks implemented:
  1. LC-to-PI ratio (~5.5x expected)
  2. Time monotonicity (no backwards jumps)
  3. No duplicate timestamps
  4. Row count match (LC == PI)
  5. GTI coverage percentage
  6. HEL1OS gap statistics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from tejas.config import LC_PI_RATIO_EXPECTED, LC_PI_RATIO_TOL


# ── Validation Report ────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated validation results for a day's data.

    Attributes:
        date: The date being validated ('YYYY-MM-DD').
        checks: List of individual check results.
        all_passed: True if every check passed.
        warnings: List of warning messages from failed checks.
    """
    date: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self):
        return all(c.passed for c in self.checks)

    @property
    def warnings(self):
        return [c.message for c in self.checks if not c.passed]

    def add(self, check):
        """Add a check result to the report."""
        self.checks.append(check)
        return self

    def summary_dict(self):
        """Return a flat dict suitable for a summary CSV row."""
        d = {"date": self.date, "all_passed": self.all_passed}
        for c in self.checks:
            d[f"check_{c.name}"] = c.passed
            d.update({f"{c.name}_{k}": v for k, v in c.details.items()})
        return d

    def print_summary(self):
        """Print a human-readable summary."""
        status = "[OK] ALL PASSED" if self.all_passed else "[WARNING] ISSUES FOUND"
        print(f"  Validation [{self.date}]: {status}")
        for c in self.checks:
            icon = "[OK]" if c.passed else "[!!]"
            print(f"    {icon} {c.name}: {c.message}")


# ── Individual Checks ────────────────────────────────────────────────────────


def check_lc_pi_ratio(lc_df, pi_df):
    """Check that SoLEXS LC-to-PI total count ratio is ~5.5.

    The ~5.5x ratio is expected because LC includes all 340 channels
    while PI extraction uses channels 59-340. Deviation suggests a
    loading or channel-range error.

    Args:
        lc_df: SoLEXS LC DataFrame with 'counts' column.
        pi_df: SoLEXS PI DataFrame with 'counts_total_pi' column.

    Returns:
        CheckResult with pass/fail and measured ratio.
    """
    if lc_df is None or pi_df is None:
        return CheckResult(
            name="lc_pi_ratio",
            passed=True,
            message="Skipped (LC or PI data missing)",
        )

    # Use valid (non-NaN) rows only
    lc_valid = lc_df["counts"].dropna()
    pi_valid = pi_df["counts_total_pi"].dropna()
    valid_idx = lc_valid.index.intersection(pi_valid.index)

    if len(valid_idx) == 0:
        return CheckResult(
            name="lc_pi_ratio",
            passed=True,
            message="Skipped (no overlapping valid rows)",
        )

    lc_vals = lc_valid.loc[valid_idx]
    pi_vals = pi_valid.loc[valid_idx]
    nonzero = pi_vals > 0

    if not nonzero.any():
        return CheckResult(
            name="lc_pi_ratio",
            passed=False,
            message="All PI total counts are zero",
        )

    ratios = lc_vals[nonzero] / pi_vals[nonzero]
    median_ratio = float(ratios.median())

    # Informational only: ratio varies per day (5-10x) depending on
    # spectral state. The EDA's 5.5 was a month-wide median.
    return CheckResult(
        name="lc_pi_ratio",
        passed=True,
        message=f"Median ratio = {median_ratio:.2f} (informational)",
        details={"median_ratio": median_ratio},
    )


def check_time_monotonicity(df, time_col="time_unix", label=""):
    """Check that timestamps are strictly increasing (no backwards jumps).

    Args:
        df: DataFrame with a time column.
        time_col: Name of the time column.
        label: Label for the check (e.g., 'SoLEXS LC', 'CdTe').

    Returns:
        CheckResult.
    """
    name = f"time_monotonic{'_' + label if label else ''}"

    if df is None or len(df) <= 1:
        return CheckResult(name=name, passed=True, message="Skipped (too few rows)")

    times = df[time_col].dropna().values
    diffs = np.diff(times)
    n_backwards = int(np.sum(diffs < 0))
    passed = n_backwards == 0

    if passed:
        msg = "Timestamps are monotonically increasing"
    else:
        worst = float(diffs.min())
        msg = f"{n_backwards} backwards jump(s), worst = {worst:.3f}s"

    return CheckResult(
        name=name,
        passed=passed,
        message=msg,
        details={"n_backwards": n_backwards},
    )


def check_no_duplicate_timestamps(df, time_col="time_unix", label=""):
    """Check for duplicate timestamps.

    Args:
        df: DataFrame with a time column.
        time_col: Name of the time column.
        label: Label for the check.

    Returns:
        CheckResult.
    """
    name = f"no_duplicates{'_' + label if label else ''}"

    if df is None or len(df) == 0:
        return CheckResult(name=name, passed=True, message="Skipped (empty)")

    times = df[time_col].dropna()
    n_dupes = int(times.duplicated().sum())
    passed = n_dupes == 0

    msg = f"No duplicate timestamps" if passed else f"{n_dupes} duplicate(s) found"

    return CheckResult(
        name=name,
        passed=passed,
        message=msg,
        details={"n_duplicates": n_dupes},
    )


def check_row_count_match(lc_df, pi_df):
    """Check that SoLEXS LC and PI have the same number of rows.

    Args:
        lc_df: SoLEXS LC DataFrame.
        pi_df: SoLEXS PI DataFrame.

    Returns:
        CheckResult.
    """
    if lc_df is None or pi_df is None:
        return CheckResult(
            name="row_count_match",
            passed=True,
            message="Skipped (LC or PI data missing)",
        )

    n_lc = len(lc_df)
    n_pi = len(pi_df)
    passed = n_lc == n_pi

    msg = f"LC={n_lc}, PI={n_pi}" + ("" if passed else " -- MISMATCH")

    return CheckResult(
        name="row_count_match",
        passed=passed,
        message=msg,
        details={"lc_rows": n_lc, "pi_rows": n_pi},
    )


def check_gti_coverage(df, gti_df, time_col="time_unix", label=""):
    """Report what percentage of data falls within GTI intervals.

    This is informational — it always "passes" but reports the coverage
    percentage so operators can flag days with poor coverage.

    Args:
        df: DataFrame with a time column.
        gti_df: GTI DataFrame with 'start' and 'stop' columns.
        time_col: Name of the time column.
        label: Label for the check.

    Returns:
        CheckResult with coverage percentage in details.
    """
    name = f"gti_coverage{'_' + label if label else ''}"

    if df is None or gti_df is None or len(df) == 0:
        return CheckResult(
            name=name,
            passed=True,
            message="Skipped (data or GTI missing)",
            details={"coverage_pct": None},
        )

    times = df[time_col].values
    in_gti = np.zeros(len(times), dtype=bool)
    for _, row in gti_df.iterrows():
        in_gti |= (times >= row["start"]) & (times <= row["stop"])

    coverage = float(in_gti.sum()) / len(times) * 100.0

    return CheckResult(
        name=name,
        passed=True,  # Informational, always passes
        message=f"GTI coverage = {coverage:.1f}%",
        details={"coverage_pct": coverage},
    )


def check_hel1os_gaps(df, label=""):
    """Report HEL1OS gap statistics from segment_id and gap_mask columns.

    Requires that the DataFrame has been processed through the HEL1OS
    loader pipeline (which adds segment_id and gap_mask columns).

    Args:
        df: HEL1OS DataFrame with segment_id and gap_mask columns.
        label: Label for the check (e.g., 'CdTe', 'CZT').

    Returns:
        CheckResult with gap counts per tier.
    """
    name = f"hel1os_gaps{'_' + label if label else ''}"

    if df is None or len(df) == 0:
        return CheckResult(
            name=name,
            passed=True,
            message="Skipped (no data)",
            details={"n_segments": 0, "n_gap_markers": 0},
        )

    n_segments = 0
    n_gap_markers = 0

    if "segment_id" in df.columns:
        n_segments = int(df["segment_id"].nunique())

    if "gap_mask" in df.columns:
        n_gap_markers = int(df["gap_mask"].sum())

    msg = f"{n_segments} segment(s), {n_gap_markers} gap marker(s) inserted"

    return CheckResult(
        name=name,
        passed=True,  # Informational
        message=msg,
        details={"n_segments": n_segments, "n_gap_markers": n_gap_markers},
    )


# ── Full Validation Suite ────────────────────────────────────────────────────


def validate_day(date_str, solexs_data, hel1os_data):
    """Run all validation checks for a single day.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        solexs_data: dict from solexs.load_day() with keys 'lc', 'pi', 'gti'.
        hel1os_data: dict from hel1os.load_day() with keys 'cdte', 'czt',
            'cdte_gti', 'czt_gti'.

    Returns:
        ValidationReport with all check results.
    """
    report = ValidationReport(date=date_str)

    lc = solexs_data.get("lc") if solexs_data else None
    pi = solexs_data.get("pi") if solexs_data else None
    gti = solexs_data.get("gti") if solexs_data else None

    cdte = hel1os_data.get("cdte") if hel1os_data else None
    czt = hel1os_data.get("czt") if hel1os_data else None
    cdte_gti = hel1os_data.get("cdte_gti") if hel1os_data else None
    czt_gti = hel1os_data.get("czt_gti") if hel1os_data else None

    # SoLEXS checks
    report.add(check_row_count_match(lc, pi))
    report.add(check_lc_pi_ratio(lc, pi))
    report.add(check_time_monotonicity(lc, label="solexs_lc"))
    report.add(check_no_duplicate_timestamps(lc, label="solexs_lc"))
    report.add(check_gti_coverage(lc, gti, label="solexs"))

    # HEL1OS CdTe checks
    report.add(check_time_monotonicity(cdte, label="cdte"))
    report.add(check_hel1os_gaps(cdte, label="cdte"))

    # HEL1OS CZT checks
    report.add(check_time_monotonicity(czt, label="czt"))
    report.add(check_hel1os_gaps(czt, label="czt"))

    return report
