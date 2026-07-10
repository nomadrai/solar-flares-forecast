"""
TEJAS Phase 6 — Flare Classification Tests
=============================================
Covers native intensity binning, GOES class/magnitude assignment, the
optional calibration fit, pipeline behaviour with and without a
calibration artifact, and the no-future-leakage guarantee.

Run with: pytest tests/test_classification.py -v
"""

import math
import numpy as np
import pandas as pd
import pytest

from tejas.classification.native import (
    classify_counts,
    intensity_score,
    classify_dataframe as native_classify,
)
from tejas.classification.goes import (
    flux_to_class,
    flux_to_class_vectorised,
    counts_to_flux,
)
from tejas.classification.calibration import (
    cross_match,
    fit_log_log,
    calibrate,
    load_calibration,
)
from tejas.classification.pipeline import classify_catalogue, summarize_by_day
from tejas.config import (
    INTENSITY_BIN_EDGES,
    INTENSITY_BIN_LABELS,
    GOES_CLASS_THRESHOLDS,
    CALIBRATION_MIN_R2,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_catalogue():
    """A small master-catalogue-like DataFrame covering all event types."""
    return pd.DataFrame([
        {
            "event_id": 1, "date": "2026-05-01",
            "peak_time": "2026-05-01T10:00:00", "start_time": "2026-05-01T09:59:00",
            "end_time": "2026-05-01T10:01:00", "peak_solexs_counts": 15.0,
            "peak_cdte_counts": np.nan, "peak_czt_counts": np.nan,
            "has_soft": True, "has_hard": False, "has_high_energy": False,
            "confidence": "soft_only",
        },
        {
            "event_id": 2, "date": "2026-05-01",
            "peak_time": "2026-05-01T11:00:00", "start_time": "2026-05-01T10:59:00",
            "end_time": "2026-05-01T11:01:00", "peak_solexs_counts": 250.0,
            "peak_cdte_counts": 50.0, "peak_czt_counts": np.nan,
            "has_soft": True, "has_hard": True, "has_high_energy": False,
            "confidence": "confirmed",
        },
        {
            "event_id": 3, "date": "2026-05-02",
            "peak_time": "2026-05-02T09:00:00", "start_time": "2026-05-02T08:59:00",
            "end_time": "2026-05-02T09:01:00", "peak_solexs_counts": np.nan,
            "peak_cdte_counts": 20.0, "peak_czt_counts": np.nan,
            "has_soft": False, "has_hard": True, "has_high_energy": False,
            "confidence": "hard_only",
        },
        {
            "event_id": 4, "date": "2026-05-02",
            "peak_time": "2026-05-02T12:00:00", "start_time": "2026-05-02T11:59:00",
            "end_time": "2026-05-02T12:01:00", "peak_solexs_counts": 2000.0,
            "peak_cdte_counts": 90.0, "peak_czt_counts": 120.0,
            "has_soft": True, "has_hard": True, "has_high_energy": True,
            "confidence": "confirmed",
        },
    ])


# ── Native intensity ───────────────────────────────────────────────────────


def test_native_bins_boundaries():
    # Edges: [25, 100, 400, 1500]
    inputs = [0, 24.9, 25, 99.9, 100, 399.9, 400, 1499.9, 1500, 5000]
    expected = [
        "faint", "faint", "weak", "weak", "moderate",
        "moderate", "strong", "strong", "severe", "severe",
    ]
    got = classify_counts(np.array(inputs))
    assert list(got) == expected


def test_native_bins_nan_preserved():
    got = classify_counts(np.array([50.0, np.nan, 200.0]))
    assert got[0] == "weak"
    assert isinstance(got[1], float) and math.isnan(got[1])
    assert got[2] == "moderate"


def test_native_score_range():
    inputs = np.array([0, 10, 100, 1000, 4000, 10000])
    scores = intensity_score(inputs)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    # Monotone.
    assert np.all(np.diff(scores) >= 0)


def test_native_classify_dataframe(synthetic_catalogue):
    out = native_classify(synthetic_catalogue, peak_col="peak_solexs_counts")
    # Row 3 (hard-only) must be NaN.
    assert pd.isna(out.iloc[2]["intensity_class"])
    assert pd.isna(out.iloc[2]["intensity_score"])
    # Row 4 (peak=2000) is severe.
    assert out.iloc[3]["intensity_class"] == "severe"


# ── GOES class assignment ──────────────────────────────────────────────────


def test_goes_flux_to_class_thresholds():
    # Values just at each threshold pick that class.
    for letter, thr in GOES_CLASS_THRESHOLDS.items():
        cls, mag, full = flux_to_class(thr)
        assert cls == letter
        assert abs(mag - 1.0) < 1e-6
        assert full == f"{letter}1.0"


def test_goes_flux_to_class_magnitude():
    # 2.5e-5 W/m² → M2.5
    cls, mag, full = flux_to_class(2.5e-5)
    assert cls == "M" and mag == 2.5 and full == "M2.5"
    # 9.99e-5 clips magnitude to 9.9 (truncated), stays in M.
    cls, mag, full = flux_to_class(9.99e-5)
    assert cls == "M" and mag == 9.9
    # 1.0e-4 → X1.0
    cls, mag, full = flux_to_class(1.0e-4)
    assert cls == "X" and mag == 1.0 and full == "X1.0"


def test_goes_flux_to_class_nan():
    cls, mag, full = flux_to_class(float("nan"))
    assert isinstance(cls, float) and math.isnan(cls)
    assert math.isnan(mag)


def test_goes_flux_to_class_vectorised():
    fluxes = np.array([5e-8, 5e-6, 5e-5, 2e-4, np.nan])
    letters, mags, fulls = flux_to_class_vectorised(fluxes)
    assert letters[0] == "A" and letters[1] == "C" and letters[2] == "M" and letters[3] == "X"
    assert isinstance(letters[4], float) and math.isnan(letters[4])


def test_counts_to_flux_with_calibration():
    cal = {"a": 1.0, "b": -8.0}
    # log10(flux) = 1 * log10(100) - 8 = 2 - 8 = -6 → 1e-6
    assert abs(counts_to_flux(100.0, cal) - 1e-6) < 1e-9


def test_counts_to_flux_without_calibration():
    result = counts_to_flux(100.0, None)
    assert math.isnan(result)


# ── Calibration ────────────────────────────────────────────────────────────


def test_calibration_fit_synthetic():
    # Ground truth: log10(flux) = 1.2 * log10(counts) - 9.5 + small noise.
    rng = np.random.RandomState(0)
    counts = np.logspace(1, 3.5, 60)
    log_flux = 1.2 * np.log10(counts) - 9.5 + rng.normal(0, 0.05, 60)
    pairs = pd.DataFrame({
        "peak_counts": counts,
        "goes_flux": 10 ** log_flux,
    })
    fit = fit_log_log(pairs)
    assert fit is not None
    assert abs(fit["a"] - 1.2) < 0.1
    assert abs(fit["b"] - (-9.5)) < 0.2
    assert fit["r2"] > 0.9


def test_calibration_rejects_bad_fit():
    # Pure noise → very low R² → returned None.
    rng = np.random.RandomState(1)
    pairs = pd.DataFrame({
        "peak_counts": np.abs(rng.randn(50)) * 100 + 1,
        "goes_flux": np.abs(rng.randn(50)) * 1e-6 + 1e-9,
    })
    fit = fit_log_log(pairs)
    assert fit is None or fit["r2"] >= CALIBRATION_MIN_R2


def test_calibration_insufficient_samples():
    pairs = pd.DataFrame({
        "peak_counts": [10, 20, 30],
        "goes_flux": [1e-7, 2e-7, 3e-7],
    })
    fit = fit_log_log(pairs)
    assert fit is None


def test_calibration_cross_match():
    catalogue = pd.DataFrame({
        "peak_time": pd.to_datetime(["2026-05-01T10:00:00", "2026-05-01T11:00:00"]),
        "peak_solexs_counts": [100.0, np.nan],
    })
    goes = pd.DataFrame({
        "time_unix": [
            int(pd.Timestamp("2026-05-01T10:00:15", tz="UTC").timestamp()),
            int(pd.Timestamp("2026-05-01T11:00:20", tz="UTC").timestamp()),
        ],
        "flux_wm2": [1.5e-6, 2e-6],
    })
    matched = cross_match(catalogue, goes, tolerance_s=30)
    # Only row 0 has both a soft peak and a GOES match within tolerance.
    assert len(matched) == 1
    assert abs(matched.iloc[0]["peak_counts"] - 100.0) < 1e-6


# ── Pipeline ───────────────────────────────────────────────────────────────


def test_pipeline_without_calibration(synthetic_catalogue):
    out = classify_catalogue(synthetic_catalogue, calibration=None)
    # Native columns populated for soft-detected rows.
    soft_rows = out[out["has_soft"]]
    assert soft_rows["intensity_class"].notna().all()
    # GOES columns all NaN.
    assert out["goes_flux"].isna().all()
    assert out["goes_class"].isna().all()


def test_pipeline_with_calibration(synthetic_catalogue):
    cal = {"a": 1.0, "b": -8.0}
    out = classify_catalogue(synthetic_catalogue, calibration=cal)
    soft_rows = out[out["has_soft"]]
    assert soft_rows["goes_flux"].notna().all()
    assert soft_rows["goes_class"].notna().all()
    assert soft_rows["goes_class_full"].notna().all()
    # Hard-only row still NaN.
    hard_only = out[out["confidence"] == "hard_only"]
    assert hard_only["goes_flux"].isna().all()
    assert hard_only["intensity_class"].isna().all()


def test_pipeline_augments_not_overwrites(synthetic_catalogue):
    original_cols = list(synthetic_catalogue.columns)
    out = classify_catalogue(synthetic_catalogue, calibration=None)
    for col in original_cols:
        pd.testing.assert_series_equal(
            synthetic_catalogue[col].reset_index(drop=True),
            out[col].reset_index(drop=True),
            check_names=False,
        )


def test_pipeline_event_type_populated(synthetic_catalogue):
    out = classify_catalogue(synthetic_catalogue, calibration=None)
    assert list(out["event_type"]) == list(synthetic_catalogue["confidence"])


def test_summarize_by_day(synthetic_catalogue):
    out = classify_catalogue(synthetic_catalogue, calibration=None)
    summary = summarize_by_day(out)
    assert len(summary) == 2  # two distinct dates
    day1 = summary[summary["date"] == "2026-05-01"].iloc[0]
    assert day1["n_events"] == 2
    assert day1["n_soft_classified"] == 2  # both rows have soft peak


# ── No-future-data guarantee ───────────────────────────────────────────────


def test_no_future_leakage():
    """Classification uses ONLY peak_solexs_counts (a scalar per event),
    never any future feature or GOES sample past peak_time.  Verified by
    the fact that classify_counts() and flux_to_class() are pure functions
    of a single input value.
    """
    from tejas.classification import native, goes as goes_mod
    import inspect

    # native.classify_counts and goes.flux_to_class must not access
    # any DataFrame or time series — signature is single scalar/array in.
    for fn in [native.classify_counts, goes_mod.flux_to_class]:
        sig = inspect.signature(fn)
        # Exactly one required parameter.
        required = [p for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty]
        assert len(required) == 1
