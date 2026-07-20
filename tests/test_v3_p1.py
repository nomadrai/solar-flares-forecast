"""Tests for optimization_plan_v3.md Phase P1 — label surgery.

Covers:
  §2.1 GOES-union: append_goes_only counts, source tags, begin_time anchor.
  §2.2 Ambiguity buffer: (h, h+buffer] rows leave the train fit only.
  §2.3 Coverage mask: rows inside uncovered GOES windows leave the train fit.
  §2.4 Decaying-row ablation flag.
Val/test rows are never excluded by any of the three mechanisms.
"""

import numpy as np
import pandas as pd
import pytest

from tejas.detection.credible import build_credible_catalogue, append_goes_only
from tejas.forecasting.training import (
    build_train_exclusion_mask,
    load_uncovered_goes_windows,
)


# ── §2.1 GOES-union ────────────────────────────────────────────────────────


def _mini_catalogue():
    return pd.DataFrame({
        "event_id": ["e1", "e2", "e3"],
        "start_time": pd.to_datetime([
            "2026-05-01 01:00:00", "2026-05-01 05:00:00", "2026-05-01 09:00:00",
        ]),
        "peak_time": pd.to_datetime([
            "2026-05-01 01:05:00", "2026-05-01 05:05:00", "2026-05-01 09:05:00",
        ]),
        "has_soft": [True, True, False],
        "peak_solexs_counts": [100.0, 10.0, np.nan],
        "goes_matched": [False, True, False],
    })


class TestGoesUnion:
    def test_build_credible_catalogue_tags_source_tejas(self):
        cred = build_credible_catalogue(_mini_catalogue())
        assert "source" in cred.columns
        assert (cred["source"] == "tejas").all()

    def test_append_goes_only_counts_and_tags(self):
        cred = build_credible_catalogue(_mini_catalogue())
        n_before = len(cred)
        goes_only = pd.DataFrame({
            "begin_time": pd.to_datetime(["2026-05-01 03:00:00"]),
            "peak_time": pd.to_datetime(["2026-05-01 03:10:00"]),
            "end_time": pd.to_datetime(["2026-05-01 03:20:00"]),
            "goes_class": ["C2.5"],
        })
        out = append_goes_only(cred, goes_only)
        assert len(out) == n_before + 1
        added = out[out["source"] == "goes_only"]
        assert len(added) == 1
        assert bool(added["is_credible"].iloc[0])
        assert bool(added["goes_matched"].iloc[0])

    def test_append_goes_only_uses_begin_time_anchor(self):
        cred = build_credible_catalogue(_mini_catalogue())
        goes_only = pd.DataFrame({
            "begin_time": pd.to_datetime(["2026-05-01 03:00:00"]),
            "peak_time": pd.to_datetime(["2026-05-01 03:10:00"]),
        })
        out = append_goes_only(cred, goes_only)
        added = out[out["source"] == "goes_only"].iloc[0]
        assert pd.Timestamp(added["start_time"]) == pd.Timestamp("2026-05-01 03:00:00")

    def test_append_goes_only_sorted_by_start_time(self):
        cred = build_credible_catalogue(_mini_catalogue())
        goes_only = pd.DataFrame({
            "begin_time": pd.to_datetime(["2026-05-01 03:00:00"]),
            "peak_time": pd.to_datetime(["2026-05-01 03:10:00"]),
        })
        out = append_goes_only(cred, goes_only)
        starts = pd.to_datetime(out["start_time"])
        assert starts.is_monotonic_increasing

    def test_append_goes_only_empty_is_noop(self):
        cred = build_credible_catalogue(_mini_catalogue())
        out = append_goes_only(cred, pd.DataFrame())
        assert len(out) == len(cred)


# ── Shared synthetic frame for exclusion tests ─────────────────────────────


def _frame_and_splits():
    """600 rows @ 1 min: 0-299 train, 300-449 val, 450-599 test."""
    n = 600
    ts = pd.date_range("2026-05-01 00:00:00", periods=n, freq="1min")
    minutes_to_next = np.full(n, np.nan)
    # Train rows 0..99 count down to a flare at minute 100.
    minutes_to_next[0:100] = np.arange(100, 0, -1)
    # Val rows 300..349 count down to a flare at minute 350.
    minutes_to_next[300:350] = np.arange(50, 0, -1)
    df = pd.DataFrame({
        "datetime": ts,
        "label_minutes_to_next_flare": minutes_to_next,
        "is_flare_active": np.zeros(n, dtype=int),
        "flare_phase": ["quiet"] * n,
        "label_flare_within_5m": (minutes_to_next <= 5).astype(float),
    })
    split = pd.Series(
        ["train"] * 300 + ["val"] * 150 + ["test"] * 150, index=df.index
    )
    return df, split


# ── §2.2 ambiguity buffer ─────────────────────────────────────────────────


class TestAmbiguityBuffer:
    def test_buffer_band_excluded_from_train(self):
        df, split = _frame_and_splits()
        # h=5, buffer=5 → exclude minutes in (5, 10]: train rows with
        # minutes_to_next 6..10 → rows 90..94 (minutes 10,9,8,7,6).
        mask, counts = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=5.0,
            uncovered_windows=pd.DataFrame(), exclude_decaying=False,
        )
        assert counts["buffer"] == 5
        m = df["label_minutes_to_next_flare"][mask]
        assert ((m > 5) & (m <= 10)).all()

    def test_buffer_never_touches_val_or_test(self):
        df, split = _frame_and_splits()
        # Val rows 300..349 include minutes 6..10 too — must stay.
        mask, _ = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=5.0,
            uncovered_windows=pd.DataFrame(), exclude_decaying=False,
        )
        assert not mask[300:].any()

    def test_positives_and_clean_negatives_kept(self):
        df, split = _frame_and_splits()
        mask, _ = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=5.0,
            uncovered_windows=pd.DataFrame(), exclude_decaying=False,
        )
        m = df["label_minutes_to_next_flare"]
        # Positives (≤5 min) kept:
        assert not mask[(m <= 5).fillna(False).to_numpy()].any()
        # Clean negatives (>10 min or NaN) kept:
        assert not mask[(m > 10).fillna(False).to_numpy()].any()
        assert not mask[m.isna().to_numpy()].any()

    def test_zero_buffer_disables(self):
        df, split = _frame_and_splits()
        mask, counts = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=0.0,
            uncovered_windows=pd.DataFrame(), exclude_decaying=False,
        )
        assert counts["buffer"] == 0


# ── §2.3 coverage mask ────────────────────────────────────────────────────


class TestCoverageMask:
    def test_rows_inside_uncovered_window_excluded_from_train(self):
        df, split = _frame_and_splits()
        windows = pd.DataFrame({
            "begin_time": [pd.Timestamp("2026-05-01 03:00:00")],  # minute 180
            "end_time": [pd.Timestamp("2026-05-01 03:10:00")],    # minute 190
        })
        mask, counts = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=0.0,
            uncovered_windows=windows, exclude_decaying=False,
        )
        assert counts["coverage"] == 11  # minutes 180..190 inclusive
        assert mask[180:191].all()

    def test_uncovered_window_in_val_not_excluded(self):
        df, split = _frame_and_splits()
        windows = pd.DataFrame({
            "begin_time": [pd.Timestamp("2026-05-01 05:30:00")],  # minute 330 (val)
            "end_time": [pd.Timestamp("2026-05-01 05:40:00")],
        })
        mask, counts = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=0.0,
            uncovered_windows=windows, exclude_decaying=False,
        )
        assert counts["coverage"] == 0
        assert not mask.any()

    def test_missing_windows_file_returns_empty(self, tmp_path):
        out = load_uncovered_goes_windows(tmp_path / "nope.csv")
        assert len(out) == 0


# ── §2.4 decaying-row ablation ────────────────────────────────────────────


class TestDecayingAblation:
    def test_flag_excludes_decaying_train_rows_only(self):
        df, split = _frame_and_splits()
        df.loc[200:204, "is_flare_active"] = 1
        df.loc[200:204, "flare_phase"] = "decaying"
        df.loc[400:404, "is_flare_active"] = 1          # val — must stay
        df.loc[400:404, "flare_phase"] = "decaying"
        mask, counts = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=0.0,
            uncovered_windows=pd.DataFrame(), exclude_decaying=True,
        )
        assert counts["decaying"] == 5
        assert mask[200:205].all()
        assert not mask[400:405].any()

    def test_flag_off_keeps_decaying_rows(self):
        df, split = _frame_and_splits()
        df.loc[200:204, "is_flare_active"] = 1
        df.loc[200:204, "flare_phase"] = "decaying"
        mask, counts = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=0.0,
            uncovered_windows=pd.DataFrame(), exclude_decaying=False,
        )
        assert counts["decaying"] == 0

    def test_rising_phase_never_excluded(self):
        df, split = _frame_and_splits()
        df.loc[210:214, "is_flare_active"] = 1
        df.loc[210:214, "flare_phase"] = "rising"
        mask, counts = build_train_exclusion_mask(
            df, split, horizon_min=5, buffer_min=0.0,
            uncovered_windows=pd.DataFrame(), exclude_decaying=True,
        )
        assert counts["decaying"] == 0
