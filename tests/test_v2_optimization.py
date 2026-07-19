"""
Tests for the v2 precision+recall optimization modules:
  - credible-event filter + month-wide relabelling (tejas/detection/credible.py)
  - GOES flare cross-matching (tejas/validation/goes_match.py)
  - event-level metrics (tejas/forecasting/event_metrics.py)
  - long-window context features (tejas/features/long_context.py)
  - per-horizon precision_at_min_recall threshold tuning
"""

import numpy as np
import pandas as pd
import pytest

from tejas.detection.credible import (
    credible_mask,
    build_credible_catalogue,
    relabel_frame,
)
from tejas.validation.goes_match import (
    cross_match_events,
    find_missed_goes_flares,
)
from tejas.forecasting.event_metrics import (
    merge_alert_episodes,
    compute_event_metrics,
)
from tejas.features.long_context import compute_long_context_features
from tejas.forecasting.evaluation import tune_threshold
from tejas.config import FORECAST_EVENT_MIN_PEAK_COUNTS


# ── Helpers ────────────────────────────────────────────────────────────────


def _mini_catalogue():
    """Catalogue with one of each flavour of event."""
    return pd.DataFrame({
        "event_id": [1, 2, 3, 4],
        "peak_time": pd.to_datetime([
            "2026-05-01 01:00:00", "2026-05-01 05:00:00",
            "2026-05-01 09:00:00", "2026-05-01 13:00:00",
        ]),
        "start_time": pd.to_datetime([
            "2026-05-01 00:55:00", "2026-05-01 04:55:00",
            "2026-05-01 08:55:00", "2026-05-01 12:55:00",
        ]),
        "peak_solexs_counts": [500.0, 10.0, np.nan, 30.0],
        "has_soft": [True, True, False, True],
        "confidence": ["confirmed", "soft_only", "hard_only", "confirmed"],
    })


# ── Credible filter ────────────────────────────────────────────────────────


class TestCredibleFilter:
    def test_strong_soft_kept_faint_and_hard_dropped(self):
        cat = _mini_catalogue()
        mask = credible_mask(cat)
        # 500 c/s -> credible; 10 c/s faint -> no; hard-only NaN -> no;
        # 30 c/s >= 25 -> credible.
        assert mask.tolist() == [True, False, False, True]

    def test_goes_match_rescues_faint_event(self):
        cat = _mini_catalogue()
        cat["goes_matched"] = [False, True, True, False]
        mask = credible_mask(cat)
        # GOES-matched faint + hard-only events are rescued.
        assert mask.tolist() == [True, True, True, True]

    def test_build_credible_catalogue_resets_index(self):
        cred = build_credible_catalogue(_mini_catalogue())
        assert list(cred.index) == list(range(len(cred)))
        assert (cred["is_credible"] == True).all()  # noqa: E712

    def test_min_peak_matches_config(self):
        cat = _mini_catalogue()
        cat.loc[1, "peak_solexs_counts"] = FORECAST_EVENT_MIN_PEAK_COUNTS
        assert credible_mask(cat).tolist()[1] is np.True_ or \
            credible_mask(cat).tolist()[1] == True  # noqa: E712


# ── Relabelling ────────────────────────────────────────────────────────────


class TestRelabelFrame:
    def _frame(self, n=7200, start="2026-05-01 00:00:00"):
        ts = pd.date_range(start, periods=n, freq="1s")
        return pd.DataFrame({"datetime": ts, "x": np.zeros(n)})

    def test_window_boundary_exact(self):
        df = self._frame()
        flare = pd.Timestamp("2026-05-01 01:00:00")  # row 3600
        df = relabel_frame(df, [flare], horizons=[5])
        lab = df["label_flare_within_5m"].values
        assert lab[3600 - 300] == 1      # exactly 5 min before
        assert lab[3600 - 1] == 1        # 1 s before
        assert lab[3600 - 301] == 0      # 5 min + 1 s before
        assert lab[3600] == 0            # the start row itself (strict >)

    def test_cross_midnight_labelling(self):
        # Flare 10 min into "tomorrow" must label the tail of today.
        df = self._frame(n=3600, start="2026-05-01 23:00:00")
        flare = pd.Timestamp("2026-05-02 00:05:00")
        df = relabel_frame(df, [flare], horizons=[15])
        # Last row of the frame is 23:59:59 — 5m01s before the flare.
        assert df["label_flare_within_15m"].values[-1] == 1

    def test_no_flares_all_zero(self):
        df = self._frame(n=100)
        df = relabel_frame(df, [], horizons=[5, 60])
        assert df["label_flare_within_5m"].sum() == 0
        assert df["label_flare_within_60m"].sum() == 0
        assert df["label_minutes_to_next_flare"].isna().all()


# ── GOES cross-match ───────────────────────────────────────────────────────


class TestGoesMatch:
    def _goes(self):
        return pd.DataFrame({
            "flare_id": [1, 2],
            "begin_time": pd.to_datetime(
                ["2026-05-01 00:50:00", "2026-05-01 20:00:00"]),
            "peak_time": pd.to_datetime(
                ["2026-05-01 01:01:00", "2026-05-01 20:10:00"]),
            "end_time": pd.to_datetime(
                ["2026-05-01 01:10:00", "2026-05-01 20:20:00"]),
            "goes_class": ["C5.0", "M1.0"],
            "peak_flux_wm2": [5e-6, 1e-5],
        })

    def test_match_within_tolerance(self):
        out = cross_match_events(_mini_catalogue(), self._goes(),
                                 tolerance_s=300)
        # Event 1 peaks 01:00, GOES flare 1 peaks 01:01 — matched.
        assert bool(out.loc[0, "goes_matched"]) is True
        assert out.loc[0, "goes_match_class"] == "C5.0"
        # Others: nearest GOES peak far away — unmatched.
        assert bool(out.loc[2, "goes_matched"]) is False

    def test_missed_goes_flares(self):
        missed = find_missed_goes_flares(_mini_catalogue(), self._goes(),
                                         tolerance_s=300)
        # The 20:10 M1.0 has no TEJAS event nearby.
        assert len(missed) == 1
        assert missed.iloc[0]["goes_class"] == "M1.0"

    def test_empty_goes_list_no_crash(self):
        out = cross_match_events(_mini_catalogue(), pd.DataFrame())
        assert (out["goes_matched"] == False).all()  # noqa: E712


# ── Event-level metrics ────────────────────────────────────────────────────


class TestEventMetrics:
    def test_episode_merging(self):
        t = [0, 10, 20, 1000, 1010]          # two episodes (gap 980 > 300)
        eps = merge_alert_episodes(t, merge_gap_s=300)
        assert len(eps) == 2
        assert eps[0] == (0, 20)
        assert eps[1] == (1000, 1010)

    def test_event_recall_and_precision(self):
        ts = pd.date_range("2026-05-01", periods=720, freq="10s")  # 2 h
        prob = np.zeros(720)
        prob[100:110] = 0.9                   # one episode, minute ~16.7
        flare = [ts[150]]                     # starts minute 25 — inside 15m
        m = compute_event_metrics(ts, prob, 0.5, flare, horizon_min=15)
        assert m["event_recall"] == 1.0
        assert m["episode_precision"] == 1.0
        assert m["n_episodes"] == 1
        assert 0 < m["median_lead_min"] <= 15

    def test_false_episode_counted(self):
        ts = pd.date_range("2026-05-01", periods=720, freq="10s")
        prob = np.zeros(720)
        prob[100:110] = 0.9                   # episode with no flare after
        m = compute_event_metrics(ts, prob, 0.5, [], horizon_min=15)
        assert m["n_episodes"] == 1
        assert m["n_episodes_true"] == 0
        assert np.isnan(m["event_recall"])    # no events to recall

    def test_missed_event(self):
        ts = pd.date_range("2026-05-01", periods=720, freq="10s")
        prob = np.zeros(720)                  # never fires
        m = compute_event_metrics(ts, prob, 0.5, [ts[300]], horizon_min=15)
        assert m["event_recall"] == 0.0
        assert m["n_episodes"] == 0


# ── Long-context features ──────────────────────────────────────────────────


class TestLongContextFeatures:
    def _frame(self, n=8000):
        ts = pd.date_range("2026-05-01", periods=n, freq="1s")
        flux = np.full(n, 10.0)
        return pd.DataFrame({"datetime": ts, "solexs_counts_smooth": flux})

    def test_columns_added(self):
        r = compute_long_context_features(self._frame())
        assert set(r["columns_added"]) == {
            "bg_pctrank_1h", "bg_pctrank_3h", "bg_pctrank_6h",
            "rise_consistency_300s", "time_elevated_s",
        }

    def test_causality_future_change_does_not_affect_past(self):
        df1 = self._frame()
        df2 = df1.copy()
        df2.loc[df2.index[-100]:, "solexs_counts_smooth"] = 1e6  # future spike
        r1 = compute_long_context_features(df1)["df"]
        r2 = compute_long_context_features(df2)["df"]
        cut = len(df1) - 100
        for col in ["bg_pctrank_1h", "rise_consistency_300s",
                    "time_elevated_s"]:
            a = r1[col].values[:cut]
            b = r2[col].values[:cut]
            mask = ~(np.isnan(a) & np.isnan(b))
            assert np.allclose(a[mask], b[mask], equal_nan=True), col

    def test_elevated_state_rises_during_step(self):
        df = self._frame(n=7200 * 2)
        # Step up for the last 10 minutes — short vs the 1-h median
        # baseline, so the median does NOT catch up (flare-scale event).
        df.loc[df.index[-600]:, "solexs_counts_smooth"] = 100.0
        out = compute_long_context_features(df)["df"]
        assert out["time_elevated_s"].iloc[-1] > 400


# ── Per-horizon threshold tuning ───────────────────────────────────────────


class TestPrecisionAtMinRecall:
    def test_recall_floor_respected(self):
        rng = np.random.default_rng(42)
        y = (rng.random(5000) < 0.15).astype(int)
        p = np.clip(y * 0.5 + rng.random(5000) * 0.6, 0, 1)
        thr = tune_threshold(y, p, metric="precision_at_min_recall",
                             min_recall=0.8)
        rec = ((p >= thr) & (y == 1)).sum() / (y == 1).sum()
        assert rec >= 0.8

    def test_fallback_highest_recall_when_floor_unreachable(self):
        # Probabilities uncorrelated with labels and floor of 1.0 —
        # tuner must fall back rather than crash.
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.3).astype(int)
        p = rng.random(500)
        thr = tune_threshold(y, p, metric="precision_at_min_recall",
                             min_recall=1.01)
        assert 0.0 < thr < 1.0
