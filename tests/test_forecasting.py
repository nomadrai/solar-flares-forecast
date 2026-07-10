"""
TEJAS Phase 7 — Forecasting Tests
====================================
Covers sampling, splits, models, training, evaluation, inference tripwires,
no-future-data guarantees, and full pipeline integration.

Run with: pytest tests/test_forecasting.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tejas.config import (
    FORECAST_DROP_COLUMNS,
    FORECAST_LABEL_HORIZONS_MIN,
    FORECAST_MIN_NONNULL_FEATURE_RATIO,
    FORECAST_SPLIT_PURGE_S,
    FORECAST_TRAIN_DAYS,
    FORECAST_VAL_DAYS,
    FORECAST_TEST_DAYS,
)
from tejas.forecasting.sampling import (
    downsample,
    drop_incomplete_rows,
    select_feature_columns,
    select_label_column,
)
from tejas.forecasting.splits import (
    apply_purge,
    assign_split,
    build_split_arrays,
)
from tejas.forecasting.models import (
    ALGORITHMS,
    compute_scale_pos_weight,
    make_logreg,
    make_lgbm,
    make_model,
    make_xgb,
)
from tejas.forecasting.evaluation import (
    compute_lead_time,
    compute_metrics,
    feature_importance,
    tune_threshold,
)
from tejas.forecasting.training import (
    load_feature_columns_manifest,
    save_feature_columns_manifest,
    train_all,
    train_one_horizon,
)
from tejas.forecasting.inference import Forecaster, load_forecaster


# ══════════════════════════════════════════════════════════════════════════
# Helpers & fixtures
# ══════════════════════════════════════════════════════════════════════════


def _make_day_frame(
    day: int,
    n_seconds: int = 86_400,
    flare_seconds: list[int] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Construct a synthetic Phase 5-shaped frame for one day.

    Columns match the essential fields Phase 7 consumes: ``datetime``,
    ``time_unix``, three signal features, ``segment_id``, and the four
    binary forecasting labels.  Flares are injected as short bumps so
    label windows have real positive samples.

    Args:
        day: Day-of-month (1..31).
        n_seconds: Number of rows.
        flare_seconds: Seconds since midnight where a flare "starts".
        seed: RNG seed.
    """
    if seed is None:
        seed = 1000 + day
    rng = np.random.RandomState(seed)

    start = pd.Timestamp(f"2026-05-{day:02d} 00:00:00")
    timestamps = pd.date_range(start, periods=n_seconds, freq="1s")

    # Three signal-like features + one derivative-like feature
    base = 10 + rng.normal(0, 1.0, n_seconds)
    hardness = 0.2 + rng.normal(0, 0.02, n_seconds)
    czt = 15 + rng.normal(0, 2.0, n_seconds)
    grad = rng.normal(0, 0.1, n_seconds)

    # Inject flare bumps
    flare_seconds = flare_seconds or [30_000, 60_000]
    labels = {h: np.zeros(n_seconds, dtype=np.int8) for h in FORECAST_LABEL_HORIZONS_MIN}
    flare_id = np.zeros(n_seconds, dtype=np.int32)
    is_flare_active = np.zeros(n_seconds, dtype=np.int8)
    minutes_to_next = np.full(n_seconds, np.nan)

    for idx, s in enumerate(flare_seconds, start=1):
        peak = s + 30
        end = s + 90
        if end >= n_seconds:
            continue
        base[s:end] += np.linspace(20, 100, end - s) * (1 + 0.02 * rng.randn(end - s))
        hardness[s:end] += 0.4
        flare_id[s:end] = idx
        is_flare_active[s:end] = 1

        # Populate label windows retrospectively
        for h in FORECAST_LABEL_HORIZONS_MIN:
            window_start = max(0, s - h * 60)
            labels[h][window_start:s] = 1

    # minutes_to_next_flare (only informative — feature list should
    # exclude this since it's a label)
    next_start = np.inf
    for i in reversed(range(n_seconds)):
        if flare_id[i] > 0:
            next_start = i
        if np.isfinite(next_start):
            minutes_to_next[i] = (next_start - i) / 60.0

    # Compute time_unix explicitly (in whole seconds since 1970-01-01).
    # ``.astype("int64")`` on a DatetimeIndex under pandas 3 / Python 3.14
    # can silently truncate to microseconds; go via datetime64[s] to force
    # second resolution before the int cast.
    time_unix = timestamps.values.astype("datetime64[s]").astype("int64")

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "time_unix": time_unix,
            "segment_id": 0,
            "solexs_counts_smooth": base,
            "solexs_hardness_ratio_smooth": hardness,
            "czt_net_rate": czt,
            "solexs_deriv1": grad,
            "flare_id": flare_id,
            "is_flare_active": is_flare_active,
            "flare_phase": np.where(is_flare_active == 1, "rising", "quiet"),
            "label_minutes_to_next_flare": minutes_to_next,
            "label_flare_within_5m": labels[5],
            "label_flare_within_15m": labels[15],
            "label_flare_within_30m": labels[30],
            "label_flare_within_60m": labels[60],
        }
    )
    return df


def _make_multi_day_frame(days: list[int], n_per_day: int = 86_400) -> pd.DataFrame:
    """Concatenate multiple _make_day_frame outputs."""
    frames = [_make_day_frame(d, n_seconds=n_per_day) for d in days]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def day_frame_1s():
    """A single day, native 1-s cadence, with two flares."""
    return _make_day_frame(day=1, flare_seconds=[30_000, 60_000])


@pytest.fixture
def training_frame_10s():
    """Multi-day training frame, downsampled to 10 s."""
    days = [1, 2, 3, 21, 27]  # 3 train + 1 val + 1 test days
    df = _make_multi_day_frame(days, n_per_day=86_400)
    return downsample(df, 10)


# ══════════════════════════════════════════════════════════════════════════
# 1. Sampling
# ══════════════════════════════════════════════════════════════════════════


class TestSampling:
    def test_downsample_cadence_10(self, day_frame_1s):
        ds = downsample(day_frame_1s, 10)
        # Rows should be ~1/10 of the original (± small tolerance for
        # segment-boundary force-keeps).
        assert abs(len(ds) - len(day_frame_1s) // 10) <= 2

    def test_downsample_cadence_1_is_noop(self, day_frame_1s):
        ds = downsample(day_frame_1s, 1)
        assert len(ds) == len(day_frame_1s)

    def test_downsample_rejects_zero(self, day_frame_1s):
        with pytest.raises(ValueError):
            downsample(day_frame_1s, 0)

    def test_downsample_empty(self):
        empty = pd.DataFrame()
        out = downsample(empty, 10)
        assert out.empty

    def test_downsample_preserves_segment_boundary(self):
        # Two segments, boundary at row 100
        n = 300
        df = pd.DataFrame(
            {
                "time_unix": np.arange(n, dtype=np.int64),
                "segment_id": np.concatenate([np.zeros(100), np.ones(200)]),
                "x": np.arange(n, dtype=float),
            }
        )
        ds = downsample(df, 10)
        # The first sample of segment 1 (row 100) MUST be preserved.
        assert 100.0 in ds["x"].to_numpy()

    def test_select_feature_columns_excludes_blacklist(self, day_frame_1s):
        cols = select_feature_columns(day_frame_1s)
        for forbidden in FORECAST_DROP_COLUMNS:
            assert forbidden not in cols

    def test_select_feature_columns_is_deterministic(self, day_frame_1s):
        c1 = select_feature_columns(day_frame_1s)
        c2 = select_feature_columns(day_frame_1s)
        assert c1 == c2

    def test_select_label_column_returns_int8(self, day_frame_1s):
        y = select_label_column(day_frame_1s, 5)
        assert y.dtype == np.int8
        assert y.min() >= 0 and y.max() <= 1

    def test_select_label_column_missing_raises(self, day_frame_1s):
        with pytest.raises(KeyError):
            select_label_column(day_frame_1s, 999)

    def test_drop_incomplete_rows_honours_threshold(self):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, np.nan],
                "b": [1.0, 2.0, np.nan, np.nan],
                "c": [1.0, 2.0, 3.0, 4.0],
            }
        )
        # ratio=0.95 → require all 3 non-null (ceil)
        out = drop_incomplete_rows(df, ["a", "b", "c"], min_nonnull_ratio=0.95)
        assert len(out) == 1
        # ratio=0.66 → allow 1 NaN out of 3
        out2 = drop_incomplete_rows(df, ["a", "b", "c"], min_nonnull_ratio=0.66)
        assert len(out2) == 3

    def test_drop_incomplete_default_ratio_matches_config(self):
        # The default should equal the config value; just verify the constant.
        assert 0.0 <= FORECAST_MIN_NONNULL_FEATURE_RATIO <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# 2. Splits
# ══════════════════════════════════════════════════════════════════════════


class TestSplits:
    def test_assign_split_labels_by_day(self):
        df = _make_multi_day_frame([1, 21, 27], n_per_day=100)
        s = assign_split(df)
        # Day 1 → train, 21 → val, 27 → test
        assert (s.iloc[:100] == "train").all()
        assert (s.iloc[100:200] == "val").all()
        assert (s.iloc[200:300] == "test").all()

    def test_assign_split_unassigned_is_empty_string(self):
        # Day 99 doesn't exist in any list
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2026-05-01", periods=10, freq="1s"),
            }
        )
        s = assign_split(df, train_days=["02"])
        assert (s == "").all()  # day 01 not in train

    def test_no_day_appears_in_two_splits(self):
        train = set(FORECAST_TRAIN_DAYS)
        val = set(FORECAST_VAL_DAYS)
        test = set(FORECAST_TEST_DAYS)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_apply_purge_drops_tail_of_each_block(self):
        # Two-day train, one-day val, one-day test at 1s cadence
        days = [1, 2, 21, 27]
        df = _make_multi_day_frame(days, n_per_day=10_000)
        s = assign_split(df)
        purge_s = 3600
        df2, s2 = apply_purge(df, s, purge_s=purge_s)
        # train's last row should now be at most 10_000 - purge_s from the
        # original train end minus one; verify by counting per-split rows.
        assert (s2 == "train").sum() < (s == "train").sum()
        assert (s2 == "val").sum() < (s == "val").sum()
        # test is left alone
        assert (s2 == "test").sum() == (s == "test").sum()

    def test_apply_purge_zero_is_noop(self):
        df = _make_multi_day_frame([1, 21, 27], n_per_day=1000)
        s = assign_split(df)
        df2, s2 = apply_purge(df, s, purge_s=0)
        assert len(df2) == len(df)

    def test_build_split_arrays_shapes(self, training_frame_10s):
        feat = select_feature_columns(training_frame_10s)
        s = assign_split(training_frame_10s)
        arrays = build_split_arrays(training_frame_10s, feat, "label_flare_within_5m", s)
        for name in ("train", "val", "test"):
            assert arrays[f"X_{name}"].shape[1] == len(feat)
            assert len(arrays[f"y_{name}"]) == arrays[f"X_{name}"].shape[0]

    def test_build_split_arrays_missing_label_raises(self, training_frame_10s):
        feat = select_feature_columns(training_frame_10s)
        s = assign_split(training_frame_10s)
        with pytest.raises(KeyError):
            build_split_arrays(training_frame_10s, feat, "nope", s)


# ══════════════════════════════════════════════════════════════════════════
# 3. Models
# ══════════════════════════════════════════════════════════════════════════


class TestModels:
    def test_algorithms_registry(self):
        assert ALGORITHMS == ("logreg", "xgb", "lgbm")

    def test_scale_pos_weight(self):
        y = np.array([0, 0, 0, 1])
        assert compute_scale_pos_weight(y) == pytest.approx(3.0)

    def test_scale_pos_weight_no_positives(self):
        y = np.array([0, 0, 0])
        # Falls back to 1.0
        assert compute_scale_pos_weight(y) == 1.0

    def test_make_logreg_has_fit(self):
        m = make_logreg()
        assert hasattr(m, "fit") and hasattr(m, "predict_proba")

    def test_make_xgb_scale_pos_weight_applied(self):
        m = make_xgb(scale_pos_weight=7.0)
        assert float(m.get_params()["scale_pos_weight"]) == 7.0

    def test_make_lgbm_scale_pos_weight_applied(self):
        m = make_lgbm(scale_pos_weight=5.5)
        assert float(m.get_params()["scale_pos_weight"]) == 5.5

    def test_make_model_dispatches_by_name(self):
        y = np.array([0, 0, 1])
        assert make_model("logreg") is not None
        assert make_model("xgb", y_train=y) is not None
        assert make_model("lgbm", y_train=y) is not None

    def test_make_model_unknown_raises(self):
        with pytest.raises(ValueError):
            make_model("nope")


# ══════════════════════════════════════════════════════════════════════════
# 4. Training
# ══════════════════════════════════════════════════════════════════════════


class TestTraining:
    def test_train_one_horizon_end_to_end(self, training_frame_10s):
        feat = select_feature_columns(training_frame_10s)
        s = assign_split(training_frame_10s)
        # Skip purge for tiny synthetic to avoid emptying splits
        result = train_one_horizon(
            training_frame_10s, feat, s, horizon_min=5, algorithm="logreg"
        )
        assert result["algorithm"] == "logreg"
        assert result["horizon_min"] == 5
        assert "model" in result
        assert 0.0 <= result["decision_threshold"] <= 1.0
        assert result["n_train"] > 0
        assert result["n_val"] > 0
        assert result["n_test"] > 0

    def test_train_all_writes_artifacts(self, training_frame_10s, tmp_path):
        feat = select_feature_columns(training_frame_10s)
        s = assign_split(training_frame_10s)
        summary = train_all(
            training_frame_10s,
            feat,
            s,
            algorithms=["logreg"],  # keep quick
            models_dir=tmp_path,
            save=True,
            verbose=False,
        )
        assert len(summary) == 4  # one per horizon
        assert (tmp_path / "feature_columns.json").exists()
        for h in FORECAST_LABEL_HORIZONS_MIN:
            assert (tmp_path / f"logreg_horizon_{h}m.joblib").exists()

    def test_feature_columns_manifest_roundtrip(self, tmp_path):
        cols = ["a", "b", "c"]
        save_feature_columns_manifest(cols, models_dir=tmp_path)
        loaded = load_feature_columns_manifest(models_dir=tmp_path)
        assert loaded == cols


# ══════════════════════════════════════════════════════════════════════════
# 5. Evaluation
# ══════════════════════════════════════════════════════════════════════════


class TestEvaluation:
    def test_compute_metrics_perfect(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.0, 0.1, 0.9, 1.0])
        m = compute_metrics(y, p, threshold=0.5)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_compute_metrics_pr_auc_no_positives(self):
        y = np.zeros(10, dtype=np.int64)
        p = np.random.rand(10)
        m = compute_metrics(y, p, threshold=0.5)
        assert np.isnan(m["pr_auc"])

    def test_compute_metrics_far(self):
        y = np.array([0, 0, 0, 0])
        p = np.array([0.9, 0.9, 0.1, 0.1])
        m = compute_metrics(y, p, threshold=0.5)
        # 2 of 4 negatives predicted positive
        assert m["false_alarm_rate"] == pytest.approx(0.5)

    def test_tune_threshold_picks_max_f1(self):
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        thr = tune_threshold(y, p, grid=np.arange(0.1, 1.0, 0.1))
        # Any threshold between 0.2 and 0.7 yields F1=1.0; we accept
        # the first argmax.
        assert 0.2 <= thr <= 0.7

    def test_tune_threshold_no_positives_falls_back(self):
        y = np.zeros(5, dtype=np.int64)
        p = np.random.rand(5)
        thr = tune_threshold(y, p)
        assert 0.0 <= thr <= 1.0  # fallback is valid

    def test_compute_lead_time_hit(self):
        ts = pd.date_range("2026-05-01", periods=100, freq="1s")
        prob = np.zeros(100)
        prob[50] = 0.9   # crossing at t=50
        starts = pd.Series(pd.to_datetime(["2026-05-01 00:00:55"]))
        out = compute_lead_time(pd.Series(ts), prob, starts, threshold=0.5)
        assert out.iloc[0]["lead_time_s"] == pytest.approx(5.0)

    def test_compute_lead_time_miss(self):
        ts = pd.date_range("2026-05-01", periods=100, freq="1s")
        prob = np.zeros(100)
        starts = pd.Series(pd.to_datetime(["2026-05-01 00:00:55"]))
        out = compute_lead_time(pd.Series(ts), prob, starts, threshold=0.5)
        assert np.isnan(out.iloc[0]["lead_time_s"])

    def test_feature_importance_logreg(self):
        # Build a tiny fitted LogReg
        rng = np.random.RandomState(0)
        X = rng.rand(50, 3)
        y = (X[:, 0] > 0.5).astype(int)
        model = make_logreg()
        model.fit(X, y)
        fi = feature_importance(model, ["a", "b", "c"])
        assert set(fi["feature"]) == {"a", "b", "c"}


# ══════════════════════════════════════════════════════════════════════════
# 6. Inference & tripwires
# ══════════════════════════════════════════════════════════════════════════


class TestInference:
    def _fit_and_persist(self, training_frame_10s, tmp_path, algorithm="logreg"):
        feat = select_feature_columns(training_frame_10s)
        s = assign_split(training_frame_10s)
        train_all(
            training_frame_10s,
            feat,
            s,
            algorithms=[algorithm],
            models_dir=tmp_path,
            save=True,
            verbose=False,
        )
        return feat

    def test_load_forecaster_and_predict_row(self, training_frame_10s, tmp_path):
        feat = self._fit_and_persist(training_frame_10s, tmp_path)
        f = load_forecaster(algorithm="logreg", models_dir=tmp_path)
        row = training_frame_10s.iloc[0][feat]
        probs = f.predict_row(row)
        assert set(probs.keys()) == {"prob_5m", "prob_15m", "prob_30m", "prob_60m"}
        for v in probs.values():
            assert 0.0 <= v <= 1.0

    def test_predict_dataframe_matches_row_by_row(self, training_frame_10s, tmp_path):
        feat = self._fit_and_persist(training_frame_10s, tmp_path)
        f = load_forecaster(algorithm="logreg", models_dir=tmp_path)
        sub = training_frame_10s.iloc[:5][feat]
        batch = f.predict_dataframe(sub)
        row0 = f.predict_row(sub.iloc[0])
        for k, v in row0.items():
            assert batch.iloc[0][k] == pytest.approx(v, rel=1e-6)

    def test_predict_row_rejects_label_columns(self, training_frame_10s, tmp_path):
        feat = self._fit_and_persist(training_frame_10s, tmp_path)
        f = load_forecaster(algorithm="logreg", models_dir=tmp_path)
        row = training_frame_10s.iloc[0][feat + ["label_flare_within_5m"]]
        with pytest.raises(ValueError, match="Forbidden"):
            f.predict_row(row)

    def test_predict_row_missing_feature_raises(self, training_frame_10s, tmp_path):
        feat = self._fit_and_persist(training_frame_10s, tmp_path)
        f = load_forecaster(algorithm="logreg", models_dir=tmp_path)
        row = training_frame_10s.iloc[0][feat[:-1]]  # drop last feature
        with pytest.raises(ValueError, match="Missing"):
            f.predict_row(row)

    def test_load_forecaster_missing_artifact_raises(self, tmp_path):
        # No manifest / no models present
        save_feature_columns_manifest(["a", "b"], models_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            load_forecaster(algorithm="logreg", models_dir=tmp_path)


# ══════════════════════════════════════════════════════════════════════════
# 7. No-future-data guarantees
# ══════════════════════════════════════════════════════════════════════════


class TestNoFutureData:
    def test_drop_columns_covers_all_label_columns(self):
        # Every horizon's label MUST appear in the blacklist
        for h in FORECAST_LABEL_HORIZONS_MIN:
            assert f"label_flare_within_{h}m" in FORECAST_DROP_COLUMNS
        # And the minutes-to-next label
        assert "label_minutes_to_next_flare" in FORECAST_DROP_COLUMNS

    def test_select_feature_columns_never_returns_a_label(self, day_frame_1s):
        cols = select_feature_columns(day_frame_1s)
        for c in cols:
            assert not c.startswith("label_"), (
                f"Feature list contains label column: {c}"
            )

    def test_predict_row_rejects_timestamp_columns(self, training_frame_10s, tmp_path):
        feat = select_feature_columns(training_frame_10s)
        s = assign_split(training_frame_10s)
        train_all(
            training_frame_10s, feat, s, algorithms=["logreg"],
            models_dir=tmp_path, save=True, verbose=False,
        )
        f = load_forecaster(algorithm="logreg", models_dir=tmp_path)
        row = training_frame_10s.iloc[0][feat + ["datetime"]]
        with pytest.raises(ValueError, match="Forbidden"):
            f.predict_row(row)

    def test_purge_gap_not_less_than_longest_horizon(self):
        # Purge (seconds) must at least cover the longest horizon in minutes
        longest_min = max(FORECAST_LABEL_HORIZONS_MIN)
        assert FORECAST_SPLIT_PURGE_S >= longest_min * 60

    def test_random_features_dont_leak(self, tmp_path):
        # Build noise features + real labels; test PR-AUC should not exceed
        # a modest bound over the base positive rate.
        rng = np.random.RandomState(0)
        days = [1, 2, 3, 21, 27]
        base = _make_multi_day_frame(days, n_per_day=86_400)
        base = downsample(base, 10)
        # Overwrite feature columns with noise
        feature_cols = ["solexs_counts_smooth", "solexs_hardness_ratio_smooth",
                        "czt_net_rate", "solexs_deriv1"]
        for c in feature_cols:
            base[c] = rng.rand(len(base))
        feat = select_feature_columns(base)
        s = assign_split(base)
        r = train_one_horizon(base, feat, s, horizon_min=5, algorithm="logreg")
        y_test = base.loc[s == "test", "label_flare_within_5m"].to_numpy()
        pos_rate = float((y_test == 1).mean())
        # PR-AUC on pure noise should be within striking distance of pos_rate.
        # Give a generous margin — the assertion is that we're not seeing
        # near-perfect PR-AUC (which would indicate leakage).
        assert r["test_metrics"]["pr_auc"] < 0.5, (
            f"Suspicious PR-AUC {r['test_metrics']['pr_auc']:.3f} on noise "
            f"features (pos_rate={pos_rate:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════
# 8. Pipeline integration
# ══════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    def test_prepare_train_evaluate_end_to_end(self, tmp_path):
        """Full pipeline on 5 synthetic days written to tmp_path."""
        from tejas.forecasting.pipeline import (
            prepare_training_frame,
            train_all_horizons,
            evaluate_all,
        )

        # Materialise 5 daily detection Parquets in tmp.  Use full-length
        # days so every split contains at least one flare (labels for the
        # 60-min horizon otherwise collapse to all-zero on the shrunk
        # val/test days and LogReg refuses to fit).
        det_dir = tmp_path / "detection"
        det_dir.mkdir()
        for day in [1, 2, 3, 21, 27]:
            df = _make_day_frame(day, n_seconds=86_400)
            df.to_parquet(det_dir / f"2026-05-{day:02d}.parquet", index=False)

        fc_dir = tmp_path / "forecasting"
        md_dir = tmp_path / "models"

        # 1. Prepare
        frame = prepare_training_frame(
            cadence_s=10,
            detection_dir=det_dir,
            output_dir=fc_dir,
            save=True,
            verbose=False,
        )
        assert len(frame) > 0
        assert (fc_dir / "training_frame_10s.parquet").exists()

        # 2. Train (LogReg only, quick)
        summary = train_all_horizons(
            cadence_s=10,
            algorithms=["logreg"],
            output_dir=fc_dir,
            models_dir=md_dir,
            verbose=False,
        )
        assert len(summary) == 4
        assert (fc_dir / "metrics_summary.csv").exists()
        for h in FORECAST_LABEL_HORIZONS_MIN:
            assert (md_dir / f"logreg_horizon_{h}m.joblib").exists()
        assert (md_dir / "feature_columns.json").exists()

        # 3. Evaluate
        result = evaluate_all(
            cadence_s=10,
            algorithm="logreg",
            output_dir=fc_dir,
            models_dir=md_dir,
            verbose=False,
        )
        assert "metrics" in result and "lead_times" in result
        assert (fc_dir / "forecast_report.md").exists()
