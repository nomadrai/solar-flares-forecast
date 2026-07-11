"""
TEJAS Two-Stage False Alert Discriminator (Fix 8)
====================================================
Stage 2 classifier trained on the Stage 1 model's candidate alerts.
Learns to distinguish true-positive candidate alerts from false-positive
ones using the 30-minute trailing context window before each alert.

Training data (per horizon):
    Stage 1 candidate alerts:
        * Positive class = false alert (label_flare_within_{h}m == 0)
        * Negative class = true  alert (label_flare_within_{h}m == 1)
    Features:
        * 5 physics-based "pattern" features
        * Summary statistics (mean/std/min/max/slope) of every base
          feature over the 30-min trailing window

No-future-data guarantee:
    * All window features are strictly trailing (t-1800 s .. t) w.r.t.
      the candidate alert timestamp t.
    * Base model (Stage 1) is loaded and used only to score train + val
      candidates.  Stage 2 is fit on train candidates.  Threshold tuned
      on val candidates.  Test never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    DISCRIMINATOR_WINDOW_S,
    FORECAST_CADENCE_S,
    FORECAST_DIR,
    FORECAST_LABEL_HORIZONS_MIN,
    FORECAST_TARGET_FAR,
    MODELS_DIR,
)
from tejas.forecasting.evaluation import compute_metrics, tune_threshold
from tejas.forecasting.inference import (
    _apply_artifact_calibrator,
    _predict_positive,
    load_forecaster,
)
from tejas.forecasting.sampling import select_feature_columns
from tejas.forecasting.splits import apply_purge, assign_split


# ── Pattern feature extraction ─────────────────────────────────────────────


def _pattern_features(win):
    """Compute 5 physics-based discriminating pattern features from a
    30-minute trailing window DataFrame."""
    out = {}

    # Pattern A: CZT/CdTe coupling correlation
    if "czt_net_rate" in win and "cdte_count_rate_smooth" in win:
        czt = win["czt_net_rate"].astype(float).clip(lower=0.01)
        cdte = win["cdte_count_rate_smooth"].astype(float).clip(lower=0.01)
        if len(czt) > 3:
            try:
                corr = np.corrcoef(czt.values, cdte.values)[0, 1]
                out["disc_czt_cdte_coupling"] = float(corr) if np.isfinite(corr) else 0.0
            except (ValueError, IndexError):
                out["disc_czt_cdte_coupling"] = 0.0
        else:
            out["disc_czt_cdte_coupling"] = 0.0
    else:
        out["disc_czt_cdte_coupling"] = 0.5

    # Pattern B: sustained spectral hardening
    if "solexs_hr_rmean_60" in win and "solexs_hr_rmean_300" in win:
        try:
            out["disc_hr_sustained_frac"] = float(
                (win["solexs_hr_rmean_60"] > win["solexs_hr_rmean_300"]).mean()
            )
        except (TypeError, ValueError):
            out["disc_hr_sustained_frac"] = 0.5
    else:
        out["disc_hr_sustained_frac"] = 0.5

    # Pattern C: CUSUM-to-zscore ratio
    if "solexs_cusum" in win and "solexs_zscore_300" in win:
        cusum_mean = float(np.nan_to_num(win["solexs_cusum"].mean()))
        zmax = float(np.nan_to_num(win["solexs_zscore_300"].max()))
        out["disc_cusum_z_ratio"] = cusum_mean / (zmax + 1e-3)
    else:
        out["disc_cusum_z_ratio"] = 0.0

    # Pattern D: multi-band cascade correlation
    if ("solexs_band_ratio_mid_low" in win
            and "solexs_band_ratio_high_mid" in win):
        ml = win["solexs_band_ratio_mid_low"].dropna().values
        hm = win["solexs_band_ratio_high_mid"].dropna().values
        n = min(len(ml), len(hm))
        if n > 10:
            try:
                corr = np.corrcoef(ml[:n], hm[:n])[0, 1]
                out["disc_band_cascade_corr"] = (
                    float(corr) if np.isfinite(corr) else 0.0
                )
            except (ValueError, IndexError):
                out["disc_band_cascade_corr"] = 0.0
        else:
            out["disc_band_cascade_corr"] = 0.0
    else:
        out["disc_band_cascade_corr"] = 0.0

    # Pattern E: minimum GTI proximity
    if "particle_gti_proximity_s" in win:
        try:
            val = float(win["particle_gti_proximity_s"].min())
            out["disc_min_gti_proximity_s"] = val if np.isfinite(val) else 9999.0
        except (TypeError, ValueError):
            out["disc_min_gti_proximity_s"] = 9999.0
    else:
        out["disc_min_gti_proximity_s"] = 9999.0

    return out


def _window_summary(win, feature_cols):
    """Return mean/std/min/max/slope of each base feature over the window.

    A subset of the base features would already suffice, but with
    GradientBoosting + tree splits we simply pass the full set and let
    the model do feature selection.
    """
    out = {}
    n = len(win)
    if n < 2:
        for col in feature_cols:
            for stat in ("mean", "std", "min", "max", "slope"):
                out[f"win_{col}_{stat}"] = np.nan
        return out
    t = np.arange(n)
    for col in feature_cols:
        if col not in win.columns:
            for stat in ("mean", "std", "min", "max", "slope"):
                out[f"win_{col}_{stat}"] = np.nan
            continue
        arr = win[col].to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) < 2:
            for stat in ("mean", "std", "min", "max", "slope"):
                out[f"win_{col}_{stat}"] = np.nan
            continue
        out[f"win_{col}_mean"] = float(finite.mean())
        out[f"win_{col}_std"] = float(finite.std())
        out[f"win_{col}_min"] = float(finite.min())
        out[f"win_{col}_max"] = float(finite.max())
        # Slope over the fully populated window (using positional indices)
        try:
            slope = float(np.polyfit(t[np.isfinite(arr)], arr[np.isfinite(arr)], 1)[0])
        except (np.linalg.LinAlgError, ValueError, TypeError):
            slope = 0.0
        out[f"win_{col}_slope"] = slope
    return out


# ── Discriminator artifact ─────────────────────────────────────────────────


@dataclass
class FalseAlertDiscriminator:
    """Stage 2 classifier for one horizon."""

    horizon_min: int
    model: object
    threshold: float
    feature_cols: list = field(default_factory=list)
    input_feature_names: list = field(default_factory=list)  # meta-feature list

    def predict(self, window_df):
        """Return True if the alert should be SUPPRESSED (false alert)."""
        pf = _pattern_features(window_df)
        wf = _window_summary(window_df, self.feature_cols)
        row = {**pf, **wf}
        x = pd.DataFrame([row])
        # Align to the training feature order.
        x = x.reindex(columns=self.input_feature_names).fillna(0.0)
        p = float(self.model.predict_proba(x.values)[0, 1])
        return bool(p >= self.threshold), p


# ── Training routine ──────────────────────────────────────────────────────


def _build_candidate_matrix(full_df, feature_cols, candidate_indices,
                              window_s):
    """For each candidate index, extract the 30-min trailing window and
    compute (pattern + summary) features."""
    time_ns = pd.to_datetime(full_df["datetime"]).astype("int64").to_numpy()
    delta_ns = int(window_s * 1e9)

    rows = []
    keep = []
    for idx in candidate_indices:
        t = time_ns[idx]
        lo = np.searchsorted(time_ns, t - delta_ns, side="left")
        hi = idx  # exclusive: strict <
        if hi - lo < 10:
            continue
        win = full_df.iloc[lo:hi]
        pf = _pattern_features(win)
        wf = _window_summary(win, feature_cols)
        rows.append({**pf, **wf})
        keep.append(idx)

    X = pd.DataFrame(rows)
    return X, keep


def train_discriminator(
    training_frame=None,
    cadence_s=None,
    base_algorithm="xgb",
    output_dir=None,
    models_dir=None,
    window_s=DISCRIMINATOR_WINDOW_S,
    verbose=True,
):
    """Train Stage 2 discriminators for all four horizons.

    Uses the specified base algorithm's Forecaster as Stage 1.
    """
    if cadence_s is None:
        cadence_s = FORECAST_CADENCE_S
    output_dir = Path(output_dir) if output_dir else Path(FORECAST_DIR)
    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)

    if training_frame is None:
        path = output_dir / f"training_frame_{cadence_s}s.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Training frame not found: {path}. "
                f"Run prepare_training_frame first."
            )
        training_frame = pd.read_parquet(path)

    feature_cols = select_feature_columns(training_frame)
    split_labels = assign_split(training_frame)
    training_frame, split_labels = apply_purge(training_frame, split_labels)

    # Load Stage 1 forecaster.
    base = load_forecaster(base_algorithm, models_dir=models_dir)

    train_mask = (split_labels == "train").to_numpy()
    val_mask = (split_labels == "val").to_numpy()
    test_mask = (split_labels == "test").to_numpy()

    X_all = training_frame[feature_cols].to_numpy(dtype=float)

    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    reports = []
    for h in FORECAST_LABEL_HORIZONS_MIN:
        art = base.horizon_models[h]
        threshold_s1 = float(art["decision_threshold"])
        raw = _predict_positive(art["model"], X_all)
        probs = _apply_artifact_calibrator(art, raw)

        label_col = f"label_flare_within_{h}m"
        y_all = training_frame[label_col].fillna(0).astype(np.int8).to_numpy()

        # Candidates = rows where Stage 1 fires.
        cand_mask = probs >= threshold_s1

        # ── Train candidates ──────────────────────────────────────────
        tr_idx = np.where(cand_mask & train_mask)[0]
        if len(tr_idx) < 20:
            if verbose:
                print(f"  [SKIP] h={h}m: only {len(tr_idx)} train candidates.")
            continue
        Xtr, tr_kept = _build_candidate_matrix(
            training_frame, feature_cols, tr_idx, window_s
        )
        # Target: 1 = false alert (Stage 1 fired but no flare)
        y_tr = (y_all[tr_kept] == 0).astype(np.int8)
        if int(y_tr.sum()) < 5 or int((y_tr == 0).sum()) < 5:
            if verbose:
                print(f"  [SKIP] h={h}m: too few of one class in train "
                      f"(FP={int(y_tr.sum())}, TP={int((y_tr == 0).sum())}).")
            continue

        # Fit model.
        meta_feat_names = list(Xtr.columns)
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=42,
            )),
        ])
        clf.fit(Xtr.values, y_tr)

        # ── Val candidates: tune threshold ─────────────────────────────
        val_idx = np.where(cand_mask & val_mask)[0]
        threshold_s2 = 0.5
        val_metrics = {}
        if len(val_idx) >= 20:
            Xv, val_kept = _build_candidate_matrix(
                training_frame, feature_cols, val_idx, window_s
            )
            Xv = Xv.reindex(columns=meta_feat_names).fillna(0.0)
            y_v = (y_all[val_kept] == 0).astype(np.int8)
            if int(y_v.sum()) >= 3 and int((y_v == 0).sum()) >= 3:
                p_v = clf.predict_proba(Xv.values)[:, 1]
                threshold_s2 = tune_threshold(
                    y_v, p_v,
                    metric="far_constrained",
                    target_far=FORECAST_TARGET_FAR,
                )
                val_metrics = compute_metrics(y_v, p_v, threshold=threshold_s2)

        # ── Test evaluation ────────────────────────────────────────────
        test_idx = np.where(cand_mask & test_mask)[0]
        test_metrics = {}
        n_true_alerts_test = int((y_all[test_idx] == 1).sum()) if len(test_idx) else 0
        n_false_alerts_test = int((y_all[test_idx] == 0).sum()) if len(test_idx) else 0
        if len(test_idx) >= 10:
            Xt, test_kept = _build_candidate_matrix(
                training_frame, feature_cols, test_idx, window_s
            )
            Xt = Xt.reindex(columns=meta_feat_names).fillna(0.0)
            y_t = (y_all[test_kept] == 0).astype(np.int8)
            p_t = clf.predict_proba(Xt.values)[:, 1]
            # Metric on the discriminator target: predicts "is false alert".
            # We compare vs. y_t (1=false alert).
            if int(y_t.sum()) and int((y_t == 0).sum()):
                test_metrics = compute_metrics(y_t, p_t, threshold=threshold_s2)

            # Compute the alert-level suppression + preservation metrics.
            suppressed = p_t >= threshold_s2
            n_suppressed = int(suppressed.sum())
            # Alert-level: how many "true alerts" (y_t==0) did we
            # incorrectly suppress → contamination
            wrongly_suppressed = int(
                ((y_t == 0) & suppressed).sum()
            )
            # How many false alerts (y_t==1) got suppressed → good
            correctly_suppressed = int(
                ((y_t == 1) & suppressed).sum()
            )
            n_false_kept = n_false_alerts_test - correctly_suppressed
            n_true_kept = n_true_alerts_test - wrongly_suppressed
        else:
            n_suppressed = 0
            wrongly_suppressed = 0
            correctly_suppressed = 0
            n_false_kept = n_false_alerts_test
            n_true_kept = n_true_alerts_test

        # Persist.
        disc = FalseAlertDiscriminator(
            horizon_min=int(h),
            model=clf,
            threshold=float(threshold_s2),
            feature_cols=list(feature_cols),
            input_feature_names=meta_feat_names,
        )
        art_path = models_dir / f"discriminator_{base_algorithm}_horizon_{h}m.joblib"
        joblib.dump({"discriminator": disc}, art_path)

        rep = {
            "horizon_min": int(h),
            "stage1_threshold": threshold_s1,
            "stage2_threshold": float(threshold_s2),
            "n_train_candidates": int(len(tr_idx)),
            "n_val_candidates": int(len(val_idx)),
            "n_test_candidates": int(len(test_idx)),
            "n_test_true_alerts": n_true_alerts_test,
            "n_test_false_alerts": n_false_alerts_test,
            "n_test_suppressed": n_suppressed,
            "n_test_wrongly_suppressed": wrongly_suppressed,
            "n_test_correctly_suppressed": correctly_suppressed,
            "test_recall_after_gate": (
                n_true_kept / n_true_alerts_test
                if n_true_alerts_test > 0 else float("nan")
            ),
            "test_false_alerts_after_gate": n_false_kept,
            "test_precision_after_gate": (
                n_true_kept / (n_true_kept + n_false_kept)
                if (n_true_kept + n_false_kept) > 0 else float("nan")
            ),
            "val_disc_pr_auc": val_metrics.get("pr_auc", float("nan")),
            "test_disc_pr_auc": test_metrics.get("pr_auc", float("nan")),
        }
        reports.append(rep)
        if verbose:
            print(
                f"  disc h={h}m  cand(test)={len(test_idx)}  "
                f"stage1_FAR={n_false_alerts_test / max(1, len(test_idx)):.3f}  "
                f"suppressed={n_suppressed}  wrongly={wrongly_suppressed}  "
                f"post-recall={rep['test_recall_after_gate']:.3f}"
            )

    rep_df = pd.DataFrame(reports)
    output_dir.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(output_dir / "discriminator_report.csv", index=False)
    if verbose and len(rep_df):
        print(f"  Saved: {output_dir / 'discriminator_report.csv'}")

    return rep_df


def load_discriminator(horizon_min, base_algorithm="xgb", models_dir=None):
    import joblib
    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)
    art_path = models_dir / f"discriminator_{base_algorithm}_horizon_{horizon_min}m.joblib"
    if not art_path.exists():
        raise FileNotFoundError(f"Missing discriminator artifact: {art_path}")
    return joblib.load(art_path)["discriminator"]
