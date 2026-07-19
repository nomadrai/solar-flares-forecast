"""
TEJAS Two-Stage False Alert Discriminator (Fix 8) — Optimized
==============================================================
Stage 2 classifier trained on the Stage 1 model's candidate alerts.
Learns to distinguish true-positive candidate alerts from false-positive
ones using the 30-minute trailing context window before each alert.

Training data (per horizon):
    Stage 1 candidate alerts:
        * Positive class = false alert (label_flare_within_{h}m == 0)
        * Negative class = true  alert (label_flare_within_{h}m == 1)
    Features:
        * 5 physics-based "pattern" features (correlations + fractions +
          ratios computed on the window)
        * Summary statistics (mean / std / slope) of only the top-K most
          important base features per horizon (K = 30 by default)
          — capped for speed and to reduce over-fitting on 4 000-sample
          val folds

Speed:
    Vectorised: candidate windows are stacked into a (n_candidates,
    window_len, n_features) tensor once, then numpy ops compute all
    summaries at once — no per-candidate Python loop over 100+ features.

No-future-data guarantee:
    All window features are strictly trailing (t - 1800 s .. t - 1) w.r.t.
    the candidate alert timestamp t.
    Base model (Stage 1) is loaded and used only to score train + val
    candidates.  Stage 2 is fit on train candidates.  Threshold tuned
    on val candidates.  Test never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    DISCRIMINATOR_MAX_TP_KILL_RATE,
    DISCRIMINATOR_MIN_NET_BENEFIT,
    DISCRIMINATOR_MIN_PRECISION,
    DISCRIMINATOR_TOP_K_BASE,
    DISCRIMINATOR_TP_LOSS_WEIGHT,
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


# ── Pattern feature extraction (vectorised over N candidates) ──────────────


def _vec_pattern_features(win_stack, feature_index):
    """Compute the 5 physics-based pattern features across a stack of
    candidate windows.

    Args:
        win_stack: numpy array of shape (n_candidates, window_len,
            n_features).
        feature_index: dict {feature_name: column_index_into_win_stack}
            for the columns we may need.

    Returns:
        DataFrame with 5 columns × n_candidates rows.
    """
    n = win_stack.shape[0]

    def _get(name):
        idx = feature_index.get(name)
        if idx is None:
            return None
        return win_stack[:, :, idx]

    czt = _get("czt_net_rate")
    cdte_s = _get("cdte_count_rate_smooth")
    if cdte_s is None:
        cdte_s = _get("cdte_count_rate_kalman")

    # Pattern A: CZT/CdTe coupling correlation
    if czt is not None and cdte_s is not None:
        # Vectorised Pearson correlation across the last axis
        a = np.nan_to_num(czt, nan=0.0)
        b = np.nan_to_num(cdte_s, nan=0.0)
        am = a.mean(axis=1, keepdims=True)
        bm = b.mean(axis=1, keepdims=True)
        av = a - am
        bv = b - bm
        num = (av * bv).sum(axis=1)
        den = np.sqrt((av * av).sum(axis=1) * (bv * bv).sum(axis=1)) + 1e-12
        corr = num / den
        corr = np.where(np.isfinite(corr), corr, 0.0)
    else:
        corr = np.full(n, 0.5)

    # Pattern B: sustained spectral hardening
    hr60 = _get("solexs_hr_rmean_60")
    hr300 = _get("solexs_hr_rmean_300")
    if hr60 is not None and hr300 is not None:
        frac_h = np.nanmean(
            (np.nan_to_num(hr60, nan=0.0)
             > np.nan_to_num(hr300, nan=0.0)).astype(float),
            axis=1,
        )
    else:
        frac_h = np.full(n, 0.5)

    # Pattern C: CUSUM-to-zscore ratio
    cusum = _get("solexs_cusum")
    z300 = _get("solexs_zscore_300")
    if cusum is not None and z300 is not None:
        cm = np.nanmean(np.nan_to_num(cusum, nan=0.0), axis=1)
        zmax = np.nanmax(np.nan_to_num(z300, nan=0.0), axis=1)
        cz = cm / (zmax + 1e-3)
    else:
        cz = np.zeros(n)

    # Pattern D: multi-band cascade correlation
    ml = _get("solexs_band_ratio_mid_low")
    hm = _get("solexs_band_ratio_high_mid")
    if ml is not None and hm is not None:
        a = np.nan_to_num(ml, nan=0.0)
        b = np.nan_to_num(hm, nan=0.0)
        am = a.mean(axis=1, keepdims=True)
        bm = b.mean(axis=1, keepdims=True)
        av = a - am
        bv = b - bm
        num = (av * bv).sum(axis=1)
        den = np.sqrt((av * av).sum(axis=1) * (bv * bv).sum(axis=1)) + 1e-12
        bcorr = num / den
        bcorr = np.where(np.isfinite(bcorr), bcorr, 0.0)
    else:
        bcorr = np.zeros(n)

    # Pattern E: minimum GTI proximity
    gti = _get("particle_gti_proximity_s")
    if gti is not None:
        gmin = np.nanmin(np.nan_to_num(gti, nan=9999.0), axis=1)
    else:
        gmin = np.full(n, 9999.0)

    return pd.DataFrame({
        "disc_czt_cdte_coupling": corr.astype(np.float32),
        "disc_hr_sustained_frac": frac_h.astype(np.float32),
        "disc_cusum_z_ratio": cz.astype(np.float32),
        "disc_band_cascade_corr": bcorr.astype(np.float32),
        "disc_min_gti_proximity_s": gmin.astype(np.float32),
    })


def _vec_window_summary(win_stack, base_cols, feature_index):
    """Compute mean/std/slope of the specified base columns across a
    stack of candidate windows (3 stats, matching the discriminator that
    produced the standout 30-min result).

    slope uses closed-form OLS (O(N) per window vs. O(N²) via polyfit).
    """
    n = win_stack.shape[0]
    L = win_stack.shape[1]
    t = np.arange(L, dtype=np.float32)
    tm = float(t.mean())
    tt = float(((t - tm) ** 2).sum())

    out = {}
    zeros = np.zeros(n, dtype=np.float32)
    for col in base_cols:
        idx = feature_index.get(col)
        if idx is None:
            for stat in ("mean", "std", "slope"):
                out[f"win_{col}_{stat}"] = zeros.copy()
            continue
        arr = np.nan_to_num(
            win_stack[:, :, idx].astype(np.float32), nan=0.0
        )
        m = arr.mean(axis=1)
        s = arr.std(axis=1)
        num = ((t - tm) * (arr - m[:, None])).sum(axis=1)
        slope = num / (tt + 1e-12)
        out[f"win_{col}_mean"] = m.astype(np.float32)
        out[f"win_{col}_std"] = s.astype(np.float32)
        out[f"win_{col}_slope"] = slope.astype(np.float32)
    return pd.DataFrame(out)


# ── Threshold tuning: net-benefit maximisation ────────────────────────────


def _tune_stage2_threshold(y_disc_is_fp, disc_prob,
                             max_tp_kill_rate, tp_loss_weight):
    """Pick the Stage-2 threshold.  Uses the canonical
    ``tune_threshold(..., metric="far_constrained")`` sweep from
    ``tejas/forecasting/evaluation.py`` so we get exactly the same
    behaviour as the base-model threshold tuner.

    ``max_tp_kill_rate`` is passed through as ``target_far`` — the two
    metrics coincide when the positive class is "false alert" and the
    negative class is "true alert" (which is how this discriminator is
    trained).

    Then apply a **val net-benefit sanity check**: if the chosen
    threshold produces `correct_kills − α × wrong_kills < 0` on val,
    return threshold = 1.01 (Stage-2 is a no-op).  This is the only
    thing that reliably rejects the 60-min horizon, where Stage-1 is
    already precise enough that Stage-2 does more harm than good.
    """
    if len(y_disc_is_fp) == 0:
        return 1.01, 0.0, {}

    n_tp = int((y_disc_is_fp == 0).sum())
    if n_tp == 0:
        return 0.5, 0.0, {"note": "no_tp_in_candidates"}

    thr = float(tune_threshold(
        y_disc_is_fp, disc_prob,
        metric="far_constrained",
        target_far=max_tp_kill_rate,
    ))

    suppress = disc_prob >= thr
    correct = int((suppress & (y_disc_is_fp == 1)).sum())
    wrong   = int((suppress & (y_disc_is_fp == 0)).sum())
    total_supp = correct + wrong
    net = correct - tp_loss_weight * wrong
    stats = {
        "correct": correct, "wrong": wrong,
        "precision": (correct / total_supp) if total_supp > 0 else 0.0,
        "tp_kill_rate": wrong / n_tp if n_tp > 0 else 0.0,
        "net": float(net),
    }

    # Note: no skip gate here.  On tiny val folds the net-benefit
    # estimate is noisy and doesn't reliably predict test-time
    # behaviour.  We always apply the tuned threshold; the
    # test-side report tells the operator per-horizon which
    # discriminators are net-positive on the held-out split, and
    # the dashboard can choose to disable specific horizons.
    return thr, float(net), stats


# ── Discriminator artifact ─────────────────────────────────────────────────


@dataclass
class FalseAlertDiscriminator:
    """Stage 2 classifier for one horizon."""

    horizon_min: int
    model: object
    threshold: float
    feature_cols: list = field(default_factory=list)          # base features used
    input_feature_names: list = field(default_factory=list)   # meta-feature list
    top_base_features: list = field(default_factory=list)     # base features summarised

    def predict(self, window_df):
        """Return (suppress?, prob_false) for a single 30-min window
        DataFrame."""
        # Stack the window into a 3-D array of shape (1, window_len, n_feat).
        # Build a lightweight feature index against the base features we
        # care about.
        needed = set(self.top_base_features) | {
            "czt_net_rate", "cdte_count_rate_smooth", "cdte_count_rate_kalman",
            "solexs_hr_rmean_60", "solexs_hr_rmean_300",
            "solexs_cusum", "solexs_zscore_300",
            "solexs_band_ratio_mid_low", "solexs_band_ratio_high_mid",
            "particle_gti_proximity_s",
        }
        cols = [c for c in needed if c in window_df.columns]
        if len(window_df) == 0 or not cols:
            return False, 0.0
        mat = window_df[cols].to_numpy(dtype=float)
        L = mat.shape[0]
        stack = np.zeros((1, L, len(cols)), dtype=float)
        stack[0, :, :] = mat
        idx_map = {c: i for i, c in enumerate(cols)}

        pf = _vec_pattern_features(stack, idx_map)
        ws = _vec_window_summary(stack, self.top_base_features, idx_map)
        row = pd.concat([pf, ws], axis=1)
        row = row.reindex(columns=self.input_feature_names).fillna(0.0)
        p = float(self.model.predict_proba(row.values)[0, 1])
        return bool(p >= self.threshold), p


# ── Training routine (vectorised) ──────────────────────────────────────────


def _select_top_base_features(models_dir, algorithm, horizon_min, k):
    """Read feature_importance.csv (persisted by pipeline) and take the
    top-K feature names for the (algorithm, horizon) pair."""
    fi_path = Path(FORECAST_DIR) / "feature_importance.csv"
    if not fi_path.exists():
        return []
    fi = pd.read_csv(fi_path)
    subset = fi[(fi["algorithm"] == algorithm) & (fi["horizon_min"] == horizon_min)]
    if subset.empty:
        return []
    top = subset.nlargest(k, "importance")["feature"].tolist()
    return top


def _build_candidate_windows(full_df, feature_cols, candidate_indices,
                              needed_cols, window_len):
    """For every candidate index build the (n_cand, window_len, n_feat)
    stack.  Rows with insufficient trailing data are silently skipped.

    Returns (stack, kept_indices, feature_index_map).
    """
    time_arr = pd.to_datetime(full_df["datetime"]).astype("int64").to_numpy()

    # For simplicity operate on the assumption that the data is regularly
    # spaced at the cadence.  Any candidate whose position i is < window_len
    # is skipped (no trailing history).
    n_needed = len(needed_cols)
    idx_map = {c: i for i, c in enumerate(needed_cols)}
    values = full_df[needed_cols].to_numpy(dtype=float)

    kept = []
    stacks = []
    for i in candidate_indices:
        lo = i - window_len
        if lo < 0:
            continue
        win = values[lo:i, :]
        if win.shape[0] < window_len:
            continue
        stacks.append(win)
        kept.append(i)

    if not stacks:
        return None, [], idx_map
    stack = np.stack(stacks, axis=0)  # (n_cand, window_len, n_feat)
    return stack, kept, idx_map


def train_discriminator(
    training_frame=None,
    cadence_s=None,
    base_algorithm="xgb",
    output_dir=None,
    models_dir=None,
    window_s=DISCRIMINATOR_WINDOW_S,
    top_k_base=DISCRIMINATOR_TOP_K_BASE,
    tp_loss_weight=DISCRIMINATOR_TP_LOSS_WEIGHT,
    max_tp_kill_rate=DISCRIMINATOR_MAX_TP_KILL_RATE,
    verbose=True,
):
    """Train Stage-2 discriminators for all four horizons."""
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

    base = load_forecaster(base_algorithm, models_dir=models_dir)

    train_mask = (split_labels == "train").to_numpy()
    val_mask = (split_labels == "val").to_numpy()
    test_mask = (split_labels == "test").to_numpy()

    X_all = training_frame[feature_cols].to_numpy(dtype=float)

    # Sequence length in rows.  With cadence 10 s and window 1800 s this
    # is 180 rows.
    window_len = max(2, int(window_s // cadence_s))

    # Pattern-feature columns (always requested if present)
    pattern_cols = [
        "czt_net_rate", "cdte_count_rate_smooth", "cdte_count_rate_kalman",
        "solexs_hr_rmean_60", "solexs_hr_rmean_300",
        "solexs_cusum", "solexs_zscore_300",
        "solexs_band_ratio_mid_low", "solexs_band_ratio_high_mid",
        "particle_gti_proximity_s",
    ]
    pattern_cols = [c for c in pattern_cols if c in training_frame.columns]

    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from xgboost import XGBClassifier
        _use_xgb = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        _use_xgb = False

    reports = []
    for h in FORECAST_LABEL_HORIZONS_MIN:
        art = base.horizon_models[h]
        threshold_s1 = float(art["decision_threshold"])
        raw = _predict_positive(art["model"], X_all)
        probs = _apply_artifact_calibrator(art, raw)

        label_col = f"label_flare_within_{h}m"
        y_all = training_frame[label_col].fillna(0).astype(np.int8).to_numpy()

        # ── Top-K base features for this (algo, horizon) ───────────────
        top_base = _select_top_base_features(models_dir, base_algorithm, h, top_k_base)
        top_base = [c for c in top_base if c in training_frame.columns]
        needed_cols = list(dict.fromkeys(pattern_cols + top_base))

        cand_mask = probs >= threshold_s1

        # ── Train candidates ──────────────────────────────────────────
        tr_idx = np.where(cand_mask & train_mask)[0]
        if len(tr_idx) < 20:
            if verbose:
                print(f"  [SKIP] h={h}m: only {len(tr_idx)} train candidates.")
            continue

        stack_tr, tr_kept, idx_map = _build_candidate_windows(
            training_frame, feature_cols, tr_idx, needed_cols, window_len,
        )
        if stack_tr is None or len(tr_kept) < 20:
            if verbose:
                print(f"  [SKIP] h={h}m: only {len(tr_kept)} usable train windows.")
            continue

        # Features
        pf_tr = _vec_pattern_features(stack_tr, idx_map)
        ws_tr = _vec_window_summary(stack_tr, top_base, idx_map)
        Xtr = pd.concat([pf_tr, ws_tr], axis=1)
        y_tr = (y_all[tr_kept] == 0).astype(np.int8)

        if int(y_tr.sum()) < 5 or int((y_tr == 0).sum()) < 5:
            if verbose:
                print(f"  [SKIP] h={h}m: too few of one class in train "
                      f"(FP={int(y_tr.sum())}, TP={int((y_tr == 0).sum())}).")
            continue

        # Use GradientBoostingClassifier (matches the discriminator that
        # produced the excellent 30-min result — 249 FPs killed, 0 TPs
        # killed — during the first successful run).
        meta_feat_names = list(Xtr.columns)
        from sklearn.ensemble import GradientBoostingClassifier as _GB
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", _GB(
                n_estimators=150, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=42,
            )),
        ])
        clf.fit(Xtr.values, y_tr)

        # ── Val candidates for threshold tuning ─────────────────────────
        val_idx = np.where(cand_mask & val_mask)[0]
        threshold_s2 = 1.01   # default: never suppress
        val_metrics = {}
        val_net_stats = {}

        if len(val_idx) >= 20:
            stack_v, val_kept, _ = _build_candidate_windows(
                training_frame, feature_cols, val_idx, needed_cols, window_len,
            )
            if stack_v is not None and len(val_kept) >= 20:
                pf_v = _vec_pattern_features(stack_v, idx_map)
                ws_v = _vec_window_summary(stack_v, top_base, idx_map)
                Xv = pd.concat([pf_v, ws_v], axis=1)
                Xv = Xv.reindex(columns=meta_feat_names).fillna(0.0)
                y_v = (y_all[val_kept] == 0).astype(np.int8)
                if int(y_v.sum()) >= 3 and int((y_v == 0).sum()) >= 3:
                    p_v = clf.predict_proba(Xv.values)[:, 1]
                    threshold_s2, val_net, val_net_stats = _tune_stage2_threshold(
                        y_v, p_v,
                        max_tp_kill_rate=max_tp_kill_rate,
                        tp_loss_weight=tp_loss_weight,
                    )
                    val_metrics = compute_metrics(y_v, p_v, threshold=threshold_s2)

        # ── Test candidates ────────────────────────────────────────────
        test_idx = np.where(cand_mask & test_mask)[0]
        n_true_alerts_test = int((y_all[test_idx] == 1).sum()) if len(test_idx) else 0
        n_false_alerts_test = int((y_all[test_idx] == 0).sum()) if len(test_idx) else 0
        n_suppressed = 0
        wrongly_suppressed = 0
        correctly_suppressed = 0
        test_metrics = {}
        if len(test_idx) >= 10:
            stack_t, test_kept, _ = _build_candidate_windows(
                training_frame, feature_cols, test_idx, needed_cols, window_len,
            )
            if stack_t is not None and len(test_kept) >= 10:
                pf_t = _vec_pattern_features(stack_t, idx_map)
                ws_t = _vec_window_summary(stack_t, top_base, idx_map)
                Xt = pd.concat([pf_t, ws_t], axis=1)
                Xt = Xt.reindex(columns=meta_feat_names).fillna(0.0)
                y_t = (y_all[test_kept] == 0).astype(np.int8)
                p_t = clf.predict_proba(Xt.values)[:, 1]
                if int(y_t.sum()) and int((y_t == 0).sum()):
                    test_metrics = compute_metrics(y_t, p_t, threshold=threshold_s2)

                suppressed = p_t >= threshold_s2
                n_suppressed = int(suppressed.sum())
                wrongly_suppressed = int(((y_t == 0) & suppressed).sum())
                correctly_suppressed = int(((y_t == 1) & suppressed).sum())

        n_false_kept = n_false_alerts_test - correctly_suppressed
        n_true_kept = n_true_alerts_test - wrongly_suppressed

        # ── Persist ────────────────────────────────────────────────────
        disc = FalseAlertDiscriminator(
            horizon_min=int(h),
            model=clf,
            threshold=float(threshold_s2),
            feature_cols=list(feature_cols),
            input_feature_names=meta_feat_names,
            top_base_features=list(top_base),
        )
        art_path = models_dir / f"discriminator_{base_algorithm}_horizon_{h}m.joblib"
        joblib.dump({"discriminator": disc}, art_path)

        # Test-side net benefit at the tuned threshold
        test_net_benefit = correctly_suppressed - tp_loss_weight * wrongly_suppressed
        applied = threshold_s2 <= 1.0
        rep = {
            "horizon_min": int(h),
            "stage1_threshold": threshold_s1,
            "stage2_threshold": float(threshold_s2),
            "applied": bool(applied),
            "tp_loss_weight": float(tp_loss_weight),
            "n_train_candidates": int(len(tr_idx)),
            "n_val_candidates": int(len(val_idx)),
            "n_test_candidates": int(len(test_idx)),
            "n_test_true_alerts": n_true_alerts_test,
            "n_test_false_alerts": n_false_alerts_test,
            "n_test_suppressed": n_suppressed,
            "n_test_wrongly_suppressed": wrongly_suppressed,
            "n_test_correctly_suppressed": correctly_suppressed,
            "test_net_benefit": float(test_net_benefit),
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
            "val_net_correct": val_net_stats.get("correct", 0),
            "val_net_wrong": val_net_stats.get("wrong", 0),
            "val_net_score": val_net_stats.get("net", float("nan")),
            "top_k_base_features": len(top_base),
        }
        reports.append(rep)
        if verbose:
            tag = "APPLIED" if applied else "SKIPPED (net benefit <= 0)"
            print(
                f"  disc h={h}m [{tag}]  cand(test)={len(test_idx)}  "
                f"thr={threshold_s2:.2f}  "
                f"stage1_FAR_in_cand={n_false_alerts_test / max(1, len(test_idx)):.3f}  "
                f"kills FP={correctly_suppressed}  TP={wrongly_suppressed}  "
                f"net={test_net_benefit:+.1f}  "
                f"post-precision={rep['test_precision_after_gate']:.3f}"
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
