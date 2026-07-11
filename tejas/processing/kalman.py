"""
TEJAS Kalman Filter Smoothing (Fix 6)
=======================================
Online 1D Kalman filter for CZT and CdTe count rate columns.

Model (local level / random walk):
    x_t = x_{t-1} + w_t       w_t ~ N(0, Q)   (process noise)
    z_t = x_t     + v_t       v_t ~ N(0, R_t) (measurement noise)

where R_t comes from the per-sample STAT_ERR column (Poisson error) when
available, and falls back to a scalar KALMAN_R_DEFAULT otherwise.

This filter is *strictly causal* — the filtered estimate at time t
only depends on observations up to time t.  It is applied per
``segment_id`` so filter state never bleeds across data gaps.

Rollback path: remove ``"_kalman"`` from the ``variants`` lists in
tejas/features/temporal.py::_SOURCE_SIGNALS.  No Phase 3 re-run needed;
the existing ``*_smooth`` columns remain untouched.
"""

import numpy as np
import pandas as pd

from tejas.config import KALMAN_Q_CZT, KALMAN_Q_CDTE, KALMAN_R_DEFAULT


# ── Core scalar filter ─────────────────────────────────────────────────────


def kalman_filter_1d(
    signal,
    stat_err=None,
    q_process_noise=1.0,
    r_default=5.0,
    x0=None,
    p0=100.0,
):
    """Apply a scalar Kalman filter to a 1D time series.

    Args:
        signal: 1D array of measurements (count rate).
        stat_err: 1D array of per-sample statistical error (sigma).
            If provided, R = stat_err**2.  Non-finite values fall back
            to ``r_default``.  If ``None``, R = r_default**2 for all
            samples.
        q_process_noise: Scalar process noise variance Q.
        r_default: Default measurement noise std when stat_err is
            missing or non-finite.
        x0: Initial state estimate.  Defaults to the first finite
            observation (or 0 if none).
        p0: Initial error covariance.  Large value => high initial
            uncertainty, so the filter converges quickly to the data.

    Returns:
        1D numpy array of filtered state estimates.
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    if n == 0:
        return sig.copy()
    out = np.empty(n, dtype=float)

    # Initialise state.  Pick the first finite observation as x0 when
    # the caller did not override it.
    if x0 is None:
        finite = sig[np.isfinite(sig)]
        x = float(finite[0]) if len(finite) > 0 else 0.0
    else:
        x = float(x0)
    p = float(p0)

    # Pre-compute per-sample R.
    if stat_err is not None:
        err = np.asarray(stat_err, dtype=float)
        # Non-finite or non-positive → default
        bad = ~np.isfinite(err) | (err <= 0)
        err[bad] = r_default
        r_arr = err ** 2
    else:
        r_arr = np.full(n, float(r_default) ** 2, dtype=float)

    q = float(q_process_noise)

    for i in range(n):
        z = sig[i]
        # Predict
        # x_pred = x (random-walk transition)
        p_pred = p + q
        if np.isfinite(z):
            R = r_arr[i]
            denom = p_pred + R
            if denom <= 0:
                K = 0.0
            else:
                K = p_pred / denom
            x = x + K * (z - x)
            p = (1.0 - K) * p_pred
        else:
            # Missing measurement — keep prediction, grow uncertainty.
            p = p_pred
        out[i] = x

    return out


# ── Day-level application ──────────────────────────────────────────────────


def smooth_day_kalman(df):
    """Apply per-signal Kalman filtering to CZT and CdTe count rates.

    Adds ``{signal}_kalman`` columns to the DataFrame in-place (via copy).
    Operates segment-by-segment (respects ``cdte_segment_id`` /
    ``czt_segment_id`` where present) to prevent filter state from
    bleeding across data gaps.

    Args:
        df: Phase 3 clean DataFrame (must contain time_unix,
            cdte_count_rate, czt_count_rate; ideally
            cdte_stat_err/czt_stat_err and cdte_segment_id/czt_segment_id).

    Returns:
        Dict with keys:
            df: DataFrame with new columns added.
            columns_added: list of new column names.
            report: dict of summary stats.
    """
    df = df.copy()
    added = []

    # (signal_col, err_col, seg_col, q)
    targets = [
        ("cdte_count_rate", "cdte_stat_err", "cdte_segment_id", KALMAN_Q_CDTE),
        ("czt_count_rate",  "czt_stat_err",  "czt_segment_id",  KALMAN_Q_CZT),
    ]

    for sig_col, err_col, seg_col, q in targets:
        if sig_col not in df.columns:
            continue
        out_col = f"{sig_col}_kalman"
        df[out_col] = np.nan

        if seg_col in df.columns:
            # Per-segment filtering (isolates state across GTI gaps).
            for _, seg_df in df.groupby(seg_col, sort=False):
                signal = seg_df[sig_col].values.astype(float)
                stat_err = (
                    seg_df[err_col].values.astype(float)
                    if err_col in df.columns else None
                )
                filtered = kalman_filter_1d(
                    signal, stat_err=stat_err,
                    q_process_noise=q, r_default=KALMAN_R_DEFAULT,
                )
                df.loc[seg_df.index, out_col] = filtered
        else:
            signal = df[sig_col].values.astype(float)
            stat_err = (
                df[err_col].values.astype(float)
                if err_col in df.columns else None
            )
            df[out_col] = kalman_filter_1d(
                signal, stat_err=stat_err,
                q_process_noise=q, r_default=KALMAN_R_DEFAULT,
            )

        added.append(out_col)

    report = {
        "columns_added": added,
        "q_cdte": KALMAN_Q_CDTE,
        "q_czt": KALMAN_Q_CZT,
        "r_default": KALMAN_R_DEFAULT,
    }
    return {"df": df, "columns_added": added, "report": report}
