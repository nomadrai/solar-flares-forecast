"""
TEJAS Particle / SEP Discriminator Features (Fix 3)
=====================================================
Features that distinguish genuine flare precursors from Solar Energetic
Particle (SEP) events, cosmic ray hits, and GTI-boundary artifacts in
the HEL1OS CZT detectors — the primary driver of 5-minute false alerts.

During a genuine pre-flare / flare event:
  * CdTe (8–70 keV) and CZT (20–150 keV) both rise together
  * The CZT/CdTe ratio evolves consistently with plasma spectral shape
  * The Neupert effect: CdTe hard X-ray emission precedes SoLEXS soft X-ray

During a particle / SEP event:
  * CZT spikes disproportionately (CdTe barely responds)
  * The CZT/CdTe ratio spikes anomalously
  * No corresponding soft X-ray (SoLEXS) response

All features use strictly causal (trailing) windows.
"""

import numpy as np
import pandas as pd

from tejas.config import (
    ZERO_COUNT_EPS,
    PARTICLE_RATIO_WINDOW_S,
)


def _resolve_cdte(df):
    """Return the best available CdTe column (Kalman-preferred)."""
    for col in (
        "cdte_count_rate_kalman",   # Fix 6 Kalman (best)
        "cdte_count_rate_smooth",
        "cdte_count_rate_clean",
        "cdte_count_rate",
    ):
        if col in df.columns:
            return col
    return None


def _resolve_czt(df):
    """Return the best available CZT column."""
    for col in (
        "czt_net_rate",              # Background-subtracted (best)
        "czt_count_rate_kalman",
        "czt_count_rate_smooth",
        "czt_count_rate_clean",
        "czt_count_rate",
    ):
        if col in df.columns:
            return col
    return None


def compute_particle_discriminator_features(df):
    """Compute particle-event discriminator features.

    Adds 5 new columns:
        particle_czt_cdte_ratio       — instantaneous CZT/CdTe count ratio
        particle_czt_cdte_zscore      — z-score of ratio vs. trailing 600s
        particle_czt_without_cdte     — binary: CZT spike w/o CdTe response
        particle_hr_without_flux      — binary: HR rise w/o total flux rise
        particle_gti_proximity_s      — seconds since last GTI-start row

    Args:
        df: DataFrame containing at minimum a resolvable CdTe and CZT
            signal column.  Optional inputs: czt_above_bg,
            solexs_hr_deriv1, solexs_deriv1, gti_start.

    Returns:
        Dict with keys df, columns_added, report.
    """
    df = df.copy()
    added = []

    cdte_col = _resolve_cdte(df)
    czt_col = _resolve_czt(df)

    if cdte_col is None or czt_col is None:
        # Missing HEL1OS data — emit neutral values so the model always
        # sees the columns.
        for col, default in [
            ("particle_czt_cdte_ratio",   0.0),
            ("particle_czt_cdte_zscore",  0.0),
            ("particle_czt_without_cdte", 0),
            ("particle_hr_without_flux",  0),
            ("particle_gti_proximity_s",  9999.0),
        ]:
            df[col] = default
            added.append(col)
        return {"df": df, "columns_added": added,
                "report": {"warning": "no CdTe/CZT columns available"}}

    # ── Feature 1: CZT/CdTe instantaneous ratio ────────────────────────
    safe_cdte = df[cdte_col].astype(float).clip(lower=ZERO_COUNT_EPS)
    ratio = df[czt_col].astype(float) / safe_cdte
    df["particle_czt_cdte_ratio"] = ratio.astype(np.float32)
    added.append("particle_czt_cdte_ratio")

    # ── Feature 2: CZT/CdTe ratio z-score (trailing 600s) ──────────────
    w = int(PARTICLE_RATIO_WINDOW_S)
    minp = max(10, w // 10)
    r_mean = ratio.rolling(w, min_periods=minp).mean()
    r_std = ratio.rolling(w, min_periods=minp).std()
    r_std = r_std.clip(lower=0.01)
    df["particle_czt_cdte_zscore"] = ((ratio - r_mean) / r_std).astype(np.float32)
    added.append("particle_czt_cdte_zscore")

    # ── Feature 3: CZT fires without CdTe coupling ─────────────────────
    cdte_deriv1 = df[cdte_col].astype(float).diff(1)
    if "czt_above_bg" in df.columns:
        czt_above = df["czt_above_bg"].astype(bool)
    else:
        # No background-subtracted column: use crude z-score proxy on CZT.
        cz = df[czt_col].astype(float)
        czt_above = (cz - cz.rolling(600, min_periods=30).mean()) > (
            3.0 * cz.rolling(600, min_periods=30).std().clip(lower=0.1)
        )
        czt_above = czt_above.fillna(False)
    cdte_flat = cdte_deriv1 <= 0.0

    czt_above_sustained = czt_above.astype(float).rolling(
        30, min_periods=5).max() > 0
    cdte_flat_sustained = cdte_flat.astype(float).rolling(
        30, min_periods=5).min() > 0
    df["particle_czt_without_cdte"] = (
        czt_above_sustained & cdte_flat_sustained
    ).fillna(False).astype(np.int8)
    added.append("particle_czt_without_cdte")

    # ── Feature 4: Spectral hardening without total flux increase ──────
    have_hr = "solexs_hr_deriv1" in df.columns
    have_flux = "solexs_deriv1" in df.columns
    if have_hr and have_flux:
        hr_rising = df["solexs_hr_deriv1"].astype(float) > 0
        flux_flat = df["solexs_deriv1"].astype(float) <= 0
        hr_rising_s = hr_rising.astype(float).rolling(
            60, min_periods=10).mean() > 0.5
        flux_flat_s = flux_flat.astype(float).rolling(
            60, min_periods=10).mean() > 0.5
        df["particle_hr_without_flux"] = (
            hr_rising_s & flux_flat_s
        ).fillna(False).astype(np.int8)
    else:
        df["particle_hr_without_flux"] = np.int8(0)
    added.append("particle_hr_without_flux")

    # ── Feature 5: GTI proximity (seconds since last gti_start) ────────
    if "gti_start" in df.columns and "time_unix" in df.columns:
        gti_mask = df["gti_start"].astype(bool).to_numpy()
        time_s = df["time_unix"].astype(float).to_numpy()
        if gti_mask.any():
            gti_start_times = time_s[gti_mask]
            # searchsorted needs a sorted array
            gti_start_times = np.sort(gti_start_times)
            pos = np.searchsorted(gti_start_times, time_s, side="right") - 1
            pos = np.clip(pos, 0, len(gti_start_times) - 1)
            last_start = gti_start_times[pos]
            since = np.clip(time_s - last_start, 0, None).astype(np.float32)
            df["particle_gti_proximity_s"] = since
        else:
            df["particle_gti_proximity_s"] = np.float32(9999.0)
    else:
        df["particle_gti_proximity_s"] = np.float32(9999.0)
    added.append("particle_gti_proximity_s")

    report = {
        "columns_added": added,
        "cdte_source": cdte_col,
        "czt_source": czt_col,
        "n_czt_without_cdte_events": int(df["particle_czt_without_cdte"].sum()),
        "n_hr_without_flux_events": int(df["particle_hr_without_flux"].sum()),
        "mean_gti_proximity_s": float(df["particle_gti_proximity_s"].mean()),
    }
    return {"df": df, "columns_added": added, "report": report}
