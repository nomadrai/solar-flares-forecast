# TEJAS — Feature Catalogue

This document describes every engineered feature produced by Phase 4 (Feature Engineering). Features are computed from Phase 3 clean daily Parquet files (40 columns → ~125 columns after feature engineering).

All features use **strictly causal (trailing) windows** — no future data is ever used, preserving scientific integrity during inference and playback.

---

## Source Column Priority

Features are computed from the best available signal variant, in priority order:

| Priority | Suffix | Origin |
|---|---|---|
| 1 (best) | `*_smooth` | Phase 3 rolling-median smoothed |
| 2 | `*_clean` | Phase 3 spike-removed |
| 3 (raw) | (no suffix) | Phase 2 unified |

**Rationale:** Smoothed signals are preferred because high-frequency noise produces noisy derivatives and rolling statistics. The fallback chain ensures features are computed even if smoothing was not applied.

---

## 1. Temporal Features (`tejas/features/temporal.py`)

Per-instrument features computed from the 3 primary signals:
- **SoLEXS** (`solexs_counts_smooth`) — soft X-ray total counts (2–22 keV)
- **CdTe** (`cdte_count_rate_smooth`) — hard X-ray counts (8–70 keV)
- **CZT** (`czt_net_rate`) — high-energy background-subtracted counts (20–150 keV)

### Rolling Mean — `{sig}_rmean_{w}`

| Feature | Windows (s) | Units | Description |
|---|---|---|---|
| `solexs_rmean_30` | 30 | c/s | Trailing 30s mean of soft X-ray flux |
| `solexs_rmean_60` | 60 | c/s | Trailing 60s mean |
| `solexs_rmean_300` | 300 | c/s | Trailing 5-min mean |
| `solexs_rmean_600` | 600 | c/s | Trailing 10-min mean |
| `cdte_rmean_{w}` | 30, 60, 300, 600 | c/s | Same for CdTe |
| `czt_rmean_{w}` | 30, 60, 300, 600 | c/s | Same for CZT |

**Physical meaning:** Smoothed recent flux level at different time scales. Short windows (30s) respond to fast transients; long windows (600s) represent the baseline trend.

### Rolling Standard Deviation — `{sig}_rstd_{w}`

| Feature | Windows (s) | Units | Description |
|---|---|---|---|
| `{sig}_rstd_30` | 30 | c/s | Trailing 30s standard deviation |
| `{sig}_rstd_60` | 60 | c/s | Trailing 60s std |
| `{sig}_rstd_300` | 300 | c/s | Trailing 5-min std |
| `{sig}_rstd_600` | 600 | c/s | Trailing 10-min std |

**Physical meaning:** Recent flux variability. Low during quiet Sun, spikes during flare onset when the signal becomes volatile.

### First Derivative — `{sig}_deriv1`

| Feature | Units | Description |
|---|---|---|
| `solexs_deriv1` | c/s² | Instantaneous rate of change (`diff(1)`) |
| `cdte_deriv1` | c/s² | Hard X-ray rate of change |
| `czt_deriv1` | c/s² | High-energy rate of change |

**Physical meaning:** Positive = flux increasing (flare onset), negative = flux decreasing (decay). The magnitude indicates how rapidly the signal is changing.

### Second Derivative — `{sig}_deriv2`

| Feature | Units | Description |
|---|---|---|
| `{sig}_deriv2` | c/s³ | Acceleration: `diff(diff(1))` |

**Physical meaning:** Detects onset curvature. Positive second derivative during a rise means the rise is accelerating — a hallmark of flare impulsive phase.

### Flux Gradient — `{sig}_grad_{h}`

| Feature | Horizons (s) | Units | Description |
|---|---|---|---|
| `{sig}_grad_30` | 30 | c/s per second | Average rate of change over 30s |
| `{sig}_grad_60` | 60 | c/s per second | Average rate over 60s |
| `{sig}_grad_300` | 300 | c/s per second | Average rate over 5 min |

**Physical meaning:** Less noisy than instantaneous derivatives for longer time scales. `(x[t] - x[t-h]) / h`.

### Rise/Decay Rates — `{sig}_rise_rate`, `{sig}_decay_rate`

| Feature | Units | Description |
|---|---|---|
| `{sig}_rise_rate` | c/s² | `max(deriv1, 0)` — zeroed during decay |
| `{sig}_decay_rate` | c/s² | `min(deriv1, 0)` — zeroed during rise |

**Physical meaning:** Separate channels for rising and falling behaviour. The rise rate is particularly useful as a forecasting feature: rapid rise in soft X-rays often precedes the flare peak.

**Total temporal features: 3 signals × 15 features = 45**

---

## 2. Spectral Features (`tejas/features/spectral.py`)

SoLEXS PI-derived features leveraging energy-resolved information.

### Hardness Ratio Features

| Feature | Description | Physical Meaning |
|---|---|---|
| `solexs_hr_rmean_30` | Trailing 30s mean of hardness ratio | Smoothed spectral shape trend |
| `solexs_hr_rmean_60` | Trailing 60s mean | Medium-term spectral trend |
| `solexs_hr_rmean_300` | Trailing 5-min mean | Long-term spectral trend |
| `solexs_hr_deriv1` | First derivative of HR | **Key precursor:** rate of spectral hardening |
| `solexs_hr_grad_30` | HR gradient over 30s | Short-term hardening rate |
| `solexs_hr_grad_60` | HR gradient over 60s | Medium-term hardening rate |
| `solexs_hr_grad_300` | HR gradient over 5 min | Long-term hardening rate |

**Why these matter:** Spectral hardening (HR increasing) is a well-documented precursor that often appears **minutes before** flare onset. The EDA phase confirmed this in several events. The `solexs_hr_deriv1` feature is expected to be among the most important for forecasting.

### Band Ratio Features

| Feature | Formula | Physical Meaning |
|---|---|---|
| `solexs_band_ratio_mid_low` | counts_mid / counts_low | Mid-to-low energy ratio (5–10 / 2.8–5 keV) |
| `solexs_band_ratio_high_mid` | counts_high / counts_mid | High-to-mid energy ratio (10–22 / 5–10 keV) |
| `solexs_band_ratio_mid_low_deriv1` | `diff(1)` on mid/low | Temporal change in low-energy spectral slope |
| `solexs_band_ratio_high_mid_deriv1` | `diff(1)` on high/mid | Temporal change in high-energy spectral slope |

### Spectral Slope

| Feature | Formula | Physical Meaning |
|---|---|---|
| `solexs_spectral_slope` | `(log(high) - log(low)) / ΔE` | Approximate power-law index proxy |
| `solexs_spectral_slope_deriv1` | `diff(1)` on slope | Rate of spectral slope change |

**Total spectral features: 13**

---

## 3. Cross-Instrument Features (`tejas/features/cross_instrument.py`)

Features combining signals across SoLEXS, CdTe, and CZT.

### Flux Ratios

| Feature | Formula | Physical Meaning |
|---|---|---|
| `ratio_solexs_cdte` | SoLEXS / CdTe | Soft-to-hard X-ray ratio (thermal/non-thermal balance) |
| `ratio_solexs_czt` | SoLEXS / CZT_net | Soft-to-high-energy ratio |
| `ratio_cdte_czt` | CdTe / CZT_net | Hard X-ray spectral hardness |
| `ratio_solexs_cdte_deriv1` | `diff(1)` on soft/hard ratio | Temporal evolution of thermal balance |
| `ratio_solexs_cdte_rmean_60` | 60s rolling mean of ratio | Smoothed short-term ratio trend |
| `ratio_solexs_cdte_rmean_300` | 300s rolling mean of ratio | Smoothed long-term ratio trend |

### Cross-Correlation (Neupert Effect)

| Feature | Description | Physical Meaning |
|---|---|---|
| `lag_solexs_cdte` | Lag at peak cross-correlation (seconds) | Neupert effect: how much soft X-ray lags hard X-ray |
| `xcorr_solexs_cdte` | Peak normalised cross-correlation value | Strength of soft-hard coupling |

**Computation:** Sliding 300s trailing window, max lag ±120s, computed every 60s and forward-filled. Positive lag means CdTe (hard) leads SoLEXS (soft), consistent with the Neupert effect.

**Total cross-instrument features: 8**

---

## 4. Statistical Features (`tejas/features/statistical.py`)

Context-aware features capturing how unusual the current signal is.

### Z-Score — `{sig}_zscore_{w}`

| Feature | Windows (s) | Description |
|---|---|---|
| `{sig}_zscore_300` | 300 | σ above trailing 5-min mean |
| `{sig}_zscore_600` | 600 | σ above trailing 10-min mean |

**Physical meaning:** During quiet Sun, z ≈ 0. During flare onset, z rises sharply (often 5–10σ). A std floor (0.1) prevents extreme values during very stable quiet periods.

### Trailing Percentile Rank — `{sig}_pctrank_{w}`

| Feature | Windows (s) | Description |
|---|---|---|
| `{sig}_pctrank_300` | 300 | Rank in trailing 5-min window [0, 1] |
| `{sig}_pctrank_600` | 600 | Rank in trailing 10-min window [0, 1] |

**Physical meaning:** Where the current value sits relative to recent history. A pctrank near 1.0 means the signal is at its recent maximum — potential flare onset.

### CUSUM — `{sig}_cusum`

| Feature | Description |
|---|---|
| `{sig}_cusum` | Cumulative sum of positive deviations from trailing 300s mean |

**Physical meaning:** Detects sustained mean shifts that rolling-window features might miss. Accumulates small positive deviations that individually are not significant but collectively indicate a rising trend. Resets every 1800s (30 min) to prevent unbounded growth.

**Total statistical features: 3 signals × 5 features = 15**

---

## Feature Count Summary

| Category | Features per Signal | Signals | Total |
|---|---|---|---|
| Temporal | 15 | 3 (SoLEXS, CdTe, CZT) | 45 |
| Spectral | 13 | 1 (SoLEXS PI) | 13 |
| Cross-instrument | 8 | — (multi-signal) | 8 |
| Statistical | 5 | 3 (SoLEXS, CdTe, CZT) | 15 |
| **Total** | | | **81** |

Combined with the 40 input columns from Phase 3, the output dataset has **~121 columns** per day.

---

## No-Future-Data Guarantee

Every feature in this catalogue satisfies the strict no-future-data constraint:

- **Rolling operations:** Use trailing windows (`center=False`, pandas default for `.rolling()`)
- **Derivatives:** Use `diff(1)` which computes `x[t] - x[t-1]`
- **Gradients:** Use `shift(h)` which looks back `h` steps
- **Cross-correlation:** Uses a trailing window of past data only
- **CUSUM:** Accumulates forward in time from past deviations

This is verified by `test_no_future_leakage_in_features` in the test suite.
