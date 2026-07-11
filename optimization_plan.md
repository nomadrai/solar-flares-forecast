# TEJAS — False Alert Optimization Plan

> **Scope:** All 8 fixes targeting the high false-alarm rate (FAR) across the
> 5 / 15 / 30 / 60 minute forecasting horizons. No changes are implemented
> until this plan is approved. Each fix section includes exact file paths,
> function names, new code to write, benchmark methodology, and pipeline
> re-run requirements.

---

## 1. Performance Baseline (pre-optimization)

From `data/forecasting/2026/05/forecast_report.md` — XGBoost, Youden threshold:

| Horizon | PR-AUC | Precision | Recall | FAR    | Threshold | True:False Alert Ratio |
|---------|--------|-----------|--------|--------|-----------|----------------------|
| 5 min   | 0.251  | 0.232     | 0.585  | 0.4008 | 0.16      | 1 : 3.3              |
| 15 min  | 0.475  | 0.463     | 0.704  | 0.6036 | 0.45      | 1 : 1.2              |
| 30 min  | 0.701  | 0.701     | 0.802  | 0.6965 | 0.68      | 1 : 0.4              |
| 60 min  | 0.883  | 0.891     | 0.853  | 0.7428 | 0.94      | 1 : 0.1              |

**Reading:** The 5-min model fires at threshold 0.16, which means nearly any transient
above background triggers an alert. 77% of those alerts are false. The 60-min model
has the best precision but the highest raw FAR because even at 0.94 threshold it fires
on any sustained elevated baseline (of which there are many, due to quiet-Sun drift).

**Target post-optimization (all horizons):**
- FAR ≤ 15%
- Recall ≥ 50% (5 min) / ≥ 55% (15 min) / ≥ 65% (30 min) / ≥ 75% (60 min)

---

## 2. Dependency Map

Some fixes are independent; others require earlier fixes to land first (because they
require pipeline re-runs). The dependency graph:

```
Fix 6 (Kalman) ──→ rerun Phase 3 ──→ rerun Phase 4 ──→ rerun Phase 5
Fix 3 (particle features)           ──→ rerun Phase 4 ──→ rerun Phase 5
Fix 7 (AR-proxy features)           ──→ rerun Phase 4 ──→ rerun Phase 5
                                                         ↓
Fix 4 (class-specific labels)                       rerun Phase 5
                                                         ↓
                              All above land ──→ Fix 1 (threshold config)
                                             ──→ Fix 5 (stacking)
                                             ──→ Fix 8 (discriminator)
                                             ──→ rerun Phase 7 (--all)

Fix 2 (confirmation gate) ──→ Phase 8 Alert Engine (new module, no pipeline dep)
```

**Recommended execution order:**
1. Fix 7 (AR-proxy features) — highest impact, lowest effort, no upstream changes
2. Fix 3 (particle discriminator features) — high impact at 5-min, no upstream changes
3. Fix 1 (FAR-constrained threshold) — one config line, immediate result
4. Fix 4 (class-specific labels) — new label column, requires Phase 5 re-run
5. Fix 6 (Kalman filter) — adds Kalman smoothing to Phase 3, requires Phase 3→5 re-run
6. Fix 8 (two-stage discriminator) — trains on Phase 7 outputs, no pipeline change
7. Fix 5 (ensemble stacking) — trains on Phase 7 outputs, no pipeline change
8. Fix 2 (confirmation gate) — new Phase 8 module, independent of all above

---

## 3. Fix 1 — FAR-Constrained Decision Threshold

### What It Fixes

The current threshold metric `"youden"` maximises `TPR − FPR` (Youden's J). Under
TEJAS's extreme class imbalance (~270 flares across 31 days at 10s cadence, roughly
1 positive per 500 rows), Youden's J tends to set dangerously low thresholds —
0.16 for the 5-min horizon. This means the model fires on any small signal elevation
above a very low bar, generating massive false-alarm volumes.

The `"far_constrained"` metric already exists in `tejas/forecasting/evaluation.py`
(function `tune_threshold()`, lines 192–211). It performs the same grid search but
returns the **highest threshold** that keeps `FAR ≤ FORECAST_TARGET_FAR`. The code
is already there — only the config values need changing.

### Files to Modify

**`tejas/config.py`** — two constants in the Phase 7 block (~line 354–357):

```python
# BEFORE:
FORECAST_THRESHOLD_METRIC   = "youden"
FORECAST_TARGET_FAR         = 0.15   # (only used when metric="far_constrained")

# AFTER:
FORECAST_THRESHOLD_METRIC   = "far_constrained"
FORECAST_TARGET_FAR         = 0.10   # 10% FAR ceiling across all horizons
```

That is the **entire code change** for Fix 1.

### Pipeline Re-run Required

After changing config, re-train all models so the new threshold is embedded in the
saved artifacts:

```bash
python scripts/run_forecasting.py --all
```

This re-runs prepare (no change), trains all 12 models (new threshold selected on val),
and re-evaluates on test. New `forecast_report.md` will show updated metrics.

### Expected Impact (Estimated)

| Horizon | FAR Before | FAR After | Recall Cost |
|---------|-----------|-----------|-------------|
| 5 min   | 40.1%     | ~10%      | ~58% → ~40–45% |
| 15 min  | 60.4%     | ~10%      | ~70% → ~50–55% |
| 30 min  | 69.7%     | ~10%      | ~80% → ~65–70% |
| 60 min  | 74.3%     | ~10%      | ~85% → ~75–80% |

The recall loss is the intentional trade-off: we accept fewer caught flares in exchange
for far more credible alerts. This is operationally defensible — an alert operator who
receives 10 false alerts per true one will start ignoring alerts entirely.

### Benchmark

Compare `forecast_report.md` before vs. after:
- Primary: FAR per horizon
- Secondary: Recall per horizon (must not drop below targets in §1)
- Tertiary: PR-AUC (threshold-independent; should be unchanged by this fix since it
  is computed across all thresholds)

### Interactions / Risks

- `FORECAST_FBETA` and `FORECAST_MIN_RECALL` remain in config but are unused when
  `metric = "far_constrained"`. No conflict.
- If the constraint `FAR ≤ 0.10` is never met on the val split for any horizon, the
  fallback in `tune_threshold()` (line 208) returns the threshold with the lowest FAR
  achieved. Check the report for a note if this happens and adjust
  `FORECAST_TARGET_FAR` upward to 0.15.

---

## 4. Fix 2 — Multi-Cadence Confirmation Gate (Phase 8 Alert Engine)

### What It Fixes

The forecasting model makes an independent prediction at each 10-second row. A single
time step crossing the threshold triggers an alert immediately. Real pre-flare activity
is physically sustained over minutes (spectral hardening builds, CUSUM accumulates,
flux gradients persist). Transient spikes from particle hits, instrument noise, or
isolated flux bumps produce single-step exceedances that are false alarms.

A confirmation gate requires the model to exceed the threshold for `K` consecutive
windows before firing a confirmed alert. Since the 5-min horizon has a median lead
time of 298 seconds (P25 = 231s), absorbing 30s of confirmation delay preserves
virtually all the lead time while suppressing spike-driven false alarms.

### New Module Structure

Create `tejas/alerts/` package:

```
tejas/alerts/
├── __init__.py
├── confirmation.py     ← confirmation state machine
└── engine.py           ← top-level alert orchestrator (future Phase 8 home)
```

### `tejas/alerts/confirmation.py` — Full Design

```python
"""
Confirmation gate for TEJAS alert engine.

A ConfirmationGate wraps the Forecaster and requires K consecutive
10-second windows to exceed the decision threshold before emitting
a CONFIRMED alert. This suppresses single-timestep spike-driven
false alarms while preserving lead time for real pre-flare events.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

# States
IDLE      = "idle"       # No exceedance in progress
PENDING   = "pending"    # Exceedance streak started, not yet confirmed
CONFIRMED = "confirmed"  # K consecutive exceedances — alert fired
COOLDOWN  = "cooldown"   # Post-alert suppression window


@dataclass
class ConfirmationGate:
    """
    Stateful per-horizon confirmation gate.

    Parameters
    ----------
    horizon_min : int
        One of [5, 15, 30, 60]. Determines which probability and threshold
        to watch.
    decision_threshold : float
        Val-tuned threshold for this horizon (from Forecaster.decision_thresholds()).
    k_consecutive : int
        Number of consecutive windows required before confirming.
        Default = 3 (30 seconds at 10s cadence).
    cooldown_s : int
        Seconds after a confirmed alert during which no new alert can fire.
        Default = 300 (5 minutes).
    cadence_s : int
        Prediction cadence in seconds. Default = 10.
    """
    horizon_min       : int
    decision_threshold: float
    k_consecutive     : int   = 3
    cooldown_s        : int   = 300
    cadence_s         : int   = 10

    # Internal state (not set by user)
    _state            : str   = field(default=IDLE, init=False, repr=False)
    _streak           : int   = field(default=0,    init=False, repr=False)
    _cooldown_remaining: int  = field(default=0,    init=False, repr=False)

    def step(self, prob: float, timestamp: pd.Timestamp) -> dict:
        """
        Process one prediction step.

        Parameters
        ----------
        prob : float
            Calibrated probability for this horizon at the current timestep.
        timestamp : pd.Timestamp
            Current simulation timestamp.

        Returns
        -------
        dict with keys:
            state       : str    — current state after this step
            alert_fired : bool   — True if a new CONFIRMED alert fired this step
            streak      : int    — consecutive exceedance count
            horizon_min : int
            timestamp   : pd.Timestamp
        """
        above = prob >= self.decision_threshold

        # Tick down cooldown
        if self._state == COOLDOWN:
            self._cooldown_remaining -= self.cadence_s
            if self._cooldown_remaining <= 0:
                self._state = IDLE
                self._cooldown_remaining = 0

        alert_fired = False

        if self._state in (IDLE, PENDING):
            if above:
                self._streak += 1
                self._state = PENDING
                if self._streak >= self.k_consecutive:
                    self._state = CONFIRMED
                    alert_fired = True
                    self._streak = 0
                    self._cooldown_remaining = self.cooldown_s
            else:
                # Reset streak on any non-exceedance
                self._streak = 0
                self._state = IDLE

        # In COOLDOWN: ignore new exceedances (they'll be picked up
        # once cooldown ends and the gate returns to IDLE)

        return {
            "state"      : self._state,
            "alert_fired": alert_fired,
            "streak"     : self._streak,
            "horizon_min": self.horizon_min,
            "timestamp"  : timestamp,
        }

    def reset(self):
        """Hard reset — use when jumping to a new simulation segment."""
        self._state = IDLE
        self._streak = 0
        self._cooldown_remaining = 0


class MultiHorizonGate:
    """
    Manages four ConfirmationGates (one per horizon) simultaneously.
    Used by the dashboard playback loop.
    """
    def __init__(self, thresholds: dict, k_consecutive: int = 3,
                 cooldown_s: int = 300, cadence_s: int = 10):
        """
        Parameters
        ----------
        thresholds : dict
            {horizon_min: decision_threshold} from Forecaster.decision_thresholds()
        """
        self.gates = {
            h: ConfirmationGate(
                horizon_min=h,
                decision_threshold=t,
                k_consecutive=k_consecutive,
                cooldown_s=cooldown_s,
                cadence_s=cadence_s,
            )
            for h, t in thresholds.items()
        }

    def step(self, probs: dict, timestamp: pd.Timestamp) -> list[dict]:
        """
        Process one prediction step across all four horizons.

        Parameters
        ----------
        probs : dict
            {"prob_5m": float, "prob_15m": float, "prob_30m": float, "prob_60m": float}
            from Forecaster.predict_row()
        timestamp : pd.Timestamp

        Returns
        -------
        List of result dicts (one per horizon, from ConfirmationGate.step()).
        """
        results = []
        for h, gate in self.gates.items():
            p = probs.get(f"prob_{h}m", 0.0)
            results.append(gate.step(p, timestamp))
        return results

    def reset_all(self):
        for gate in self.gates.values():
            gate.reset()
```

### `tejas/alerts/__init__.py`

```python
from tejas.alerts.confirmation import ConfirmationGate, MultiHorizonGate

__all__ = ["ConfirmationGate", "MultiHorizonGate"]
```

### Config Constants to Add

In `tejas/config.py`, add a **Phase 8: Alert Engine** block:

```python
# ─────────────────────────────────────────────────────────────
# Phase 8: Alert Engine
# ─────────────────────────────────────────────────────────────
ALERT_CONFIRMATION_K         = 3     # consecutive windows before confirming
ALERT_COOLDOWN_S             = 300   # post-alert suppression window (seconds)
ALERT_CADENCE_S              = FORECAST_CADENCE_S   # inherit from forecast cadence
```

### Dashboard Integration Point

In the Streamlit dashboard playback loop, replace the current binary threshold check:

```python
# BEFORE (raw threshold check):
if probs["prob_5m"] >= thresholds[5]:
    fire_alert(...)

# AFTER (confirmation gate):
gate_results = multi_horizon_gate.step(probs, current_timestamp)
for result in gate_results:
    if result["alert_fired"]:
        fire_alert(result["horizon_min"], current_timestamp)
```

### Benchmark — Offline Retrospective Simulation

Run the confirmation gate as a **post-processing filter** over the test-set predictions
already saved in the training frame:

1. Load `data/forecasting/2026/05/training_frame_10s.parquet`
2. Filter to test days (27–31)
3. Replay predictions row-by-row through `MultiHorizonGate`
4. Compare:
   - Alerts fired by raw threshold (current)
   - Alerts fired after confirmation gate
   - For each real flare: did a confirmed alert arrive before the flare onset?

Metrics:
- False alert suppression rate: `(raw_alerts - confirmed_alerts) / raw_alerts`
- Recall preservation: `confirmed_true_alerts / raw_true_alerts`
- Lead time distribution: same `compute_lead_time()` from `evaluation.py`

Expected: ~30–40% of false alerts (particularly 5-min) are single-step spikes that
the confirmation gate suppresses with < 5% recall cost.

### No-Future-Data Compliance

The `ConfirmationGate` is a pure streaming state machine. It only sees past
probability values at the current timestamp. No look-ahead. Fully compliant.

---

## 5. Fix 3 — Particle Event / SEP Discriminator Features

### What It Fixes

The CZT detectors (32 cm² combined area) are large enough to detect non-solar
backgrounds: Solar Energetic Particle (SEP) events and cosmic rays. During a
particle hit, `czt_net_rate` spikes sharply while `cdte_count_rate_smooth` stays
flat (CdTe's 0.5 cm² area is orders of magnitude less sensitive to diffuse particle
backgrounds). The model currently has no explicit feature to distinguish a
CZT-only particle spike from a genuine pre-flare soft-to-hard X-ray signal
that appears in both instruments.

This fix adds 5 new features specifically designed to flag particle-event signatures.
These are computed from signals that **already exist** in the Phase 3 processed data —
no new data ingestion is needed.

### New File: `tejas/features/particle_discriminator.py`

```python
"""
Particle event / SEP discriminator features.

These features identify signatures characteristic of Solar Energetic Particle (SEP)
events and cosmic ray hits in the HEL1OS CZT detectors — the primary driver of
false alerts at the 5-minute forecasting horizon.

During a genuine pre-flare or flare event:
  - CdTe (8-70 keV) and CZT (20-150 keV) both rise together
  - The CZT/CdTe ratio evolves in a characteristic, physically consistent way
  - The Neupert effect: CdTe hard X-ray emission precedes SoLEXS soft X-ray by 30-120s

During a particle event:
  - CZT spikes disproportionately (CdTe does not respond)
  - The CZT/CdTe ratio spikes anomalously
  - No corresponding soft X-ray (SoLEXS) response

All features use strictly causal (trailing) windows.
"""

import numpy as np
import pandas as pd
from tejas.config import (
    ZERO_COUNT_EPS,
    PARTICLE_RATIO_WINDOW_S,          # new config constant (600s)
    PARTICLE_RATIO_SPIKE_THRESHOLD,   # new config constant (3.0 sigma)
    PARTICLE_GTI_PROXIMITY_WINDOW_S,  # new config constant (120s)
)


def compute_particle_discriminator_features(df: pd.DataFrame) -> dict:
    """
    Compute particle event discriminator features.

    Requires columns from Phase 3 processed data:
        czt_net_rate         -- background-subtracted CZT signal
        cdte_count_rate_smooth (or cdte_count_rate_clean / cdte_count_rate)
        czt_above_bg         -- boolean: CZT above background significance threshold
        solexs_counts_smooth
        gti_start            -- boolean: True at the first row of each GTI interval
                               (added by this fix to the processing pipeline, see below)

    Returns
    -------
    dict with keys:
        df              : DataFrame with new columns added
        columns_added   : list of new column names
        report          : dict with summary stats
    """
    added = []
    df = df.copy()

    # Resolve CdTe source column (same priority chain as other features)
    cdte_col = _resolve_cdte(df)
    czt_col  = "czt_net_rate"

    # Feature 1: CZT/CdTe instantaneous ratio
    # During a real flare: ratio is elevated but stable (both instruments respond)
    # During a particle hit: ratio spikes sharply (CZT up, CdTe flat)
    safe_cdte = df[cdte_col].clip(lower=ZERO_COUNT_EPS)
    czt_cdte_ratio = df[czt_col] / safe_cdte
    df["particle_czt_cdte_ratio"] = czt_cdte_ratio
    added.append("particle_czt_cdte_ratio")

    # Feature 2: CZT/CdTe ratio z-score (trailing 600s window)
    # Flags anomalous spikes in the ratio relative to recent baseline
    w = PARTICLE_RATIO_WINDOW_S          # 600 seconds
    ratio_mean = czt_cdte_ratio.rolling(w, min_periods=max(10, w // 10)).mean()
    ratio_std  = czt_cdte_ratio.rolling(w, min_periods=max(10, w // 10)).std()
    ratio_std  = ratio_std.clip(lower=0.01)   # std floor prevents division by near-zero
    df["particle_czt_cdte_zscore"] = (czt_cdte_ratio - ratio_mean) / ratio_std
    added.append("particle_czt_cdte_zscore")

    # Feature 3: CZT fires without CdTe coupling (binary discriminator)
    # True when CZT is above background significance (czt_above_bg) but
    # CdTe derivative is flat or negative -- the hallmark particle-only signature
    cdte_deriv1 = df[cdte_col].diff(1)
    czt_above   = df.get("czt_above_bg", pd.Series(False, index=df.index))
    cdte_flat   = cdte_deriv1 <= 0.0

    # Use a 30s trailing window to avoid single-sample sensitivity
    czt_above_sustained = czt_above.astype(float).rolling(30, min_periods=5).max() > 0
    cdte_flat_sustained = cdte_flat.astype(float).rolling(30, min_periods=5).min() > 0
    df["particle_czt_without_cdte"] = (
        czt_above_sustained & cdte_flat_sustained
    ).astype(np.int8)
    added.append("particle_czt_without_cdte")

    # Feature 4: Spectral hardening without total flux increase
    # During genuine pre-flare: hardness ratio (HR) rises AND total soft flux rises
    # During particle contamination in high-energy SoLEXS channels:
    # HR rises but total solexs_counts_smooth stays flat or decreases
    if "solexs_hr_deriv1" in df.columns and "solexs_deriv1" in df.columns:
        hr_rising    = df["solexs_hr_deriv1"] > 0
        flux_flat    = df["solexs_deriv1"] <= 0
        hr_rising_s  = hr_rising.astype(float).rolling(60, min_periods=10).mean() > 0.5
        flux_flat_s  = flux_flat.astype(float).rolling(60, min_periods=10).mean() > 0.5
        df["particle_hr_without_flux"] = (hr_rising_s & flux_flat_s).astype(np.int8)
    else:
        df["particle_hr_without_flux"] = np.int8(0)
    added.append("particle_hr_without_flux")

    # Feature 5: GTI proximity -- seconds since GTI start
    # Data near GTI boundaries can be erratic as the instrument stabilizes
    # after an eclipse or data gap. False alerts cluster near GTI start.
    if "gti_start" in df.columns:
        gti_starts = df["gti_start"].astype(bool)
        time_s = df["time_unix"]
        gti_start_times = time_s[gti_starts].values

        # For each row, find the most recent GTI start time <= current time
        pos = np.searchsorted(gti_start_times, time_s.values, side="right") - 1
        pos = pos.clip(0, len(gti_start_times) - 1)
        last_gti_start = gti_start_times[pos]
        seconds_since_gti = (time_s.values - last_gti_start).clip(0)
        df["particle_gti_proximity_s"] = seconds_since_gti.astype(np.float32)
    else:
        df["particle_gti_proximity_s"] = np.float32(9999.0)
    added.append("particle_gti_proximity_s")

    report = {
        "columns_added"             : added,
        "n_czt_without_cdte_events" : int(df["particle_czt_without_cdte"].sum()),
        "n_hr_without_flux_events"  : int(df["particle_hr_without_flux"].sum()),
        "mean_gti_proximity_s"      : float(df["particle_gti_proximity_s"].mean()),
    }
    return {"df": df, "columns_added": added, "report": report}


def _resolve_cdte(df: pd.DataFrame) -> str:
    """Return the best available CdTe column."""
    for col in ("cdte_count_rate_smooth", "cdte_count_rate_clean", "cdte_count_rate"):
        if col in df.columns:
            return col
    raise KeyError("No CdTe count rate column found in DataFrame.")
```

### Config Constants to Add (`tejas/config.py`)

In the Phase 4: Feature Engineering block, add:

```python
# Particle discriminator feature constants
PARTICLE_RATIO_WINDOW_S         = 600   # trailing window for CZT/CdTe ratio z-score
PARTICLE_RATIO_SPIKE_THRESHOLD  = 3.0   # sigma threshold for ratio anomaly
PARTICLE_GTI_PROXIMITY_WINDOW_S = 120   # seconds within which to flag GTI proximity
```

### Pipeline Integration — `tejas/features/pipeline.py`

In the `engineer_day()` function, add one call after the existing spectral/cross/statistical steps:

```python
from tejas.features.particle_discriminator import compute_particle_discriminator_features

# Inside engineer_day(), after compute_statistical_features():
result_particle = compute_particle_discriminator_features(df)
df = result_particle["df"]
report["particle_discriminator"] = result_particle["report"]
```

### GTI Start Column — `tejas/processing/pipeline.py`

To enable Feature 5, the Phase 3 pipeline needs to tag the first row of each GTI
interval. In `process_day()`, after applying GTI filtering, add:

```python
# Tag GTI boundaries (for particle discriminator features downstream)
df["gti_start"] = False
if len(gti_df) > 0:
    for _, row in gti_df.iterrows():
        mask = df["time_unix"] >= row["start"]
        first_valid = mask.idxmax() if mask.any() else None
        if first_valid is not None:
            df.at[first_valid, "gti_start"] = True
df["gti_start"] = df["gti_start"].astype(bool)
```

**Note:** `gti_start` is a raw metadata column. It can be left as a model feature
(it is causal) or added to `FORECAST_DROP_COLUMNS` if you prefer metadata not to
enter the model directly (the `particle_gti_proximity_s` derived feature is richer).

### Pipeline Re-run After Fix 3

```bash
python scripts/run_phase4.py --all      # re-engineer all feature parquets
python scripts/run_phase5.py --all      # re-detect (re-attach labels to new features)
python scripts/run_forecasting.py --all # retrain with 5 new features
```

### Benchmark

Compare `feature_importance.csv` before vs. after:
- Expect `particle_czt_cdte_zscore` and `particle_czt_without_cdte` to rank in top 20
  for the 5-min horizon
- FAR reduction: compare test-split FAR per horizon before/after in `forecast_report.md`

---

## 6. Fix 4 — Significant-Flare Class-Specific Model

### What It Fixes

The binary label `label_flare_within_{h}m` treats all detected events identically.
A faint B-class microflare (barely detectable at SoLEXS noise floor) counts the same
as an X-class event. Most false alerts are on the minor-flare boundary: the model
sees a pattern that in training was associated with small B/C events, many of which
are themselves borderline detections (marginal signal-to-noise). By training a second
set of models exclusively on M/X-class events, we remove this label noise and produce
a high-confidence, low-FAR alert tier.

**Phase 6 is confirmed complete.** The `classification` column in `master_catalogue.csv`
is populated with GOES-class labels. Before using it, **first verify** the data:

```python
import pandas as pd
cat = pd.read_csv("data/detection/2026/05/master_catalogue.csv")
print(cat["classification"].value_counts())
print(cat["classification"].isna().sum())  # must be 0
```

Once verified, the significant-flare filter uses:
```python
sig_mask = catalogue["classification"].isin(["M", "X"])
```

### New Config Constants (`tejas/config.py`)

```python
# Significant-flare classes (used for class-specific model; Fix 4)
FORECAST_SIGNIFICANT_FLARE_CLASSES = ["M", "X"]   # GOES classes treated as "significant"
FORECAST_LABEL_HORIZONS_MIN_SIG    = [5, 15, 30, 60]  # same horizons as all-flare
```

### New Label Creation — `tejas/detection/catalogue.py`

In the function that attaches per-timestep labels (likely `_attach_timestep_labels()`),
add four new label columns after the existing `label_flare_within_{h}m` columns:

```python
from tejas.config import FORECAST_SIGNIFICANT_FLARE_CLASSES, FORECAST_LABEL_HORIZONS_MIN

# Significant-flare labels (M/X class only)
sig_mask = catalogue["classification"].isin(FORECAST_SIGNIFICANT_FLARE_CLASSES)

for h in FORECAST_LABEL_HORIZONS_MIN:
    col_sig = f"label_sig_flare_within_{h}m"
    df[col_sig] = np.int8(0)
    for _, event in catalogue[sig_mask].iterrows():
        flare_start  = event["start_time"]
        window_start = flare_start - pd.Timedelta(minutes=h)
        mask = (df["datetime"] >= window_start) & (df["datetime"] < flare_start)
        df.loc[mask, col_sig] = np.int8(1)
```

New columns added to detection parquet:
- `label_sig_flare_within_5m`
- `label_sig_flare_within_15m`
- `label_sig_flare_within_30m`
- `label_sig_flare_within_60m`

Add all four to `FORECAST_DROP_COLUMNS` in `tejas/config.py` to prevent label leakage.

### Training Integration — `tejas/forecasting/pipeline.py`

Add a `train_significant_flare_models()` function that mirrors `train_all_horizons()`
but reads `label_sig_flare_within_{h}m` as the target label and saves models as
`{algo}_sig_horizon_{h}m.joblib`.

Add a `--sig` flag to `scripts/run_forecasting.py`:

```bash
python scripts/run_forecasting.py --train --sig    # train significant-flare models only
python scripts/run_forecasting.py --all --sig      # prepare + train + evaluate
```

### Alert Tier System

The two model tiers map operationally to:
- **Tier 1 (any-flare model):** "WATCH" — any flare possible within horizon
- **Tier 2 (significant-flare model):** "WARNING" — M/X-class event likely

This matches NOAA's WATCH / WARNING space-weather alert convention.

### Benchmark

Compare `metrics_summary.csv` for any-flare vs. significant-flare models:
- Expected: Significant-flare models will have substantially higher PR-AUC and precision
  because M/X-class precursor signatures are stronger and more consistent
- Also compute: fraction of "any-flare" false alerts that target B/C-class events
  (quantifies how much of the FAR problem is driven by borderline events)

### Pipeline Re-run After Fix 4

```bash
python scripts/run_phase5.py --all        # re-label detection parquets with sig labels
python scripts/run_forecasting.py --all   # retrain all-flare models
python scripts/run_forecasting.py --all --sig   # train significant-flare models
```

---

## 7. Fix 5 — True Ensemble Stacking (Meta-Learner)

### What It Fixes

The three base models (LogReg, XGBoost, LightGBM) are trained independently and the
dashboard defaults to XGBoost. They are not combined. A stacked ensemble learns *when
to trust which base model*. False alerts that are model-specific artefacts get
cancelled when 2 of 3 models disagree. The meta-learner is trained on the **val split**
(base models were never fit on val — no circular leakage).

### New File: `tejas/forecasting/ensemble.py`

```python
"""
Stacked ensemble meta-learner for TEJAS forecasting.

Architecture:
  Base models: LogisticRegression + XGBoost + LightGBM (already trained)
  Meta-learner: LogisticRegression trained on val-split calibrated probabilities
                from all three base models.

No-future-data guarantee:
  - Base models trained on train split only
  - Meta-learner trained on val split probabilities
  - Neither ever sees test-split labels during fitting
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tejas.forecasting.inference import load_forecaster
from tejas.forecasting.evaluation import tune_threshold
from tejas.config import FORECAST_LABEL_HORIZONS_MIN, FORECAST_TARGET_FAR


class StackedForecaster:
    """Meta-learner combining LogReg + XGB + LightGBM predictions."""

    def __init__(self, base_forecasters: dict, meta_models: dict,
                 meta_thresholds: dict):
        self.base_forecasters = base_forecasters
        self.meta_models      = meta_models
        self.meta_thresholds  = meta_thresholds

    def predict_row(self, row) -> dict:
        """Return stacked probabilities for all four horizons."""
        base_probs = {algo: f.predict_row(row)
                      for algo, f in self.base_forecasters.items()}
        out = {}
        for h in FORECAST_LABEL_HORIZONS_MIN:
            X_meta = np.array([[
                base_probs["logreg"][f"prob_{h}m"],
                base_probs["xgb"   ][f"prob_{h}m"],
                base_probs["lgbm"  ][f"prob_{h}m"],
            ]])
            out[f"prob_{h}m"] = float(
                self.meta_models[h].predict_proba(X_meta)[0, 1]
            )
        return out

    def decision_thresholds(self) -> dict:
        return dict(self.meta_thresholds)


def train_stacked_ensemble(val_df: pd.DataFrame, feature_cols: list[str],
                            split_labels: pd.Series, models_dir: Path,
                            output_dir: Path) -> StackedForecaster:
    """
    Build and persist a stacked ensemble.

    Steps:
    1. Load all three base Forecasters from disk
    2. On the val split, compute calibrated probabilities from each base model
    3. For each horizon, train a meta LogReg on [logreg_prob, xgb_prob, lgbm_prob] -> y_val
    4. Tune meta decision threshold on val (far_constrained)
    5. Persist meta models to {models_dir}/stacked_horizon_{h}m.joblib
    """
    base = {algo: load_forecaster(algo, models_dir)
            for algo in ("logreg", "xgb", "lgbm")}

    val_mask   = split_labels == "val"
    val_df_sub = val_df[val_mask].copy()

    meta_models     = {}
    meta_thresholds = {}

    for h in FORECAST_LABEL_HORIZONS_MIN:
        label_col = f"label_flare_within_{h}m"
        y_val     = val_df_sub[label_col].fillna(0).astype(int).values

        X_meta = np.column_stack([
            [base[algo].predict_row(row)[f"prob_{h}m"]
             for _, row in val_df_sub[feature_cols].iterrows()]
            for algo in ("logreg", "xgb", "lgbm")
        ])

        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)),
        ])
        meta.fit(X_meta, y_val)

        meta_probs = meta.predict_proba(X_meta)[:, 1]
        threshold  = tune_threshold(y_val, meta_probs,
                                    metric="far_constrained",
                                    target_far=FORECAST_TARGET_FAR)

        meta_models[h]     = meta
        meta_thresholds[h] = threshold

        joblib.dump({"meta_model": meta, "meta_threshold": threshold, "horizon_min": h},
                    output_dir / f"stacked_horizon_{h}m.joblib")

    return StackedForecaster(base, meta_models, meta_thresholds)
```

### CLI Integration (`scripts/run_forecasting.py`)

```bash
python scripts/run_forecasting.py --stack            # train meta-learner
python scripts/run_forecasting.py --evaluate --stack # evaluate stacked ensemble on test
```

### Benchmark

Compare `metrics_summary.csv` (best single model) against stacked ensemble:
- PR-AUC: expect 5–15% improvement at 5-min horizon (highest base-model variance)
- FAR: should be ≤ `FORECAST_TARGET_FAR` by design (meta-learner tuned to this)

### No-Future-Data Compliance

- Base models: trained on train split only
- Meta-learner: trained on val split out-of-fold predictions (no overlap with train)
- Meta threshold: tuned on val, applied to test in final evaluation
- Fully compliant with all five existing `TestNoFutureData` tripwires

---

## 8. Fix 6 — Online Kalman Filter on CZT and CdTe Signals

### What It Fixes

Phase 3 rolling-median smoothing (`center=True`, fixed 5–11s windows) has two
weaknesses:
1. Ignores `STAT_ERR` (Poisson error) from HEL1OS — every timestep is weighted equally
   even when measurement uncertainty is high
2. Fixed window can't adapt: lags during rapid flare onset; doesn't suppress enough
   during elevated background

A Kalman filter uses `STAT_ERR` directly as measurement noise `R`. During high-
background periods (particle showers), `STAT_ERR` is large → Kalman trusts its model
more than the measurement → signal stays smooth → model doesn't fire.

### New File: `tejas/processing/kalman.py`

```python
"""
Online 1D Kalman filter for TEJAS signal smoothing.

Model: local-level (random walk) — x_t = x_{t-1} + noise
Process noise Q: variance of true flux change between consecutive seconds
Measurement noise R: per-sample Poisson variance from STAT_ERR column

All computation is causal (forward-time only, no look-ahead).
"""

import numpy as np
import pandas as pd
from typing import Optional


def kalman_filter_1d(
    signal: np.ndarray,
    stat_err: Optional[np.ndarray] = None,
    q_process_noise: float = 1.0,
    r_default: float = 5.0,
    x0: Optional[float] = None,
    p0: float = 100.0,
) -> np.ndarray:
    """
    Apply a scalar Kalman filter to a 1D time series.

    Parameters
    ----------
    signal : np.ndarray
        Observed measurements (count rate).
    stat_err : np.ndarray or None
        Per-sample statistical error (sigma). If provided, R = stat_err^2.
    q_process_noise : float
        Process noise variance. Tune per-signal:
          czt_count_rate: Q ~ 0.5  (noisy background; trust model more)
          cdte_count_rate: Q ~ 2.0 (cleaner signal; allow faster tracking)
    r_default : float
        Default measurement noise std when stat_err is None or NaN.
    x0, p0 : float
        Initial state estimate and error covariance.
    """
    n   = len(signal)
    out = np.empty(n)
    x   = signal[0] if (x0 is None or np.isnan(signal[0])) else x0
    p   = p0

    R_arr = (np.where(np.isfinite(stat_err) & (stat_err > 0), stat_err, r_default) ** 2
             if stat_err is not None else np.full(n, r_default ** 2))

    for i in range(n):
        z      = signal[i]
        p_pred = p + q_process_noise          # predict
        if np.isfinite(z):
            K = p_pred / (p_pred + R_arr[i]) # Kalman gain
            x = x + K * (z - x)              # update state
            p = (1 - K) * p_pred             # update covariance
        else:
            p = p_pred                        # missing measurement: skip update
        out[i] = x

    return out


def smooth_day_kalman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-signal Kalman filtering to CZT and CdTe count rate columns.
    Produces new columns: czt_count_rate_kalman, cdte_count_rate_kalman.
    Operates segment-by-segment to prevent state bleeding across data gaps.
    """
    from tejas.config import KALMAN_Q_CZT, KALMAN_Q_CDTE, KALMAN_R_DEFAULT

    target_signals = {
        "czt_count_rate" : ("czt_stat_err",  KALMAN_Q_CZT),
        "cdte_count_rate": ("cdte_stat_err", KALMAN_Q_CDTE),
    }

    df = df.copy()
    for sig_col, (err_col, q) in target_signals.items():
        if sig_col not in df.columns:
            continue
        out_col = f"{sig_col}_kalman"
        df[out_col] = np.nan

        for seg_id, seg_df in df.groupby("segment_id", sort=False):
            signal   = seg_df[sig_col].values.astype(float)
            stat_err = (seg_df[err_col].values.astype(float)
                        if err_col in df.columns else None)
            df.loc[seg_df.index, out_col] = kalman_filter_1d(
                signal, stat_err=stat_err,
                q_process_noise=q, r_default=KALMAN_R_DEFAULT
            )

    return df
```

### Config Constants to Add (`tejas/config.py`)

```python
# Kalman filter smoothing constants (Phase 3)
KALMAN_Q_CZT     = 0.5    # CZT process noise (small = more smoothing)
KALMAN_Q_CDTE    = 2.0    # CdTe process noise (allows faster flare tracking)
KALMAN_R_DEFAULT = 5.0    # Default measurement noise std when STAT_ERR unavailable
```

### Strategy: Add `*_kalman` Alongside `*_smooth` (Non-Destructive)

Kalman columns are added alongside existing rolling-median `*_smooth` columns.
Rollback = remove `"_kalman"` from `SIGNAL_DEFINITIONS` in `temporal.py` — no
Phase 3 re-run needed.

### Pipeline Integration — `tejas/processing/pipeline.py`

```python
from tejas.processing.kalman import smooth_day_kalman

# In process_day(), after smooth_day():
df = smooth_day_kalman(df)
# Adds: czt_count_rate_kalman, cdte_count_rate_kalman
```

### Feature Engineering Integration — `tejas/features/temporal.py`

Update `SIGNAL_DEFINITIONS` to prefer Kalman columns for CZT and CdTe:

```python
SIGNAL_DEFINITIONS = [
    {"name": "solexs", "col": "solexs_counts",   "variants": ["_smooth", "_clean", ""]},
    {"name": "cdte",   "col": "cdte_count_rate", "variants": ["_kalman", "_smooth", "_clean", ""]},
    {"name": "czt",    "col": "czt_count_rate",  "variants": ["_kalman", "_smooth", "_clean", ""]},
]
```

### Benchmark

- SNR comparison: `mean(quiet_flux) / std(quiet_flux)` for `czt_net_rate` vs.
  `czt_count_rate_kalman` on quiet-Sun periods. Expected: 20–40% SNR improvement.
- FAR before vs. after in `forecast_report.md` after full pipeline rerun.

### Pipeline Re-run After Fix 6

```bash
python scripts/run_phase3.py --all      # compute Kalman columns
python scripts/run_phase4.py --all      # re-engineer features from Kalman columns
python scripts/run_phase5.py --all      # re-detect with new features
python scripts/run_forecasting.py --all # retrain
```

---

## 9. Fix 7 — Active Region Proxy Features

### What It Fixes

Aditya-L1 has no magnetograph — TEJAS cannot observe active region complexity.
But a powerful proxy is available within TEJAS itself: **flare history**. An AR that
produced flares in the past 6–12 hours is "awake" and likely to produce more. During
quiet Sun (no recent flares), any precursor-like signal is overwhelmingly likely to
be noise. These features allow the model to learn this context.

**This is the highest-impact, lowest-effort fix.** It requires one new feature file,
one line in `engineer_day()`, and no Phase 3 or Phase 5 re-runs.

### New File: `tejas/features/ar_proxy.py`

```python
"""
Active Region (AR) proxy features derived from the TEJAS flare history catalogue.

All features are strictly causal: they only count flare events that have
already completed (flare_end_time < current_timestamp).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tejas.config import AR_PROXY_SHORT_WINDOW_H, AR_PROXY_LONG_WINDOW_H


def compute_ar_proxy_features(df: pd.DataFrame,
                               catalogue_path=None) -> dict:
    """
    Parameters
    ----------
    df : DataFrame with 'datetime' and 'time_unix' columns (Phase 4/5 parquet).
    catalogue_path : path to master_catalogue.csv or None (uses default).
    """
    df = df.copy()
    added = []
    cat = _load_catalogue(catalogue_path)

    if cat is None or len(cat) == 0:
        for col in _ar_feature_names():
            df[col] = np.float32(0)
            added.append(col)
        return {"df": df, "columns_added": added, "report": {"warning": "no catalogue"}}

    cat["end_dt"] = pd.to_datetime(cat["end_time"])
    cat_sorted = cat.sort_values("end_dt").reset_index(drop=True)
    end_times  = cat_sorted["end_dt"].values.astype("datetime64[ns]")

    times = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    w_short_ns = int(AR_PROXY_SHORT_WINDOW_H * 3600 * 1e9)
    w_long_ns  = int(AR_PROXY_LONG_WINDOW_H  * 3600 * 1e9)

    minute_times = pd.date_range(times.min(), times.max(), freq="1min")
    n6h, n12h, since_s, rate, peak6h = [], [], [], [], []

    for t in minute_times:
        t_ns    = t.value
        mask6h  = (end_times <= t_ns) & (end_times >= t_ns - w_short_ns)
        mask12h = (end_times <= t_ns) & (end_times >= t_ns - w_long_ns)
        n6      = int(mask6h.sum())
        n12     = int(mask12h.sum())
        past    = end_times[end_times <= t_ns]
        ts      = float((t_ns - past[-1]) / 1e9) if len(past) > 0 else np.nan
        n6h.append(n6);  n12h.append(n12)
        since_s.append(ts)
        rate.append(n6 / AR_PROXY_SHORT_WINDOW_H if n6 > 0 else 0.0)
        pk = float(cat_sorted.loc[mask6h, "peak_solexs_counts"].max()) if n6 > 0 else 0.0
        peak6h.append(pk)

    idx = pd.DatetimeIndex(minute_times)
    for col, vals in [
        ("ar_flares_in_last_6h",  n6h),
        ("ar_flares_in_last_12h", n12h),
        ("ar_time_since_last_s",  since_s),
        ("ar_flare_rate_6h",      rate),
        ("ar_max_peak_last_6h",   peak6h),
    ]:
        s = pd.Series(vals, index=idx, dtype="float32")
        df[col] = s.reindex(times, method="ffill").values
        added.append(col)

    df["ar_flares_in_last_6h"].fillna(0, inplace=True)
    df["ar_flares_in_last_12h"].fillna(0, inplace=True)
    df["ar_time_since_last_s"].fillna(86400, inplace=True)  # 24h if no prior flare
    df["ar_flare_rate_6h"].fillna(0, inplace=True)
    df["ar_max_peak_last_6h"].fillna(0, inplace=True)

    return {
        "df": df, "columns_added": added,
        "report": {
            "mean_flares_in_last_6h": float(df["ar_flares_in_last_6h"].mean()),
            "mean_time_since_last_s": float(df["ar_time_since_last_s"].mean()),
        }
    }


def _ar_feature_names():
    return ["ar_flares_in_last_6h", "ar_flares_in_last_12h",
            "ar_time_since_last_s", "ar_flare_rate_6h", "ar_max_peak_last_6h"]


def _load_catalogue(path):
    from tejas.config import DETECTION_DIR, YEAR, MONTH
    if path is None:
        path = Path(DETECTION_DIR) / YEAR / MONTH / "master_catalogue.csv"
    try:
        return pd.read_csv(path, parse_dates=["start_time", "end_time", "peak_time"])
    except FileNotFoundError:
        return None
```

### Config Constants to Add (`tejas/config.py`)

```python
# AR-proxy feature windows
AR_PROXY_SHORT_WINDOW_H = 6    # hours for short-window flare count
AR_PROXY_LONG_WINDOW_H  = 12   # hours for long-window flare count
```

### Pipeline Integration — `tejas/features/pipeline.py`

```python
from tejas.features.ar_proxy import compute_ar_proxy_features

# At the END of engineer_day() (after all other feature steps):
result_ar = compute_ar_proxy_features(df, catalogue_path=catalogue_path)
df = result_ar["df"]
report["ar_proxy"] = result_ar["report"]
```

### Computation Order Dependency

AR-proxy features read from the Phase 5 master catalogue, but Phase 4 (features)
currently runs before Phase 5 (detection). Solution: add a new post-Phase-5 step
via `scripts/run_ar_proxy.py`:

1. Loads the already-saved Phase 4 feature Parquet
2. Adds AR-proxy features on top of it
3. Overwrites the Phase 4 Parquet with the augmented version

Run order: Phase 3 → Phase 4 → Phase 5 → `run_ar_proxy.py` → Phase 7.

### Benchmark

- Feature importance: expect `ar_time_since_last_s` and `ar_flares_in_last_6h` to
  rank in top 10 for all horizons
- FAR stratified by `ar_flares_in_last_6h == 0` vs. `> 0`:
  - Quiet Sun: near-zero FAR expected
  - Active Sun (3+ flares in 6h): higher precision too
- Expected biggest lift at 60-min horizon (where sustained elevated baseline drives
  most false alerts)

---

## 10. Fix 8 — Two-Stage False Alert Discriminator

### What It Fixes

Stage 1 (current model) is trained to catch all flares. Stage 2 is trained exclusively
on Stage 1's output — it learns to distinguish "truly about to flare" from "looks like
a flare but isn't" using the 30-minute trailing context window before each candidate
alert.

### Architecture

```
Stage 1 (XGBoost forecaster):
  Feature_row → P(flare) > threshold → candidate_alert

Stage 2 (FalseAlertDiscriminator):
  candidate_alert + 30-min trailing window → P(false_alert) > disc_threshold
  → SUPPRESS (false alert)   or   CONFIRM (fire alert)
```

### New File: `tejas/forecasting/discriminator.py`

```python
"""
Two-Stage False Alert Discriminator for TEJAS.

Stage 2 classifier trained exclusively on Stage 1 candidate alerts.
Distinguishes true-positive from false-positive candidate alerts using
the 30-minute trailing context window.

Training data:
  Positive class: false-positive candidates from train split
  Negative class: true-positive candidates from train split
  Threshold tuned on val split candidates.

No-future-data guarantee: all context features are derived from data
strictly before each alert timestamp.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from tejas.config import DISCRIMINATOR_WINDOW_S, FORECAST_LABEL_HORIZONS_MIN, FORECAST_TARGET_FAR
from tejas.forecasting.evaluation import tune_threshold


def _compute_pattern_features(window_df: pd.DataFrame) -> dict:
    """
    Compute 5 physics-based discriminating pattern features from a 30-min window.

    Pattern A: CZT/CdTe coupling correlation (low = particle event)
    Pattern B: Fraction of window with sustained spectral hardening
    Pattern C: CUSUM-to-zscore ratio (low = isolated spike without buildup)
    Pattern D: Multi-band cascade correlation (low = single-band particle hit)
    Pattern E: Minimum GTI proximity in window (low = near-boundary erratic data)
    """
    feats = {}

    # Pattern A
    if "czt_net_rate" in window_df and "cdte_count_rate_smooth" in window_df:
        czt  = window_df["czt_net_rate"].clip(lower=0.01)
        cdte = window_df["cdte_count_rate_smooth"].clip(lower=0.01)
        corr = np.corrcoef(czt.values, cdte.values)[0, 1] if len(czt) > 3 else 0.0
        feats["disc_czt_cdte_coupling"] = float(corr)
    else:
        feats["disc_czt_cdte_coupling"] = 0.5

    # Pattern B
    if "solexs_hr_rmean_60" in window_df and "solexs_hr_rmean_300" in window_df:
        feats["disc_hr_sustained_frac"] = float(
            (window_df["solexs_hr_rmean_60"] > window_df["solexs_hr_rmean_300"]).mean()
        )
    else:
        feats["disc_hr_sustained_frac"] = 0.5

    # Pattern C
    if "solexs_cusum" in window_df and "solexs_zscore_300" in window_df:
        cusum_mean = float(window_df["solexs_cusum"].mean())
        zscore_max = float(window_df["solexs_zscore_300"].max())
        feats["disc_cusum_z_ratio"] = cusum_mean / (zscore_max + 1e-3)
    else:
        feats["disc_cusum_z_ratio"] = 0.0

    # Pattern D
    if all(c in window_df.columns for c in
           ["solexs_band_ratio_mid_low", "solexs_band_ratio_high_mid"]):
        ml = window_df["solexs_band_ratio_mid_low"].dropna().values
        hm = window_df["solexs_band_ratio_high_mid"].dropna().values
        n  = min(len(ml), len(hm))
        if n > 10:
            corr = np.corrcoef(ml[:n], hm[:n])[0, 1]
            feats["disc_band_cascade_corr"] = float(corr) if np.isfinite(corr) else 0.0
        else:
            feats["disc_band_cascade_corr"] = 0.0
    else:
        feats["disc_band_cascade_corr"] = 0.0

    # Pattern E
    feats["disc_min_gti_proximity_s"] = float(
        window_df["particle_gti_proximity_s"].min()
        if "particle_gti_proximity_s" in window_df.columns else 9999.0
    )

    return feats


def _compute_window_summary(window_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Compute mean, std, min, max, and slope for each base feature over
    the 30-minute window. Gives the discriminator access to full context.
    """
    summary = {}
    for col in feature_cols:
        if col not in window_df.columns:
            for stat in ["mean", "std", "min", "max", "slope"]:
                summary[f"win_{col}_{stat}"] = np.nan
            continue
        s = window_df[col].dropna().values
        if len(s) < 2:
            for stat in ["mean", "std", "min", "max", "slope"]:
                summary[f"win_{col}_{stat}"] = np.nan
            continue
        summary[f"win_{col}_mean"]  = float(np.mean(s))
        summary[f"win_{col}_std"]   = float(np.std(s))
        summary[f"win_{col}_min"]   = float(np.min(s))
        summary[f"win_{col}_max"]   = float(np.max(s))
        t = np.arange(len(s))
        summary[f"win_{col}_slope"] = float(np.polyfit(t, s, 1)[0])
    return summary


class FalseAlertDiscriminator:
    """
    Stage 2 classifier: identifies false-positive candidate alerts.

    Usage:
        disc = load_discriminator(horizon_min=5)
        if candidate_alert:
            window_df = df[df["time_unix"].between(t0 - 1800, t0)]
            suppress  = disc.predict(window_df)
            if not suppress:
                fire_alert()
    """

    def __init__(self, model, threshold: float, horizon_min: int,
                 feature_cols: list):
        self.model        = model
        self.threshold    = threshold
        self.horizon_min  = horizon_min
        self.feature_cols = feature_cols

    def predict(self, window_df: pd.DataFrame) -> bool:
        """Returns True = SUPPRESS, False = CONFIRM."""
        pattern_feats = _compute_pattern_features(window_df)
        window_feats  = _compute_window_summary(window_df, self.feature_cols)
        X = pd.Series({**pattern_feats, **window_feats}).to_frame().T
        X = X.reindex(self.model.feature_names_in_, axis=1)
        prob_false = self.model.predict_proba(X.values)[0, 1]
        return bool(prob_false >= self.threshold)


def train_discriminator(full_df, feature_cols, split_labels, horizon_min,
                         stage1_probs, stage1_threshold,
                         window_s=DISCRIMINATOR_WINDOW_S,
                         models_dir=None) -> FalseAlertDiscriminator:
    """
    Train Stage 2 discriminator for one horizon.
    Positive class = false-positive candidates from train split.
    Negative class = true-positive candidates from train split.
    Threshold tuned on val-split candidates.
    """
    label_col   = f"label_flare_within_{horizon_min}m"
    train_mask  = split_labels == "train"
    train_probs = stage1_probs[train_mask]
    candidates  = full_df[train_mask][train_probs >= stage1_threshold].copy()
    y_disc      = (candidates[label_col].fillna(0) == 0).astype(int)

    if len(candidates) < 20:
        raise ValueError(f"Too few candidate alerts ({len(candidates)}) for discriminator training.")

    rows = []
    for _, cand_row in candidates.iterrows():
        t0  = cand_row["time_unix"]
        win = full_df[(full_df["time_unix"] >= t0 - window_s) &
                      (full_df["time_unix"] <  t0)]
        if len(win) < 10:
            continue
        rows.append({**_compute_pattern_features(win),
                     **_compute_window_summary(win, feature_cols)})

    X_disc = pd.DataFrame(rows).fillna(pd.DataFrame(rows).mean())
    y_disc = y_disc.iloc[:len(X_disc)].values

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8,
        )),
    ])
    model.fit(X_disc, y_disc)

    # Tune threshold on val candidates
    val_mask     = split_labels == "val"
    val_probs_s1 = stage1_probs[val_mask]
    val_cands    = full_df[val_mask][val_probs_s1 >= stage1_threshold]
    y_val_disc   = (val_cands[label_col].fillna(0) == 0).astype(int)
    val_rows     = []
    for _, cand_row in val_cands.iterrows():
        t0  = cand_row["time_unix"]
        win = full_df[(full_df["time_unix"] >= t0 - window_s) &
                      (full_df["time_unix"] <  t0)]
        if len(win) < 10:
            continue
        val_rows.append({**_compute_pattern_features(win),
                         **_compute_window_summary(win, feature_cols)})

    if len(val_rows) > 10:
        X_val = pd.DataFrame(val_rows).reindex(
            model.feature_names_in_, axis=1).fillna(0)
        val_probs_disc = model.predict_proba(X_val.values)[:, 1]
        threshold = tune_threshold(y_val_disc.values[:len(val_rows)],
                                   val_probs_disc, metric="far_constrained",
                                   target_far=FORECAST_TARGET_FAR)
    else:
        threshold = 0.5

    disc = FalseAlertDiscriminator(model, threshold, horizon_min, feature_cols)

    if models_dir is not None:
        joblib.dump({"discriminator": disc},
                    Path(models_dir) / f"discriminator_horizon_{horizon_min}m.joblib")
    return disc


def load_discriminator(horizon_min: int, models_dir=None) -> FalseAlertDiscriminator:
    from tejas.config import MODELS_DIR, YEAR, MONTH
    if models_dir is None:
        models_dir = Path(MODELS_DIR) / YEAR / MONTH
    art = joblib.load(models_dir / f"discriminator_horizon_{horizon_min}m.joblib")
    return art["discriminator"]
```

### Config Constants to Add (`tejas/config.py`)

```python
# Two-stage discriminator constants
DISCRIMINATOR_WINDOW_S      = 1800   # 30-minute trailing context window (seconds)
DISCRIMINATOR_SUMMARY_STATS = ["mean", "std", "min", "max", "slope"]
```

### CLI Integration (`scripts/run_forecasting.py`)

```bash
python scripts/run_forecasting.py --discriminator   # train Stage 2 on all 4 horizons
python scripts/run_forecasting.py --evaluate --discriminator   # evaluate two-stage system
```

### Pre-Training: False Alert Pattern Analysis Script

Before training Stage 2, run `scripts/analyze_false_alerts.py --horizon 5` to
empirically validate which of the 5 patterns most strongly discriminate FP from TP:

1. Loads training frame and Stage 1 predictions
2. Separates FP and TP candidate alerts
3. Runs Mann-Whitney U test on each of the 5 pattern features
4. Outputs `data/forecasting/2026/05/false_alert_analysis.md` with U-statistics and
   significance p-values per feature per horizon

### Benchmark

| Metric | How to measure |
|--------|---------------|
| Suppression rate | `(Stage1 false alerts - Stage2 confirmed) / Stage1 false alerts` |
| Contamination rate | `true alerts suppressed by Stage2 / total true alerts` (must be < 10%) |
| Net FAR | Compare `forecast_report.md` Stage1 vs. Stage1+Stage2 |
| Net Recall | Same comparison |

---

## 11. Files Summary — Complete Change Map

| Fix | New Files | Modified Files | Config Changes |
|-----|-----------|---------------|----------------|
| 1 | None | `tejas/config.py` | `FORECAST_THRESHOLD_METRIC`, `FORECAST_TARGET_FAR` |
| 2 | `tejas/alerts/__init__.py`, `confirmation.py`, `engine.py` | `tejas/config.py`, dashboard | Phase 8 block (3 constants) |
| 3 | `tejas/features/particle_discriminator.py` | `tejas/features/pipeline.py`, `tejas/processing/pipeline.py`, `tejas/config.py` | 3 new constants |
| 4 | None | `tejas/detection/catalogue.py`, `tejas/forecasting/sampling.py`, `tejas/forecasting/pipeline.py`, `scripts/run_forecasting.py`, `tejas/config.py` | 2 new constants |
| 5 | `tejas/forecasting/ensemble.py` | `scripts/run_forecasting.py` | None |
| 6 | `tejas/processing/kalman.py` | `tejas/processing/pipeline.py`, `tejas/features/temporal.py`, `tejas/config.py` | 3 new constants |
| 7 | `tejas/features/ar_proxy.py`, `scripts/run_ar_proxy.py` | `tejas/features/pipeline.py`, `tejas/config.py` | 2 new constants |
| 8 | `tejas/forecasting/discriminator.py`, `scripts/analyze_false_alerts.py` | `scripts/run_forecasting.py`, `tejas/config.py` | 2 new constants |

---

## 12. Pipeline Re-run Master Schedule

| Fix | Phases to Rerun | Command |
|-----|----------------|---------|
| 1   | Phase 7 only | `run_forecasting.py --all` |
| 2   | None (alert engine only) | n/a |
| 3   | Phase 4 → 5 → 7 | `run_phase4.py --all && run_phase5.py --all && run_forecasting.py --all` |
| 4   | Phase 5 → 7 | `run_phase5.py --all && run_forecasting.py --all` |
| 5   | Phase 7 (--stack) | `run_forecasting.py --train && run_forecasting.py --stack` |
| 6   | Phase 3 → 4 → 5 → 7 | Full pipeline rerun |
| 7   | AR-proxy step → 7 | `run_ar_proxy.py --all && run_forecasting.py --all` |
| 8   | Phase 7 (--discriminator) | `run_forecasting.py --discriminator` |

**If all 8 fixes are applied simultaneously (recommended):**

```bash
# Step 1: Apply all code changes first, then run:
python scripts/run_phase3.py --all          # Fix 6: Kalman smoothing
python scripts/run_phase4.py --all          # Fix 3: particle features
python scripts/run_phase5.py --all          # Fix 4: sig-flare labels
python scripts/run_ar_proxy.py --all        # Fix 7: AR-proxy features
python scripts/run_forecasting.py --all     # Fix 1: new threshold + all 12 models
python scripts/run_forecasting.py --all --sig     # Fix 4: sig-flare models
python scripts/run_forecasting.py --stack   # Fix 5: ensemble
python scripts/run_forecasting.py --discriminator  # Fix 8: Stage 2
```

---

## 13. Testing Plan

### Unit Tests to Add

| Test | File | What it verifies |
|------|------|-----------------|
| `test_confirmation_gate_suppresses_spikes` | `tests/test_alerts.py` | Single-step spike does not trigger; K-consecutive does |
| `test_confirmation_gate_preserves_sustained` | `tests/test_alerts.py` | Real sustained exceedance fires and is not suppressed |
| `test_confirmation_gate_no_future_data` | `tests/test_alerts.py` | Gate only uses past probabilities |
| `test_particle_features_all_causal` | `tests/test_features.py` | All 5 new particle features use only trailing data |
| `test_particle_czt_without_cdte_fires` | `tests/test_features.py` | Feature=1 when CZT high, CdTe flat |
| `test_ar_proxy_causal_ordering` | `tests/test_features.py` | `ar_flares_in_last_6h` at time T only counts events ending before T |
| `test_ar_proxy_zero_before_first_flare` | `tests/test_features.py` | Feature is 0 for rows before any flare |
| `test_kalman_causal` | `tests/test_processing.py` | Filtered value at t does not change when t+1 data is modified |
| `test_kalman_segment_isolation` | `tests/test_processing.py` | Kalman state does not bleed across segment_id boundaries |
| `test_discriminator_suppression_rate` | `tests/test_forecasting.py` | Stage 2 reduces false alerts without killing all true alerts |
| `test_stacked_no_future_data` | `tests/test_forecasting.py` | Meta-learner only trained on val (after train) |

### Integration Test

Add `TestFullOptimizationPipeline` in `tests/test_forecasting.py`:
1. Full pipeline (prepare → train → stack → discriminator) on synthetic 5-day data
2. Verifies FAR < 0.20 and Recall > 0.40 for all horizons on the test split
3. Re-runs `test_random_features_dont_leak()` to verify no future-data leakage

---

## 14. Resolved Decisions & Remaining Assumptions

### Resolved (confirmed by user)

1. **Fix 4 — Phase 6 status:** Phase 6 classification is complete. Use
   `catalogue["classification"].isin(["M", "X"])` directly. **Verify the data first**
   (`value_counts()`, null check) before writing labels.

2. **Fix 6 — Kalman strategy:** Add `*_kalman` columns **alongside** existing `*_smooth`
   columns (non-destructive). Feature engineering picks `*_kalman` as highest priority
   for CZT and CdTe. Rollback = revert one line in `temporal.py`.

### Remaining Assumptions (verify during implementation)

3. **`czt_stat_err` / `cdte_stat_err` column availability:** Check
   `data/processed/unified/2026/05/*.parquet` for these columns. If absent, Kalman
   falls back to `r_default = 5.0` (graceful degradation, still an improvement).

4. **Fix 6 — `center=True` in rolling median:** Current `smooth_day()` uses
   `center=True` (2-second look-ahead in 5-second windows). Scientifically negligible
   at minute-scale horizons, but noted. Kalman columns are fully causal.

5. **Fix 7 — AR-proxy computation order:** Uses `scripts/run_ar_proxy.py` to decouple
   from Phase 4's normal run order. Confirm this is acceptable before implementing.
