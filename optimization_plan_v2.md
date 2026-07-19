# TEJAS — Precision + Recall Optimization Plan (v2)

> **Goal:** Increase BOTH precision and recall of flare alarms, with flares
> confirmed against a trusted external public reference (NOAA GOES).
> Predecessor (`optimization_plan.md`) reduced FAR by threshold-tightening,
> which traded recall away. This plan attacks the actual bottleneck.

## 0. Diagnosis (why precision AND recall are both low)

Measured on the current test split (`test_metrics.csv`):

| Horizon | ROC-AUC | PR-AUC | Base rate | Verdict |
|---|---|---|---|---|
| 5 min | 0.603 | 0.232 | 0.172 | weak signal |
| 15 min | 0.543 | 0.457 | 0.425 | barely above random |
| 30 min | 0.516 | 0.664 | 0.671 | AT random |
| 60 min | 0.588 | 0.915 | 0.877 | base-rate artifact |

ROC-AUC ≈ 0.5 everywhere: the models barely discriminate. No threshold,
ensemble, or discriminator tuning can fix that. Root cause analysis:

1. **Label pollution.** `label_flare_within_{h}m` is generated from ALL
   1,751 catalogue events (`tejas/detection/pipeline.py:377-412`):
   1,136 hard-only events (mostly CZT/CdTe noise blips, NULL
   classification) and 364 "faint" events (peak < 25 c/s, at the SoLEXS
   noise floor). The model is being trained to predict noise, and
   evaluated on predicting noise. ~86% of "flares" in the labels are
   events no operator would ever alert on.
2. **Self-referential ground truth.** Labels come from TEJAS's own Phase-5
   detector on the same data. No external validation has ever run:
   `data/reference/goes/2026/05/` is empty, `goes_class` populated for
   0/615 events.
3. **Row-level scoring punishes timing.** A model that fires 7 min before
   a flare scores as FP under the 5-min label even though the alert was
   operationally correct. Event-level metrics don't exist yet.
4. **Threshold objective mismatch.** `far_constrained` maximizes recall
   s.t. FAR ≤ 10% — the wrong objective per the user's goal. The right
   mode (`precision_at_min_recall`) already exists in
   `tejas/forecasting/evaluation.py:213` but is unused.

**User decisions (confirmed):**
- Ground truth = **GOES-confirmed flares ∪ solid TEJAS soft events
  (peak ≥ 25 c/s)** — external anchor + SoLEXS sensitivity, ~251 events.
- **Tree models first**; sequence model (GRU/TCN) is a follow-up if
  targets unmet.
- Recall target: **≥ 0.8 (event-level)** with precision maximized.

---

## Phase A — External ground truth (GOES/NOAA)  [the "trusted source"]

**Source verified live:** NCEI GOES-R archive has May 2026 for GOES-18:
- `xrsf-l2-flsum_science` — NOAA's official flare event summary
  (begin/peak/end + GOES class per flare, one netCDF/day, ~44 KB):
  `https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-flsum_science/2026/05/sci_xrsf-l2-flsum_g18_dYYYYMMDD_v2-2-1.nc`
- `xrsf-l2-flx1s_science` — 1-s XRS flux (already targeted by the
  existing `scripts/fetch_goes_data.py`).
- GOES-19 equivalents exist as fallback.

**A1. Install deps.** `pip install xarray netCDF4 h5netcdf` into `.venv`
(missing — verified); update `requirements.txt`.

**A2. New script `scripts/fetch_goes_flares.py`.** Download + parse the 30
daily flsum files → `data/reference/goes/2026/05/goes_flare_events.csv`
(columns: begin/peak/end UTC, class, magnitude, satellite). Skip day 22
(`MISSING_DAYS`).

**A3. Run existing flux fetch.** `python scripts/fetch_goes_data.py`
(GOES-18, 1-s XRS-B) → per-day CSVs for calibration.

**A4. Run calibration.** `python scripts/run_classification.py --calibrate`
— populates `goes_flux`/`goes_class` on the 615 soft events via the
existing cross-match + Huber log-log fit (`tejas/classification/calibration.py`).

**A5. New module `tejas/validation/goes_match.py` + report.** Cross-match
TEJAS catalogue vs GOES flare list (peak-time tolerance ±5 min,
configurable). Output `goes_validation_report.md`:
- match rate of TEJAS events vs GOES list (by intensity class),
- GOES flares TEJAS missed (detection-level recall vs external truth),
- TEJAS-only events (SoLEXS-sensitive or spurious).
This is the deliverable that externally confirms which flares are real.

## Phase B — Label surgery (the core fix)

**B1. Credible-event definition** (new config block):
```python
FORECAST_EVENT_MIN_PEAK_COUNTS = 25    # "weak"+ intensity bins
FORECAST_EVENT_REQUIRE_SOFT    = True  # drop hard-only noise events
# credible = (has_soft & peak >= 25) | matched-to-GOES
```
→ ~251 TEJAS events ∪ any GOES-matched events → filtered catalogue
`master_catalogue_credible.csv`.

**B2. Regenerate forecasting labels** from the credible catalogue only
(re-run Phase-5 label attachment; `label_minutes_to_next_flare` and
`label_flare_within_{h}m` recomputed). Detection parquets keep a
`label_*_all` copy of old labels for comparison. Expected effect: base
rates drop to sane levels (60-min no longer 88% positive), labels stop
teaching the model to predict noise.

**B3. AR-proxy features keep using the FULL catalogue** (activity context
from all events, incl. hard-only, is legitimate causal input — only the
*targets* change).

## Phase C — Objective & metrics aligned with the goal

**C1. Threshold mode switch:**
```python
FORECAST_THRESHOLD_METRIC = "precision_at_min_recall"
FORECAST_MIN_RECALL_PER_HORIZON = {5: 0.80, 15: 0.80, 30: 0.80, 60: 0.80}
```
(extend `tune_threshold()` to accept per-horizon floors; keep existing
modes intact).

**C2. Event-level metrics** (new `tejas/forecasting/event_metrics.py`) —
the honest, operator-meaningful scoreboard:
- **Event recall**: fraction of credible events with ≥1 alert inside the
  h-window before start.
- **Episode precision**: consecutive alert rows merged into episodes; an
  episode is TRUE if a credible event starts within h after it began.
- Median warning lead time per caught event.
Reported alongside row-level metrics in `forecast_report.md`, plus a
**GOES-only column** (recall/precision measured against GOES-confirmed
events exclusively — the external headline number).

## Phase D — Discrimination-lifting features (trees, no new model class)

- `bg_percentile_1h/3h/6h`: current SoLEXS flux percentile vs trailing
  1/3/6-h distribution (quiet-vs-active context; current max trailing
  window is only 600 s).
- `rise_consistency_300s`: fraction of positive 10-s derivatives in last
  5 min (sustained-rise vs spike).
- `time_in_elevated_state_s`: seconds since flux last below trailing
  1-h median (event-phase context).
- All strictly causal; added in Phase-4 pipeline alongside existing
  families; re-run features → detection → forecasting.

## Phase E — Retrain, evaluate, compare

- Retrain 12 base models + stacked ensemble on credible labels
  (~10–15 min total, per prior runs). Isotonic calibration unchanged.
- Produce `forecast_report_v2.md` with before/after on the SAME test days
  (27–31): row-level P/R/PR-AUC/ROC-AUC vs base rate, event-level
  recall/episode-precision per horizon, GOES-confirmed subset, lead times.
- Success criteria (honest reporting, not guaranteed): event recall ≥ 0.8
  at 30/60 min with episode precision materially above the before
  snapshot; at 5/15 min report the achievable precision at recall 0.8 and
  state plainly if the signal doesn't support it (feeds the go/no-go on
  the sequence-model follow-up).

## Phase F — Tests & integrity

- All existing tests (incl. 5 no-future-data tripwires) must stay green.
- New tests: credible-event filter counts; GOES flsum parser (fixture
  netCDF); event-metrics correctness on a synthetic day; per-horizon
  `precision_at_min_recall` tuning.
- No-leakage audit of new features (trailing-only windows).

## Execution order & est. effort

A (GOES fetch+match, ~1 h incl. downloads) → B (labels, ~30 min) →
C (objective/metrics, ~45 min) → D (features + pipeline re-run,
~1 h) → E (retrain+report, ~30 min) → F (tests, ~45 min).

## Risks

- **May 2026 GOES activity unknown** — if the GOES list is sparse, the
  union definition still works (TEJAS ≥25 c/s carries the labels) but the
  external-validated subset will be small; reported as-is.
- **Calibration match sparsity** — Huber fit needs ≳10 matched pairs;
  fall back to reporting match-only validation without count→flux fit.
- **Recall 0.8 at 5 min may be physically unreachable** — the plan
  measures and reports this honestly rather than hiding it; it defines
  the sequence-model follow-up scope.
- Uncommitted changes in `tejas/config.py` / `discriminator.py` will be
  committed first so every change in this plan is cleanly diffable.
