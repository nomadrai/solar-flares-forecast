# TEJAS — Optimization Plan v3: Precision ∧ Recall, Scientifically Honest

> **Goal:** Simultaneously raise Precision AND Recall (plus PR-AUC, ROC-AUC,
> calibration, and lead time; lower FAR and miss rate) for the multi-horizon
> flare alarm system — without violating the strict no-future-data constraint,
> and with every headline number externally validated against NOAA GOES.
>
> **Status:** PLAN ONLY. No code changes have been made. Every claim below is
> grounded in a measurement taken from the current repo state (2026-07-19).
>
> Predecessors: `optimization_plan.md` (Fixes 1–8, FAR-focused),
> `optimization_plan_v2.md` (credible-event relabelling + GOES external truth).
> This plan starts from the v2 state: credible labels, GOES flare list,
> `precision_at_min_recall` thresholds, event-level metrics, and the
> calibration artifact (`data/reference/calibration/solexs_to_goes.json`,
> fit 2026-07-19) all exist and ran.

---

## 0. Measured Diagnosis — where the current system actually stands

All numbers below were measured directly from the repo's current artifacts
(`training_frame_10s.parquet`, `metrics_summary.csv`, `event_metrics_v2.csv`,
`event_metrics_goes_v2.csv`, `feature_importance.csv`, catalogues).

### 0.1 Row-level test metrics (XGB, calibrated, v2 credible labels)

| Horizon | ROC-AUC (val→test) | PR-AUC (val→test) | Test base rate | Alarm duty cycle* |
|---|---|---|---|---|
| 5 min  | 0.741 → **0.650** | 0.114 → **0.053** | 0.025 | **39.5%** |
| 15 min | 0.610 → **0.495** | 0.160 → **0.068** | 0.068 | **79.7%** |
| 30 min | 0.619 → **0.494** | 0.292 → **0.130** | 0.130 | **80.0%** |
| 60 min | 0.538 → **0.564** | 0.424 → **0.267** | 0.236 | **100.0%** |

\* duty cycle = fraction of all test rows above the tuned threshold
(`test_n_predicted_positive / n_test` from `metrics_summary.csv`).

### 0.2 Event-level test metrics (the operator view)

| Horizon | Event recall (credible) | Episode precision | Event recall (GOES-only) | Median lead |
|---|---|---|---|---|
| 5 min  | 77.5% | 12.7% | 47.2% | 4.8 min |
| 15 min | 75.0% | 35.6% | 63.9% | 14.8 min |
| 30 min | 80.0% | 37.2% | 69.4% | 29.8 min |
| 60 min | 82.5% | 100.0% (5 episodes) | 75.0% | 59.8 min |

### 0.3 The five root problems this plan attacks

**P1 — Metadata leakage into the feature matrix (memorization, not physics).**
The frozen manifest (`outputs/models/2026/05/feature_columns.json`, 136
features) contains `cdte_segment_id`, `czt_segment_id`, `cdte_stat_err`,
`czt_stat_err`. `FORECAST_DROP_COLUMNS` (`tejas/config.py:278`) blacklists the
unified `segment_id` but **not the per-instrument variants**. Measured impact:
`cdte_segment_id` is the **#2 XGBoost feature at the 30-min horizon**
(importance 0.043) and `czt_segment_id` is top-15. Segment IDs count telemetry
dumps — they encode *which part of the month a row is from*, i.e. day
identity. The model partially memorizes "active days" instead of learning
precursors. This is the most likely driver of the val→test collapse
(ROC 0.62 → 0.49 at 15/30 min). It is not future-leakage, but it is
generalization-poison and must die first.

**P2 — Degenerate threshold objective.** `precision_at_min_recall` with a
**row-level** recall floor of 0.8 (`FORECAST_MIN_RECALL_PER_HORIZON`) forces
the tuner into an always-on corner: duty cycle 80% at 15/30 min and **100% at
60 min** (the 60-min "alarm" fires on every single test row — recall 1.0, FAR
1.0, meaningless). Consequence: the reported median leads (4.8/14.8/29.8/59.8
min ≈ the horizon maximum) are largely an *artifact* — with the alarm nearly
always on, the "first crossing" is trivially near the window start. Honest
post-fix lead times will be smaller and real. Expectations must be set now.

**P3 — Label incompleteness and ambiguity.** Measured: **28 GOES flares have
no credible-catalogue start within ±10 min** (23 of them have Aditya frame
coverage; by class: 17 C, 8 B, 3 M). Every one is a *label false negative*:
the model is punished for correctly alerting before a real (externally
confirmed) flare. Additionally, rows just outside a horizon window (flare in
h+1…h+10 min) are labeled 0 while being physically near-identical to
positives, and in-flare decay rows (4.3% of rows) carry confusing targets.

**P4 — Regime shift between splits, amplified by a single month of data.**
Train/val mean SoLEXS flux 22.9/21.7 c/s vs **test 16.6 c/s**; test base rates
are ≈ half of train's (e.g. 30-min: 21.1% train vs 13.0% test). Val-tuned
thresholds and isotonic calibrators systematically mis-transfer to the quieter
test days. One month cannot fix this; more months can (§12).

**P5 — Val split is triple-dipped.** Early stopping, calibrator fitting,
threshold tuning, stacking meta-fit, and model selection all consume the same
5 val days (`ensemble.py` even *trains* on val). Every additional consumer
overfits val a little more, and val is regime-shifted from test to begin with.
The protocol needs day-blocked rotation inside val (and eventually more
months), not more val-fitted components.

---

## Mapping: requested optimization area → plan section → phase

| # | Requested area | Section | Phase |
|---|---|---|---|
| 1 | Feature engineering | §1 | P2 |
| 2 | Label engineering | §2 | P1 |
| 3 | Threshold optimization | §3 | P0 + P3 |
| 4 | Event-level post-processing | §4 | P7 (+P3) |
| 5 | Probability calibration | §5 | P3 |
| 6 | Hard negative mining | §6 | P5 |
| 7 | Temporal consistency filtering | §7 | P3 + P7 |
| 8 | Model comparison & selection | §8 | P4 |
| 9 | Ensemble strategy | §9 | P6 |
| 10 | Explainability-based error analysis | §10 | P5 |
| 11 | Class-imbalance optimization | §11 | P4 |
| 12 | Dataset expansion | §12 | P8 |

---

## 1. Feature Engineering v3

**Current state.** 136 model features: temporal (45), spectral (13),
cross-instrument (8), statistical (15), particle (5), AR-proxy (5),
long-context (5), plus raw/clean/smooth/Kalman variants. All causal. Gaps
found by audit: no EMA family, no log-space gradients, no Poisson-aware
significance, no lag/snapshot features, hardness-ratio dynamics stop at 300 s,
no rolling-max/range, and the P1 metadata columns pollute the matrix.

### 1.0 Hygiene first (P0, one config edit — highest impact/effort in the plan)

- Add to `FORECAST_DROP_COLUMNS`: `cdte_segment_id`, `czt_segment_id`,
  `cdte_stat_err`, `czt_stat_err` (stat_err is an instrument-noise metadata
  channel, ~all-NaN upstream, partially imputed in the frame — not physics).
  Additionally harden `select_feature_columns` (`sampling.py:105`) to drop any
  column matching `*_segment_id` / `*_stat_err` by pattern, and add a tripwire
  test asserting no such column ever appears in the manifest.
- Replace what segment IDs *legitimately* encode with causal data-quality
  features (P2): `seconds_since_segment_start` (time since last data gap —
  rolling features are less trustworthy right after a gap) and
  `hel1os_ffill_fraction_300s` (fraction of forward-filled HEL1OS samples in
  the last 5 min). These let the model discount unreliable stretches without
  learning day identity.

### 1.1 New feature families (all trailing-window, causal; ~35–40 columns)

| Family | Features | Physics rationale |
|---|---|---|
| **EMA + crossover** | `{sig}_ema_{30,120,600}` (span-based), `{sig}_ema_xover = ema_30 − ema_600`, and its sign-persistence | EMA responds faster than the same-length SMA and never "forgets" a burst at the window edge; the fast-minus-slow crossover is a classic sustained-rise detector (MACD analog) |
| **Log-space gradients** | `{sig}_loggrad_{60,300}` = Δlog(x+eps)/Δt | Flares are multiplicative phenomena; a 20→40 c/s rise (B-class precursor) and 200→400 c/s rise are the *same* log-gradient. Linear c/s gradients under-weight weak-event precursors — a direct recall lever for B/C flares |
| **Poisson significance** | `{sig}_poisson_sig_{300}` = (x − bg)/√(bg+eps), bg = trailing 25th-percentile (600 s) | Statistically correct "is this excess real" for count data; superior to z-score with an arbitrary 0.1 std floor at low counts (CZT/CdTe quiet rates) |
| **Baseline excess ratio** | `solexs_excess_1h`, `solexs_excess_ratio_1h` = (x − P25_1h)/P25_1h | `bg_pctrank_*` saturates at 1.0 early in a rise; the *ratio* keeps growing and carries magnitude information through the whole precursor phase |
| **HR long dynamics** | `solexs_hr_ema_600`, `hr_pctrank_3h`, `hr_rise_consistency_300s` | Spectral hardening is THE documented pre-flare signature (see `feature_catalog.md` §2); current HR features stop at 300 s and cannot see slow hardening trends |
| **Lag snapshots** | `{sig}_ratio_lag_{60,300,600}` = x(t)/x(t−Δ) | Gives trees an explicit trajectory shape (three points of the rise curve) without a sequence model — the cheapest possible "temporal pattern" encoding |
| **Rolling extrema** | `{sig}_rmax_600`, `{sig}_range_ratio_600` = (max−min)/mean, `time_since_rmax_600` | Distinguishes "elevated and still climbing" from "elevated and decaying past a peak" — the decay phase is a major current FP source (alarm stays on after the flare) |
| **CdTe rise-consistency** | `cdte_rise_consistency_300s` | Exists only for SoLEXS (`long_context.py`); hard X-ray sustained rise is the Neupert-effect early warning |
| **Data-quality** | `seconds_since_segment_start`, `hel1os_ffill_fraction_300s` | §1.0 — legitimate replacement for segment IDs |

Explicitly rejected: time-of-day / day-of-month features (no physical basis at
L1, pure schedule-memorization risk), any GOES-derived inference-time feature
(policy §13), future-window statistics (obviously).

### 1.2 Pruning protocol

After adding ~40 features, prune once per horizon using native tree
contributions (XGBoost `pred_contribs`, LightGBM `pred_contrib` — zero new
dependencies): drop features with near-zero mean |contribution| on val AND
unstable sign across val days. Target a stable ~110–130 column manifest.
Re-freeze `feature_columns.json`. Fewer, stabler features also narrows the
val→test gap (P4).

**Expected benefit:** ROC/PR lift at all horizons (this is where genuine new
discrimination comes from); better weak-flare recall via log-gradients;
faster post-peak alarm release via extrema features.
**Risk:** feature bloat → overfit; mitigated by the pruning pass.
**Effort:** ~1–2 days incl. full features→detection→frame re-run.

---

## 2. Label Engineering v3

**Current state.** v2 labels: `flare_within_{h}m` from 370 credible events
(soft ≥ 25 c/s ∪ GOES-matched; `tejas/detection/credible.py`), month-wide
relabelling (cross-day correct), `_all` backups blacklisted. Sound foundation,
three measured defects (P3).

### 2.1 Complete the ground truth with the GOES union (label false negatives)

Add the **23 coverage-verified GOES flares** that have no credible TEJAS
counterpart as label events, using GOES `begin_time` as the start anchor.
Rationale: labels are *training targets*, always retrospective by design
(`CONTEXT.md`: training may use complete historical data). Using externally
confirmed flares as targets is exactly what the external reference is for —
it does NOT leak anything into inference inputs. Implementation: extend
`build_credible_catalogue` to append GOES-unmatched-by-TEJAS events with a
`source="goes_only"` tag → `master_catalogue_credible.csv` grows ~370 → ~393.
The 5 GOES flares *without* Aditya data coverage are handled by §2.3, not
labeled.

### 2.2 Ambiguity buffer ("don't-care" band) — training only

Rows whose `label_minutes_to_next_flare` falls in `(h, h + buffer]` are
**excluded from the training fit** for horizon h (not relabeled — excluded).
Recommended `buffer = max(5 min, 0.25·h)` per horizon, config-driven
(`FORECAST_LABEL_AMBIGUITY_BUFFER_MIN`). These rows are physically
near-identical to positives (the precursor is already underway) yet labeled 0;
they inject pure gradient noise and depress both precision and recall.
**Val/test evaluation keeps strict labels — no evaluation softening, ever.**
Expected training-row loss: a few % per horizon.

### 2.3 Coverage masking

Rows inside a GOES-confirmed flare window where Aditya has no/degraded data
(the 5 uncovered flares + any GTI gap overlapping a GOES event) are dropped
from training for all horizons: their true label is unknowable from the
instrument's viewpoint. Currently they sit in the frame as (wrong) negatives.

### 2.4 In-flare row policy

Rows with `is_flare_active=1` and `flare_phase="decaying"` (measured: in-flare
= 4.3% of rows) are candidates for exclusion from *training*: during a decay
the operational question is handled by the nowcaster, and the forecast label
("next flare within h") is dominated by sympathetic-flaring statistics rather
than precursor physics. Run as an A/B ablation (in/out) on val — adopt only if
it helps; keep evaluation untouched either way.

### 2.5 Explicitly NOT changed

- The 25 c/s credible floor stays (v2 decision, externally validated at 81.9%
  detection recall vs GOES).
- `label_minutes_to_next_flare` recomputed against the completed event set for
  consistency; remains blacklisted.
- Detection-level misses (3/8 M-class missed by the Phase-5 detector, see
  `goes_validation_report.md`) are an adjacent detection-tuning workstream —
  §2.1 already ensures the *forecaster* trains on those events regardless.

**Expected benefit:** removes systematic label FNs (+recall, +precision
simultaneously — the model stops being penalized for correct alerts);
cleaner gradients from the buffer (+both).
**Risk:** GOES begin_time vs TEJAS start_time timing offsets (GOES begins
typically earlier); mitigated: the event-level scoreboard already tolerates
this, and begin-anchored labels are the *conservative* choice for lead time.
**Effort:** ~1 day incl. relabel + retrain + before/after report.

---

## 3. Decision Threshold Optimization (event-level, per horizon)

**Current state.** Per-horizon val sweep exists (`evaluation.py:tune_threshold`,
5 metrics, 0.01 grid). The objective is row-level → degenerate corner (P2).

### 3.1 New primary objective: episode precision at event-recall floor

For each horizon, sweep thresholds on **val** and compute, via the existing
`event_metrics.compute_event_metrics`:

- event recall vs credible events in the val window,
- episode precision (episodes via the existing 300 s merge),
- **alarm duty cycle** (fraction of val rows above threshold),
- false episodes per day.

Objective: `maximize episode_precision subject to event_recall ≥ floor(h) AND
duty_cycle ≤ cap(h)`. Recommended caps: 5m: 0.10, 15m: 0.15, 30m: 0.20,
60m: 0.30 (config: `FORECAST_MAX_DUTY_CYCLE_PER_HORIZON`). The duty-cycle cap
is the structural fix that makes the 60-min always-on corner *infeasible*
rather than merely disfavored. Recall floors stay per-horizon config
(`FORECAST_MIN_RECALL_PER_HORIZON`, now interpreted event-level).

### 3.2 Mechanics

- Sweep on **calibrated** probabilities with a **quantile-spaced grid**
  (probabilities concentrate near 0 after calibration under imbalance; the
  linear 0.01 grid wastes resolution where nothing happens).
- Tie-break to the highest threshold (existing convention, keep).
- **Stability check:** leave-one-val-day-out re-tuning → report the threshold
  spread per horizon. A horizon whose optimal threshold swings wildly across
  val days gets flagged in the report (operator trust signal), and the median
  of the LODO thresholds is used instead of the全-val argmax.
- Fallbacks preserved for degenerate val folds (existing behavior).

### 3.3 Operator utility table (reported, not optimized)

Alongside the chosen threshold, emit a small table per horizon: for cost
ratios miss:false-episode ∈ {1:1, 3:1, 10:1}, the utility-optimal threshold
and its (event recall, episode precision, FE/day). This documents the
trade-off surface for the SOC instead of hiding it inside one scalar choice.

**Expected benefit:** kills the degenerate 60-min alarm; honest precision
gains at 15/30 min; lead times become real (and will *look* smaller — §0/P2
expectation-setting).
**Risk:** event-level tuning on 71 val events has variance; mitigated by LODO
median + duty cap.
**Effort:** ~0.5 day (P0 minimal version: duty cap + event-level objective);
joint tuning with persistence in P3.

---

## 4. Event-Level Post-Processing & Operational Alert Generation

**Current state.** Episode merge (300 s) exists *for evaluation*;
`ConfirmationGate` (K=3 consecutive, 300 s cooldown) exists *for runtime* —
but evaluation scores raw threshold crossings, so the thing being measured is
not the thing an operator would see.

### 4.1 Unify: evaluate the alert stream, not raw crossings

Route offline evaluation through `simulate_gate_on_frame` so the headline
event metrics describe the *post-gate* alert stream. From P3 onward, the
official scoreboard is: gate(threshold, K, cooldown, hysteresis) → episodes →
event recall / episode precision / FE-per-day / lead time.

### 4.2 Episode construction with hysteresis

Mirror the Phase-5 detector's asymmetric design (3.5σ start / 1.5σ end): an
episode *opens* at the confirmed onset threshold and *closes* only when
calibrated probability falls below a **release threshold** = r·onset (r ≈
0.5–0.8, tuned on val). Prevents flicker-alarms around a single threshold and
formalizes "merge consecutive positives into one flare alert." The 300 s
post-close merge stays as the outer dedup.

### 4.3 Cross-horizon deduplication → one operational alert object

Four horizons currently imply four independent alert streams for the same
physical flare. Introduce escalation levels on a single master alert:

`WATCH (60m) → ADVISORY (30m) → WARNING (15m) → IMMINENT (5m)`

A shorter-horizon crossing inside an active longer-horizon episode *escalates*
the existing alert instead of spawning a duplicate. Alert object carries:
first-issued time, current level, level history, peak calibrated probability,
top-3 feature contributions (§10). This is the Phase-8 deliverable the
dashboard consumes; duplicate-alarm reduction falls out of it by construction.

### 4.4 New operational metrics (report additions)

False episodes **per day** and duty cycle join the scoreboard (operators
reason in interruptions-per-shift, not FP-row-rate). Target: ≤ 2–3 false
episodes/day at 15/30 min (current: 45 episodes/16 true over 5 test days at
15 min ≈ 5.8 false/day).

**Expected benefit:** large episode-precision gains at zero event-recall cost
(dedup + hysteresis only reorganize positives); the operational product
becomes measurable.
**Risk:** minimal — post-processing is evaluated on val like everything else.
**Effort:** ~1 day (P7), on top of P3's parameter tuning.

---

## 5. Probability Calibration

**Current state.** Isotonic on val (default), Platt available, Brier reported,
sensible fallbacks (`training.py:_fit_calibrator`). Two issues: (a) method is
a global config, not chosen per horizon by evidence; (b) val regime shift
means the calibrator itself mis-transfers (the 60-min isotonic currently maps
everything above threshold — FAR 1.0).

### Plan

1. **Per-horizon method selection by day-blocked CV inside val:** fit each
   candidate (none / Platt / isotonic / beta-calibration) on 4 of 5 val days,
   score Brier + ECE (15-bin) on the held-out day, rotate, pick the winner per
   horizon. Beta calibration is a ~30-line pure-NumPy addition (no new dep)
   that handles skewed-toward-0 probability distributions better than Platt.
   Expected outcome: isotonic keeps 5/15 min (plenty of val positives),
   Platt/beta likely wins 30/60 min (isotonic's step-overfit on shifted val is
   the current failure).
2. **Reliability artifacts:** per-horizon reliability diagrams + ECE in the
   forecast report (currently only scalar Brier). Calibration is a stated
   deliverable for operational decision-making — it needs visible evidence.
3. **Protocol note (documented in `forecasting_catalog.md`):** calibrators
   remain val-fit only (chronologically after train, before test — no
   leakage); train-split probabilities are in-sample and must never feed the
   calibrator.
4. Threshold sweep (§3) runs on the *selected* calibrator's outputs, so the
   quantile grid adapts to the calibrated distribution.

**Expected benefit:** honest probabilities at 30/60 min (currently saturated),
lower Brier/ECE, and a threshold axis that actually has resolution.
**Risk:** ECE on 5 val days is noisy — report with bin counts.
**Effort:** ~0.5 day, inside P3.

---

## 6. Hard Negative Mining (training-time)

**Current state.** Absent. The Stage-2 discriminator (Fix 8) is a *post-hoc
alert filter* — it suppresses FPs after the base model made them, and it
kills TPs at 15/60 min (376 and 290 respectively on test). Training-time HNM
attacks the same FP families at the source without a second model in the
loop.

### Plan

1. **Out-of-fold FP harvesting on train only:** split train days 01–20 into 4
   contiguous day-blocks; for each block, train on the other 3 (+purge),
   predict the held-out block; collect negatives with calibrated p ≥ the
   val-tuned threshold. Out-of-fold prediction is mandatory — in-sample FPs
   of a boosted tree are nearly empty by construction.
2. **Exclusions:** rows inside the §2.2 don't-care buffer are not "hard
   negatives" — they're mislabeled positives-adjacent rows and stay excluded.
3. **Reweight, don't resample:** `sample_weight = w_hn` for harvested rows
   (sweep w_hn ∈ {1.5, 2, 3, 4} on val), total hard-negative weight mass
   capped at ≤ 20% of the negative class. No SMOTE, no synthetic series —
   consistent with the project's stated policy and the user's constraint.
4. **Hard-negative library:** persist harvested FP episodes with timestamps +
   §10 SHAP cause-tags to `data/forecasting/YYYY/MM/hard_negative_library.csv`.
   Doubles as a regression suite: future model versions are checked against
   last generation's failure modes.
5. **Guardrail:** accept the reweighted model only if val event recall drops
   < 2 pts while episode precision rises — otherwise shrink w_hn.

Anticipated FP families (from Fix-3/Fix-8 evidence): GTI-boundary artifacts,
particle spikes (CZT-without-CdTe signature), gradual background swells on
active days (AR-proxy over-reliance), post-peak decay tails.

**Expected benefit:** precision/FAR improvement with minimal recall cost —
the discrimination boundary sharpens exactly where it currently fails.
**Risk:** overweighting quiet-period lookalikes suppresses weak-flare
precursors (recall hit) — the guardrail + cap manage this.
**Effort:** ~1 day (P5).

---

## 7. Temporal Consistency Filtering

**Current state.** `ConfirmationGate` K=3 consecutive @10 s + 300 s cooldown —
both values are guesses, never tuned, and not part of evaluation (§4.1).

### Plan

1. **Add an m-of-n gate mode** (e.g., 4-of-6 above threshold) alongside
   strict-K: robust to a single dropout sample in a real precursor (strict-K
   resets the streak on one bad sample; m-of-n doesn't).
2. **Add probability-EMA smoothing as a third mode:** threshold on
   EMA(p, span≈30–60 s) instead of raw p. All three modes are causal.
3. **Joint 2-D tuning with the threshold (P3):** per horizon, grid over
   (threshold × persistence-config) scored on val at the *event level*. A
   lower threshold + stronger persistence frequently beats a higher threshold
   + no persistence on BOTH precision and recall — this coupling is exactly
   what sequential tuning misses.
4. **Lead-time cost accounting:** every persistence setting delays
   confirmation by a computable amount (e.g., K=3 @10 s = 30 s). The report
   shows the (FE/day ↓, median lead ↓) trade per candidate so the choice is
   explicit. At 5 min the budget is tight (30 s ≈ 10% of the window); expect
   small K or EMA there, heavier settings at 30/60 min.

**Expected benefit:** isolated-spike alarms die; episode precision rises at
bounded, visible lead-time cost.
**Risk:** none beyond the quantified lead trade.
**Effort:** folded into P3 (~0.5 day incremental).

---

## 8. Model Comparison and Selection

**Current state.** LogReg / XGBoost 3.3 / LightGBM 4.6 (+val-fit stack).
Python **3.14** venv — relevant constraint below.

### Plan

1. **Candidates to add:**
   - **CatBoost** — *attempt* installation. Honest caveat: CatBoost wheels
     historically lag new CPython releases and 3.14 support is unverified;
     if no wheel exists, record that in the report and proceed — do not
     build from source.
   - **sklearn `HistGradientBoostingClassifier`** — zero new dependencies,
     native NaN handling, fast, and supports **monotonic constraints** (a
     scientifically interesting option: e.g., forcing P(flare) non-decreasing
     in `solexs_poisson_sig` — run as an ablation, not a default).
   - LogReg stays (calibrated linear baseline + explainability anchor);
     GRU/TCN sequence model remains the documented follow-up *only if* the
     tree ceiling persists after P1–P5 (torch 2.12 already installed).
2. **Hyperparameter search:** randomized search, 30–50 configs per
   (algorithm × horizon), fixed seed, early stopping on val AP as today.
   Search space: depth/leaves, learning rate, min_child_weight/samples,
   subsample/colsample, L1/L2. Budget: hours. Log every trial to
   `data/forecasting/YYYY/MM/hpo_trials.csv`. Plain NumPy sampling — Optuna
   not required (avoids another 3.14-compat unknown).
3. **Selection protocol (anti-P5):** rank by **leave-one-val-day-out mean of
   the event-level objective** (episode precision at recall floor, §3) plus
   Brier — NOT single-shot val PR-AUC. Prefer the simpler model within one
   standard error of the best (stability beats a lucky val fit under regime
   shift). Winner chosen **per horizon** — nothing forces one algorithm to
   serve all four.
4. Test-set numbers are computed once, after selection, and reported as-is.

**Expected benefit:** measurable but secondary — model choice matters far
less than labels/features/thresholds here (the v2 diagnosis already showed
all three algorithms collapse together when the target is noisy).
**Risk:** HPO overfitting val — LODO scoring is the mitigation.
**Effort:** ~1–2 days compute-bound (P4).

---

## 9. Ensemble Strategy

**Current state.** Stacked LR meta on val-calibrated base probs: +0.006 to
+0.015 test PR-AUC over XGB — real but marginal, and it deepens the val
triple-dip (meta trains on the same 5 days that tune everything else).

### Plan

1. Re-evaluate **after** P1–P4 land (base-model improvements change the
   ensemble calculus entirely).
2. Compare three combiners per horizon:
   (a) **logit-mean blend** of calibrated probs — zero fitted parameters,
   nothing to overfit, robust under regime shift;
   (b) rank-average blend — same, scale-free;
   (c) existing stacked LR meta (val-fit).
3. **Adoption rule (pre-registered):** the ensemble ships for a horizon only
   if LODO-val event-level score beats the best single model by ≥ 2 pts
   episode precision at the recall floor. Marginal PR-AUC deltas do not
   justify the extra artifact chain. Otherwise: single best model per horizon.
4. **Diversity worth testing:** a SoLEXS-only feature-subset model as a third
   voice — decorrelated errors AND graceful degradation during HEL1OS gaps
   (operational robustness argument, not just metrics).

**Expected benefit:** possibly +2–5 pts episode precision; possibly nothing —
the rule keeps us honest.
**Risk:** none (adoption is gated).
**Effort:** ~0.5 day (P6).

---

## 10. Explainability-Based Error Analysis

**Current state.** Gain-based importance CSV only. No per-prediction
attribution, no error-class analysis. SHAP not installed.

### Plan

1. **Attribution engine with zero new dependencies:** XGBoost
   `predict(..., pred_contribs=True)` and LightGBM `predict(..., 
   pred_contrib=True)` are native TreeSHAP implementations. Use them as the
   primary path; the `shap` package (plots) is optional-if-installable on
   py3.14, never load-bearing.
2. **Error-class protocol (per horizon, val + test):**
   - FP episodes (no credible/GOES event followed) vs TP episodes: rank
     features by mean |contribution| in each class; the divergence list =
     "misleading features".
   - FN events (missed flares): attribute the *sub-threshold* probability —
     which features suppressed it, and was the precursor visible at all
     (SoLEXS z-score > 2 pre-onset?) → separates model failures from
     instrument-invisible events (feeds §2.3 masking and honest reporting).
3. **Specific hypotheses to test** (from current evidence):
   - AR-proxy features (`ar_max_peak_last_6h` = #1 global XGB importance)
     dominating FPs at 30/60 min → the model fires on "active day"
     climatology, not imminent precursors. If confirmed: cap/monotonize or
     down-prune AR features at long horizons.
   - `particle_gti_proximity_s` behaving as designed (suppressing, not
     causing, alerts near GTI edges).
   - Post-P0 check: no residual identity-proxy features in top-20.
4. **Artifacts:** `error_analysis_report.md` + per-class contribution CSVs +
   cause-tags written into the §6 hard-negative library. Feature refinements
   discovered here loop back into §1 pruning.
5. **Operator-facing:** top-3 contributions attached to every alert object
   (§4.3) — this is USP #1 (explainable alerts) getting its backend.

**Expected benefit:** converts anecdotal FP/FN hunches into ranked, actionable
feature fixes; recurring value every retrain.
**Risk:** none.
**Effort:** ~1 day (P5, shared with HNM).

---

## 11. Class-Imbalance Optimization

**Current state.** Global sqrt-tempered `scale_pos_weight` capped at 12
(`FORECAST_USE_SQRT_SPW`, `FORECAST_SPW_CAP`) — one policy for all horizons,
never systematically tuned. LogReg uses `class_weight="balanced"`.

### Plan

1. **Per-horizon SPW sweep** in the P4 HPO: policy ∈ {1.0 (none), sqrt-capped
   (current), raw-capped} × cap ∈ {4, 8, 12, 20}, scored on LODO-val
   event-level objective. Post-P1 label surgery shifts every base rate
   (60-min will drop once GOES-union events and masking land), so current
   settings are stale regardless.
2. **Event-importance sample weights (experiment):** weight positive rows by
   their source event's class — moderate+/GOES-C+ events ×2 vs weak events
   ×1 — so the loss prioritizes flares operators most need caught. Tuned on
   val; adopted only if the event-recall mix improves (weak-event recall must
   not collapse).
3. **Interaction rule (documented):** heavy SPW inflates raw probabilities →
   calibration then compresses them into a narrow band → threshold resolution
   suffers. Preference order: *milder SPW + good calibration* over aggressive
   SPW. The sweep scores the combined pipeline, so this is enforced by
   construction rather than by hand.
4. **No SMOTE / no synthetic time series** — reaffirmed. Temporal
   autocorrelation makes interpolated minority samples physically meaningless
   here; weighting achieves the same loss-shaping without fabricating data.
5. Focal loss (XGB custom objective) parked as a stretch experiment only if
   weighting plateaus.

**Expected benefit:** modest standalone; compounds with §5 to fix the
saturated long-horizon probability distributions.
**Risk:** none beyond HPO cost.
**Effort:** inside P4.

---

## 12. Dataset Expansion Strategy (beyond May 2026)

**Current state.** One month, keyed by `YEAR`/`MONTH` singletons in
`tejas/config.py`. Day 22 missing. 370 credible events; 43 "significant";
8 M-class total. The train/test regime gap (P4) and the under-powered
sig-flare model (Fix-4 verdict) are both, at root, sample-size problems.
**This is the single highest-ceiling item in the plan** — but it depends on
external data availability, so it runs as a parallel track.

### Roadmap (compatibility-preserving)

1. **R1 — Ingest June 2026 (+July to date) as-is.** All pipeline paths
   already parameterize by `YEAR/MONTH`; processing per month requires *zero
   code change* (flip config, run the stage scripts per month). GOES fetch
   scripts (`fetch_goes_data.py`, `fetch_goes_flares.py`) already take
   arbitrary YYYY/MM. Deliverable: per-month processed/feature/detection
   trees + GOES reference.
2. **R2 — Multi-month training frames (small code change).** Generalize the
   split config from day-strings to `(month, day)` tuples (or a list of
   `YYYY-MM-DD` dates) in `splits.assign_split`; concatenate per-month
   detection parquets in `prepare_training_frame`. Purge logic unchanged.
   Chronological protocol: train = May+June, val = early July, test = late
   July. Backward compatible: a single-month config reproduces today's
   behavior exactly.
3. **R3 — Rolling-origin (walk-forward) evaluation.** Train ≤ month k → test
   month k+1, rolled forward as months accumulate. This is the honest
   operational protocol, kills the quiet-tail-of-May artifact permanently,
   and yields metric *stability bands* instead of one lucky/unlucky split.
4. **R4 — Sig-flare model revival.** With ≥ 3 months and the now-existing
   GOES calibration (`solexs_to_goes.json`), switch Fix-4 labels from native
   intensity bins to the GOES **M/X boundary** (`run_sig_labels.py
   --use-goes`, already scaffolded) — the plan's original intent, viable once
   M-class counts leave single digits.
5. **R5 — Recalibrate physics constants on the wider baseline** (quiet-Sun
   thresholds `CZT_BG_QUIET_THRESHOLD`, intensity bin edges) — solar-cycle
   activity drift makes May-only constants gradually stale.

**Expected benefit:** the generalization gap (P4) is structurally closed;
every other optimization's val decisions become trustworthy.
**Risk:** Aditya-L1 L1-product release lag is outside project control; GOES
side is guaranteed available.
**Effort:** R1 ~0.5 day/month (mostly compute); R2+R3 ~1 day.

---

## 13. External Validation Policy (NOAA / GOES / HEK)

The leakage boundary, stated precisely (and to be added to
`forecasting_catalog.md`):

| Use of external (GOES/HEK/SWPC) data | Verdict |
|---|---|
| Training **labels** (retrospective targets) | ✅ Legitimate — labels are always retrospective; that is what ground truth is for |
| Evaluation reference (event recall vs GOES list, per-class breakdown) | ✅ Legitimate — this is the headline external validation |
| Count→flux calibration of *outputs* (Phase 6) | ✅ Legitimate — presentation-layer mapping |
| **Inference-time features** (any GOES-derived input to `predict_row`) | ❌ Forbidden — TEJAS must forecast from Aditya-L1 data alone; enforced by keeping GOES columns out of detection parquets entirely |
| Threshold/calibrator fitting on *test-window* GOES truth | ❌ Forbidden — all tuning stays on val |

Concrete additions:

1. **GOES-19 secondary fetch** as cross-check where GOES-18 has gaps
   (fallback URLs already documented in v2 plan).
2. **HEK (Heliophysics Event Knowledgebase) cross-reference** — optional
   secondary validation only: match TEJAS/GOES events against HEK SSW flare
   entries (begin/peak/class) in the validation report. Adds independence to
   the external check; never touches the model.
3. **Per-class external scoreboard** stays the headline: event recall / lead
   time vs GOES-confirmed flares, broken out by B/C/M — extended with the
   walk-forward protocol in R3.
4. Every optimization phase's report includes the GOES-only columns
   (`event_metrics_goes_v2.csv` pattern) so internal-label drift can never
   silently flatter the results.

---

## 14. Target Metrics (test split, honest-reporting commitments)

Baselines from §0. Targets are goals, not guarantees — any miss is reported
as a miss, per project policy.

| Metric | Baseline (worst→best across horizons) | Target |
|---|---|---|
| ROC-AUC | 0.49–0.65 | ≥ 0.70 all horizons |
| PR-AUC / base rate | 1.0–2.1× | ≥ 2× all horizons |
| Event recall (credible) | 75–82.5% | ≥ 80% all horizons |
| Event recall (GOES-only) | 47–75% | ≥ 70% (5 min: ≥ 55%) |
| Episode precision | 12.7% (5m) – 37.2% (30m) | ≥ 25% (5m), ≥ 50% (15/30m) |
| False episodes / day | ~5.8 (15m) | ≤ 3 all horizons |
| Alarm duty cycle | 40–100% | ≤ 10–30% (per-horizon caps) |
| Brier / ECE | Brier 0.02–0.20, ECE unmeasured | ECE ≤ 0.05, Brier ↓ per horizon |
| Median lead | 4.8/14.8/29.8/59.8 min (duty-cycle-inflated — see P2) | ≥ 50–60% of horizon, measured on the *post-gate* stream |

**Pre-registered caveat:** current median leads are artifacts of near-always-on
alarms and WILL decrease when duty cycles are fixed. The honest number is the
post-gate lead on a ≤30%-duty-cycle stream. This is an accounting correction,
not a regression.

---

## 15. Phased Implementation Roadmap (impact ÷ effort ordered)

Every phase: all 242 existing tests stay green; the five no-future-data
tripwires extend to new code; before/after metrics on the same test days.

| Phase | Contents | Effort | Expected impact | Key risk | Depends on |
|---|---|---|---|---|---|
| **P0 — Hygiene & degeneracy kill** | Blacklist `*_segment_id`/`*_stat_err` (+pattern guard + tripwire test); event-level threshold objective with duty-cycle caps; retrain; before/after | ~0.5 day | **XL.** Removes identity memorization (val→test gap) AND the always-on 60-min alarm — the two most damaging measured defects | None — pure removals + objective change | — |
| **P1 — Label surgery v3** | GOES-union completion (23 events), don't-care buffer (train only), coverage masking, in-flare ablation; relabel + retrain | ~1 day | **XL.** Removes systematic label FNs → lifts precision AND recall simultaneously | GOES/TEJAS timing offsets (mitigated §2) | GOES list (present) |
| **P2 — Features v3** | EMA, log-gradients, Poisson significance, baseline excess, HR long-dynamics, lag snapshots, extrema, CdTe rise-consistency, data-quality features; contribution-based pruning; pipeline re-run | ~1–2 days | **L.** The genuinely new discrimination signal | Feature bloat → prune pass | P0 (manifest churn) |
| **P3 — Joint decision-layer tuning** | 2-D (threshold × persistence) event-level sweep; per-horizon calibration selection via day-blocked CV; reliability report; LODO stability bands | ~1 day | **L.** Converts model skill into operator-visible precision at held recall | Val variance (71 events) → LODO medians | P1 (labels), P2 helpful |
| **P4 — Models & imbalance** | +HistGB (+CatBoost if py3.14 wheel exists); randomized HPO; per-horizon SPW/sample-weight tuning; LODO-based per-horizon selection | ~1–2 days | **M.** Real but secondary to labels/features | HPO-overfit on val → LODO scoring | P2 (frame) |
| **P5 — Hard negatives & explainability** | Out-of-fold FP mining + reweight (guardrailed); native-TreeSHAP error-class analysis; hard-negative library; `error_analysis_report.md` | ~1 day | **M.** Precision push at held recall; recurring diagnostic value | Recall erosion → 2-pt guardrail | P4 (final model) |
| **P6 — Ensemble re-evaluation** | Logit-mean vs rank-blend vs stack; pre-registered ≥2-pt adoption rule; SoLEXS-only diversity member | ~0.5 day | **S–M.** Possibly nothing — gated honestly | None (gated) | P4 |
| **P7 — Operational alert engine** | Hysteresis episodes, cross-horizon escalation (WATCH→IMMINENT), per-alert explanations, post-gate scoreboard = official metrics, FE/day | ~1 day | **L (operational).** The product an operator actually receives | None | P3 |
| **P8 — Dataset expansion** (parallel track) | June+ ingest (config-only) → multi-month splits → walk-forward eval → GOES-M/X sig-model | ~0.5 day/month + 1 day code | **XL (ceiling).** Structurally closes the regime gap; makes every val decision trustworthy | External data availability | External releases |

**Suggested execution order:** P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7, with P8
running in parallel from day one (fetches are compute-bound, not
attention-bound). P0+P1 alone are expected to deliver the largest single jump
in test-set honesty and quality; each subsequent phase is measured against the
same frozen test days before the next begins.

### Per-phase deliverables

Every phase produces: updated `metrics_summary.csv` + event-level CSVs +
GOES-only columns, a diff-style section in `optimization_report_v3.md`
(before/after on identical test days), and new/updated tests. Phases P0–P2
additionally re-freeze `feature_columns.json` (manifest version noted in the
report). `CONTEXT.md` / `forecasting_catalog.md` / `feature_catalog.md` are
updated in the same change that causes drift, per the Document Maintenance
policy.

### New tests (Phase F-equivalent, spread across phases)

- Manifest tripwire: no `*_segment_id` / `*_stat_err` / blacklist-pattern
  column in `feature_columns.json` (P0).
- Don't-care buffer excluded from train, present in eval (P1).
- GOES-union credible catalogue counts + `source` tags (P1).
- Event-level tuner: duty-cap feasibility, recall-floor fallback, LODO median
  (P0/P3).
- Gate modes (strict-K / m-of-n / EMA) causality + hysteresis release (P3/P7).
- Calibration selection: per-horizon method persisted and applied at
  inference (P3).
- Hard-negative harvest is out-of-fold (assert no train-block overlap) (P5).
- Escalation dedup: one master alert per overlapping multi-horizon episode
  (P7).
- Multi-month split disjointness + cross-month purge (P8).

---

## 16. Integrity Statement

Everything in this plan preserves the inviolable constraint (`CONTEXT.md`):

- All new features are trailing-window/causal; the existing
  noise-features tripwire test extends to the new families.
- All tuning (thresholds, persistence, calibration, HPO, ensemble adoption,
  class weights) consumes **val only**; test is touched once per phase for
  reporting.
- External data (GOES/HEK) enters **labels and evaluation only** — never
  inference inputs (§13 table becomes part of the catalog docs).
- No synthetic samples of any kind.
- Honest-reporting rules are pre-registered: the lead-time accounting
  correction (§14), the ensemble adoption gate (§9), the CatBoost/py3.14
  availability caveat (§8), and any missed target are reported as-is.
