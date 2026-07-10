# TEJAS — Forecasting Catalogue

This document describes the Phase 7 forecasting module: how the models are trained, what their outputs mean, how the streaming inference API is used, and what guarantees the module makes about the strict no-future-data constraint.

---

## Design Principles

1. **One model per horizon.** Four independent binary classifiers, one for each of 5 / 15 / 30 / 60 minutes. Independent decision thresholds and independent precision/recall trade-offs, so operations teams can dial each horizon separately.
2. **Three algorithms, same feature contract.** Logistic Regression, XGBoost, LightGBM. All read the same frozen feature list. The dashboard defaults to XGBoost; LogReg + LightGBM are kept for benchmarking and explainability.
3. **Chronological blocked splits with a purge gap.** Train = days 01–20, Val = 21+23–26 (22 missing), Test = 27–31. A 1-hour purge is applied to the tail of the train and val blocks so a 60-min-horizon label at the boundary cannot depend on truth from the next block.
4. **No-future-data is enforced at two independent choke points.** Feature/label separation is checked once in `sampling.select_feature_columns` (blacklist) and once again in `Forecaster.predict_row` (allowlist + blacklist). See the [Tripwires](#no-future-data-tripwires) section.
5. **Class imbalance handled via weighting only.** No synthetic samples (SMOTE, ADASYN). LogReg uses `class_weight="balanced"`; XGBoost and LightGBM compute `scale_pos_weight = n_neg / n_pos` from the train fold.

---

## Inputs & outputs

### Inputs

The module consumes Phase 5 labelled feature Parquets at `data/detection/YYYY/MM/{YYYY-MM-DD}.parquet`. Each file contains:

- Phase 2 unified signals (`solexs_counts`, `cdte_count_rate`, `czt_count_rate`, ...) and their `_clean` / `_smooth` variants.
- The 81 Phase 4 engineered features (see `feature_catalog.md`).
- Phase 5 per-timestep labels: `flare_id`, `is_flare_active`, `flare_phase`, `label_minutes_to_next_flare`, `label_flare_within_{5,15,30,60}m`.

Only feature columns are consumed as input to the model. Labels are used strictly for training and evaluation.

### Outputs

| Path | Description |
|---|---|
| `data/forecasting/YYYY/MM/training_frame_{c}s.parquet` | Downsampled + purged training frame (concatenated across all days). |
| `data/forecasting/YYYY/MM/metrics_summary.csv` | One row per (horizon × algorithm) with val + test metrics. |
| `data/forecasting/YYYY/MM/feature_importance.csv` | Gain-based importance for tree models; \|coef\| for LogReg. |
| `data/forecasting/YYYY/MM/test_metrics.csv` | Test-set metrics for the evaluated algorithm. |
| `data/forecasting/YYYY/MM/lead_time_distribution.csv` | Per-event lead time on the test split. |
| `data/forecasting/YYYY/MM/forecast_report.md` | Human-readable evaluation summary. |
| `outputs/models/YYYY/MM/{algo}_horizon_{h}m.joblib` | 12 persisted model artifacts. |
| `outputs/models/YYYY/MM/feature_columns.json` | Frozen feature-list manifest (single source of truth for inference). |

`outputs/` is git-ignored — model artifacts are not committed.

---

## Configuration reference

All tunables live in `tejas/config.py` under the `Phase 7: Forecasting` block:

| Constant | Default | Purpose |
|---|---|---|
| `FORECAST_DIR` | `data/forecasting/YYYY/MM/` | Frame + report output directory |
| `MODELS_DIR` | `outputs/models/YYYY/MM/` | Persisted joblib artifacts |
| `FORECAST_CADENCE_S` | `10` | Primary training cadence (seconds) |
| `FORECAST_BENCHMARK_CADENCE_S` | `30` | Optional secondary cadence for ablation |
| `FORECAST_LABEL_HORIZONS_MIN` | `[5, 15, 30, 60]` | Binary horizons in minutes |
| `FORECAST_TRAIN_DAYS` | `01..20` | Train split day strings |
| `FORECAST_VAL_DAYS` | `21, 23, 24, 25, 26` | Val split (22 missing) |
| `FORECAST_TEST_DAYS` | `27, 28, 29, 30, 31` | Test split |
| `FORECAST_SPLIT_PURGE_S` | `3600` | Purge gap between blocks (must ≥ longest horizon × 60) |
| `FORECAST_DROP_COLUMNS` | (11 items) | Blacklist of columns never fed to a model |
| `FORECAST_MIN_NONNULL_FEATURE_RATIO` | `0.95` | Threshold for dropping rows with too many NaNs |
| `FORECAST_XGB_PARAMS`, `FORECAST_LGBM_PARAMS`, `FORECAST_LOGREG_PARAMS` | — | Model hyperparameters |
| `FORECAST_DECISION_THRESHOLD_DEFAULT` | `0.5` | Fallback if val-based tuning fails |
| `FORECAST_LEAD_TIME_MIN_PROB` | `0.5` | Probability crossing threshold used for lead-time computation |

---

## Pipeline steps

### 1. `prepare_training_frame(cadence_s=10)`

Loads every detection Parquet, downsamples to the requested cadence (aligned to whole-second boundaries; segment-boundary rows are force-kept), concatenates, and drops rows whose feature-nonnull ratio is below `FORECAST_MIN_NONNULL_FEATURE_RATIO`. The output frame is written once to `training_frame_{c}s.parquet` for reproducible training.

### 2. `train_all_horizons(cadence_s=10, algorithms=…)`

For each `(horizon, algorithm)` combination:

1. Assign each row a `train` / `val` / `test` label from the day-of-month in `datetime`.
2. Apply the 1-hour purge to the tail of the `train` and `val` blocks.
3. Build X / y arrays.
4. Fit the model on `X_train`.
5. Tune the decision threshold on val by sweeping `[0.05, 0.95]` in `0.05` steps and picking the argmax F1.
6. Compute PR-AUC / ROC-AUC / precision / recall / F1 / false-alarm rate / Brier score on both val and test.
7. Persist the model + threshold + frozen feature list via `joblib.dump`.

A single `feature_columns.json` manifest is written once per training run.

### 3. `evaluate_all(algorithm="xgb")`

Reloads the four horizon models for one algorithm, computes test-set metrics using the tuned thresholds, computes per-flare lead-time distribution against the Phase 5 master catalogue, and writes `forecast_report.md`.

---

## Streaming Inference API

```python
from tejas.forecasting import load_forecaster

f = load_forecaster(algorithm="xgb")   # or "logreg" / "lgbm"

# Per-row prediction (for the dashboard's playback loop):
probs = f.predict_row(feature_row)
# → {"prob_5m": 0.57, "prob_15m": 0.60, "prob_30m": 0.72, "prob_60m": 0.93}

# Vectorised prediction (for playback batch replay):
probs_df = f.predict_dataframe(feature_frame)

# Tuned thresholds (per horizon):
thresholds = f.decision_thresholds()
# → {5: 0.30, 15: 0.05, 30: 0.05, 60: 0.05}
```

`Forecaster.forbidden_columns` is exposed so downstream code can construct clean inputs:

```python
clean = feature_row.drop(list(f.forbidden_columns), errors="ignore")
probs = f.predict_row(clean)
```

---

## Metric reference

| Metric | Interpretation |
|---|---|
| **PR-AUC** (average precision) | Primary metric under class imbalance. Reflects precision/recall trade-off across all thresholds. |
| **ROC-AUC** | Threshold-independent ranking quality. Less informative at extreme class imbalance. |
| **Precision** | Fraction of positive predictions that are true positives at the chosen threshold. |
| **Recall** | Fraction of true positives caught. |
| **F1** | Harmonic mean of precision and recall. What the threshold is tuned to maximise on val. |
| **False alarm rate** | `FP / (FP + TN)`. The operational cost knob. |
| **Brier score** | Mean squared probability error. Lower = better-calibrated probabilities. |
| **Median lead time** | Seconds between the earliest probability crossing and the flare start. |

---

## No-future-data tripwires

The strict no-future-data constraint (see `CONTEXT.md`) is enforced by **five independent tripwires**, tested in `tests/test_forecasting.py::TestNoFutureData`:

1. **Feature list is frozen at training time.** `feature_columns.json` is written once; inference asserts row columns match. Any label column that entered by accident would fail the assertion.
2. **`FORECAST_DROP_COLUMNS` blacklist in sampling.** `select_feature_columns` never returns a column in the blacklist.
3. **`FORECAST_DROP_COLUMNS` blacklist in inference.** `Forecaster.predict_row` re-checks that no forbidden column is present in the input row.
4. **Split purge.** `FORECAST_SPLIT_PURGE_S` (1 hour) drops the last hour of `train` and `val` before the next split begins. Tested to be ≥ `max(horizons_min) × 60`.
5. **Noise-features leakage test.** A dedicated test replaces the entire feature matrix with uniform noise and asserts the test PR-AUC stays below 0.5. If any part of the pipeline were leaking future information, the model would score suspiciously well on noise.

---

## Running the pipeline

```
python scripts/run_forecasting.py --prepare               # build training frame (~2s for 30 days at 10s cadence)
python scripts/run_forecasting.py --train                 # fit all 12 models
python scripts/run_forecasting.py --train --algorithm xgb # fit only XGBoost's 4 horizons
python scripts/run_forecasting.py --evaluate              # test-set metrics + lead times
python scripts/run_forecasting.py --all                   # do everything
python scripts/run_forecasting.py --all --cadence 30      # 30s benchmark run
```

The typical May 2026 run takes:

- `--prepare`: ~2 s (I/O bound).
- `--train --algorithm all`: ~4 min (12 fits, LogReg dominates via LBFGS).
- `--evaluate`: ~1 s.

---

## Tests

`tests/test_forecasting.py` — 48 tests across 8 groups:

| Group | Tests | What it covers |
|---|---|---|
| `TestSampling` | 11 | Downsampling, segment-boundary preservation, blacklist enforcement, NaN dropping. |
| `TestSplits` | 6 | Split assignment, day-disjointness, purge behaviour. |
| `TestModels` | 8 | Factory outputs, `scale_pos_weight`, algorithm dispatch. |
| `TestTraining` | 3 | End-to-end horizon training, artifact writing, manifest roundtrip. |
| `TestEvaluation` | 8 | Metrics vs sklearn references, threshold tuning, lead-time correctness. |
| `TestInference` | 5 | Row/dataframe parity, label-leak & missing-feature rejection. |
| `TestNoFutureData` | 5 | The five tripwires above. |
| `TestPipelineIntegration` | 1 | Full prepare → train → evaluate on synthetic days into `tmp_path`. |

All 48 pass in ~14 s.
