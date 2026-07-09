# TEJAS — Project Task Tracker

This file tracks all phases and tasks for the TEJAS project. Use this to jump back into development at any session.

**Legend:** `[x]` = done, `[/]` = in progress, `[ ]` = not started

---

## Phase 0: Project Setup & Data Acquisition
- [x] Create project repository and initial structure
- [x] Write CONTEXT.md (project philosophy, architecture, constraints)
- [x] Write data_used.md (file-by-file reference, data flow diagram)
- [x] Download SoLEXS data from PRADAN (May 2026, 30 days)
- [x] Download HEL1OS data from PRADAN (May 2026, 60 dumps)
- [x] Extract SoLEXS SDD2 files to `data/raw/solexs/YYYY/MM/DD/` (script: `scripts/extract_solexs_data.py`)
  - [x] 90 files: 30 × (.lc.gz + .pi.gz + .gti.gz), 213 MB total
- [x] Extract HEL1OS lightcurve + GTI files to `data/raw/hel1os/YYYY/MM/DD/dump_HHMMSS/` (script: `scripts/extract_hel1os_data.py`)
  - [x] 480 files: 60 dumps × 8 files, 2.67 GB total
- [x] Discard unnecessary data (SDD1, spectral files, event lists, housekeeping, CalDB)
- [x] Document FITS file internal structures in CONTEXT.md
- [x] Configure .gitignore (exclude data/, __pycache__, etc.)
- [x] Create data viewer script (`scripts/view_solexs_hel1os_data.py`)

---

## Phase 1: Exploratory Data Analysis (EDA)
- [x] **SoLEXS Light Curve Exploration**
  - [x] Plot full-month overview of daily SoLEXS light curves (all 30 days)
  - [x] Identify days with significant flare activity vs quiet days
  - [x] Characterize baseline (quiet Sun) count rates
  - [x] Identify and catalogue visible flare events (start, peak, end times)
  - [x] Assess data gaps (missing day 22, GTI coverage per day)
- [x] **SoLEXS Spectral (PI) Exploration**
  - [x] Plot spectrograms (time vs energy channel vs counts) for flare days
  - [x] Verify multi-band extraction (low/mid/high) produces sensible light curves
  - [x] Examine hardness ratio behaviour before, during, and after flares
  - [x] Check for spectral hardening as a precursor signal
- [x] **HEL1OS CdTe Exploration**
  - [x] Plot CdTe1 and CdTe2 light curves, compare for consistency
  - [x] Overlay CdTe with SoLEXS on same time axis to observe soft/hard X-ray relationship
  - [x] Check individual energy bands (5-20, 20-30, 30-40, 40-60 keV) during flares
  - [x] Verify dump-to-dump continuity (no gaps or overlaps between consecutive dumps)
- [x] **HEL1OS CZT Exploration**
  - [x] Plot CZT1 and CZT2 light curves, characterize background noise level
  - [x] Identify flares where CZT signal clearly rises above background
  - [x] Compare CZT signal with CdTe and SoLEXS for the same events
- [x] **Cross-Instrument Analysis**
  - [x] Time-align SoLEXS (Unix time) and HEL1OS (MJD) on common UTC axis
  - [x] Plot combined 3-panel view: SoLEXS + CdTe + CZT for selected flares
  - [x] Observe Neupert effect (hard X-ray peak leading soft X-ray peak)
  - [x] Quantify typical time offsets between soft and hard X-ray peaks
- [x] **Data Quality Report**
  - [x] Summarize data availability per day (valid hours from GTIs)
  - [x] Flag any anomalous days (unexpected count rates, instrument issues)
  - [x] Document findings in an EDA notebook or report

---

## Phase 2: Data Ingestion Module
- [x] **SoLEXS Loader**
  - [x] Write function to load LC file → DataFrame with `time_unix`, `counts`, `datetime`
  - [x] Write function to load PI file → DataFrame with `time_unix`, `counts_low`, `counts_mid`, `counts_high`, `hardness_ratio`
  - [x] Write function to load GTI file → DataFrame with `start`, `stop`
  - [x] Write function to apply GTI mask to LC/PI DataFrames
- [x] **HEL1OS Loader**
  - [x] Write function to load CdTe light curves (both detectors, all dumps for a day) → combined DataFrame
  - [x] Write function to load CZT light curves (both detectors, all dumps for a day) → combined DataFrame
  - [x] Write function to load HEL1OS GTI files
  - [x] Handle MJD-to-Unix time conversion
  - [x] Handle multi-dump merging (concatenate dumps, remove overlaps if any)
  - [x] Extract per-band light curves (not just full range) for feature engineering
- [x] **Unified Dataset Builder**
  - [x] Time-synchronize SoLEXS and HEL1OS onto a common 1-second UTC grid
  - [x] Handle missing data (NaN for gaps, forward-fill or interpolation strategy)
  - [x] Combine into a single unified DataFrame per day
  - [x] Save unified dataset to `data/processed/` (Parquet or CSV)
- [x] **Batch Processing Script**
  - [x] Process all 30 days in batch, produce unified daily files
  - [x] Generate a data availability summary (which days have complete data from both instruments)

---

## Phase 3: Data Processing & Cleaning
- [x] **Signal Cleaning**
  - [x] Apply GTI filtering to discard invalid observations
  - [x] Handle data gaps between GTI intervals (NaN, segmentation, or interpolation)
  - [x] Remove outliers / spurious spikes (cosmic ray hits, telemetry glitches)
  - [x] Smooth noisy signals where appropriate (configurable window)
- [x] **Normalization & Resampling**
  - [x] Normalize count rates if needed (or keep raw and normalize during feature engineering)
  - [x] Support resampling to different cadences (1s, 10s, 30s, 60s) for different use cases
- [x] **CZT Background Handling**
  - [x] Estimate CZT background level during quiet periods
  - [x] Implement background subtraction or background-aware features
- [x] **Data Validation**
  - [x] Verify time monotonicity (no backwards jumps)
  - [x] Verify no duplicate timestamps
  - [x] Cross-check SoLEXS LC total counts vs PI band sum for consistency
  - [x] Save clean processed dataset to `data/processed/`

---

## Phase 4: Feature Engineering
- [x] **Temporal Features (per instrument)**
  - [x] Rolling mean (multiple windows: 30s, 60s, 300s, 600s)
  - [x] Rolling variance / standard deviation
  - [x] First derivative (flux rate of change)
  - [x] Second derivative (acceleration)
  - [x] Flux gradient over multiple time horizons
  - [x] Rate of rise / rate of decay
- [x] **Spectral Features (SoLEXS PI-derived)**
  - [x] Multi-band counts (low, mid, high)
  - [x] Hardness ratio (high/low) and its rolling derivative
  - [x] Band ratios (mid/low, high/mid)
  - [x] Spectral slope changes
- [x] **Cross-Instrument Features**
  - [x] Soft-to-hard X-ray ratio (SoLEXS / CdTe)
  - [x] SoLEXS-to-CZT ratio
  - [x] CdTe-to-CZT ratio
  - [x] Temporal lag between soft and hard X-ray changes
  - [x] Cross-correlation features
- [x] **Statistical Features**
  - [x] Z-score relative to recent baseline
  - [x] Percentile rank within recent history
  - [x] Change-point detection features (CUSUM)
- [x] **Feature Storage**
  - [x] Save engineered feature dataset to `data/features/`
  - [x] Document all features with descriptions, units, and physical meaning (`feature_catalog.md`)

---

## Phase 5: Flare Detection Module (Nowcasting)
- [x] **Threshold-Based Detection**
  - [x] Define adaptive threshold for flare start (e.g., N-sigma above rolling baseline)
  - [x] Detect flare peak (local maximum after start)
  - [x] Detect flare end (return to within threshold of baseline)
  - [x] Handle nested/overlapping flares
- [x] **Independent Catalogues**
  - [x] Generate soft X-ray flare catalogue (from SoLEXS)
  - [x] Generate hard X-ray flare catalogue (from HEL1OS CdTe)
  - [x] Generate high-energy catalogue (from HEL1OS CZT, if signal above background)
- [x] **Master Catalogue**
  - [x] Cross-match soft and hard X-ray detections
  - [x] Merge into unified master flare catalogue with columns: start, peak, end, duration, peak_counts, classification
- [x] **Evaluation**
  - [x] Compare detected events against known flare lists (e.g., GOES flare catalogue for the same period)
  - [x] Tune detection parameters

---

## Phase 6: Flare Classification
- [ ] Classify detected flares by peak soft X-ray flux (A, B, C, M, X classes)
- [ ] Map SoLEXS SDD2 count rates to GOES flux equivalents (calibration/correlation)
- [ ] Add classification to the master flare catalogue
- [ ] Validate classifications against GOES data if available

---

## Phase 7: Forecasting Module
- [ ] **Training Data Preparation**
  - [ ] Label time series with "flare within next N minutes" (binary label for multiple horizons)
  - [ ] Create train/validation/test splits (chronological, no data leakage)
  - [ ] Balance classes (flare vs non-flare periods)
- [ ] **Model Development**
  - [ ] Baseline model: logistic regression on engineered features
  - [ ] Tree-based models: XGBoost / LightGBM
  - [ ] Evaluate need for sequence models (LSTM, Transformer) — only if tree models insufficient
  - [ ] Multi-horizon output: predict probability for 5, 15, 30, 60 minute windows
- [ ] **Training & Evaluation**
  - [ ] Train with strict temporal separation (no future leakage)
  - [ ] Evaluate: precision, recall, F1, false alarm rate, lead time
  - [ ] Feature importance analysis
  - [ ] Save trained models to `models/`
- [ ] **Inference Pipeline**
  - [ ] Build streaming inference function (input: current + historical features → output: probabilities)
  - [ ] Enforce no-future-data constraint during inference

---

## Phase 8: Alert Engine
- [ ] Define alert levels: Normal, Watch, Warning, Critical
- [ ] Map forecast probabilities to alert levels (configurable thresholds)
- [ ] Generate alert messages with context (current readings, probability, contributing features)
- [ ] Alert history log

---

## Phase 9: Explainability Module
- [ ] Compute feature importance (from tree models or SHAP)
- [ ] Generate human-readable prediction explanations
  - [ ] Example: "Flare probability 82% — soft X-ray rising rapidly, hardness ratio increased 40% in 5 min"
- [ ] Link explanations to alert engine output
- [ ] Flare similarity search: find historical events with similar feature profiles

---

## Phase 10: Flare Catalogue Database
- [ ] Design database schema (SQLite)
- [ ] Store: timestamp, start, peak, end, classification, confidence, prediction history, explanation
- [ ] Automatically populate from detection + classification modules
- [ ] Query interface for historical lookup

---

## Phase 11: Visualization Dashboard
- [ ] **Core Panels**
  - [ ] Live soft X-ray monitoring graph (SoLEXS)
  - [ ] Live hard X-ray monitoring graph (HEL1OS CdTe + CZT)
  - [ ] Forecasting panel (probability gauges, risk level)
  - [ ] Detection panel (active flare status, current classification)
  - [ ] Alert panel (generated warnings with timestamps)
- [ ] **Playback System**
  - [ ] Timer-based dataset playback (simulate real-time data arrival)
  - [ ] Configurable playback speed
  - [ ] Strict no-future-data enforcement during playback
  - [ ] Progress bar showing position in dataset
- [ ] **Historical Views**
  - [ ] Searchable flare catalogue table
  - [ ] Event replay (jump to a historical flare and watch model behaviour)
  - [ ] Prediction timeline (how forecast confidence evolved before each event)
- [ ] **Analytics**
  - [ ] Performance metrics panel (TPR, FAR, precision, recall, F1)
  - [ ] Feature importance visualization
  - [ ] Flare similarity search UI
- [ ] **Technical**
  - [ ] Streamlit app structure
  - [ ] Responsive layout
  - [ ] Deploy / run instructions

---

## Current Status

**Last updated:** 2026-07-09

| Phase | Status | Notes |
|---|---|---|
| Phase 0: Setup & Data | ✅ Complete | All raw data in place, 30 days SoLEXS + 60 dumps HEL1OS |
| Phase 1: EDA | ✅ Complete | 23 flare-day plots, data quality report, preliminary flare catalogue, GTI/noise/availability CSVs |
| Phase 2: Data Ingestion | ✅ Complete | 30/30 days ingested, 60/60 tests passing, unified Parquet files in data/processed/unified/ |
| Phase 3: Data Processing | ✅ Complete | 30/30 days processed, 23/23 tests passing, clean Parquet files in data/processed/clean/. Spike removal (62 spikes across 30 days), rolling-median smoothing, CZT background subtraction (avg ~17 c/s), log1p + percentile normalization. |
| Phase 4: Feature Engineering | ✅ Complete | 30/30 days processed, 115/115 tests passing (32 new + 83 existing). 81 features added per day (45 temporal + 13 spectral + 8 cross-instrument + 15 statistical), 121 total columns. Feature Parquet files in data/features/. Avg 5.1s/day, 153s total. |
| Phase 5: Flare Detection | ✅ Complete | 30/30 days processed, 40/40 tests passing. 1751 master events (471 confirmed, 144 soft-only, 1136 hard-only). 13/13 EDA recall (100%). Adaptive threshold (3.5σ start, 1.5σ end), segment-aware baseline, per-timestep labels (flare_id, flare_phase, forecasting labels). Avg 2.7s/day, 82s total. |
| Phase 6: Classification | ⬜ Not started | — |
| Phase 7: Forecasting | ⬜ Not started | — |
| Phase 8: Alert Engine | ⬜ Not started | — |
| Phase 9: Explainability | ⬜ Not started | — |
| Phase 10: Catalogue DB | ⬜ Not started | — |
| Phase 11: Dashboard | ⬜ Not started | — |

**Next step:** Phase 6 (Flare Classification) — classify detected flares by peak soft X-ray flux (A, B, C, M, X classes), map SoLEXS count rates to GOES flux equivalents.

