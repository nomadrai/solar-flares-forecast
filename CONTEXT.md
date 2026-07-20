# TEJAS: Temporal Evaluation and Joint Analysis of Solar-flares

## Agent Guidelines
**⚠ IMPORTANT:** Do not make any changes until you have 95% confidence in what you need to build. Ask me follow-up questions until you reach that confidence.

**⚠ Keep the docs in sync:** `CONTEXT.md` (this file) and `data_used.md` are living documents. **They must be updated on every major project update** — a completed phase, a new module/data source, a changed file format or directory layout, or a new reference/optimization document. Treat a stale doc as a defect and update it in the same change that caused the drift. See *Document Maintenance* at the end of this file.

## Project Status (as of the latest update)
| Phase | Module | Status |
|---|---|---|
| 1 | Exploratory Data Analysis | ✅ Complete (`outputs/eda/`) |
| 2 | Data Ingestion | ✅ Complete (`tejas/ingestion/`, `tejas/loaders/`) |
| 3 | Data Processing (incl. Kalman smoothing, CZT background) | ✅ Complete (`tejas/processing/`) |
| 4 | Feature Engineering (81 features + SEP/AR-proxy/long-context) | ✅ Complete (`tejas/features/`) |
| 5 | Flare Detection / Nowcasting | ✅ Complete (`tejas/detection/`) |
| 6 | Flare Classification (native + GOES calibration) | ✅ Complete (`tejas/classification/`) |
| 7 | Forecasting (multi-horizon, +v2 optimization) | ✅ Complete (`tejas/forecasting/`) |
| 8 | Alert Engine (confirmation gate + cooldown) | ✅ Implemented (`tejas/alerts/confirmation.py`) |
| 9–11 | Explainability, Catalogue DB, Dashboard | ◻ Pending |

**Optimization track:** A precision/recall optimization effort is documented in `optimization_plan.md`, `optimization_plan_v2.md`, `optimization_report.md`, and `optimization_plan_v3.md`. The v2 stage rebuilds forecasting targets from a *credible* event subset (soft-detected or GOES-confirmed) — see `tejas/detection/credible.py`, `tejas/features/long_context.py`, and the v2 config block in `tejas/config.py`. The v3 plan (measured diagnosis + 12-area coverage + P0–P8 phased roadmap) is the current implementation target. External detection recall vs GOES truth is tracked in `goes_validation_report.md`.

## Project Overview
TEJAS is an AI-powered solar flare nowcasting and forecasting platform using combined SoLEXS (Soft X-ray) and HEL1OS (Hard X-ray) observations from the Aditya-L1 mission.
**Goals:** Detect flares (nowcasting), forecast probabilities, auto-generate flare catalogues, provide explainable alerts, and visualize activity via an operational dashboard.

## Core Philosophy & Scientific Integrity
*   **Algorithm-Driven Operational Realism:** TEJAS is not a static machine-learning demo; it must simulate a real-time space-weather operations center. The algorithms are the core product, and the UI is purely the presentation layer.
*   **Strict No-Future-Data Constraint:** The application must never cheat by using future information when making predictions.
    *   **Training Phase:** The model may use complete historical datasets to learn temporal patterns and precursor signatures that occur before a flare.
    *   **Inference/Playback Phase:** During playback or simulation, the model may only consume observations available at the current simulated timestamp (e.g., at hour 2, data from hour 3+ is inaccessible). Predictions must rely solely on historical context. Any look-ahead bias or future data leakage is scientifically invalid.

## Project Development Phases and System Layers

### 1. Exploratory Data Analysis (EDA)
Visually analyze baseline trends and precursor patterns in soft and hard X-ray light curves to build scientific intuition and assess data quality.

### 2. Data Ingestion Module
Load and parse SoLEXS and HEL1OS data. Accurately extract timestamps and flux values, and synchronize observations from both instruments into a unified dataset.

### 3. Data Processing Module
Clean and align observations by handling missing values, reducing noise, resampling, synchronizing time arrays, and normalizing the signals.

### 4. Feature Engineering Module
Generate physically meaningful features (e.g., rolling averages, flux gradients, soft-to-hard X-ray ratios, energy trends) instead of relying purely on raw values.

> **Reference:** `feature_catalog.md` is the complete reference for all 81 engineered features — names, formulas, units, and physical meaning.


### 5. Flare Detection Module (Nowcasting)
Detect flare states (start, peak, end, active state) independently in both soft and hard X-rays, combining them to generate a master real-time flare catalogue.

### 6. Flare Classification Module
Classify detected solar flare events into standard categories (e.g., A, B, C, M, X-class) based on their peak flux. TEJAS assigns a **native intensity class** (faint/weak/moderate/strong/severe) directly from SoLEXS peak counts, and — when optional GOES XRS data is present — an empirically calibrated **GOES-equivalent letter class** via `data/reference/calibration/solexs_to_goes.json`.

> **Reference:** `classification_catalog.md` is the complete reference for the classification columns and GOES calibration methodology used by this module.

### 7. Forecasting Module
Predict the probability of a flare occurring within multiple future time windows (e.g., next 5, 15, 30 mins) strictly using historical context available at the prediction time.

> **Reference:** `forecasting_catalog.md` is the complete reference for the forecasting module — model architectures, training/split protocol, inference API, and the no-future-data tripwires enforced by the code.

### 8. Alert Engine
Generate actionable operational alerts based on the model's output, such as high flare probability warnings or active flare detections.

### 9. Explainability Module
Provide transparent reasoning for predictions to avoid black-box behaviour (e.g., using feature importance, SHAP values, or specific precursor signatures).

### 10. Flare Catalogue Database
Automatically maintain records of detected flares, storing timestamps, classifications, prediction history, and confidence scores.

### 11. Visualization Dashboard
Provide operational visibility through live monitoring panels, forecasting and detection metrics, generated alerts, and a searchable historical catalogue.

## FITS File Format Reference

### SoLEXS (SDD2 only)
*   **Time format:** Unix seconds. One file per day.
*   **Light Curve (`.lc.gz`):** `RATE` extension. Columns: `TIME`, `COUNTS` (2–22 keV). 1-second cadence.
*   **Spectrum (`.pi.gz`):** `SPECTRUM` extension. Columns: `TSTART`, `COUNTS` (2D: N_seconds × 340 channels). Useful energy range starts at ~2.8 keV.
*   **Good Time Interval (`.gti.gz`):** `GTI` extension. Columns: `START`, `STOP`.

### HEL1OS (CdTe1, CdTe2, CZT1, CZT2)
*   **Time format:** MJD (Modified Julian Date). Convert to Unix: `(MJD - 40587) × 86400`.
*   **Light Curve (`lightcurve_*.fits`):** Dump-wise data (~12h per dump). Uncompressed FITS. **⚠ Multiple extensions per file** (one per energy band). **⚠ Each band has its own time array** with different row counts. Columns: `MJD`, `ISOT`, `CTR`, `STAT_ERR`.
*   **Data characteristics:** 1-second cadence. ~85-88% of bins have zero counts outside flares.
*   **Good Time Interval (`gti*.fits`):** Intervals span the entire dump for Level-1 data.

### External Reference: NOAA GOES (optional ground truth)
*   **Source:** NOAA NCEI archive. Stored under `data/reference/goes/YYYY/MM/`. Fetched by `scripts/fetch_goes_data.py` (XRS 1s flux CSVs) and `scripts/fetch_goes_flares.py` (`goes_flare_events.csv`).
*   **Role:** Empirical SoLEXS-count → GOES-flux calibration (Phase 6) and independent confirmation of TEJAS detections (v2 forecasting). Fully optional — the core pipeline runs without it.
*   **Reference:** `data_used.md` describes both GOES products in detail.

## Unique Selling Propositions (USPs)

### 1. Explainable AI for Solar Flares
TEJAS avoids black-box modelling by explaining its predictions. It justifies forecasts based on physical signatures like rapid soft X-ray increases or soft-to-hard X-ray ratios, building scientific trust.

### 2. Multi-Horizon Forecasting
The platform behaves like a real space-weather center by predicting flare probabilities across multiple future windows simultaneously (e.g., next 5, 15, 30, and 60 minutes).

### 3. Flare Similarity Search
TEJAS maintains a historical database and can identify past flare events that resemble current conditions in real time. This provides operators with valuable context and scientific insight during active events.

## Technical Stack
*   **Backend & Data:** Python, Pandas, NumPy, Astropy (FITS), PyArrow (Parquet)
*   **External data:** requests, xarray, netCDF4 (NOAA GOES fetch)
*   **ML:** Scikit-learn, XGBoost, LightGBM, PyTorch (if needed)
*   **Visualization & UI:** Plotly, Streamlit
*   **Database:** SQLite
*   **Testing:** pytest (`tests/`)

## Architectural Priority

The algorithms are the product. The dashboard is only the presentation layer.

All features must preserve scientific integrity and operational realism.

No feature should violate the strict no-future-data-leakage requirement during inference, playback, simulation, evaluation, or deployment.

Future AI agents must treat this as an inviolable constraint throughout all phases of development.

## Document Maintenance

`CONTEXT.md` and `data_used.md` are the two canonical living documents for this project. **Both must be updated on every major update**, including:

- A pipeline phase is completed or a new module is added (update the *Project Status* table).
- A new data source, reference artifact, or optimization document is introduced.
- A file format, column, directory layout, or config contract changes.
- A USP, technical-stack choice, or architectural constraint changes.

The rule: whatever change causes drift, fix the affected doc **in the same change**. A stale `CONTEXT.md` or `data_used.md` is treated as a defect, not a cosmetic issue. Companion reference files (`feature_catalog.md`, `classification_catalog.md`, `forecasting_catalog.md`) follow the same principle for their respective modules.
