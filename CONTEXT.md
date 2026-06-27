# TEJAS: Temporal Evaluation and Joint Analysis of Solar-flares

## Agent Guidelines
**⚠ IMPORTANT:** Do not make any changes until you have 95% confidence in what you need to build. Ask me follow-up questions until you reach that confidence.

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

### 5. Flare Detection Module (Nowcasting)
Detect flare states (start, peak, end, active state) independently in both soft and hard X-rays, combining them to generate a master real-time flare catalogue.

### 6. Flare Classification Module
Classify detected solar flare events into standard categories (e.g., A, B, C, M, X-class) based on their peak flux.

### 7. Forecasting Module
Predict the probability of a flare occurring within multiple future time windows (e.g., next 5, 15, 30 mins) strictly using historical context available at the prediction time.

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

## Unique Selling Propositions (USPs)

### 1. Explainable AI for Solar Flares
TEJAS avoids black-box modelling by explaining its predictions. It justifies forecasts based on physical signatures like rapid soft X-ray increases or soft-to-hard X-ray ratios, building scientific trust.

### 2. Multi-Horizon Forecasting
The platform behaves like a real space-weather center by predicting flare probabilities across multiple future windows simultaneously (e.g., next 5, 15, 30, and 60 minutes).

### 3. Flare Similarity Search
TEJAS maintains a historical database and can identify past flare events that resemble current conditions in real time. This provides operators with valuable context and scientific insight during active events.

## Technical Stack
*   **Backend & Data:** Python, Pandas, NumPy
*   **ML:** Scikit-learn, XGBoost, LightGBM, PyTorch (if needed)
*   **Visualization & UI:** Plotly, Streamlit
*   **Database:** SQLite

## Architectural Priority

The algorithms are the product. The dashboard is only the presentation layer.

All features must preserve scientific integrity and operational realism.

No feature should violate the strict no-future-data-leakage requirement during inference, playback, simulation, evaluation, or deployment.

Future AI agents must treat this as an inviolable constraint throughout all phases of development.
