# TEJAS: Temporal Evaluation and Joint Analysis of Solar-flares

## Project Overview

TEJAS is an AI-powered solar flare nowcasting and forecasting platform built using combined SoLEXS and HEL1OS X-ray observations from the Aditya-L1 mission.

The objective is to:
1. Detect solar flares as they occur (nowcasting).
2. Forecast the probability of future solar flares before they occur.
3. Generate an automated flare catalogue.
4. Provide explainable alerts.
5. Visualize X-ray activity through a web-based operational dashboard.

The project is intended to resemble a real space-weather monitoring system rather than a static machine-learning demonstration.

## Scientific Background

Solar flares are sudden releases of magnetic energy from the Sun that appear as bursts in X-ray observations.

The project uses:
* **SoLEXS** (Soft X-ray observations)
* **HEL1OS** (Hard X-ray observations)
* **Supplementary Datasets** (Optional integration of other relevant datasets to enhance predictive accuracy)

The system must combine information from both instruments. The project should exploit relationships between soft and hard X-ray behaviour instead of relying on a single source.

## Core Philosophy

*   **This project MUST be algorithm-driven.**
*   **The application must behave like a real-time operational monitoring system.**
*   **The application must never cheat.**
*   **The application must never use future information when making a prediction.**
*   **The application must never generate alerts simply because the future data already exists inside the dataset.**

This is a strict requirement.

## Non-Negotiable Scientific Integrity Requirement

The system must not access future observations during inference, forecasting, playback, simulation, evaluation, demonstration, or deployment. Training and inference must be treated differently.

### Training Phase
During training, the model is allowed to use complete historical datasets containing both pre-flare and flare observations. The purpose of training is to learn precursor signatures and temporal patterns that typically occur before solar flare events.

The training pipeline may analyze:
* Historical flare events
* Pre-flare behaviour
* Soft X-ray evolution
* Hard X-ray evolution
* Combined SoLEXS and HEL1OS trends
* Temporal relationships preceding flare onset

The model should learn what changes commonly occur minutes or hours before a flare. Training may therefore access historical future observations because those observations are part of the labelled dataset used for learning.

### Inference / Operational Phase
During playback, simulation, evaluation, demonstration, or deployment, the model may only consume observations that would have been available at the current simulated timestamp.

**For example:**
If the current playback time is 02:00:00, the model may only use observations available up to 02:00:00. It must not inspect observations from 02:00:01 or later.

Predictions must be generated solely from historical context available at the prediction time. Future observations must remain inaccessible until simulated time reaches them.

### Prohibited Behaviour
Any implementation that accesses future records, future labels, future flare timestamps, future classifications, future observations, or future ground-truth information during prediction is considered invalid.

Any forecasting result generated using information that would not have been available at the prediction timestamp is scientifically invalid and violates the core philosophy of TEJAS. The objective of TEJAS is to perform genuine solar flare forecasting based on learned precursor behaviour, not retrospective event recognition.

## Critical Real-Time Constraint

Assume a dataset contains 12 hours of observations. The system must simulate the passage of time.

*   **At hour 2:** The model may only use data that would have been available up to hour 2. It must not inspect data from hour 3, 4, 5, 6, etc.
*   **At hour 6:** The model may only use data available up to hour 6.
*   **At hour 10:** The model may only use data available up to hour 10.

**Future observations must remain inaccessible until simulated time reaches them.** The system must behave exactly like a real operational environment. Predictions must be generated using only historical context.

*   No look-ahead bias is allowed.
*   No future leakage is allowed.
*   No hidden access to future records is allowed.

This is one of the most important architectural requirements of TEJAS.

## Desired Simulation Behaviour

The web application should support playback of historical datasets as if they were arriving in real time.

**Example:**
Dataset Duration = 12 Hours. Playback begins at T=0. The dashboard progresses through the dataset using a timer.

As new records arrive:
* X-ray graphs update
* Detection algorithms execute
* Forecasting models execute
* Alerts may trigger
* Flare catalogue updates

The system should never know what happens later in the dataset until the playback reaches that point. The goal is to reproduce the behaviour of a real space-weather operations center.

## Project Development Phases and System Layers

### 1. Exploratory Data Analysis (EDA) Phase
**Responsibilities:** Visually understand typical solar flare light curves observed in soft and hard X-rays. Analyze baseline trends, identify precursor patterns, and assess data quality.
**Outputs:** Scientific insights, visual intuition of flare signatures, and baseline characteristics.

### 2. Data Ingestion Module
**Responsibilities:** Load SoLEXS data, load HEL1OS data, parse timestamps, parse flux values, synchronize observations.
**Outputs:** Unified time-series dataset.

### 3. Data Processing Module
**Responsibilities:** Missing value handling, signal cleaning, noise reduction, resampling, time synchronization, normalization.
**Outputs:** Clean aligned observations.

### 4. Feature Engineering Module
**Responsibilities:** Generate meaningful physical and statistical features.
**Examples:** Rolling averages, rolling variance, flux gradients, first derivatives, second derivatives, soft-to-hard X-ray ratio, growth rate, energy trends, temporal behaviour metrics.
The project should prioritize meaningful engineered features instead of relying purely on raw values.

### 5. Flare Detection Module (Nowcasting)
**Responsibilities:** Build algorithms to detect flares independently in both soft (SoLEXS) and hard (HEL1OS) X-rays. Combine these independent catalogues to generate a master catalogue. Detect flare start, detect flare peak, detect flare end, detect active flare state.
**Outputs:** Independent soft/hard X-ray catalogues, Master real-time flare catalogue.

### 6. Flare Classification Module
**Responsibilities:** Classify detected events.
**Potential classes:** B-class, C-class, M-class, X-class.
**Outputs:** Classified flare catalogue.

### 7. Forecasting Module
**Responsibilities:** Predict probability of flare occurrence within future time windows.
**Examples:** Next 5 minutes, next 10 minutes, next 15 minutes, next 30 minutes.
*The forecasting model must only use historical observations available at prediction time.*
**Outputs:** Probability scores, predicted flare risk levels.

### 8. Alert Engine
**Responsibilities:** Generate operational alerts.
**Examples:** High flare probability, active flare detected, major flare detected.
**Outputs:** Actionable alerts.

### 9. Explainability Module
**Responsibilities:** Explain why predictions were generated.
**Possible approaches:** Feature importance, SHAP values, decision summaries.
**Example:** Prediction = 82%. Reasons: Rapid increase in soft X-ray flux, elevated hard X-ray activity, similarity to previous flare patterns.
The system should avoid black-box behaviour whenever possible.

### 10. Flare Catalogue Database
**Responsibilities:** Maintain automatically generated flare records.
**Store:** Timestamp, start time, peak time, end time, classification, confidence, prediction history.
This catalogue must be generated automatically.

### 11. Visualization Dashboard
**Responsibilities:** Provide operational visibility.
**Suggested Panels:**
*   **Live Monitoring:** Soft X-ray graph, hard X-ray graph.
*   **Forecasting Panel:** Flare probability, risk level.
*   **Detection Panel:** Active flare status, current classification.
*   **Alerts Panel:** Generated warnings.
*   **Historical Catalogue:** Searchable flare database.

## FITS File Format Reference

This section documents the actual internal structure of the FITS files used by TEJAS, as confirmed by direct inspection during data exploration. This is essential for writing correct data loaders.

### SoLEXS (SDD2 only)

**Light Curve (`.lc.gz`):**
- Extension: `RATE`
- Columns: `TIME` (Unix seconds), `COUNTS` (counts/s in 2–22 keV)
- Cadence: 1 second. One file per day.

**Pulse Invariant Spectral File (`.pi.gz`):**
- Extension: `SPECTRUM`
- Columns: `TSTART` (Unix seconds), `TELAPSE`, `SPEC_NUM`, `CHANNEL`, `COUNTS` (2D: N_seconds × 340 channels), `EXPOSURE`
- Channel mapping: Channels 1–168 at ~47.6 eV width, channels 169–340 at ~95.2 eV width
- Usable energy range starts at channel 59 (~2.8 keV); below this the instrument response is uncertain
- Cadence: 1 second. One file per day.

**Good Time Interval (`.gti.gz`):**
- Extension: `GTI`
- Columns: `START` (Unix seconds), `STOP` (Unix seconds)

### HEL1OS (CdTe1, CdTe2, CZT1, CZT2)

**Light Curve (`lightcurve_*.fits`):**
- **Multiple extensions per file** — one per energy band, NOT a single data table
- **⚠ Each band has its own time array with different row counts** — do NOT assume shared timestamps across extensions
- CdTe bands (actual FITS extension names):
  - `CDTE1_LC_BAND_5.00KEV_TO_20.00KEV`, `CDTE1_LC_BAND_20.00KEV_TO_30.00KEV`
  - `CDTE1_LC_BAND_30.00KEV_TO_40.00KEV`, `CDTE1_LC_BAND_40.00KEV_TO_60.00KEV`
  - `CDTE1_LC_BAND_1.80KEV_TO_90.00KEV` (full range)
- CZT bands follow the same naming convention
- Columns in each extension: `MJD` (Modified Julian Date), `ISOT` (ISO timestamp string), `CTR` (count rate), `STAT_ERR` (Poisson error)
- **Time is MJD, not Unix time.** Convert: `Unix = (MJD - 40587) × 86400`
- Cadence: 1 second. Data is dump-wise (~12 hours per dump), not day-wise.
- **~85-88% of 1-second bins have zero counts** (expected for small-area detectors outside flares)

**Good Time Interval (`gti*.fits`):**
- For Level-1 data, GTI intervals span the entire dump duration (they become meaningful only in Level-2 data)

### Key Differences Between Instruments

| Property | SoLEXS | HEL1OS |
|---|---|---|
| Time format | Unix seconds | MJD |
| Data organization | Day-wise (1 file per day) | Dump-wise (~12h per dump, 2 dumps per day) |
| File format | Gzip-compressed FITS (`.gz`) | Uncompressed FITS (`.fits`) |
| Count rate column | `COUNTS` | `CTR` |
| Energy bands | Single broadband + 340-channel spectrum | 5 separate extensions per energy band |

## Technical Direction

**Preferred Stack:**
*   **Backend:** Python
*   **Data:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost, LightGBM, PyTorch (if necessary)
*   **Visualization:** Plotly
*   **Dashboard:** Streamlit
*   **Database:** SQLite initially


## What Makes TEJAS Different

The project must not be treated as a simple dashboard. The primary focus is:
1. Real-time simulation
2. Scientific signal processing
3. Flare detection
4. Forecasting accuracy
5. Explainable predictions
6. Operational realism

The dashboard exists to visualize the intelligence generated by the algorithms. **The algorithms are the product. The UI is only the presentation layer.**

Future AI agents must preserve this philosophy throughout development.

## Unique Selling Propositions (USPs)

### 1. Explainable AI for Solar Flares

TEJAS must provide explainable predictions instead of acting as a black-box model.

The system should not only predict the probability of a solar flare but should also explain why the prediction was made.

Possible explanations include:

*   Rapid increase in soft X-ray flux
*   Increased hard X-ray activity
*   Significant change in soft-to-hard X-ray ratio
*   Similarity to previously observed flare patterns
*   Important features identified through feature importance or SHAP values

The objective is to build trust and scientific interpretability into the forecasting system.

### 2. Multi-Horizon Forecasting

TEJAS should support predictions across multiple future time windows instead of only a single forecast horizon.

Examples:

*   Probability of flare in the next 5 minutes
*   Probability of flare in the next 15 minutes
*   Probability of flare in the next 30 minutes
*   Probability of flare in the next 1 hour

This should make TEJAS behave more like a real operational space-weather forecasting platform.

### 3. Flare Similarity Search

The system should maintain a searchable historical flare database and identify flare events that resemble the current conditions.

For a detected or forecasted event, TEJAS should be able to provide:

*   Similar historical flare events
*   Similar X-ray signatures
*   Similar precursor behaviour
*   Similar flare classifications

The objective is to provide context and scientific insight for operators and researchers.

## Features Offered by TEJAS

The platform should provide the following capabilities:

### 1. Real-Time Solar Monitoring

*   Live Soft X-ray monitoring
*   Live Hard X-ray monitoring
*   Combined visualisation

### 2. Real-Time Dataset Playback

*   Timer-based simulation
*   Progressive data arrival
*   Strict prevention of future-data leakage

### 3. Automated Flare Detection

*   Flare start detection
*   Flare peak detection
*   Flare end detection
*   Active flare identification

### 4. Flare Classification

Support all standard solar flare classes:

*   A-Class
*   B-Class
*   C-Class
*   M-Class
*   X-Class

### 5. Multi-Horizon Forecasting

Forecast probabilities for multiple future windows.

### 6. Probability-Based Predictions

Generate probability scores and confidence levels.

### 7. Explainable Predictions

Provide explanations for every prediction and alert.

### 8. Alert Engine

Support operational alert levels such as:

*   Normal
*   Watch
*   Warning
*   Critical

### 9. Historical Flare Catalogue

Maintain an automatically generated flare database.

### 10. Event Replay System

Replay historical flare events and observe model behaviour over time.

### 11. Prediction Timeline

Visualize how forecast confidence evolves before a flare occurs.

### 12. Flare Similarity Search

Find historical flare events that resemble the current situation.

### 13. Performance Metrics

Display metrics such as:

*   True Positive Rate
*   False Alarm Rate
*   Precision
*   Recall
*   F1 Score
*   Lead Time

### 14. Scientific Analytics

Provide advanced scientific analysis including:

*   Soft and Hard X-ray correlations
*   Precursor analysis
*   Feature importance analysis
*   Statistical summaries
*   Forecast evaluation

## Architectural Priority

The algorithms are the product. The dashboard is only the presentation layer.

All features must preserve scientific integrity and operational realism.

No feature should violate the strict no-future-data-leakage requirement during inference, playback, simulation, evaluation, or deployment.

Future AI agents must treat this as an inviolable constraint throughout all phases of development.
