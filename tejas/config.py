"""
TEJAS Configuration
===================
Central configuration for paths, constants, and tunable parameters.
All configurable values are defined here to avoid magic numbers
scattered across the codebase.
"""

from pathlib import Path


# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Raw FITS data (read-only)
RAW_DIR = DATA_DIR / "raw"
SOLEXS_DIR = RAW_DIR / "solexs"
HEL1OS_DIR = RAW_DIR / "hel1os"

# Processed output root
PROCESSED_DIR = DATA_DIR / "processed"

# EDA outputs (existing, untouched)
EDA_OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"


# ── Dataset Scope ────────────────────────────────────────────────────────────

YEAR = "2026"
MONTH = "05"

# Day 22 is missing from both instruments
MISSING_DAYS = {"22"}

# Per-month output subdirectories (type → year → month).
# Update YEAR / MONTH above to switch to a different month.
# Structure: data/processed/unified/YYYY/MM/
UNIFIED_DIR = PROCESSED_DIR / "unified" / YEAR / MONTH


# ── SoLEXS Configuration ────────────────────────────────────────────────────

# PI channel ranges for multi-band extraction (0-indexed numpy slices).
# Channels are 1-indexed in the SoLEXS manual; these are the Python slice
# equivalents.
#
# Channels 1-58 (~0-2.8 keV): excluded due to uncertain instrument response.
# See data_used.md Section 2 and CONTEXT.md.
PI_CH_LOW = slice(58, 105)      # ~2.8–5 keV  (cool thermal plasma)
PI_CH_MID = slice(105, 189)     # ~5–10 keV   (hotter plasma, transition)
PI_CH_HIGH = slice(189, 340)    # ~10–22 keV  (hottest thermal, early non-thermal)

# LC-to-PI ratio validation.
# LC includes all 340 channels; PI extraction uses channels 59-340.
# EDA measured median ratio ≈ 5.5 with correlation 0.90.
LC_PI_RATIO_EXPECTED = 5.5
LC_PI_RATIO_TOL = 1.0  # warn if ratio deviates beyond 5.5 ± 1.0


# ── HEL1OS Configuration ────────────────────────────────────────────────────

# Default resampling cadence for HEL1OS data (seconds).
# EDA showed ~87% zero-count bins at 1s. 10s produces cleaner signals
# while preserving temporal resolution for impulsive hard X-ray bursts.
HEL1OS_RESAMPLE_CADENCE_S = 10

# MJD-to-Unix conversion constant.
# Unix = (MJD - MJD_UNIX_EPOCH) × SECONDS_PER_DAY
MJD_UNIX_EPOCH = 40587.0
SECONDS_PER_DAY = 86400.0


# ── Unified Grid Configuration ──────────────────────────────────────────────

# The unified grid is at 1-second cadence.
# SoLEXS maps 1:1. HEL1OS (at 10s) is forward-filled between bins.
UNIFIED_GRID_CADENCE_S = 1


# ── Gap Handling (Tiered Strategy) ───────────────────────────────────────────
#
# EDA found inter-dump gaps of 22-650s. Tiered handling:
#   - Short (≤ GAP_SHORT_S):  routine telemetry gaps, forward-fill OK
#   - Medium (GAP_SHORT_S < gap ≤ GAP_MEDIUM_S): configurable, default NaN
#   - Long (> GAP_MEDIUM_S):  always NaN + gap mask
#
# The segment_id column increments at medium and long gaps so downstream
# code can process contiguous segments independently.

GAP_SHORT_S = 30       # ≤30s: forward-fill / interpolate
GAP_MEDIUM_S = 120     # 30-120s: NaN insertion (configurable)


# ── Zero-Count Handling ──────────────────────────────────────────────────────

# Guard value for divisions and log transforms on Poisson count data.
# eps = 0.5 is motivated by the Anscombe transform convention for
# variance-stabilising Poisson data. Applied via np.maximum(counts, eps)
# before any division or log operation.
ZERO_COUNT_EPS = 0.5


# ── Phase 3: Processing & Cleaning ──────────────────────────────────────────

# Output directory for clean processed data.
# Structure: data/processed/clean/YYYY/MM/
CLEAN_DIR = PROCESSED_DIR / "clean" / YEAR / MONTH

# Spike removal parameters (per instrument).
# window:          rolling-median window in seconds (must be odd)
# n_sigma:         MAD-based threshold multiplier for spike candidates
# min_consecutive: minimum consecutive flagged bins to be considered a REAL
#                  signal (flare) rather than a spike.  Isolated runs shorter
#                  than this are replaced.
SPIKE_REMOVAL = {
    "solexs": {"window": 61, "n_sigma": 7, "min_consecutive": 3},
    "cdte":   {"window": 61, "n_sigma": 5, "min_consecutive": 3},
    "czt":    {"window": 61, "n_sigma": 5, "min_consecutive": 3},
}

# Smoothing windows (seconds, must be odd).
# Wider windows reduce noise but blur fast transients.
SMOOTHING_WINDOWS = {
    # SoLEXS channels
    "solexs_counts":          5,
    "solexs_counts_low":      5,
    "solexs_counts_mid":      5,
    "solexs_counts_high":     5,
    "solexs_hardness_ratio": 11,   # wider: ratios amplify noise
    # HEL1OS channels
    "cdte_count_rate":        5,
    "czt_count_rate":        11,   # wider: noisier CZT data
}

# CZT background estimation
CZT_BG_QUIET_THRESHOLD = 18.0   # SoLEXS c/s; above → Sun is "active"
CZT_BG_WINDOW_S = 3600          # 1-hour sliding window for background est.
CZT_BG_PERCENTILE = 25          # percentile of quiet-period CZT rates
CZT_BG_SIGNIFICANCE = 3.0       # n × background noise → "above background"


# ── Phase 4: Feature Engineering ─────────────────────────────────────────

# Output directory for engineered features.
# Structure: data/features/YYYY/MM/
FEATURES_DIR = DATA_DIR / "features" / YEAR / MONTH

# Rolling window sizes (seconds) for temporal features.
# All windows are strictly trailing (causal) — no future data leakage.
FEATURE_ROLLING_WINDOWS = [30, 60, 300, 600]

# Gradient horizons (seconds) for flux gradient computation.
FEATURE_GRADIENT_HORIZONS = [30, 60, 300]

# Cross-correlation parameters (SoLEXS ↔ CdTe Neupert-effect detection)
XCORR_WINDOW_S = 300        # Trailing window for cross-correlation (seconds)
XCORR_MAX_LAG_S = 120       # Maximum lag to search (± seconds)
XCORR_COMPUTE_EVERY_S = 60  # Compute cross-corr every N seconds (forward-fill)

# Z-score and percentile rank windows for statistical features (seconds)
STAT_ZSCORE_WINDOWS = [300, 600]
STAT_PCTRANK_WINDOWS = [300, 600]

# CUSUM (Cumulative Sum) reset interval (seconds).
# Prevents unbounded accumulation during long quiet periods.
CUSUM_RESET_INTERVAL_S = 1800

# Z-score denominator floor.  Prevents divide-by-zero when trailing
# std is near zero (very stable quiet periods).
ZSCORE_STD_FLOOR = 0.1


# ── Phase 5: Flare Detection ────────────────────────────────────────────────

# Output directory for detection results.
# Structure: data/detection/YYYY/MM/
DETECTION_DIR = DATA_DIR / "detection" / YEAR / MONTH

# Adaptive threshold parameters.
# The detector uses a trailing rolling baseline (mean + std) and triggers
# when the signal's z-score exceeds DETECTION_SIGMA_START.  The flare ends
# when the z-score falls back below DETECTION_SIGMA_END.
DETECTION_BASELINE_WINDOW_S = 600     # 10-min trailing window for baseline
DETECTION_SIGMA_START = 3.5           # σ above baseline to trigger start
DETECTION_SIGMA_END = 1.5             # σ above baseline to declare end

# Event filtering.
# EDA catalogue: median duration = 11s, min = 6s.  MIN_DURATION must be
# below the shortest known event.
DETECTION_MIN_DURATION_S = 5          # discard events shorter than 5s
DETECTION_MIN_GAP_S = 60             # merge events within 60s of each other
DETECTION_MAX_DURATION_S = 10800     # 3h — discard pathologically long events

# CZT significance threshold.
# CZT detection only fires when czt_net_rate exceeds this multiple of
# the estimated background noise level (from Phase 3).
DETECTION_CZT_SIGNIFICANCE = 3.0

# Cross-matching window for building the master catalogue.
# A soft and hard X-ray event are considered the same physical event
# if their time ranges overlap or their peaks are within this window.
DETECTION_CROSS_MATCH_WINDOW_S = 300  # ±5 min

# Forecasting label horizons (minutes).
# Binary labels: "is there a flare start within h minutes?"
# These are retrospective labels for training only — never used at inference.
DETECTION_LABEL_HORIZONS_MIN = [5, 15, 30, 60]


# ── Phase 6: Flare Classification ───────────────────────────────────────────

# Output directory for classification results.
# Structure: data/classification/YYYY/MM/
CLASSIFICATION_DIR = DATA_DIR / "classification" / YEAR / MONTH

# Optional GOES reference data (only populated if user runs fetch_goes_data.py).
GOES_REFERENCE_DIR = DATA_DIR / "reference" / "goes" / YEAR / MONTH

# Persisted count → flux calibration artifact (produced by calibration module).
CALIBRATION_ARTIFACT = DATA_DIR / "reference" / "calibration" / "solexs_to_goes.json"

# Native TEJAS intensity bins.  Applied to peak_solexs_counts (c/s).
# Observed range across May 2026: 14–3667 c/s.  Bin edges scale ~4-5× per step,
# mirroring the order-of-magnitude structure of GOES letter classes.
INTENSITY_BIN_EDGES = [25, 100, 400, 1500]
INTENSITY_BIN_LABELS = ["faint", "weak", "moderate", "strong", "severe"]

# Log-scale ceiling for the intensity_score.  score = log(counts+1) / log(ceiling).
INTENSITY_SCORE_MAX_COUNTS = 4000

# GOES class flux thresholds (W/m^2, 1–8 Å channel).
# Standard NOAA definition: A < 1e-7, B 1e-7–1e-6, C 1e-6–1e-5, M 1e-5–1e-4, X ≥ 1e-4.
# The magnitude within a class is flux / threshold (rounded to 1 decimal).
GOES_CLASS_THRESHOLDS = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

# Calibration parameters.
CALIBRATION_MIN_SAMPLES = 20        # need ≥20 matched events to attempt a fit
CALIBRATION_MIN_R2 = 0.5            # reject fits below this
GOES_MATCH_TOLERANCE_S = 30         # ±30s search window around peak_time
