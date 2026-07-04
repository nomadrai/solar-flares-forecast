"""
TEJAS Configuration
===================
Central configuration for paths, constants, and tunable parameters.
All configurable values are defined here to avoid magic numbers
scattered across the codebase.
"""

from pathlib import Path


# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(r"D:\SOLAR FLARE FORECAST\solar-flares-forecast")
DATA_DIR = PROJECT_DIR / "data"

# Raw FITS data (read-only)
RAW_DIR = DATA_DIR / "raw"
SOLEXS_DIR = RAW_DIR / "solexs"
HEL1OS_DIR = RAW_DIR / "hel1os"

# Processed output
PROCESSED_DIR = DATA_DIR / "processed"
UNIFIED_DIR = PROCESSED_DIR / "unified"

# EDA outputs (existing, untouched)
EDA_OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"


# ── Dataset Scope ────────────────────────────────────────────────────────────

YEAR = "2026"
MONTH = "05"

# Day 22 is missing from both instruments
MISSING_DAYS = {"22"}


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

# Output directory for clean processed data (separate from Phase 2 unified)
CLEAN_DIR = PROCESSED_DIR / "clean"

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
