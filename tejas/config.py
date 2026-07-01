"""
TEJAS Configuration
===================
Central configuration for paths, constants, and tunable parameters.
All configurable values are defined here to avoid magic numbers
scattered across the codebase.
"""

from pathlib import Path


# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(r"C:\Users\aadit\Documents\Projects\TEJAS")
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
