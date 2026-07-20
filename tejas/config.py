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


# ── Phase 7: Forecasting ────────────────────────────────────────────────────

# Output directories.  Model artifacts go under outputs/ (git-ignored);
# derived training frames + evaluation reports go under data/forecasting/.
FORECAST_DIR = DATA_DIR / "forecasting" / YEAR / MONTH
MODELS_DIR = PROJECT_DIR / "outputs" / "models" / YEAR / MONTH

# Training cadence.  10s is the primary cadence (keeps ~250k rows across 30
# days, preserves precursor dynamics without redundant 1s samples).  30s is
# an optional benchmark run.  1s cadence is preserved upstream but not used
# for training (excessive redundancy, positive-class dilution).
FORECAST_CADENCE_S = 10
FORECAST_BENCHMARK_CADENCE_S = 30

# Multi-horizon binary forecast targets (minutes).  One model per horizon.
# Mirrors DETECTION_LABEL_HORIZONS_MIN so downstream label columns align.
FORECAST_LABEL_HORIZONS_MIN = [5, 15, 30, 60]

# Chronological blocked split.  Day strings ("01".."31").  Day 22 is missing
# and is intentionally excluded from every split.  Ordering is train → val →
# test with no overlap.
FORECAST_TRAIN_DAYS = [f"{d:02d}" for d in range(1, 21)]        # 01..20
FORECAST_VAL_DAYS = ["21", "23", "24", "25", "26"]              # 22 missing
FORECAST_TEST_DAYS = ["27", "28", "29", "30", "31"]

# Purge gap (seconds) applied to the tail of each split before the next
# split begins.  Prevents a 60-min horizon label at the end of train from
# depending on truth from the beginning of val.  Set slightly above the
# longest horizon (60 min) to be safe.
FORECAST_SPLIT_PURGE_S = 3600

# Columns that must NEVER enter the feature matrix.  Enforced independently
# in sampling.select_feature_columns AND inference.Forecaster.predict_row
# so a bug in one choke point does not create a leak.
FORECAST_DROP_COLUMNS = {
    "datetime", "time_unix", "segment_id",
    "flare_id", "is_flare_active", "flare_phase",
    "label_minutes_to_next_flare",
    "label_flare_within_5m", "label_flare_within_15m",
    "label_flare_within_30m", "label_flare_within_60m",
    # Fix 4: significant-flare labels
    "label_sig_flare_within_5m", "label_sig_flare_within_15m",
    "label_sig_flare_within_30m", "label_sig_flare_within_60m",
    # v2: backed-up original (all-event) labels — pure leakage if kept
    "label_minutes_to_next_flare_all",
    "label_flare_within_5m_all", "label_flare_within_15m_all",
    "label_flare_within_30m_all", "label_flare_within_60m_all",
    # Metadata
    "gti_start",
    # v3 P0: per-instrument metadata that leaked into the v2 manifest.
    # Segment IDs count telemetry dumps — they encode *which part of the
    # month* a row is from (day identity), so the model memorizes active
    # days instead of learning precursors (cdte_segment_id was the #2 XGB
    # feature at 30 min).  stat_err is an instrument-noise channel, not
    # physics.  A suffix guard (FORECAST_DROP_COLUMN_PATTERNS) additionally
    # blocks any future *_segment_id / *_stat_err variant.
    "cdte_segment_id", "czt_segment_id",
    "cdte_stat_err", "czt_stat_err",
}

# Suffix patterns enforced on top of FORECAST_DROP_COLUMNS in
# sampling.select_feature_columns and inference.Forecaster: a column whose
# name ends with one of these can never enter the feature matrix, even if
# someone forgets to add its literal name to the blacklist above.
FORECAST_DROP_COLUMN_PATTERNS = ("_segment_id", "_stat_err")

# Rolling-window features produce NaN for the first N-1 samples of each
# segment.  Rows missing more than (1 - this ratio) of features are dropped
# before training rather than forward-filled (matches Phase 5 convention).
FORECAST_MIN_NONNULL_FEATURE_RATIO = 0.95

# Model hyperparameters.  Trees are moderately deep with explicit L1/L2
# regularisation and min_child_weight to guard against over-fitting on
# small folds.  Higher n_estimators is intentional — early stopping caps
# the effective depth using val (see FORECAST_EARLY_STOPPING_ROUNDS).
FORECAST_XGB_PARAMS = dict(
    max_depth=7, n_estimators=1200, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, colsample_bylevel=0.85,
    min_child_weight=5.0, gamma=0.1,
    reg_alpha=0.05, reg_lambda=1.5,
    objective="binary:logistic", tree_method="hist", eval_metric="aucpr",
    n_jobs=-1,
)
FORECAST_LGBM_PARAMS = dict(
    num_leaves=127, max_depth=-1, n_estimators=1500, learning_rate=0.03,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
    min_child_samples=40, min_split_gain=0.0,
    reg_alpha=0.05, reg_lambda=1.5,
    objective="binary", metric="average_precision",
    n_jobs=-1,
)
FORECAST_LOGREG_PARAMS = dict(
    C=0.5, class_weight="balanced",
    solver="lbfgs", max_iter=3000,
)

# Class-weighting policy for tree models.  The raw ``n_neg / n_pos`` ratio
# tends to collapse predictions to trivially positive on the majority
# class; ``sqrt`` softens that.  The cap prevents one horizon from
# dominating the fit when the imbalance is extreme.
FORECAST_USE_SQRT_SPW = True
FORECAST_SPW_CAP = 12.0

# Early stopping for tree models.  Uses the val split as the eval set.
# ``0`` disables early stopping.  Sits well below n_estimators to allow
# genuine convergence to be caught by the plateau detector.
FORECAST_EARLY_STOPPING_ROUNDS = 60

# Probability calibration.  Applied to raw model probs using the val
# split as the reference distribution.  Options:
#   ``"none"``      — pass through the raw probabilities.
#   ``"isotonic"``  — non-parametric monotonic calibration (recommended
#                     for tree ensembles; needs enough val positives).
#   ``"sigmoid"``   — Platt scaling (single-parameter, more stable when
#                     positives are scarce).
# When the requested calibrator cannot be fit (e.g. no val positives) the
# training path silently falls back to ``"none"`` for that horizon.
# v3 P0b: switched isotonic → sigmoid.  Isotonic's step function collapsed
# val probabilities to 3–36 unique values (60m: 37 raw scores → 3), leaving
# the event-level threshold tuner almost no operating points — most of the
# duty-cycle budget was unreachable.  Sigmoid is strictly monotonic, so
# every unique raw score stays a distinct threshold candidate.
FORECAST_CALIBRATION_METHOD = "sigmoid"

# Evaluation & threshold-optimisation policy.
#
# ``FORECAST_THRESHOLD_METRIC`` controls how per-horizon decision
# thresholds are picked on the val split.  Options:
#   ``"youden"``    — Youden's J = TPR - FPR.  Directly maximises "high
#                     recall AND low FAR" and does NOT collapse to the
#                     trivial "always positive" solution when the class
#                     is imbalanced.  Preferred default.
#   ``"fbeta"``     — F-beta with beta = FORECAST_FBETA (default 2 =
#                     recall-oriented).
#   ``"f1"``        — Classical F1 (previous default; retained for
#                     backward compatibility).
#   ``"far_constrained"`` — max recall subject to FAR <= FORECAST_TARGET_FAR.
#   ``"precision_at_min_recall"`` — max precision subject to recall >=
#                                    FORECAST_MIN_RECALL.
FORECAST_THRESHOLD_METRIC = "precision_at_min_recall"  # v2: max precision s.t. recall >= floor
FORECAST_THRESHOLD_GRID_STEP = 0.01     # finer grid than the legacy 0.05
FORECAST_FBETA = 2.0                    # only used when metric == "fbeta"
FORECAST_TARGET_FAR = 0.10              # Fix 1: 10% FAR ceiling
FORECAST_MIN_RECALL = 0.85              # only used when metric == "precision_at_min_recall"

# DECISION_THRESHOLD_DEFAULT is only used if val-based tuning fails
# (e.g., zero positives in val fold).
FORECAST_DECISION_THRESHOLD_DEFAULT = 0.5
FORECAST_LEAD_TIME_MIN_PROB = 0.5     # first ts with p >= this counts as lead


# ── Fix 6: Kalman Filter Smoothing (Phase 3 augmentation) ──────────────────
# Kalman columns are added alongside the existing *_smooth columns.
# Feature engineering prefers *_kalman as highest-priority variant.
KALMAN_Q_CZT     = 0.5    # CZT process noise (small = more smoothing)
KALMAN_Q_CDTE    = 2.0    # CdTe process noise (allows faster flare tracking)
KALMAN_R_DEFAULT = 5.0    # Default measurement noise std when STAT_ERR unavailable


# ── Fix 3: Particle / SEP Discriminator Features (Phase 4 augmentation) ───
PARTICLE_RATIO_WINDOW_S         = 600   # trailing window for CZT/CdTe ratio z-score
PARTICLE_RATIO_SPIKE_THRESHOLD  = 3.0   # sigma threshold for ratio anomaly
PARTICLE_GTI_PROXIMITY_WINDOW_S = 120   # seconds within which to flag GTI proximity


# ── Fix 7: AR-Proxy Feature Windows (post-Phase-5 augmentation) ───────────
AR_PROXY_SHORT_WINDOW_H = 6    # hours for short-window flare count
AR_PROXY_LONG_WINDOW_H  = 12   # hours for long-window flare count


# ── Fix 4: Significant-Flare Class-Specific Model ─────────────────────────
# Phase 6 provides `intensity_class` (native TEJAS bins) since GOES calibration
# was not run.  We treat moderate/strong/severe events as "significant" — this
# gives ~43 events across the May 2026 catalogue.  GOES calibration would
# populate `goes_class`, in which case the plan document recommends switching
# to `["M", "X"]`.
FORECAST_SIGNIFICANT_INTENSITY_CLASSES = ["moderate", "strong", "severe"]
FORECAST_LABEL_HORIZONS_MIN_SIG        = [5, 15, 30, 60]


# ── Fix 8: Two-Stage False Alert Discriminator ────────────────────────────
DISCRIMINATOR_WINDOW_S       = 1800   # 30-minute trailing context window (seconds)
DISCRIMINATOR_SUMMARY_STATS  = ["mean", "std", "min", "max", "slope"]
DISCRIMINATOR_TOP_K_BASE     = 30     # top-K feature-importance columns to summarise
DISCRIMINATOR_TP_LOSS_WEIGHT  = 1.2    # cost of killing a true alert relative to
                                        # killing a false one (used for net-benefit
                                        # net gain check when SKIPPING a horizon).
DISCRIMINATOR_MIN_NET_BENEFIT = 3     # skip the horizon if val net-benefit
                                       # (correct_kills − α × wrong_kills) < this.
DISCRIMINATOR_MIN_PRECISION   = 0.60   # minimum val precision on suppressed
                                        # alerts (correctly_killed /
                                        # total_killed).  Above this floor
                                        # every kill lifts precision.
DISCRIMINATOR_MAX_TP_KILL_RATE = 0.10  # fraction of Stage-1 true positives we
                                        # allow the discriminator to
                                        # inadvertently kill.  0.10 recovers
                                        # the original behaviour that got
                                        # 249 FPs killed / 0 TPs killed at
                                        # 30 min.  Tightening this (0.05 or
                                        # 0.03) trades away some of that
                                        # aggressive-but-lucky behaviour for
                                        # more conservative suppression.


# ── v2 Optimization: Precision + Recall (credible-event relabelling) ──────
# The original forecasting labels were built from ALL 1,751 catalogue events,
# ~86% of which are hard-only noise blips or sub-noise-floor "faint" events.
# Training and evaluating on those caps discrimination near random.  v2
# rebuilds the forecasting targets from a *credible* subset and confirms
# events against the external GOES flare list.

# A TEJAS event is "credible" if it has a soft (SoLEXS) detection with peak
# count rate at or above this floor (the "weak" intensity bin edge), OR it
# cross-matches a GOES flare (see GOES_VALIDATION_* below).  Hard-only events
# are excluded from the forecasting targets (they remain in the catalogue and
# still feed the AR-proxy activity-context features).
FORECAST_EVENT_MIN_PEAK_COUNTS = 25      # c/s; == INTENSITY_BIN_EDGES[0]
FORECAST_EVENT_REQUIRE_SOFT    = True    # drop hard-only events from targets
CREDIBLE_CATALOGUE_NAME        = "master_catalogue_credible.csv"

# GOES cross-match (external confirmation) — tejas/validation/goes_match.py.
# Wider than the ±30 s calibration tolerance because GOES flare *begin/peak*
# timing and TEJAS peak timing differ by minutes for gradual events.
GOES_VALIDATION_TOLERANCE_S = 300        # ±5 min peak-to-peak match window
GOES_FLARE_LIST_NAME        = "goes_flare_events.csv"
GOES_VALIDATION_REPORT      = PROJECT_DIR / "goes_validation_report.md"

# v2 threshold objective: maximise precision subject to a per-horizon recall
# floor (the user's "recall ≥ 0.8, then best precision" goal).  Overrides the
# scalar FORECAST_MIN_RECALL when FORECAST_THRESHOLD_METRIC ==
# "precision_at_min_recall".  A horizon absent here falls back to
# FORECAST_MIN_RECALL.
FORECAST_MIN_RECALL_PER_HORIZON = {5: 0.80, 15: 0.80, 30: 0.80, 60: 0.80}

# v3 P0: duty-cycle caps for the event-level threshold optimizer.  The v2
# row-level 0.8 recall floor drove the tuner into an always-on corner (duty
# cycles 40%/80%/80%/100%).  The new event-level objective maximizes episode
# precision s.t. event recall ≥ floor AND duty cycle ≤ cap.  The cap makes
# the 60-min always-on corner infeasible by construction.
FORECAST_MAX_DUTY_CYCLE_PER_HORIZON = {5: 0.10, 15: 0.15, 30: 0.20, 60: 0.30}

# ── v3 P1: Label surgery (optimization_plan_v3.md §2) ─────────────────────
# §2.1 GOES-union: GOES flares with no TEJAS counterpart become label events
# (start anchor = GOES begin_time, tagged source="goes_only") IF Aditya data
# covers the flare window at or above this fraction of expected rows.
# Flares below the floor are *not labeled*; their windows go to the §2.3
# coverage mask instead (their truth is unknowable from the instrument).
FORECAST_GOES_UNION_MIN_COVERAGE = 0.5
# Window end fallback when the GOES list has no end_time: peak + this many
# minutes (typical C-class decay scale).
FORECAST_GOES_WINDOW_FALLBACK_MIN = 15.0
# File (in CLASSIFICATION_DIR) listing GOES flare windows with no/degraded
# Aditya coverage.  Written by scripts/run_credible_relabel.py; consumed at
# training time to drop in-window rows from the fit (§2.3).
GOES_UNCOVERED_WINDOWS_NAME = "goes_uncovered_windows.csv"

# §2.2 Ambiguity buffer ("don't-care" band), training fit only: rows with
# label_minutes_to_next_flare in (h, h + buffer(h)] are excluded from the
# horizon-h fit.  They are physically near-identical to positives (precursor
# already underway) yet labeled 0 — pure gradient noise.  buffer(h) =
# max(5 min, 0.25·h).  Val/test evaluation keeps strict labels — never
# softened.
FORECAST_LABEL_AMBIGUITY_BUFFER_MIN = {5: 5.0, 15: 5.0, 30: 7.5, 60: 15.0}

# §2.4 In-flare row policy (A/B ablation flag): exclude decaying-phase
# in-flare rows from the training fit (the nowcaster owns that regime; the
# forecast label there reflects sympathetic-flaring statistics, not
# precursors).  Adopt True only if the val event metrics improve.
# P1d ablation (2026-07): measured HARMFUL — val event recall/precision
# dropped at 15/30/60m (e.g. 60m ep_prec 0.727 -> 0.557).  Decaying rows
# evidently carry real signal about follow-on flares.  Keep False.
FORECAST_EXCLUDE_DECAYING_FROM_TRAIN = False

# Event-level evaluation (tejas/forecasting/event_metrics.py).  Consecutive
# alert rows within this gap are merged into one alert "episode" for
# episode-precision accounting.
EVENT_ALERT_MERGE_GAP_S = 300            # merge alerts < 5 min apart

# Long-window context features (tejas/features/long_context.py).  Hour-scale
# quiet-vs-active context; the longest pre-v2 trailing window was only 600 s.
LONG_CONTEXT_PCTRANK_WINDOWS_S = [3600, 10800, 21600]   # 1 h, 3 h, 6 h
LONG_CONTEXT_RISE_WINDOW_S     = 300                    # rise-consistency window
LONG_CONTEXT_MEDIAN_WINDOW_S   = 3600                   # elevated-state baseline


# ── Phase 8: Alert Engine (Fix 2 — confirmation gate) ─────────────────────
ALERT_CONFIRMATION_K = 3     # consecutive windows before confirming
ALERT_COOLDOWN_S     = 300   # post-alert suppression window (seconds)
ALERT_CADENCE_S      = FORECAST_CADENCE_S   # inherit from forecast cadence
