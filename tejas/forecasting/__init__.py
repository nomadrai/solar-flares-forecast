"""
TEJAS Forecasting Module
============================
Phase 7: Multi-horizon flare forecasting.

Consumes the Phase 5 labelled feature Parquets in ``data/detection/YYYY/MM/``
and produces four binary probability models — one per horizon (5 / 15 / 30 /
60 minutes) — for each of three algorithms (LogReg, XGBoost, LightGBM).

Design principles:
  * Strict no-future-data.  Features are already causal (Phase 4).  Labels
    are used only during training; feature/label separation is enforced at
    two independent choke points (:mod:`tejas.forecasting.sampling` and
    :mod:`tejas.forecasting.inference`).
  * One model per horizon.  Independent thresholds and independent
    precision/recall trade-offs.
  * Chronological blocked splits with a 1-hour purge gap between blocks.

Public API:
  * :func:`prepare_training_frame` — build the downsampled training frame.
  * :func:`train_all_horizons`     — fit all 12 models and persist artifacts.
  * :func:`evaluate_all`           — reload artifacts, evaluate on the test
    split, and write ``forecast_report.md``.
  * :func:`load_forecaster`        — streaming inference for the dashboard.
  * :func:`load_detection_day`     — convenience loader for a labelled day.
"""

from tejas.forecasting.sampling import load_detection_day
from tejas.forecasting.inference import load_forecaster, Forecaster
from tejas.forecasting.pipeline import (
    prepare_training_frame,
    train_all_horizons,
    evaluate_all,
)

__all__ = [
    "prepare_training_frame",
    "train_all_horizons",
    "evaluate_all",
    "load_forecaster",
    "Forecaster",
    "load_detection_day",
]
