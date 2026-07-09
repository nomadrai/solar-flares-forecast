"""
TEJAS Feature Engineering Module
==================================
Phase 4: Feature Engineering.

Transforms Phase 3 clean daily Parquet files into feature-rich datasets
for downstream flare detection, classification, and forecasting.

Feature groups:
  1. Temporal — rolling stats, derivatives, gradients (per instrument)
  2. Spectral — PI-derived band ratios, hardness evolution (SoLEXS)
  3. Cross-instrument — soft/hard ratios, lag, cross-correlation
  4. Statistical — z-score, percentile rank, CUSUM

Public API:
  - engineer_day(date_str)      → engineer features for a single day
  - engineer_all_days()         → batch process all available days
"""

from tejas.features.pipeline import engineer_day, engineer_all_days

__all__ = ["engineer_day", "engineer_all_days"]
