"""
TEJAS Processing Module
========================
Phase 3: Data Processing & Cleaning.

Transforms Phase 2 unified daily Parquet files into clean, analysis-ready
datasets by applying:
  1. Spike removal (cosmic rays, telemetry glitches)
  2. Signal smoothing (configurable rolling median)
  3. CZT background estimation & subtraction
  4. Normalization (log1p, percentile rank)

Public API:
  - process_day(date_str)      → process a single day
  - process_all_days()         → batch process all available days
"""

from tejas.processing.pipeline import process_day, process_all_days

__all__ = ["process_day", "process_all_days"]
