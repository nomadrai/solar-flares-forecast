"""
TEJAS Flare Detection Module
================================
Phase 5: Flare Detection (Nowcasting).

Detects solar flare events (start, peak, end) independently from
SoLEXS (soft X-ray) and HEL1OS CdTe/CZT (hard X-ray) signals,
then cross-matches them into a unified master flare catalogue.

Detection components:
  1. Threshold — adaptive baseline engine (segment-aware)
  2. Detector — per-instrument state machine (QUIET ↔ ACTIVE)
  3. Catalogue — independent + master catalogue builders
  4. Pipeline — single-day and batch orchestration

Public API:
  - detect_day(date_str)      → detect flares for a single day
  - detect_all_days()         → batch process all available days
"""

from tejas.detection.pipeline import detect_day, detect_all_days

__all__ = ["detect_day", "detect_all_days"]
