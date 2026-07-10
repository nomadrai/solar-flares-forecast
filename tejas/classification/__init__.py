"""
TEJAS Flare Classification Module
====================================
Phase 6: Flare Classification.

Assigns each detected event in the Phase 5 master catalogue:

  1. A native TEJAS intensity class (faint / weak / moderate / strong / severe)
     derived directly from SoLEXS peak count rate — always available.
  2. An optional GOES-equivalent class (A/B/C/M/X + magnitude) derived from an
     empirical count → flux calibration.  Populated only when the calibration
     artifact exists (produced by scripts/fetch_goes_data.py + calibration).

Hard-only events keep NaN in all classification columns — GOES classification
is fundamentally defined by soft X-ray flux, so no fabrication.

Public API:
  - classify_catalogue(catalogue, calibration=None) → augmented DataFrame
  - classify_all_days(...)                           → batch, load + save
"""

from tejas.classification.pipeline import (
    classify_catalogue,
    classify_all_days,
    summarize_by_day,
)

__all__ = ["classify_catalogue", "classify_all_days", "summarize_by_day"]
