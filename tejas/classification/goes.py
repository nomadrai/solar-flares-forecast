"""
TEJAS GOES Class Assigner
==========================
Converts an X-ray flux value (W/m^2, 1-8 Å band) into the standard NOAA
GOES letter class + magnitude, e.g. 2.5e-5 W/m^2 -> "M2.5".

This module is purely rule-based (fixed physics thresholds).  The mapping
from SoLEXS counts to flux is done by the calibration module — this file
only handles flux -> class.
"""

import math
import numpy as np

from tejas.config import GOES_CLASS_THRESHOLDS


_CLASS_ORDER = ["A", "B", "C", "M", "X"]


def flux_to_class(flux):
    """Convert a scalar flux (W/m^2) to (letter, magnitude, full_string).

    Standard NOAA definition, 1-8 Å channel:
      A: 1e-8   ≤ flux < 1e-7
      B: 1e-7   ≤ flux < 1e-6
      C: 1e-6   ≤ flux < 1e-5
      M: 1e-5   ≤ flux < 1e-4
      X: flux   ≥ 1e-4

    Fluxes below 1e-8 are labelled "A" with magnitude < 1.0 (sub-A).
    Magnitude is truncated to 1 decimal (never rounded up beyond 9.9
    within a letter class — 1.0e-4 is X1.0, not M10.0).

    Args:
        flux: Scalar flux in W/m^2, or NaN.

    Returns:
        Tuple (letter, magnitude, full_string).  All NaN if input is NaN.
    """
    if flux is None or (isinstance(flux, float) and math.isnan(flux)):
        return (float("nan"), float("nan"), float("nan"))
    if flux <= 0:
        return ("A", 0.0, "A0.0")

    # Determine letter class.
    letter = "A"
    for cls in _CLASS_ORDER:
        if flux >= GOES_CLASS_THRESHOLDS[cls]:
            letter = cls

    threshold = GOES_CLASS_THRESHOLDS[letter]
    magnitude = flux / threshold

    # Magnitude for A/B/C/M is bounded to [0, 9.9].  X has no upper bound.
    if letter != "X":
        magnitude = min(magnitude, 9.9)

    # Truncate to 1 decimal (floor to avoid inflating class).
    magnitude = math.floor(magnitude * 10) / 10.0

    full = f"{letter}{magnitude:.1f}"
    return (letter, magnitude, full)


def flux_to_class_vectorised(flux_array):
    """Vectorised version of flux_to_class over an array/Series of fluxes.

    Args:
        flux_array: Array-like of flux values (W/m^2).  NaN preserved.

    Returns:
        Tuple of three arrays (letters, magnitudes, full_strings), each
        same length as input.
    """
    arr = np.asarray(flux_array, dtype=float)
    letters = np.empty(arr.shape, dtype=object)
    mags = np.full(arr.shape, np.nan, dtype=float)
    fulls = np.empty(arr.shape, dtype=object)

    for i, f in enumerate(arr):
        letter, mag, full = flux_to_class(f)
        letters[i] = letter if isinstance(letter, str) else np.nan
        mags[i] = mag
        fulls[i] = full if isinstance(full, str) else np.nan

    return letters, mags, fulls


def counts_to_flux(peak_counts, calibration):
    """Convert SoLEXS peak counts to predicted GOES flux using a fit.

    Uses log-log linear model: log10(flux) = a * log10(counts) + b.

    Args:
        peak_counts: Scalar or array of peak count rates (c/s).
        calibration: dict with keys 'a' and 'b' (from the calibration
            artifact).  If None, returns NaN.

    Returns:
        Same shape as input.  NaN preserved.
    """
    if calibration is None:
        arr = np.atleast_1d(peak_counts).astype(float)
        out = np.full(arr.shape, np.nan)
        return float(out[0]) if np.isscalar(peak_counts) else out

    a = calibration["a"]
    b = calibration["b"]

    scalar_input = np.isscalar(peak_counts)
    arr = np.atleast_1d(peak_counts).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_flux = a * np.log10(np.where(arr > 0, arr, np.nan)) + b
        flux = np.power(10.0, log_flux)

    flux[np.isnan(arr)] = np.nan

    if scalar_input:
        return float(flux[0])
    return flux
