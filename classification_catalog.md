# TEJAS — Classification Catalogue

This document describes every classification column produced by Phase 6 (Flare Classification). Classification consumes the Phase 5 master catalogue at `data/detection/YYYY/MM/master_catalogue.csv` and writes an augmented copy to `data/classification/YYYY/MM/master_catalogue.csv`.

The Phase 6 output preserves every Phase 5 column untouched and appends **seven new columns**. Two are physically defined (native intensity from SoLEXS counts) and are **always** populated for soft-detected events; five carry the optional GOES-equivalent labels and are populated only when a valid calibration artifact exists.

---

## Design Principles

1. **SoLEXS-native first.** Detection and forecasting must never depend on external GOES data. Native intensity classification is derived directly from `peak_solexs_counts` using count-based bin edges, and is available for every soft-detected event.
2. **GOES calibration is optional and separable.** GOES XRS fetching, cross-matching, log-log fitting, and letter/magnitude assignment live in a distinct module (`tejas/classification/calibration.py` + `scripts/fetch_goes_data.py`). Absence of GOES data does not degrade the rest of the pipeline — the columns simply stay `NaN`.
3. **Hard-only events are not fabricated.** The GOES class is defined by soft X-ray flux; hard-only detections (~65% of the master catalogue) legitimately have no soft peak, so all intensity + GOES columns remain `NaN`. The `event_type` column preserves their category.
4. **Causal.** Classification uses only the single scalar `peak_solexs_counts` at `peak_time`. No lookahead, no smoothing across future samples, no post-flare information.

---

## Columns Added

### 1. `event_type` (str)

Clone of the Phase 5 `confidence` field, kept explicitly under the classification namespace so downstream code has a stable name. Values:

| Value | Meaning |
|---|---|
| `confirmed` | Soft + hard match within cross-match window |
| `soft_only` | SoLEXS detection with no HEL1OS counterpart |
| `hard_only` | HEL1OS detection with no SoLEXS counterpart |

Always populated.

---

### 2. `intensity_class` (str)

Native TEJAS intensity bin derived from `peak_solexs_counts`. Populated when `peak_solexs_counts` is not NaN (i.e. `event_type ∈ {confirmed, soft_only}`). Bin edges are set in `tejas/config.py` (`INTENSITY_BIN_EDGES = [25, 100, 400, 1500]`) and were chosen from the May 2026 observed range (14–3667 c/s) so that each bin covers roughly one order-of-magnitude step in flux, mirroring the GOES letter-class ladder.

| Label | peak_solexs_counts (c/s) | Physical meaning |
|---|---|---|
| `faint` | < 25 | Barely above baseline noise; sub-A GOES equivalent |
| `weak` | 25 – 100 | Small enhancement over quiet Sun |
| `moderate` | 100 – 400 | Distinct flare, elevated hardness ratio expected |
| `strong` | 400 – 1500 | Large flare, hard X-ray counterpart likely |
| `severe` | ≥ 1500 | Major flare, CZT signal typically detected |

Observed distribution across the May 2026 catalogue (615 soft-detected events): faint 364, weak 208, moderate 34, strong 8, severe 1.

---

### 3. `intensity_score` (float, 0–1)

Log-scaled continuous severity score. Formula:

`score = min(log(peak_counts + 1) / log(INTENSITY_SCORE_MAX_COUNTS), 1.0)`

where `INTENSITY_SCORE_MAX_COUNTS = 4000` (roughly the ceiling of the observed range). Useful as a single-number input for downstream forecasting models — monotone in flux, saturates gracefully at extreme events.

---

### 4. `goes_flux` (float, W/m², 1–8 Å)

Predicted GOES XRS-B flux computed from `peak_solexs_counts` using the fitted log-log calibration:

`log10(flux) = a · log10(peak_counts) + b`

where `(a, b)` come from the calibration artifact at `data/reference/calibration/solexs_to_goes.json`. Populated only when:

- the calibration artifact exists, **and**
- `peak_solexs_counts` is not NaN.

`NaN` otherwise — no fabrication.

---

### 5. `goes_class` (str, one of A / B / C / M / X)

GOES letter class derived from `goes_flux` using the standard NOAA thresholds (1–8 Å channel):

| Letter | Flux threshold (W/m²) |
|---|---|
| A | 1×10⁻⁸ |
| B | 1×10⁻⁷ |
| C | 1×10⁻⁶ |
| M | 1×10⁻⁵ |
| X | 1×10⁻⁴ |

Populated iff `goes_flux` is populated.

---

### 6. `goes_magnitude` (float, 1.0–9.9)

Multiplier within the letter class. Computed as `flux / class_threshold`, floored to 1 decimal to avoid inflating class boundaries. Capped at 9.9 for A–M (so `9.99e-5 W/m²` → `M9.9`, not `M10.0`). No upper cap for X.

---

### 7. `goes_class_full` (str)

Concatenation of letter + magnitude for display, e.g. `"M2.5"`, `"X1.4"`. Populated iff `goes_flux` is populated.

---

## Calibration Methodology

Implemented in `tejas/classification/calibration.py`. Steps:

1. **Fetch GOES XRS 1-second data** for the target month via `scripts/fetch_goes_data.py`. Downloads L2 netCDF files from NOAA NCEI (`data.ngdc.noaa.gov`, GOES-16/17/18), extracts the `xrsb_flux` (1–8 Å) variable, and saves per-day CSVs to `data/reference/goes/YYYY/MM/YYYY-MM-DD.csv` (columns: `time_unix`, `flux_wm2`). Requires `xarray + netCDF4` (optional dependencies).
2. **Cross-match** each Phase 5 event with `peak_solexs_counts > 0` against the nearest GOES sample within `GOES_MATCH_TOLERANCE_S = 30` seconds of `peak_time`.
3. **Fit** a robust log-log linear model using `sklearn.linear_model.HuberRegressor` (falls back to `numpy.polyfit` if sklearn is unavailable).
4. **Quality gate.** Accept the fit only if `n_matches ≥ 20` and `R² ≥ 0.5`. Below these thresholds the artifact is not written and the pipeline runs native-only.
5. **Persist** to `data/reference/calibration/solexs_to_goes.json`. A one-off human-readable report is written to `data/classification/YYYY/MM/calibration_report.md`.

---

## Running the Pipeline

```
python scripts/run_classification.py                   # native only
python scripts/fetch_goes_data.py                      # optional, fetch GOES
python scripts/run_classification.py --calibrate       # fit calibration, then classify
python scripts/run_classification.py --validate        # compare vs GOES XRS
```

Outputs:
- `data/classification/YYYY/MM/master_catalogue.csv` — augmented catalogue (25 columns).
- `data/classification/YYYY/MM/classification_summary.csv` — per-day counts by intensity bin and GOES letter.
- `data/classification/YYYY/MM/calibration_report.md` — only if `--calibrate` succeeded.

---

## Tests

`tests/test_classification.py` covers native bin boundaries, GOES threshold and magnitude arithmetic, calibration fit quality gates, cross-matching within tolerance, pipeline behaviour with and without a calibration artifact, non-overwrite guarantee for Phase 5 columns, and the no-future-leakage invariant. All 20 tests pass.
