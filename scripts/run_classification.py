"""
TEJAS Phase 6 — Flare Classification Batch Runner
====================================================
Loads the Phase 5 master catalogue, assigns intensity + (optionally) GOES
class to every event, and writes an augmented catalogue to
data/classification/YYYY/MM/.

Usage:
    python scripts/run_classification.py                  # native only
    python scripts/run_classification.py --calibrate      # fit calibration, then classify
    python scripts/run_classification.py --validate       # compare vs GOES XRS if available

The --calibrate step requires fetch_goes_data.py to have been run first
(GOES CSVs must exist in data/reference/goes/).  If calibration fails
(too few matches, low R²), the pipeline falls back to native-only.
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from tejas.classification import classify_all_days
from tejas.classification.calibration import (
    calibrate,
    load_calibration,
    load_goes_reference,
)
from tejas.classification.goes import flux_to_class
from tejas.config import (
    DETECTION_DIR,
    CLASSIFICATION_DIR,
    GOES_REFERENCE_DIR,
    CALIBRATION_ARTIFACT,
    INTENSITY_BIN_LABELS,
)


def run_calibration(save_report=True):
    """Attempt to fit the SoLEXS→GOES calibration.

    Returns:
        Calibration dict or None.
    """
    print("\n" + "=" * 60)
    print("Calibration (SoLEXS peak counts → GOES flux)")
    print("=" * 60)

    goes_df = load_goes_reference()
    if goes_df.empty:
        print(f"  [WARN] No GOES reference data found in {GOES_REFERENCE_DIR}")
        print("  Run scripts/fetch_goes_data.py first.  Falling back to native-only.")
        return None
    print(f"  Loaded {len(goes_df):,} GOES samples from {GOES_REFERENCE_DIR}")

    master_path = Path(DETECTION_DIR) / "master_catalogue.csv"
    if not master_path.exists():
        print(f"  [ERROR] Master catalogue not found: {master_path}")
        return None
    catalogue = pd.read_csv(master_path)

    fit = calibrate(catalogue, goes_df=goes_df, save_artifact=True)
    if fit is None:
        print("  [WARN] Calibration fit failed (insufficient matches or low R²).")
        print("  Falling back to native-only classification.")
        return None

    print(f"  Fit: log10(flux) = {fit['a']:.3f} * log10(counts) + {fit['b']:.3f}")
    print(f"  R² = {fit['r2']:.3f}  |  n_samples = {fit['n_samples']}")
    print(f"  Saved to {CALIBRATION_ARTIFACT}")

    if save_report:
        report_path = Path(CLASSIFICATION_DIR) / "calibration_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_calibration_report(report_path, fit, catalogue, goes_df)
        print(f"  Wrote {report_path}")

    return fit


def _write_calibration_report(path, fit, catalogue, goes_df):
    """Emit a markdown summary of the calibration fit."""
    from tejas.classification.calibration import cross_match

    pairs = cross_match(catalogue, goes_df)

    lines = [
        "# SoLEXS → GOES Calibration Report",
        "",
        f"- **Fit date**: {fit['fit_date']}",
        f"- **Source**: {fit['goes_source']}",
        f"- **Model**: `log10(flux_Wm2) = a * log10(peak_counts_cps) + b`",
        f"- **a** = {fit['a']:.4f}",
        f"- **b** = {fit['b']:.4f}",
        f"- **R²** = {fit['r2']:.4f}",
        f"- **N matched events** = {fit['n_samples']}",
        "",
        "## Matched-pair statistics",
        "",
        f"- Total soft-detected events: {catalogue['peak_solexs_counts'].notna().sum()}",
        f"- Successful GOES matches: {len(pairs)}",
        f"- Peak-counts range in matches: "
        f"{pairs['peak_counts'].min():.1f} – {pairs['peak_counts'].max():.1f} c/s",
        f"- GOES flux range in matches: "
        f"{pairs['goes_flux'].min():.2e} – {pairs['goes_flux'].max():.2e} W/m²",
        "",
        "## Class-by-class comparison (confirmed events only)",
        "",
        "| Event | peak_counts | GOES flux (matched) | Predicted class | Actual class |",
        "|---|---|---|---|---|",
    ]

    confirmed = catalogue[catalogue["confidence"] == "confirmed"].head(20)
    for _, row in confirmed.iterrows():
        c = row["peak_solexs_counts"]
        if pd.isna(c):
            continue
        pred_flux = 10 ** (fit["a"] * np.log10(c) + fit["b"])
        pred_class = flux_to_class(pred_flux)[2]
        # Look up matched flux
        matches = pairs[np.isclose(pairs["peak_counts"], c, rtol=1e-3)]
        actual_flux = matches["goes_flux"].iloc[0] if len(matches) else None
        actual_class = flux_to_class(actual_flux)[2] if actual_flux else "—"
        actual_str = f"{actual_flux:.2e}" if actual_flux else "—"
        lines.append(
            f"| {row['event_id']} | {c:.0f} | {actual_str} | {pred_class} | {actual_class} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_against_goes():
    """Compare TEJAS classifications against actual GOES XRS labels."""
    print("\n" + "=" * 60)
    print("Validation against GOES XRS")
    print("=" * 60)

    goes_df = load_goes_reference()
    if goes_df.empty:
        print("  [WARN] No GOES reference data.  Skipping validation.")
        return

    classified_path = Path(CLASSIFICATION_DIR) / "master_catalogue.csv"
    if not classified_path.exists():
        print(f"  [ERROR] Classified catalogue not found: {classified_path}")
        return

    df = pd.read_csv(classified_path)
    df = df.dropna(subset=["goes_class"])

    if df.empty:
        print("  [WARN] No GOES-classified events (calibration not run).")
        return

    print(f"  Comparing {len(df)} classified events against actual GOES peak flux...")
    print(f"  Predicted class distribution:")
    for letter in ["A", "B", "C", "M", "X"]:
        n = (df["goes_class"] == letter).sum()
        print(f"    {letter}: {n}")


def print_summary(result):
    summary = result["summary"]
    catalogue = result["catalogue"]
    calibration = result["calibration"]

    print("\n=== CLASSIFICATION SUMMARY ===\n")
    print(f"Total events: {len(catalogue)}")
    print(f"Soft-classified (intensity_*): "
          f"{catalogue['intensity_class'].notna().sum()}")
    print(f"Hard-only (NaN classification): "
          f"{catalogue['intensity_class'].isna().sum()}")
    print(f"\nIntensity distribution:")
    for lbl in INTENSITY_BIN_LABELS:
        n = (catalogue["intensity_class"] == lbl).sum()
        print(f"  {lbl}: {n}")

    if calibration is not None:
        print(f"\nGOES class distribution:")
        for letter in ["A", "B", "C", "M", "X"]:
            n = (catalogue["goes_class"] == letter).sum()
            print(f"  {letter}: {n}")
        max_row = catalogue.loc[catalogue["goes_flux"].idxmax()] \
            if catalogue["goes_flux"].notna().any() else None
        if max_row is not None:
            print(f"\nStrongest event: {max_row['peak_time']} "
                  f"(peak={max_row['peak_solexs_counts']:.0f} c/s → "
                  f"{max_row['goes_class_full']})")
    else:
        print(f"\nGOES columns: NULL (calibration not available)")

    print(f"\nElapsed: {result['elapsed_s']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="TEJAS Phase 6 — Flare Classification")
    parser.add_argument("--calibrate", action="store_true",
                        help="Fit SoLEXS→GOES calibration before classifying.")
    parser.add_argument("--validate", action="store_true",
                        help="Compare classifications against GOES XRS after running.")
    args = parser.parse_args()

    print("=" * 60)
    print("TEJAS — Phase 6: Flare Classification")
    print("=" * 60)

    if args.calibrate:
        run_calibration()

    result = classify_all_days(use_calibration=True)
    print_summary(result)

    print(f"\nOutputs:")
    print(f"  {CLASSIFICATION_DIR / 'master_catalogue.csv'}")
    print(f"  {CLASSIFICATION_DIR / 'classification_summary.csv'}")

    if args.validate:
        validate_against_goes()

    return 0


if __name__ == "__main__":
    sys.exit(main())
