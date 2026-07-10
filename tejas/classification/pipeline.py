"""
TEJAS Classification Pipeline
================================
Loads the Phase 5 master catalogue, augments each event with:

  * event_type       — clone of Phase 5 'confidence', kept for clarity
  * intensity_class  — TEJAS native bin (faint/weak/moderate/strong/severe)
  * intensity_score  — log-scaled severity in [0, 1]
  * goes_flux        — predicted 1-8 Å flux (W/m²) — only if calibration loaded
  * goes_class       — GOES letter (A/B/C/M/X)
  * goes_magnitude   — magnitude within class (1.0-9.9)
  * goes_class_full  — combined display (e.g. "M2.5")

Hard-only events keep NaN in all intensity/GOES columns.  Original Phase 5
columns are preserved untouched.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    DETECTION_DIR,
    CLASSIFICATION_DIR,
    INTENSITY_BIN_LABELS,
)
from tejas.classification.native import classify_dataframe as native_classify
from tejas.classification.goes import counts_to_flux, flux_to_class_vectorised
from tejas.classification.calibration import load_calibration


def classify_catalogue(catalogue, calibration=None):
    """Augment a master catalogue DataFrame with classification columns.

    Args:
        catalogue: Master catalogue DataFrame (Phase 5 output).
        calibration: Optional calibration dict from load_calibration().
            If None, only native columns are populated; goes_* stay NaN.

    Returns:
        Augmented DataFrame.  Row order and existing columns preserved.
    """
    df = catalogue.copy()

    # event_type: explicit clone of confidence for downstream clarity.
    df["event_type"] = df["confidence"]

    # Native intensity — populated for any row with peak_solexs_counts.
    df = native_classify(df, peak_col="peak_solexs_counts")

    # GOES columns — populated only if a calibration is available.
    if calibration is not None:
        flux = counts_to_flux(df["peak_solexs_counts"].values, calibration)
        letters, mags, fulls = flux_to_class_vectorised(flux)
        df["goes_flux"] = flux
        df["goes_class"] = letters
        df["goes_magnitude"] = mags
        df["goes_class_full"] = fulls
    else:
        df["goes_flux"] = np.nan
        df["goes_class"] = np.nan
        df["goes_magnitude"] = np.nan
        df["goes_class_full"] = np.nan

    return df


def summarize_by_day(classified_df):
    """Build a per-day summary of classification counts.

    Columns: date, n_events, n_soft_classified,
             n_faint, n_weak, n_moderate, n_strong, n_severe,
             n_goes_A, n_goes_B, n_goes_C, n_goes_M, n_goes_X,
             max_peak_counts, max_goes_class_full.

    Args:
        classified_df: Output of classify_catalogue().

    Returns:
        Summary DataFrame indexed by date.
    """
    df = classified_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    rows = []
    for date, grp in df.groupby("date", sort=True):
        row = {
            "date": date,
            "n_events": len(grp),
            "n_soft_classified": grp["intensity_class"].notna().sum(),
        }
        for lbl in INTENSITY_BIN_LABELS:
            row[f"n_{lbl}"] = (grp["intensity_class"] == lbl).sum()
        for letter in ["A", "B", "C", "M", "X"]:
            row[f"n_goes_{letter}"] = (grp["goes_class"] == letter).sum()
        row["max_peak_counts"] = grp["peak_solexs_counts"].max()

        goes_full = grp["goes_class_full"].dropna()
        if len(goes_full) and grp["goes_flux"].notna().any():
            idx = grp["goes_flux"].idxmax()
            row["max_goes_class_full"] = grp.loc[idx, "goes_class_full"]
        else:
            row["max_goes_class_full"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def classify_all_days(
    input_catalogue=None,
    output_dir=None,
    use_calibration=True,
    save=True,
):
    """Batch classification for the full master catalogue.

    Pipeline:
      1. Load Phase 5 master_catalogue.csv (all days).
      2. Optionally load calibration artifact.
      3. Augment every row with classification columns.
      4. Write augmented catalogue + per-day summary to CLASSIFICATION_DIR.

    Args:
        input_catalogue: Path to the Phase 5 master catalogue.
            Defaults to DETECTION_DIR / master_catalogue.csv.
        output_dir: Directory for classification outputs.
            Defaults to CLASSIFICATION_DIR.
        use_calibration: If True, attempt to load the calibration artifact.
        save: If True, write outputs to disk.

    Returns:
        dict {catalogue, summary, calibration, elapsed_s}.
    """
    t0 = time.time()

    if input_catalogue is None:
        input_catalogue = Path(DETECTION_DIR) / "master_catalogue.csv"
    input_catalogue = Path(input_catalogue)
    if not input_catalogue.exists():
        raise FileNotFoundError(
            f"Master catalogue not found: {input_catalogue}. "
            "Run Phase 5 detection first."
        )

    if output_dir is None:
        output_dir = CLASSIFICATION_DIR
    output_dir = Path(output_dir)

    catalogue = pd.read_csv(input_catalogue)

    calibration = None
    if use_calibration:
        calibration = load_calibration()

    classified = classify_catalogue(catalogue, calibration=calibration)
    summary = summarize_by_day(classified)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        classified.to_csv(output_dir / "master_catalogue.csv", index=False)
        summary.to_csv(output_dir / "classification_summary.csv", index=False)

    return {
        "catalogue": classified,
        "summary": summary,
        "calibration": calibration,
        "elapsed_s": time.time() - t0,
    }
