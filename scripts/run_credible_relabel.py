"""
Phase B runner — credible-event catalogue + forecasting relabel.

Steps:
  1. Load the Phase-6 classification master catalogue (has intensity + confidence).
  2. Attach GOES cross-match flag (external confirmation).
  3. Build the credible subset and write master_catalogue_credible.csv.
  4. Relabel every day's Phase-5 detection parquet in place:
       - back up original labels to label_flare_within_{h}m_all (once)
       - recompute label_flare_within_{h}m + label_minutes_to_next_flare
         from the month-wide credible start times.

Idempotent: re-running recomputes from the *_all backup, never double-backs-up.
Run:  python -m scripts.run_credible_relabel
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tejas.config import (
    DETECTION_DIR, CLASSIFICATION_DIR, YEAR, MONTH, MISSING_DAYS,
    DETECTION_LABEL_HORIZONS_MIN, CREDIBLE_CATALOGUE_NAME,
    GOES_VALIDATION_TOLERANCE_S,
)
from tejas.detection.credible import build_credible_catalogue, relabel_frame, summarise
from tejas.validation.goes_match import load_goes_flare_list, build_validation_report


def main():
    cat_path = Path(CLASSIFICATION_DIR) / "master_catalogue.csv"
    catalogue = pd.read_csv(cat_path)
    print(f"Loaded classification catalogue: {len(catalogue)} events")

    # ── GOES cross-match flag ──────────────────────────────────────────
    try:
        goes = load_goes_flare_list()
        catalogue = build_validation_report(catalogue, goes,
                                             tolerance_s=GOES_VALIDATION_TOLERANCE_S)
        print(f"GOES flare list: {len(goes)} flares; "
              f"TEJAS matched: {int(catalogue['goes_matched'].sum())}")
    except FileNotFoundError:
        print("GOES flare list not found — credible filter uses soft-peak only.")
        catalogue["goes_matched"] = False

    # ── Credible subset ────────────────────────────────────────────────
    stats = summarise(catalogue)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    credible = build_credible_catalogue(catalogue)
    out_path = Path(CLASSIFICATION_DIR) / CREDIBLE_CATALOGUE_NAME
    credible.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(credible)} credible events)")

    credible_starts = pd.to_datetime(credible["start_time"]).tolist()

    # ── Relabel each day's detection parquet ───────────────────────────
    days = [f"{d:02d}" for d in range(1, 32) if f"{d:02d}" not in MISSING_DAYS]
    for dd in days:
        date_str = f"{YEAR}-{MONTH}-{dd}"
        pq = Path(DETECTION_DIR) / f"{date_str}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)

        # Back up originals once.
        for h in DETECTION_LABEL_HORIZONS_MIN:
            col, bak = f"label_flare_within_{h}m", f"label_flare_within_{h}m_all"
            if col in df.columns and bak not in df.columns:
                df[bak] = df[col]
        if "label_minutes_to_next_flare" in df.columns and \
           "label_minutes_to_next_flare_all" not in df.columns:
            df["label_minutes_to_next_flare_all"] = df["label_minutes_to_next_flare"]

        df = relabel_frame(df, credible_starts, DETECTION_LABEL_HORIZONS_MIN)
        df.to_parquet(pq, index=False, engine="pyarrow")
        pos = {h: int(df[f"label_flare_within_{h}m"].sum())
               for h in DETECTION_LABEL_HORIZONS_MIN}
        print(f"  {date_str}: relabelled, positives={pos}")

    print("Phase B relabel complete.")


if __name__ == "__main__":
    main()
