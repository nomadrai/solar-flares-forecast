"""
TEJAS Fix 4 — Significant-Flare Label Augmentation
======================================================
Reads the Phase 6 classification catalogue and adds
label_sig_flare_within_{h}m columns to every Phase 5 parquet.

Usage:
    python scripts/run_sig_labels.py --all
    python scripts/run_sig_labels.py --date 2026-05-01
    python scripts/run_sig_labels.py --all --use-goes    # if Phase 6 calibration was run
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (OSError, LookupError):
            pass

from tejas.config import (
    CLASSIFICATION_DIR,
    DETECTION_DIR,
    FORECAST_SIGNIFICANT_INTENSITY_CLASSES,
)
from tejas.detection.sig_labels import (
    augment_all_days_with_sig_labels,
    augment_day_with_sig_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add sig-flare labels to Phase 5 parquets (Fix 4)."
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--use-goes", action="store_true",
                        help="Use goes_class instead of intensity_class "
                             "(requires GOES calibration).")
    args = parser.parse_args()

    if not (args.all or args.date):
        parser.print_help()
        return 0

    label_source = ("goes_class (M/X)" if args.use_goes
                    else f"intensity_class ({FORECAST_SIGNIFICANT_INTENSITY_CLASSES})")
    print(f"Significant-flare label augmentation (Fix 4)")
    print(f"Detection dir: {DETECTION_DIR}")
    print(f"Catalogue:     {CLASSIFICATION_DIR / 'master_catalogue.csv'}")
    print(f"Label source:  {label_source}")
    print("=" * 60)

    if args.date:
        res = augment_day_with_sig_labels(args.date, use_goes=args.use_goes)
        if res is None:
            print(f"[ERROR] No parquet for {args.date}")
            return 1
        print(f"OK: {res['path']}  counts={res['sig_counts']}")
        return 0

    reports = augment_all_days_with_sig_labels(use_goes=args.use_goes)
    print(f"\nAugmented {len(reports)} days.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
