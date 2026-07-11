"""
TEJAS Fix 7 — AR-Proxy Feature Augmentation
=============================================
Reads Phase 5 labelled feature parquets, computes AR-proxy features from
the master flare catalogue, and writes the augmented parquets back in
place.

Must be run after Phase 5 (which produces master_catalogue.csv).  Before
Phase 7 training.

Usage:
    python scripts/run_ar_proxy.py --all
    python scripts/run_ar_proxy.py --date 2026-05-01
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

from tejas.config import DETECTION_DIR
from tejas.features.ar_proxy import (
    augment_day_with_ar_proxy,
    augment_all_days_with_ar_proxy,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add AR-proxy features to Phase 5 parquets (Fix 7)."
    )
    parser.add_argument("--all", action="store_true",
                        help="Augment all days.")
    parser.add_argument("--date", type=str, default=None,
                        help="Augment a single day (YYYY-MM-DD).")
    args = parser.parse_args()

    if not (args.all or args.date):
        parser.print_help()
        return 0

    print(f"AR-proxy augmentation (Fix 7)")
    print(f"Detection dir: {DETECTION_DIR}")
    print(f"Catalogue:     {DETECTION_DIR / 'master_catalogue.csv'}")
    print("=" * 60)

    if args.date:
        res = augment_day_with_ar_proxy(args.date)
        if res is None:
            print(f"[ERROR] No parquet for {args.date}")
            return 1
        print(f"OK: {res['path']}")
        return 0

    reports = augment_all_days_with_ar_proxy()
    print(f"\nAugmented {len(reports)} days.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
