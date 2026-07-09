"""
TEJAS Batch Ingestion Script
=============================
Processes all available days in the dataset, producing unified daily
Parquet files and an ingestion summary CSV.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --days 01 02 03
    python scripts/run_ingestion.py --cadence 30
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path so tejas package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tejas.config import YEAR, MONTH, MISSING_DAYS, UNIFIED_DIR
from tejas.ingestion.unified_builder import save_unified_day


def get_available_days():
    """Get list of day strings available for processing.

    Scans both SoLEXS and HEL1OS directories, returns days that
    have data in at least one instrument (excluding known missing days).

    Returns:
        Sorted list of 2-digit day strings ('01', '02', ...).
    """
    from tejas.config import SOLEXS_DIR, HEL1OS_DIR

    solexs_base = SOLEXS_DIR / YEAR / MONTH
    hel1os_base = HEL1OS_DIR / YEAR / MONTH

    days = set()
    if solexs_base.exists():
        days |= {d.name for d in solexs_base.iterdir() if d.is_dir()}
    if hel1os_base.exists():
        days |= {d.name for d in hel1os_base.iterdir() if d.is_dir()}

    # Exclude known missing days
    days -= MISSING_DAYS

    return sorted(days)


def main():
    parser = argparse.ArgumentParser(
        description="TEJAS Batch Ingestion — process raw FITS data into unified Parquet files."
    )
    parser.add_argument(
        "--days", nargs="+", default=None,
        help="Specific days to process (2-digit strings, e.g., 01 02 03). "
             "If not specified, processes all available days."
    )
    parser.add_argument(
        "--cadence", type=int, default=None,
        help="HEL1OS resampling cadence in seconds (default: 10)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for Parquet files. Default: data/processed/unified/YYYY/MM/"
    )
    args = parser.parse_args()

    # Determine which days to process
    if args.days:
        days = args.days
    else:
        days = get_available_days()

    if not days:
        print("No days found to process. Check data directories.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else UNIFIED_DIR

    print(f"+{'='*58}+")
    print(f"|  TEJAS Batch Ingestion -- {YEAR}-{MONTH}{'':<26}|")
    print(f"+{'='*58}+")
    print(f"|  Days to process: {len(days):<39}|")
    print(f"|  HEL1OS cadence:  {args.cadence or 'default (10s)':<39}|")
    print(f"|  Output:          {str(output_dir)[-39:]:<39}|")
    print(f"+{'='*58}+")

    # Process each day
    results = []
    t_start_total = time.time()

    for i, day in enumerate(days, 1):
        date_str = f"{YEAR}-{MONTH}-{day}"
        print(f"\n[{i}/{len(days)}] Processing {date_str}...")

        t_start = time.time()
        try:
            result = save_unified_day(
                date_str,
                output_dir=output_dir,
                hel1os_cadence_s=args.cadence,
            )

            elapsed = time.time() - t_start
            validation = result["validation"]
            unified = result["unified"]

            # Collect summary row
            summary = validation.summary_dict()
            summary["elapsed_s"] = round(elapsed, 1)
            summary["n_rows"] = len(unified)
            summary["solexs_valid_pct"] = round(
                unified["solexs_valid"].sum() / len(unified) * 100, 1
            ) if "solexs_valid" in unified.columns else None
            summary["hel1os_valid_pct"] = round(
                unified["hel1os_valid"].sum() / len(unified) * 100, 1
            ) if "hel1os_valid" in unified.columns else None

            results.append(summary)
            print(f"  Done in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  [FAIL] FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "date": date_str,
                "all_passed": False,
                "elapsed_s": round(elapsed, 1),
                "error": str(e),
            })

    # Save ingestion summary
    total_elapsed = time.time() - t_start_total
    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "ingestion_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    # Print final summary
    n_success = sum(1 for r in results if r.get("all_passed", False))
    n_failed = len(results) - n_success

    print(f"\n{'='*60}")
    print(f"BATCH INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:  {total_elapsed:.1f}s")
    print(f"  Processed:   {len(results)} days")
    print(f"  Passed:      {n_success}")
    print(f"  Issues:      {n_failed}")
    print(f"  Summary:     {summary_path}")
    print(f"  Parquet dir: {output_dir}")


if __name__ == "__main__":
    main()
