"""
TEJAS Phase 3 — Batch Processing Script
=========================================
Runs the full data processing & cleaning pipeline on all unified daily
Parquet files.

Usage:
    python scripts/run_processing.py

Output:
    - Clean Parquet files in data/processed/clean/
    - processing_summary.csv alongside the Parquet files
"""

import sys
from pathlib import Path

# Add project root to path so imports work when run as script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tejas.processing import process_all_days


def main():
    print("=" * 60)
    print("TEJAS — Phase 3: Data Processing & Cleaning")
    print("=" * 60)

    summary = process_all_days()

    if len(summary) > 0:
        print("\n\n=== PROCESSING SUMMARY ===\n")
        print(f"Days processed: {len(summary)}")
        print(f"Total spikes removed: {int(summary['total_spikes_removed'].sum())}")
        print(f"Average CZT background: "
              f"{summary['czt_bg_median'].mean():.1f} c/s")
        print(f"Average quiet fraction: "
              f"{summary['czt_quiet_fraction'].mean()*100:.1f}%")
        print(f"Average processing time: "
              f"{summary['elapsed_s'].mean():.1f}s per day")
        print(f"Total processing time: "
              f"{summary['elapsed_s'].sum():.0f}s")


if __name__ == "__main__":
    main()
