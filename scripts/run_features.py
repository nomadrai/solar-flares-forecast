"""
TEJAS Phase 4 — Feature Engineering Batch Runner
==================================================
Processes all clean daily Parquet files through the feature engineering
pipeline.  Produces output in data/features/.

Usage:
    python scripts/run_features.py

Output:
    - Feature Parquet files in data/features/
    - feature_summary.csv alongside the Parquet files
"""

import sys
from pathlib import Path

# Add project root to path so imports work when run as script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tejas.features import engineer_all_days


def main():
    print("=" * 60)
    print("TEJAS — Phase 4: Feature Engineering")
    print("=" * 60)

    summary = engineer_all_days()

    if len(summary) > 0:
        print("\n\n=== FEATURE ENGINEERING SUMMARY ===\n")
        print(f"Days processed: {len(summary)}")
        print(f"Average features added: "
              f"{summary['n_features_added'].mean():.0f}")
        print(f"Average output columns: "
              f"{summary['n_cols_output'].mean():.0f}")
        print(f"Average processing time: "
              f"{summary['elapsed_s'].mean():.1f}s per day")
        print(f"Total processing time: "
              f"{summary['elapsed_s'].sum():.0f}s")


if __name__ == "__main__":
    main()
