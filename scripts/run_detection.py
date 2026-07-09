"""
TEJAS Phase 5 — Flare Detection Batch Runner
================================================
Processes all feature-engineered daily Parquet files through the flare
detection pipeline.  Produces output in data/detection/.

Usage:
    python scripts/run_detection.py               # run detection on all days
    python scripts/run_detection.py --validate     # validate against EDA catalogue

Output:
    - Labelled feature Parquet files in data/detection/YYYY/MM/
    - Per-day catalogue CSVs (master + independent)
    - Consolidated master_catalogue.csv
    - detection_summary.csv
"""

import sys
import argparse
from pathlib import Path

# Add project root to path so imports work when run as script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tejas.detection import detect_all_days
from tejas.config import DETECTION_DIR


def validate_against_eda(detection_dir=None):
    """Validate Phase 5 detections against the Phase 1 EDA catalogue.

    Loads the preliminary_flare_catalogue.csv from Phase 1 EDA and checks
    how many of those 13 manually identified events are recovered by the
    Phase 5 automatic detector.

    Args:
        detection_dir: Directory containing detection outputs.
    """
    if detection_dir is None:
        detection_dir = DETECTION_DIR

    detection_dir = Path(detection_dir)
    eda_path = (
        project_root / "outputs" / "eda" / "2026" / "05"
        / "preliminary_flare_catalogue.csv"
    )

    if not eda_path.exists():
        print(f"\n[WARN] EDA catalogue not found: {eda_path}")
        print("  Cannot validate. Skipping.")
        return

    # Load EDA catalogue
    eda = pd.read_csv(eda_path)
    eda["peak_time"] = pd.to_datetime(eda["peak_time"], utc=True).dt.tz_localize(None)
    print(f"\n{'='*60}")
    print(f"Validation Against Phase 1 EDA Catalogue")
    print(f"{'='*60}")
    print(f"  EDA events: {len(eda)} across {eda['date'].nunique()} days")

    # Load Phase 5 master catalogue
    master_path = detection_dir / "master_catalogue.csv"
    if not master_path.exists():
        print(f"  [ERROR] Master catalogue not found: {master_path}")
        print("  Run detection first.")
        return

    master = pd.read_csv(master_path)
    master["peak_time"] = pd.to_datetime(master["peak_time"], utc=True).dt.tz_localize(None)
    master["start_time"] = pd.to_datetime(master["start_time"], utc=True).dt.tz_localize(None)
    master["end_time"] = pd.to_datetime(master["end_time"], utc=True).dt.tz_localize(None)
    print(f"  Phase 5 events: {len(master)}")

    # Match each EDA event to a Phase 5 event
    match_window = pd.Timedelta(seconds=60)
    matched = 0
    unmatched_eda = []

    print(f"\n  Cross-matching (+/-60s of EDA peak time)...\n")
    print(f"  {'EDA Date':<14} {'EDA Peak':<28} {'Peak c/s':>10} "
          f"{'Match?':>8} {'P5 Peak':>12}")
    print(f"  {'-'*14} {'-'*28} {'-'*10} {'-'*8} {'-'*12}")

    for _, eda_event in eda.iterrows():
        eda_peak = eda_event["peak_time"]
        eda_date = eda_event["date"]

        # Find overlapping Phase 5 events
        day_master = master[master["date"] == eda_date]
        found = False

        for _, p5_event in day_master.iterrows():
            # Check if the EDA peak falls within the P5 event +/- window
            if (p5_event["start_time"] - match_window <= eda_peak <=
                    p5_event["end_time"] + match_window):
                found = True
                p5_peak = p5_event.get("peak_solexs_counts", "N/A")
                p5_peak_str = (
                    f"{p5_peak:.0f}" if pd.notna(p5_peak) else "N/A"
                )
                print(f"  {eda_date:<14} {str(eda_peak):<28} "
                      f"{eda_event['peak_counts_cps']:>10.0f} "
                      f"{'[Y]':>8} {p5_peak_str:>12}")
                matched += 1
                break

        if not found:
            unmatched_eda.append(eda_event)
            print(f"  {eda_date:<14} {str(eda_peak):<28} "
                  f"{eda_event['peak_counts_cps']:>10.0f} "
                  f"{'[N]':>8} {'---':>12}")

    # Summary
    recall = matched / len(eda) if len(eda) > 0 else 0
    print(f"\n  {'-'*60}")
    print(f"  Recall: {matched}/{len(eda)} = {recall:.1%}")

    if unmatched_eda:
        print(f"\n  Unmatched EDA events ({len(unmatched_eda)}):")
        for e in unmatched_eda:
            print(f"    {e['date']} {e['peak_time']} "
                  f"(peak={e['peak_counts_cps']:.0f} c/s, "
                  f"dur={e['duration_s']:.0f}s)")

    # Extra Phase 5 detections on EDA days
    eda_dates = set(eda["date"].unique())
    extra_events = master[master["date"].isin(eda_dates)]
    n_extra = len(extra_events) - matched
    if n_extra > 0:
        print(f"\n  Extra Phase 5 detections on EDA days: {n_extra}")
        print(f"  (These may be real events the EDA missed, or "
              f"false positives)")

    print(f"\n  Target: >=92% recall (>=12/13)")
    if recall >= 0.92:
        print(f"  PASSED")
    else:
        print(f"  BELOW TARGET -- consider tuning detection parameters")


def main():
    parser = argparse.ArgumentParser(
        description="TEJAS Phase 5 — Flare Detection"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate against Phase 1 EDA flare catalogue",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate (skip detection)",
    )
    args = parser.parse_args()

    if not args.validate_only:
        print("=" * 60)
        print("TEJAS — Phase 5: Flare Detection")
        print("=" * 60)

        summary = detect_all_days()

        if len(summary) > 0:
            print("\n\n=== DETECTION SUMMARY ===\n")
            print(f"Days processed: {len(summary)}")
            print(f"Total master events: "
                  f"{summary['n_master_events'].sum():.0f}")
            print(f"  Confirmed (soft+hard): "
                  f"{summary['n_confirmed'].sum():.0f}")
            print(f"  Soft-only: {summary['n_soft_only'].sum():.0f}")
            print(f"  Hard-only: {summary['n_hard_only'].sum():.0f}")
            print(f"Average processing time: "
                  f"{summary['elapsed_s'].mean():.1f}s per day")
            print(f"Total processing time: "
                  f"{summary['elapsed_s'].sum():.0f}s")

    if args.validate or args.validate_only:
        validate_against_eda()


if __name__ == "__main__":
    main()
