"""
TEJAS Phase 7 — Forecasting Batch Runner
==========================================
Assembles the training frame, fits 12 models (3 algorithms × 4 horizons),
evaluates them on the held-out test split, and writes reports.

Usage:
    python scripts/run_forecasting.py --prepare
    python scripts/run_forecasting.py --train
    python scripts/run_forecasting.py --evaluate
    python scripts/run_forecasting.py --all               # prepare + train + evaluate
    python scripts/run_forecasting.py --all --cadence 30  # 30s benchmark
    python scripts/run_forecasting.py --train --algorithm xgb

Outputs:
    - outputs/models/YYYY/MM/*.joblib             (persisted models)
    - outputs/models/YYYY/MM/feature_columns.json (frozen feature contract)
    - data/forecasting/YYYY/MM/training_frame_{c}s.parquet
    - data/forecasting/YYYY/MM/metrics_summary.csv
    - data/forecasting/YYYY/MM/feature_importance.csv
    - data/forecasting/YYYY/MM/test_metrics.csv
    - data/forecasting/YYYY/MM/lead_time_distribution.csv
    - data/forecasting/YYYY/MM/forecast_report.md
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work when run as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Windows consoles default to cp1252 and choke on Unicode arrows/checkmarks.
# Reconfigure both streams to UTF-8 up front.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (OSError, LookupError):
            pass

from tejas.config import (
    FORECAST_CADENCE_S,
    FORECAST_BENCHMARK_CADENCE_S,
    FORECAST_DIR,
    MODELS_DIR,
)
from tejas.forecasting.models import ALGORITHMS
from tejas.forecasting.pipeline import (
    prepare_training_frame,
    train_all_horizons,
    evaluate_all,
)


def _banner(title: str) -> None:
    print("=" * 60)
    print(f"TEJAS — {title}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TEJAS Phase 7 — Multi-horizon flare forecasting"
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="Build the downsampled training frame.",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Fit all (horizon, algorithm) combos and persist artifacts.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Reload models, evaluate on the test split, write forecast_report.md.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run prepare, train, and evaluate in sequence.",
    )
    parser.add_argument(
        "--cadence", type=int, default=FORECAST_CADENCE_S,
        choices=[FORECAST_CADENCE_S, FORECAST_BENCHMARK_CADENCE_S, 1],
        help=f"Training cadence in seconds (default: {FORECAST_CADENCE_S}).",
    )
    parser.add_argument(
        "--algorithm", type=str, default="all",
        choices=["all", *ALGORITHMS],
        help="Algorithm to train (default: all).",
    )
    args = parser.parse_args()

    if not (args.prepare or args.train or args.evaluate or args.all):
        parser.print_help()
        return 0

    if args.all:
        args.prepare = args.train = args.evaluate = True

    algorithms = None if args.algorithm == "all" else [args.algorithm]

    if args.prepare:
        _banner(f"Phase 7 · prepare training frame (cadence={args.cadence}s)")
        prepare_training_frame(cadence_s=args.cadence)

    if args.train:
        _banner(f"Phase 7 · train (cadence={args.cadence}s, algorithms={args.algorithm})")
        summary = train_all_horizons(
            cadence_s=args.cadence,
            algorithms=algorithms,
        )
        print("\n=== TRAINING SUMMARY ===\n")
        cols = [
            "algorithm", "horizon_min", "n_train", "n_val", "n_test",
            "scale_pos_weight", "decision_threshold",
            "val_pr_auc", "val_f1", "test_pr_auc", "test_f1",
        ]
        cols_present = [c for c in cols if c in summary.columns]
        print(summary[cols_present].to_string(index=False))

    if args.evaluate:
        _banner("Phase 7 · evaluate")
        # If a specific algorithm was requested, evaluate that one;
        # otherwise default to XGBoost for the primary report.
        eval_algo = args.algorithm if args.algorithm != "all" else "xgb"
        evaluate_all(cadence_s=args.cadence, algorithm=eval_algo)
        print(f"\nReports written to: {FORECAST_DIR}")

    print(f"\nModel artifacts: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
