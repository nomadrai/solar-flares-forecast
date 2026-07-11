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
    train_significant_flare_models,
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
    parser.add_argument(
        "--sig", action="store_true",
        help="Fix 4: also train significant-flare (moderate/strong/severe) "
             "class-specific models per horizon.",
    )
    parser.add_argument(
        "--stack", action="store_true",
        help="Fix 5: train stacking meta-learner over LogReg + XGB + LGBM.",
    )
    parser.add_argument(
        "--discriminator", action="store_true",
        help="Fix 8: train Stage-2 false-alert discriminator.",
    )
    args = parser.parse_args()

    if not (
        args.prepare or args.train or args.evaluate or args.all
        or args.sig or args.stack or args.discriminator
    ):
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

    # ── Fix 4 ── Significant-flare class-specific models
    if args.sig:
        _banner(f"Phase 7 · train significant-flare models (cadence={args.cadence}s)")
        try:
            sig_summary = train_significant_flare_models(
                cadence_s=args.cadence,
                algorithms=algorithms,
            )
            if len(sig_summary):
                print("\n=== SIG-FLARE TRAINING SUMMARY ===\n")
                cols = [
                    "algorithm", "horizon_min", "n_train", "n_val", "n_test",
                    "n_positive_train", "decision_threshold",
                    "val_pr_auc", "val_recall", "val_false_alarm_rate",
                    "test_pr_auc", "test_recall", "test_false_alarm_rate",
                ]
                cols_present = [c for c in cols if c in sig_summary.columns]
                print(sig_summary[cols_present].to_string(index=False))
            else:
                print("[NOTE] No sig-flare models produced (likely no positives).")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[ERROR] Sig-flare training failed: {exc}")

    # ── Fix 5 ── Stacked ensemble
    if args.stack:
        _banner(f"Phase 7 · train stacked ensemble (cadence={args.cadence}s)")
        from tejas.forecasting.ensemble import train_stacked_ensemble
        try:
            stack_summary = train_stacked_ensemble(cadence_s=args.cadence)
            if len(stack_summary):
                print("\n=== STACKED ENSEMBLE SUMMARY ===\n")
                cols = [
                    "algorithm", "horizon_min", "decision_threshold",
                    "val_pr_auc", "test_pr_auc",
                    "test_recall", "test_false_alarm_rate",
                ]
                cols_present = [c for c in cols if c in stack_summary.columns]
                print(stack_summary[cols_present].to_string(index=False))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[ERROR] Stacked ensemble training failed: {exc}")

    # ── Fix 8 ── Two-stage discriminator
    if args.discriminator:
        _banner(f"Phase 7 · train false-alert discriminator (cadence={args.cadence}s)")
        from tejas.forecasting.discriminator import train_discriminator
        eval_algo = args.algorithm if args.algorithm != "all" else "xgb"
        try:
            disc_summary = train_discriminator(
                cadence_s=args.cadence,
                base_algorithm=eval_algo,
            )
            if len(disc_summary):
                print("\n=== DISCRIMINATOR SUMMARY ===\n")
                cols = [
                    "horizon_min", "n_test_candidates",
                    "n_test_true_alerts", "n_test_false_alerts",
                    "n_test_suppressed", "n_test_wrongly_suppressed",
                    "test_recall_after_gate", "test_precision_after_gate",
                ]
                cols_present = [c for c in cols if c in disc_summary.columns]
                print(disc_summary[cols_present].to_string(index=False))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[ERROR] Discriminator training failed: {exc}")

    print(f"\nModel artifacts: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
