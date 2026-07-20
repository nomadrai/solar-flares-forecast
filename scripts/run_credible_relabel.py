"""
Phase B runner — credible-event catalogue + forecasting relabel.

Steps:
  1. Load the Phase-6 classification master catalogue (has intensity + confidence).
  2. Attach GOES cross-match flag (external confirmation).
  3. Build the credible subset and write master_catalogue_credible.csv.
  4. v3 P1 (§2.1/§2.3): union in coverage-verified GOES flares that have no
     TEJAS counterpart (source="goes_only"); GOES windows without Aditya
     coverage are written to goes_uncovered_windows.csv for the training-time
     coverage mask.
  5. Relabel every day's Phase-5 detection parquet in place:
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
    FORECAST_GOES_UNION_MIN_COVERAGE,
    FORECAST_GOES_WINDOW_FALLBACK_MIN,
    GOES_UNCOVERED_WINDOWS_NAME,
)
from tejas.detection.credible import (
    build_credible_catalogue, append_goes_only, relabel_frame, summarise,
)
from tejas.validation.goes_match import (
    load_goes_flare_list, build_validation_report, find_missed_goes_flares,
)


def _flare_window(row):
    """(begin, end) for one GOES flare; end falls back to peak + config min."""
    begin = pd.to_datetime(row["begin_time"])
    end = pd.to_datetime(row.get("end_time"))
    if pd.isna(end):
        end = pd.to_datetime(row["peak_time"]) + pd.Timedelta(
            minutes=FORECAST_GOES_WINDOW_FALLBACK_MIN
        )
    return begin, end


def check_aditya_coverage(goes_flares):
    """Fraction of expected rows Aditya actually has inside each flare window.

    Reads the per-day detection parquet for each flare's date and compares
    the row count inside [begin, end] against the count implied by the
    day's median cadence.  Missing day file → coverage 0.

    Returns:
        goes_flares copy with added ``coverage_fraction`` column.
    """
    out = goes_flares.copy().reset_index(drop=True)
    out["coverage_fraction"] = 0.0
    day_cache: dict[str, pd.Series | None] = {}

    for i, row in out.iterrows():
        begin, end = _flare_window(row)
        date_str = begin.strftime("%Y-%m-%d")

        if date_str not in day_cache:
            pq = Path(DETECTION_DIR) / f"{date_str}.parquet"
            if pq.exists():
                ts = pd.to_datetime(
                    pd.read_parquet(pq, columns=["datetime"])["datetime"]
                )
                try:
                    ts = ts.dt.tz_localize(None)
                except (TypeError, AttributeError):
                    pass
                day_cache[date_str] = ts.sort_values().reset_index(drop=True)
            else:
                day_cache[date_str] = None

        ts = day_cache[date_str]
        if ts is None or len(ts) < 2:
            continue

        in_window = ts[(ts >= begin) & (ts <= end)]
        window_s = max((end - begin).total_seconds(), 1.0)
        cadence_s = max(ts.diff().dt.total_seconds().median(), 1e-6)
        expected = window_s / cadence_s
        out.at[i, "coverage_fraction"] = min(len(in_window) / expected, 1.0)

    return out


def main():
    cat_path = Path(CLASSIFICATION_DIR) / "master_catalogue.csv"
    catalogue = pd.read_csv(cat_path)
    print(f"Loaded classification catalogue: {len(catalogue)} events")

    # ── GOES cross-match flag ──────────────────────────────────────────
    goes = pd.DataFrame()
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
    n_tejas_credible = len(credible)

    # ── v3 P1 §2.1: GOES-union — DISABLED (P1 rollback) ────────────────
    # P1 evaluation showed GOES-only flares are feature-silent events.
    # Training on them degrades precision without buying recall.
    # Keep infrastructure for future use, but don't append for now.
    uncovered = pd.DataFrame()
    n_goes_only_added = 0
    if len(goes):
        missed = find_missed_goes_flares(catalogue, goes,
                                          tolerance_s=GOES_VALIDATION_TOLERANCE_S)
        missed = check_aditya_coverage(missed)
        covered_mask = missed["coverage_fraction"] >= FORECAST_GOES_UNION_MIN_COVERAGE
        covered = missed.loc[covered_mask].reset_index(drop=True)
        uncovered = missed.loc[~covered_mask].reset_index(drop=True)

        # ROLLBACK: Do NOT append goes_only events to the credible catalogue
        # credible = append_goes_only(credible, covered)
        n_goes_only_added = 0  # was: len(covered)
        print(f"GOES-union: {len(missed)} GOES flares without TEJAS counterpart "
              f"-> {len(covered)} coverage-verified, {len(uncovered)} uncovered "
              f"(union DISABLED per P1 rollback)")
    else:
        print("GOES-union skipped (no GOES list).")

    out_path = Path(CLASSIFICATION_DIR) / CREDIBLE_CATALOGUE_NAME
    credible.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({n_tejas_credible} tejas + "
          f"{n_goes_only_added} goes_only = {len(credible)} events)")

    # ── v3 P1 §2.3: uncovered GOES windows → training-time mask file ──
    unc_path = Path(CLASSIFICATION_DIR) / GOES_UNCOVERED_WINDOWS_NAME
    if len(uncovered):
        windows = []
        for _, row in uncovered.iterrows():
            begin, end = _flare_window(row)
            windows.append({
                "begin_time": begin, "end_time": end,
                "goes_class": row.get("goes_class"),
                "coverage_fraction": row["coverage_fraction"],
            })
        pd.DataFrame(windows).to_csv(unc_path, index=False)
        print(f"Wrote {unc_path} ({len(windows)} uncovered GOES windows)")
    else:
        # Truthful empty file so training's mask loader sees current state.
        pd.DataFrame(
            columns=["begin_time", "end_time", "goes_class", "coverage_fraction"]
        ).to_csv(unc_path, index=False)
        print(f"Wrote {unc_path} (0 uncovered GOES windows)")

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
