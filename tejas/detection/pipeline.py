"""
TEJAS Flare Detection Pipeline
==================================
Orchestrates all Phase 5 detection steps for a single day or all days.

Pipeline order:
  1. Load feature Parquet (Phase 4 output)
  2. Detect flares in each instrument (SoLEXS, CdTe, CZT)
  3. Build master catalogue (cross-match soft and hard detections)
  4. Attach per-timestep labels (flare_id, is_flare_active, flare_phase,
     label_flare_within_Xm)
  5. Save labelled Parquet + catalogue CSVs
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from tejas.config import (
    FEATURES_DIR,
    DETECTION_DIR,
    YEAR,
    MONTH,
    MISSING_DAYS,
    DETECTION_LABEL_HORIZONS_MIN,
)
from tejas.detection.catalogue import (
    detect_solexs_flares,
    detect_cdte_flares,
    detect_czt_flares,
    build_master_catalogue,
    events_to_dataframe,
)


# ── Single Day Pipeline ────────────────────────────────────────────────────


def detect_day(date_str, input_dir=None, output_dir=None, save=True):
    """Run the full Phase 5 detection pipeline for a single day.

    Pipeline:
      Load → Detect (3 instruments) → Master catalogue → Labels → Save

    Args:
        date_str: Date in 'YYYY-MM-DD' format.
        input_dir: Directory containing feature Parquet files (Phase 4 output).
            Defaults to config.FEATURES_DIR.
        output_dir: Directory for detection outputs.
            Defaults to config.DETECTION_DIR.
        save: If True, save outputs to disk.

    Returns:
        dict with keys:
          'df': DataFrame — labelled feature dataset.
          'catalogue': DataFrame — master catalogue for this day.
          'report': dict — detection statistics for this day.
        Returns None if the input file does not exist.
    """
    if input_dir is None:
        input_dir = FEATURES_DIR
    if output_dir is None:
        output_dir = DETECTION_DIR

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"Flare Detection: {date_str}")
    print(f"{'='*60}")

    # ── Step 0: Load feature Parquet ──────────────────────────────────
    input_path = input_dir / f"{date_str}.parquet"
    if not input_path.exists():
        print(f"  [SKIP] Input file not found: {input_path}")
        return None

    df = pd.read_parquet(input_path)
    n_rows = len(df)
    print(f"  Loaded: {n_rows} rows, {len(df.columns)} columns")

    # Ensure datetime column is proper datetime type
    if "datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["datetime"]
    ):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # ── Step 1: Independent detections ────────────────────────────────
    print(f"  [1/4] SoLEXS detection...")
    solexs_events = detect_solexs_flares(df)
    print(f"    --> {len(solexs_events)} events")

    print(f"  [2/4] CdTe detection...")
    cdte_events = detect_cdte_flares(df)
    print(f"    --> {len(cdte_events)} events")

    print(f"  [3/4] CZT detection...")
    czt_events = detect_czt_flares(df)
    print(f"    --> {len(czt_events)} events")

    # ── Step 2: Master catalogue ──────────────────────────────────────
    print(f"  [4/4] Building master catalogue...")
    catalogue = build_master_catalogue(
        solexs_events, cdte_events, czt_events, date_str
    )
    n_events = len(catalogue)
    n_confirmed = (catalogue["confidence"] == "confirmed").sum() if n_events > 0 else 0
    n_soft_only = (catalogue["confidence"] == "soft_only").sum() if n_events > 0 else 0
    n_hard_only = (catalogue["confidence"] == "hard_only").sum() if n_events > 0 else 0
    print(f"    --> {n_events} master events "
          f"({n_confirmed} confirmed, {n_soft_only} soft-only, "
          f"{n_hard_only} hard-only)")

    # ── Step 3: Attach per-timestep labels ────────────────────────────
    df = _attach_timestep_labels(df, catalogue)

    # ── Step 4: Save outputs ──────────────────────────────────────────
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save labelled feature Parquet
        parquet_path = output_dir / f"{date_str}.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {parquet_path.name} ({size_mb:.1f} MB)")

        # Save master catalogue CSV
        cat_path = output_dir / f"{date_str}_catalogue.csv"
        catalogue.to_csv(cat_path, index=False)
        print(f"  Saved: {cat_path.name}")

        # Save independent catalogues
        for name, events in [
            ("solexs", solexs_events),
            ("cdte", cdte_events),
            ("czt", czt_events),
        ]:
            ind_df = events_to_dataframe(events)
            ind_path = output_dir / f"{date_str}_{name}.csv"
            ind_df.to_csv(ind_path, index=False)

    # ── Build report ──────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Compute flare duty cycle
    n_valid = df["is_flare_active"].notna().sum()
    n_flare = (df["is_flare_active"] == 1).sum()
    duty_cycle = n_flare / n_valid if n_valid > 0 else 0.0

    max_peak = (
        catalogue["peak_solexs_counts"].max()
        if n_events > 0 and "peak_solexs_counts" in catalogue.columns
        else 0.0
    )

    total_flare_dur = (
        catalogue["duration_s"].sum() if n_events > 0 else 0.0
    )

    report = {
        "date": date_str,
        "n_solexs_events": len(solexs_events),
        "n_cdte_events": len(cdte_events),
        "n_czt_events": len(czt_events),
        "n_master_events": n_events,
        "n_confirmed": int(n_confirmed),
        "n_soft_only": int(n_soft_only),
        "n_hard_only": int(n_hard_only),
        "max_peak_counts": float(max_peak) if not pd.isna(max_peak) else 0.0,
        "total_flare_duration_s": float(total_flare_dur),
        "flare_duty_cycle": round(duty_cycle, 6),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"  Done in {elapsed:.1f}s")

    return {"df": df, "catalogue": catalogue, "report": report}


# ── Batch Processing ───────────────────────────────────────────────────────


def detect_all_days(input_dir=None, output_dir=None):
    """Detect flares for all available days.

    Discovers daily feature Parquet files and processes each one.
    Generates detection_summary.csv and master_catalogue.csv.

    Args:
        input_dir: Directory containing feature Parquet files.
        output_dir: Directory for detection outputs.

    Returns:
        pd.DataFrame — detection summary with one row per day.
    """
    if input_dir is None:
        input_dir = FEATURES_DIR
    if output_dir is None:
        output_dir = DETECTION_DIR

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Discover available days
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {input_dir}")
        return pd.DataFrame()

    dates = [f.stem for f in parquet_files]
    print(f"\nFound {len(dates)} days to process")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Process each day
    reports = []
    all_catalogues = []
    event_id_counter = 1  # Global event ID counter

    for date_str in dates:
        result = detect_day(
            date_str, input_dir=input_dir, output_dir=output_dir
        )
        if result is not None:
            reports.append(result["report"])

            # Renumber catalogue event_ids to be globally unique
            cat = result["catalogue"].copy()
            if len(cat) > 0:
                cat["event_id"] = range(
                    event_id_counter, event_id_counter + len(cat)
                )
                event_id_counter += len(cat)
            all_catalogues.append(cat)

    # Build summary DataFrame
    if not reports:
        print("\nNo days processed successfully.")
        return pd.DataFrame()

    summary = pd.DataFrame(reports)

    # Save summary CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "detection_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Save consolidated master catalogue
    if all_catalogues:
        master = pd.concat(all_catalogues, ignore_index=True)
        master_path = output_dir / "master_catalogue.csv"
        master.to_csv(master_path, index=False)
        total_events = len(master)
        print(f"\nConsolidated master catalogue: {total_events} events "
              f"across {len(reports)} days")
        print(f"Saved: {master_path}")

    print(f"\n{'='*60}")
    print(f"Detection complete: {len(reports)}/{len(dates)} days")
    print(f"Summary saved: {summary_path}")
    print(f"{'='*60}")

    return summary


# ── Per-Timestep Label Attachment ──────────────────────────────────────────


def _attach_timestep_labels(df, catalogue):
    """Attach per-timestep detection labels to the feature DataFrame.

    Adds columns:
      - flare_id: int, 0 = no active flare, else master event ID
      - is_flare_active: int, 1 during any flare window, 0 otherwise
      - flare_phase: str, "quiet" / "rising" / "peak" / "decaying"
      - label_minutes_to_next_flare: float, minutes until next flare start
      - label_flare_within_{h}m: int, binary forecasting labels

    Args:
        df: Feature DataFrame with 'datetime' column.
        catalogue: Master catalogue DataFrame for this day.

    Returns:
        df with new label columns added.
    """
    n = len(df)
    timestamps = df["datetime"]

    # Initialize columns
    df["flare_id"] = 0
    df["is_flare_active"] = 0
    df["flare_phase"] = "quiet"

    if len(catalogue) == 0:
        # No flares — all labels are quiet/0/NaN
        df["label_minutes_to_next_flare"] = np.nan
        for h in DETECTION_LABEL_HORIZONS_MIN:
            df[f"label_flare_within_{h}m"] = 0
        return df

    # ── Normalize timezone awareness ─────────────────────────────────
    # Real data may be tz-aware (UTC); catalogue timestamps from detector
    # are tz-naive.  Ensure consistency before comparison.
    ts_tz = getattr(timestamps.dt, "tz", None)

    def _tz_compat(t):
        """Make timestamp t tz-compatible with the DataFrame."""
        t = pd.Timestamp(t)
        if ts_tz is not None and t.tzinfo is None:
            t = t.tz_localize(ts_tz)
        elif ts_tz is None and t.tzinfo is not None:
            t = t.tz_localize(None)
        return t

    # ── Mark active flare windows ─────────────────────────────────────
    for _, event in catalogue.iterrows():
        event_id = event["event_id"]
        start = _tz_compat(event["start_time"])
        end = _tz_compat(event["end_time"])
        peak = _tz_compat(event["peak_time"])

        # Find rows within this event's time range
        mask = (timestamps >= start) & (timestamps <= end)

        df.loc[mask, "flare_id"] = int(event_id)
        df.loc[mask, "is_flare_active"] = 1

        # Determine flare_phase based on derivative
        # Use the best available SoLEXS derivative
        deriv_col = None
        for col in ["solexs_deriv1", "cdte_deriv1"]:
            if col in df.columns:
                deriv_col = col
                break

        if deriv_col is not None:
            event_rows = df.loc[mask]
            deriv = event_rows[deriv_col]

            # Peak: within ±2 seconds of peak time
            peak_mask = mask & (
                (timestamps >= peak - pd.Timedelta(seconds=2)) &
                (timestamps <= peak + pd.Timedelta(seconds=2))
            )
            df.loc[peak_mask, "flare_phase"] = "peak"

            # Rising: before peak (derivative > 0)
            rising_mask = mask & (timestamps < peak - pd.Timedelta(seconds=2))
            df.loc[rising_mask, "flare_phase"] = "rising"

            # Decaying: after peak (derivative < 0)
            decaying_mask = mask & (
                timestamps > peak + pd.Timedelta(seconds=2)
            )
            df.loc[decaying_mask, "flare_phase"] = "decaying"
        else:
            # No derivative available — just mark as "rising" before peak,
            # "decaying" after
            df.loc[mask & (timestamps < peak), "flare_phase"] = "rising"
            peak_mask = mask & (
                (timestamps >= peak - pd.Timedelta(seconds=2)) &
                (timestamps <= peak + pd.Timedelta(seconds=2))
            )
            df.loc[peak_mask, "flare_phase"] = "peak"
            df.loc[
                mask & (timestamps > peak + pd.Timedelta(seconds=2)),
                "flare_phase",
            ] = "decaying"

    # ── Forecasting labels (retrospective — uses future data) ─────────
    # These are ONLY valid for training, never for inference.

    # Get all flare start times (tz-compatible)
    flare_starts_raw = catalogue["start_time"].sort_values()
    flare_starts = np.array([_tz_compat(t) for t in flare_starts_raw])

    # label_minutes_to_next_flare
    ts_values = timestamps.values  # numpy datetime64
    minutes_to_next = np.full(n, np.nan)

    # Convert flare starts to numpy datetime64 for fast comparison
    flare_starts_np = np.array(
        [pd.Timestamp(t).asm8 for t in flare_starts],
        dtype="datetime64[ns]",
    )
    # Ensure ts_values is also in ns resolution
    if hasattr(ts_values, "astype"):
        ts_values_ns = ts_values.astype("datetime64[ns]")
    else:
        ts_values_ns = ts_values

    for i in range(n):
        # Find the next flare start after this timestamp
        future_starts = flare_starts_np[flare_starts_np > ts_values_ns[i]]
        if len(future_starts) > 0:
            delta_ns = (future_starts[0] - ts_values_ns[i])
            # Convert timedelta64 to minutes
            delta_min = delta_ns / np.timedelta64(1, "m")
            minutes_to_next[i] = delta_min

    df["label_minutes_to_next_flare"] = minutes_to_next

    # label_flare_within_{h}m
    for h in DETECTION_LABEL_HORIZONS_MIN:
        col_name = f"label_flare_within_{h}m"
        df[col_name] = (minutes_to_next <= h).astype(int)
        # Set to 0 where minutes_to_next is NaN (no future flare)
        df.loc[np.isnan(minutes_to_next), col_name] = 0

    return df
