"""
TEJAS EDA Script 2: Flare Day Deep Dives
=========================================
Auto-detects days with significant flare activity from SoLEXS
peak count rates, then generates detailed multi-panel interactive
Plotly plots for each active day, including per-band HEL1OS data.

Also produces a preliminary flare catalogue CSV.

Usage:  python scripts/eda_02_flare_days.py
        (set PYTHONIOENCODING=utf-8 on Windows)

Outputs:
  outputs/eda/flare_day_YYYYMMDD.html          -- per-day plots
  outputs/eda/preliminary_flare_catalogue.csv   -- detected events
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eda_utils as utils

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -- Configuration --------------------------------------------------------

FLARE_DAY_THRESHOLD_CPS = 100   # Min peak SoLEXS c/s to qualify as flare day

# Flare detection parameters (preliminary -- refined in Phase 5)
BASELINE_WINDOW_S = 600         # Rolling median window (seconds)
THRESHOLD_FACTOR = 3.0          # Counts > factor * baseline
MIN_PEAK_CPS = 50               # Absolute minimum peak to count as a flare
MIN_DURATION_S = 5              # Minimum sustained seconds above threshold

# Resampling for HEL1OS per-band plots (smooths integer count data)
HELIOS_RESAMPLE_S = 10

# CdTe individual band columns & labels (excluding full-range)
CDTE_BAND_COLS   = ["ctr_5_20", "ctr_20_30", "ctr_30_40", "ctr_40_60"]
CDTE_BAND_LABELS = ["5-20 keV", "20-30 keV", "30-40 keV", "40-60 keV"]
CDTE_BAND_COLORS = ["#FFEE58", "#FFA726", "#FF7043", "#EF5350"]
CDTE_FULL_COL    = "ctr_1p8_90"

# CZT individual band columns & labels (excluding full-range)
CZT_BAND_COLS   = ["ctr_20_40", "ctr_40_60", "ctr_60_80", "ctr_80_150"]
CZT_BAND_LABELS = ["20-40 keV", "40-60 keV", "60-80 keV", "80-150 keV"]
CZT_BAND_COLORS = ["#26C6DA", "#42A5F5", "#5C6BC0", "#7E57C2"]
CZT_FULL_COL    = "ctr_18_160"


# -- Flare Day Identification ---------------------------------------------

def identify_flare_days():
    """Quick pass: load each SoLEXS LC, return days with peak > threshold.

    Returns list of (day_str, peak_cps) sorted by peak descending.
    """
    solexs_days, _ = utils.get_available_days()
    active = []

    for day in solexs_days:
        date_str = utils.day_to_date_str(day)
        lc = utils.read_solexs_lc(date_str)
        if lc is not None:
            peak = lc["counts"].max()
            if peak >= FLARE_DAY_THRESHOLD_CPS:
                active.append((day, peak))

    active.sort(key=lambda x: x[1], reverse=True)
    return active


# -- Flare Event Detection ------------------------------------------------

def detect_flares_in_day(lc_df, date_str):
    """Detect individual flare events in a single day's SoLEXS light curve.

    Method:
      1. Rolling median as baseline (window = BASELINE_WINDOW_S).
      2. A point is 'active' if counts > max(baseline * THRESHOLD_FACTOR,
         MIN_PEAK_CPS).
      3. Group consecutive active points into events.
      4. Require minimum duration.

    Returns list of dicts with flare metadata.
    """
    df = lc_df.copy().reset_index(drop=True)

    df["baseline"] = df["counts"].rolling(
        window=BASELINE_WINDOW_S, center=True, min_periods=60
    ).median()
    df["baseline"] = df["baseline"].bfill().ffill()

    df["threshold"] = np.maximum(
        df["baseline"] * THRESHOLD_FACTOR, MIN_PEAK_CPS
    )
    df["above"] = df["counts"] > df["threshold"]
    df["group"] = (df["above"] != df["above"].shift()).cumsum()

    flares = []
    flare_id = 0

    for _, grp in df[df["above"]].groupby("group"):
        duration = grp["time_unix"].iloc[-1] - grp["time_unix"].iloc[0]
        if duration < MIN_DURATION_S or len(grp) < MIN_DURATION_S:
            continue

        peak_idx = grp["counts"].idxmax()
        flare_id += 1

        flares.append({
            "date": date_str,
            "flare_id": flare_id,
            "start_time": str(grp["datetime"].iloc[0]),
            "peak_time": str(grp.loc[peak_idx, "datetime"]),
            "end_time": str(grp["datetime"].iloc[-1]),
            "start_unix": grp["time_unix"].iloc[0],
            "peak_unix": grp.loc[peak_idx, "time_unix"],
            "end_unix": grp["time_unix"].iloc[-1],
            "duration_s": round(duration, 1),
            "peak_counts_cps": round(float(grp.loc[peak_idx, "counts"]), 1),
            "mean_counts_cps": round(float(grp["counts"].mean()), 1),
        })

    return flares


# -- Detector Combining ---------------------------------------------------

def combine_detector_bands(df, band_cols):
    """Average per-band count rates across detectors at each 1-s timestamp."""
    available = [c for c in band_cols if c in df.columns]
    if not available:
        return None

    df = df.copy()
    df["time_rounded"] = np.round(df["time_unix"])
    agg = {c: "mean" for c in available}
    result = df.groupby("time_rounded").agg(agg).reset_index()
    result.rename(columns={"time_rounded": "time_unix"}, inplace=True)
    result["datetime"] = utils.unix_to_datetime(result["time_unix"])
    return result


# -- Plotting --------------------------------------------------------------

def plot_flare_day(date_str, lc, pi, gti, cdte_df, czt_df, flares):
    """Generate a 6-panel interactive Plotly plot for one flare day.

    Panels:
      1. SoLEXS total LC (with GTI shading)
      2. SoLEXS multi-band (low / mid / high from PI)
      3. Hardness ratio (60-s median smoothed)
      4. SoLEXS + CdTe full-range overlay (dual y-axis)
      5. CdTe individual energy bands (10-s resampled)
      6. CZT individual energy bands (10-s resampled)
    """
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],   # dual y for overlay
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        subplot_titles=[
            "SoLEXS SDD2 - Total Light Curve (2-22 keV)",
            "SoLEXS SDD2 - Multi-Band Light Curves (from PI)",
            "Hardness Ratio (10-22 keV / 2.8-5 keV) [60s median]",
            "Soft + Hard X-ray Overlay (SoLEXS vs CdTe full-range)",
            "HEL1OS CdTe - Individual Energy Bands (10s bins)",
            "HEL1OS CZT - Individual Energy Bands (10s bins)",
        ],
    )

    # ---- Panel 1: SoLEXS total LC + GTI shading ----
    if lc is not None and len(lc) > 0:
        fig.add_trace(
            go.Scattergl(
                x=lc["datetime"], y=lc["counts"],
                mode="lines", line=dict(width=0.5, color="#42A5F5"),
                name="SoLEXS total",
                hovertemplate="%{x|%H:%M:%S}<br>%{y:.1f} c/s<extra></extra>",
            ),
            row=1, col=1,
        )
        if gti is not None:
            for _, g in gti.iterrows():
                fig.add_vrect(
                    x0=g["start_dt"], x1=g["stop_dt"],
                    fillcolor="green", opacity=0.05,
                    layer="below", line_width=0,
                    row=1, col=1,
                )
    fig.update_yaxes(type="log", title_text="c/s", row=1, col=1)

    # ---- Panel 2: SoLEXS multi-band ----
    if pi is not None and len(pi) > 0:
        for col, label, color in [
            ("counts_low",  "2.8-5 keV (low)",   "#66BB6A"),
            ("counts_mid",  "5-10 keV (mid)",     "#FFA726"),
            ("counts_high", "10-22 keV (high)",   "#EF5350"),
        ]:
            fig.add_trace(
                go.Scattergl(
                    x=pi["datetime"], y=pi[col],
                    mode="lines", line=dict(width=0.4, color=color),
                    name=label,
                    hovertemplate="%{x|%H:%M:%S}<br>%{y:.1f} c/s<extra></extra>",
                ),
                row=2, col=1,
            )
    fig.update_yaxes(type="log", title_text="c/s", row=2, col=1)

    # ---- Panel 3: Hardness ratio ----
    if pi is not None and len(pi) > 0:
        hr_smooth = pi["hardness_ratio"].rolling(
            window=60, center=True, min_periods=10
        ).median()
        fig.add_trace(
            go.Scattergl(
                x=pi["datetime"], y=hr_smooth,
                mode="lines", line=dict(width=0.7, color="#AB47BC"),
                name="Hardness ratio",
                hovertemplate="%{x|%H:%M:%S}<br>HR: %{y:.3f}<extra></extra>",
            ),
            row=3, col=1,
        )
    fig.update_yaxes(title_text="High / Low", row=3, col=1)

    # ---- Panel 4: SoLEXS + CdTe overlay (dual y) ----
    if lc is not None and len(lc) > 0:
        lc_10s = utils.resample_to_cadence(lc, 10, value_cols="counts")
        fig.add_trace(
            go.Scattergl(
                x=lc_10s["datetime"], y=lc_10s["counts"],
                mode="lines", line=dict(width=0.8, color="#42A5F5"),
                name="SoLEXS (10s)",
                hovertemplate="%{x|%H:%M:%S}<br>SoLEXS: %{y:.1f} c/s<extra></extra>",
            ),
            row=4, col=1, secondary_y=False,
        )

    if cdte_df is not None and CDTE_FULL_COL in cdte_df.columns:
        cdte_10s = utils.resample_to_cadence(
            cdte_df, 10, value_cols=CDTE_FULL_COL
        )
        fig.add_trace(
            go.Scattergl(
                x=cdte_10s["datetime"], y=cdte_10s[CDTE_FULL_COL],
                mode="lines", line=dict(width=0.8, color="#FF7043"),
                name="CdTe full (10s)",
                hovertemplate="%{x|%H:%M:%S}<br>CdTe: %{y:.3f} c/s<extra></extra>",
            ),
            row=4, col=1, secondary_y=True,
        )

    fig.update_yaxes(type="log", title_text="SoLEXS c/s",
                     row=4, col=1, secondary_y=False)
    fig.update_yaxes(type="log", title_text="CdTe c/s",
                     row=4, col=1, secondary_y=True)

    # ---- Panel 5: CdTe individual bands ----
    if cdte_df is not None and len(cdte_df) > 0:
        avail = [c for c in CDTE_BAND_COLS if c in cdte_df.columns]
        if avail:
            cdte_rs = utils.resample_to_cadence(
                cdte_df, HELIOS_RESAMPLE_S, value_cols=avail
            )
            for col, label, color in zip(
                CDTE_BAND_COLS, CDTE_BAND_LABELS, CDTE_BAND_COLORS
            ):
                if col in cdte_rs.columns:
                    fig.add_trace(
                        go.Scattergl(
                            x=cdte_rs["datetime"], y=cdte_rs[col],
                            mode="lines", line=dict(width=0.5, color=color),
                            name=f"CdTe {label}",
                            hovertemplate=(
                                f"%{{x|%H:%M:%S}}<br>"
                                f"{label}: %{{y:.3f}} c/s<extra></extra>"
                            ),
                        ),
                        row=5, col=1,
                    )
    fig.update_yaxes(type="log", title_text="c/s", row=5, col=1)

    # ---- Panel 6: CZT individual bands ----
    if czt_df is not None and len(czt_df) > 0:
        avail = [c for c in CZT_BAND_COLS if c in czt_df.columns]
        if avail:
            czt_rs = utils.resample_to_cadence(
                czt_df, HELIOS_RESAMPLE_S, value_cols=avail
            )
            for col, label, color in zip(
                CZT_BAND_COLS, CZT_BAND_LABELS, CZT_BAND_COLORS
            ):
                if col in czt_rs.columns:
                    fig.add_trace(
                        go.Scattergl(
                            x=czt_rs["datetime"], y=czt_rs[col],
                            mode="lines", line=dict(width=0.5, color=color),
                            name=f"CZT {label}",
                            hovertemplate=(
                                f"%{{x|%H:%M:%S}}<br>"
                                f"{label}: %{{y:.3f}} c/s<extra></extra>"
                            ),
                        ),
                        row=6, col=1,
                    )
    fig.update_yaxes(type="log", title_text="c/s", row=6, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=6, col=1)

    # ---- Flare peak annotations on Panel 1 ----
    for fl in flares:
        peak_dt = pd.Timestamp(fl["peak_time"])
        fig.add_annotation(
            x=peak_dt, y=fl["peak_counts_cps"],
            text=f"F{fl['flare_id']}",
            showarrow=True, arrowhead=2,
            arrowcolor="yellow", arrowwidth=1.5,
            font=dict(color="yellow", size=10),
            bgcolor="rgba(0,0,0,0.5)",
            row=1, col=1,
        )

    # ---- Layout ----
    n_ev = len(flares)
    fig.update_layout(
        title=dict(
            text=(f"TEJAS - Flare Day Deep Dive: {date_str}  "
                  f"({n_ev} event{'s' if n_ev != 1 else ''} detected)"),
            font=dict(size=16),
        ),
        height=1600,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.005,
            xanchor="center", x=0.5, font=dict(size=9),
        ),
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(t=100, b=60, l=70, r=70),
    )

    return fig


# -- Main ------------------------------------------------------------------

def main():
    t0 = time.time()
    utils.ensure_output_dir()

    print("=" * 72)
    print("  TEJAS EDA Script 2 - Flare Day Deep Dives")
    print("=" * 72)

    # -- Step 1: identify flare days ------------------------------------
    print(f"\n-- Identifying flare days (threshold: {FLARE_DAY_THRESHOLD_CPS} c/s) --")
    flare_days = identify_flare_days()
    print(f"  Found {len(flare_days)} active day(s):\n")
    for day, peak in flare_days:
        print(f"    May {day}:  {peak:>8.1f} c/s peak")

    if not flare_days:
        print("\n  No flare days found. Exiting.")
        return

    # -- Step 2: process each flare day ---------------------------------
    print(f"\n-- Processing {len(flare_days)} flare day(s) --")
    all_flares = []

    for idx, (day, peak) in enumerate(flare_days, 1):
        date_str = utils.day_to_date_str(day)
        compact = f"2026{utils.MONTH}{day}"

        print(f"\n  [{idx}/{len(flare_days)}] {date_str}  (peak: {peak:.1f} c/s)")

        # Load SoLEXS
        lc = utils.read_solexs_lc(date_str)
        print(f"    SoLEXS LC:  {len(lc) if lc is not None else 0:>6,} rows")

        pi = utils.read_solexs_pi(date_str)
        print(f"    SoLEXS PI:  {len(pi) if pi is not None else 0:>6,} spectra")

        gti = utils.read_solexs_gti(date_str)
        n_gti = len(gti) if gti is not None else 0
        gti_h = round(gti["duration_s"].sum() / 3600, 1) if gti is not None else 0
        print(f"    SoLEXS GTI: {n_gti} interval(s), {gti_h}h valid")

        # Load HEL1OS all bands
        cdte_raw = utils.read_hel1os_all_bands(date_str, "cdte")
        czte_raw = utils.read_hel1os_all_bands(date_str, "czt")

        if cdte_raw is not None:
            n_det = cdte_raw["detector"].nunique()
            bands = [c for c in cdte_raw.columns if c.startswith("ctr_")]
            print(f"    CdTe bands: {len(cdte_raw):>6,} rows, "
                  f"{n_det} det(s), {len(bands)} band cols")
        else:
            print(f"    CdTe bands: NO DATA")

        if czte_raw is not None:
            n_det = czte_raw["detector"].nunique()
            bands = [c for c in czte_raw.columns if c.startswith("ctr_")]
            print(f"    CZT  bands: {len(czte_raw):>6,} rows, "
                  f"{n_det} det(s), {len(bands)} band cols")
        else:
            print(f"    CZT  bands: NO DATA")

        # Combine detectors
        all_cdte_cols = CDTE_BAND_COLS + [CDTE_FULL_COL]
        cdte_combined = (
            combine_detector_bands(cdte_raw, all_cdte_cols)
            if cdte_raw is not None else None
        )
        all_czt_cols = CZT_BAND_COLS + [CZT_FULL_COL]
        czt_combined = (
            combine_detector_bands(czte_raw, all_czt_cols)
            if czte_raw is not None else None
        )

        # Detect flares
        day_flares = detect_flares_in_day(lc, date_str) if lc is not None else []
        all_flares.extend(day_flares)
        print(f"    Flares:     {len(day_flares)} event(s) detected")

        for fl in day_flares:
            print(f"      F{fl['flare_id']}: peak {fl['peak_counts_cps']:.0f} c/s  "
                  f"dur {fl['duration_s']:.0f}s  "
                  f"@ {fl['peak_time'][:19]}")

        # Generate plot
        fig = plot_flare_day(
            date_str, lc, pi, gti,
            cdte_combined, czt_combined, day_flares,
        )
        plot_path = utils.OUTPUT_DIR / f"flare_day_{compact}.html"
        fig.write_html(str(plot_path), include_plotlyjs=True)
        print(f"    Plot saved: {plot_path.name}")

    # -- Step 3: save flare catalogue -----------------------------------
    print(f"\n-- Flare Catalogue --")
    if all_flares:
        cat = pd.DataFrame(all_flares)
        # Sort by peak counts descending
        cat = cat.sort_values("peak_counts_cps", ascending=False).reset_index(
            drop=True
        )
        cat_path = utils.OUTPUT_DIR / "preliminary_flare_catalogue.csv"
        cat.to_csv(cat_path, index=False)

        print(f"  {len(cat)} total events across {len(flare_days)} days")
        print(f"  Saved -> {cat_path}")
        print(f"\n  Top 10 events:")
        for _, fl in cat.head(10).iterrows():
            print(f"    {fl['date']} F{fl['flare_id']:>2d}:  "
                  f"{fl['peak_counts_cps']:>7.1f} c/s  "
                  f"dur {fl['duration_s']:>6.0f}s  "
                  f"@ {fl['peak_time'][:19]}")
    else:
        print("  No flare events detected.")

    elapsed = time.time() - t0
    print(f"\n-- Completed in {elapsed:.1f}s --")


if __name__ == "__main__":
    main()
