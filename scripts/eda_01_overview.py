"""
TEJAS EDA Script 1: Month-Wide Light Curve Overview
====================================================
Loads all 30 days of SoLEXS and HEL1OS data, plots them on a
continuous time axis using interactive Plotly, and produces a
data availability summary.

Usage:  python scripts/eda_01_overview.py

Outputs:
  outputs/eda/month_overview.html   — interactive 3-panel Plotly plot
  outputs/eda/data_availability.csv — per-day availability statistics
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Ensure eda_utils is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eda_utils as utils

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_solexs_lc():
    """Load and concatenate SoLEXS light curves for all available days."""
    solexs_days, _ = utils.get_available_days()
    frames = []

    for day in solexs_days:
        date_str = utils.day_to_date_str(day)
        df = utils.read_solexs_lc(date_str)
        if df is not None:
            df["day"] = day
            frames.append(df)
            print(f"  Day {day}: {len(df):>7,} rows  "
                  f"[{df['datetime'].min().strftime('%H:%M:%S')} – "
                  f"{df['datetime'].max().strftime('%H:%M:%S')}]  "
                  f"median {df['counts'].median():.1f} c/s  "
                  f"max {df['counts'].max():.1f} c/s")
        else:
            print(f"  Day {day}: NO DATA")

    if not frames:
        print("  ERROR: No SoLEXS data found!")
        return None

    combined = (pd.concat(frames, ignore_index=True)
                .sort_values("time_unix")
                .reset_index(drop=True))
    print(f"\n  Total: {len(combined):,} rows  |  "
          f"{combined['datetime'].min().strftime('%Y-%m-%d %H:%M')} → "
          f"{combined['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
    return combined


def load_all_hel1os(detector_type):
    """Load and concatenate HEL1OS full-range light curves for all days."""
    _, hel1os_days = utils.get_available_days()
    frames = []

    for day in hel1os_days:
        date_str = utils.day_to_date_str(day)
        df = utils.read_hel1os_full_range(date_str, detector_type)
        if df is not None:
            df["day"] = day
            frames.append(df)
            n_dumps = df["dump"].nunique()
            n_dets = df["detector"].nunique()
            print(f"  Day {day}: {len(df):>8,} rows  "
                  f"{n_dumps} dump(s), {n_dets} det(s)  "
                  f"median {df['count_rate'].median():.4f} c/s  "
                  f"max {df['count_rate'].max():.4f} c/s")
        else:
            print(f"  Day {day}: NO DATA")

    if not frames:
        print(f"  ERROR: No HEL1OS {detector_type} data found!")
        return None

    combined = (pd.concat(frames, ignore_index=True)
                .sort_values("time_unix")
                .reset_index(drop=True))
    print(f"\n  Total: {len(combined):,} rows  |  "
          f"{combined['datetime'].min().strftime('%Y-%m-%d %H:%M')} → "
          f"{combined['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
    return combined


# ── Data Availability ────────────────────────────────────────────────────────

def generate_data_availability(solexs_all, cdte_all, czt_all):
    """Generate per-day data availability summary CSV."""
    solexs_days, hel1os_days = utils.get_available_days()
    all_days = sorted(set(solexs_days) | set(hel1os_days))

    rows = []
    for day in all_days:
        date_str = utils.day_to_date_str(day)
        row = {"date": date_str, "day": int(day)}

        # ── SoLEXS LC ──
        if solexs_all is not None:
            s = solexs_all[solexs_all["day"] == day]
            row["solexs_rows"] = len(s)
            if len(s) > 0:
                row["solexs_span_h"] = round(
                    (s["time_unix"].max() - s["time_unix"].min()) / 3600, 2)
                row["solexs_median_cps"] = round(s["counts"].median(), 2)
                row["solexs_max_cps"] = round(s["counts"].max(), 2)
            else:
                row["solexs_span_h"] = 0
                row["solexs_median_cps"] = 0
                row["solexs_max_cps"] = 0
        else:
            row.update(solexs_rows=0, solexs_span_h=0,
                       solexs_median_cps=0, solexs_max_cps=0)

        # ── SoLEXS GTI ──
        gti = utils.read_solexs_gti(date_str)
        if gti is not None and len(gti) > 0:
            row["solexs_gti_intervals"] = len(gti)
            row["solexs_gti_valid_h"] = round(
                gti["duration_s"].sum() / 3600, 2)
        else:
            row["solexs_gti_intervals"] = 0
            row["solexs_gti_valid_h"] = 0

        # ── HEL1OS CdTe ──
        if cdte_all is not None:
            c = cdte_all[cdte_all["day"] == day]
            row["cdte_rows"] = len(c)
            if len(c) > 0:
                row["cdte_span_h"] = round(
                    (c["time_unix"].max() - c["time_unix"].min()) / 3600, 2)
                row["cdte_dumps"] = c["dump"].nunique()
                row["cdte_median_cps"] = round(c["count_rate"].median(), 4)
                row["cdte_max_cps"] = round(c["count_rate"].max(), 4)
            else:
                row.update(cdte_span_h=0, cdte_dumps=0,
                           cdte_median_cps=0, cdte_max_cps=0)
        else:
            row.update(cdte_rows=0, cdte_span_h=0, cdte_dumps=0,
                       cdte_median_cps=0, cdte_max_cps=0)

        # ── HEL1OS CZT ──
        if czt_all is not None:
            z = czt_all[czt_all["day"] == day]
            row["czt_rows"] = len(z)
            if len(z) > 0:
                row["czt_span_h"] = round(
                    (z["time_unix"].max() - z["time_unix"].min()) / 3600, 2)
                row["czt_median_cps"] = round(z["count_rate"].median(), 4)
                row["czt_max_cps"] = round(z["count_rate"].max(), 4)
            else:
                row.update(czt_span_h=0, czt_median_cps=0, czt_max_cps=0)
        else:
            row.update(czt_rows=0, czt_span_h=0,
                       czt_median_cps=0, czt_max_cps=0)

        rows.append(row)

    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_month_overview(solexs_60s, cdte_60s, czt_60s):
    """Generate an interactive 3-panel Plotly month overview plot.

    Uses Scattergl (WebGL) for performance with large point counts.
    All panels share the x-axis for synchronized zooming.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "SoLEXS SDD2 — Soft X-ray (2–22 keV)",
            "HEL1OS CdTe — Hard X-ray (full range, combined detectors)",
            "HEL1OS CZT — High-Energy X-ray (full range, combined detectors)",
        ],
    )

    # ── Panel 1: SoLEXS ──
    if solexs_60s is not None and len(solexs_60s) > 0:
        fig.add_trace(
            go.Scattergl(
                x=solexs_60s["datetime"],
                y=solexs_60s["counts"],
                mode="lines",
                line=dict(width=0.8, color="#42A5F5"),
                name="SoLEXS (2–22 keV)",
                hovertemplate="%{x|%b %d %H:%M:%S}<br>"
                              "Counts: %{y:.1f} c/s<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Panel 2: CdTe ──
    if cdte_60s is not None and len(cdte_60s) > 0:
        fig.add_trace(
            go.Scattergl(
                x=cdte_60s["datetime"],
                y=cdte_60s["count_rate"],
                mode="lines",
                line=dict(width=0.8, color="#FF7043"),
                name="CdTe (full range)",
                hovertemplate="%{x|%b %d %H:%M:%S}<br>"
                              "Rate: %{y:.4f} c/s<extra></extra>",
            ),
            row=2, col=1,
        )

    # ── Panel 3: CZT ──
    if czt_60s is not None and len(czt_60s) > 0:
        fig.add_trace(
            go.Scattergl(
                x=czt_60s["datetime"],
                y=czt_60s["count_rate"],
                mode="lines",
                line=dict(width=0.8, color="#66BB6A"),
                name="CZT (full range)",
                hovertemplate="%{x|%b %d %H:%M:%S}<br>"
                              "Rate: %{y:.4f} c/s<extra></extra>",
            ),
            row=3, col=1,
        )

    # ── Axes ──
    fig.update_yaxes(type="log", title_text="Counts/s", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Counts/s", row=2, col=1)
    fig.update_yaxes(type="log", title_text="Counts/s", row=3, col=1)
    fig.update_xaxes(title_text="Date (UTC)", row=3, col=1)

    # ── Range slider on bottom x-axis for easy navigation ──
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04),
        row=3, col=1,
    )

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text="TEJAS — May 2026 Full-Month X-ray Overview  (60 s bins)",
            font=dict(size=18),
        ),
        height=950,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(t=100, b=60),
    )

    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    utils.ensure_output_dir()

    print("=" * 72)
    print("  TEJAS EDA Script 1 — Month-Wide Light Curve Overview")
    print("=" * 72)

    # ── 1. Load SoLEXS ──────────────────────────────────────────────────
    print("\n── Loading SoLEXS Light Curves ──")
    solexs_all = load_all_solexs_lc()

    # ── 2. Load HEL1OS CdTe ─────────────────────────────────────────────
    print("\n── Loading HEL1OS CdTe (full range) ──")
    cdte_all = load_all_hel1os("cdte")

    # ── 3. Load HEL1OS CZT ──────────────────────────────────────────────
    print("\n── Loading HEL1OS CZT (full range) ──")
    czt_all = load_all_hel1os("czt")

    # ── 4. Combine detectors ────────────────────────────────────────────
    print("\n── Combining detectors (averaging det1 + det2) ──")
    cdte_combined = None
    if cdte_all is not None:
        cdte_combined = utils.combine_detectors_mean(cdte_all)
        print(f"  CdTe combined: {len(cdte_combined):,} timestamps")

    czt_combined = None
    if czt_all is not None:
        czt_combined = utils.combine_detectors_mean(czt_all)
        print(f"  CZT  combined: {len(czt_combined):,} timestamps")

    # ── 5. Resample to 60 s for Plotly ──────────────────────────────────
    print("\n── Resampling to 60 s bins ──")
    solexs_60s = None
    if solexs_all is not None:
        solexs_60s = utils.resample_to_cadence(
            solexs_all, 60, value_cols="counts")
        print(f"  SoLEXS 60 s: {len(solexs_60s):,} points")

    cdte_60s = None
    if cdte_combined is not None:
        cdte_60s = utils.resample_to_cadence(
            cdte_combined, 60, value_cols="count_rate")
        print(f"  CdTe   60 s: {len(cdte_60s):,} points")

    czt_60s = None
    if czt_combined is not None:
        czt_60s = utils.resample_to_cadence(
            czt_combined, 60, value_cols="count_rate")
        print(f"  CZT    60 s: {len(czt_60s):,} points")

    # ── 6. Generate month overview plot ─────────────────────────────────
    print("\n── Generating month overview plot ──")
    fig = plot_month_overview(solexs_60s, cdte_60s, czt_60s)
    plot_path = utils.OUTPUT_DIR / "month_overview.html"
    fig.write_html(str(plot_path), include_plotlyjs=True)
    print(f"  Saved → {plot_path}")

    # ── 7. Data availability CSV ────────────────────────────────────────
    print("\n── Generating data availability summary ──")
    avail = generate_data_availability(solexs_all, cdte_all, czt_all)
    avail_path = utils.OUTPUT_DIR / "data_availability.csv"
    avail.to_csv(avail_path, index=False)
    print(f"  Saved → {avail_path}")

    # ── 8. Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    if solexs_all is not None:
        print(f"\n  SoLEXS (2–22 keV):")
        print(f"    Rows:         {len(solexs_all):>10,}")
        print(f"    Count range:  {solexs_all['counts'].min():.1f} – "
              f"{solexs_all['counts'].max():.1f} c/s")
        print(f"    Median:       {solexs_all['counts'].median():.1f} c/s")
        print(f"    Time span:    {solexs_all['datetime'].min().strftime('%Y-%m-%d %H:%M')} → "
              f"{solexs_all['datetime'].max().strftime('%Y-%m-%d %H:%M')}")

    if cdte_all is not None:
        print(f"\n  HEL1OS CdTe (full range):")
        print(f"    Rows:         {len(cdte_all):>10,}  (all detectors)")
        print(f"    Rate range:   {cdte_all['count_rate'].min():.4f} – "
              f"{cdte_all['count_rate'].max():.4f} c/s")
        print(f"    Median:       {cdte_all['count_rate'].median():.4f} c/s")

    if czt_all is not None:
        print(f"\n  HEL1OS CZT (full range):")
        print(f"    Rows:         {len(czt_all):>10,}  (all detectors)")
        print(f"    Rate range:   {czt_all['count_rate'].min():.4f} – "
              f"{czt_all['count_rate'].max():.4f} c/s")
        print(f"    Median:       {czt_all['count_rate'].median():.4f} c/s")

    # ── Top active days ──
    if solexs_all is not None:
        print(f"\n  ── Top 10 most active days (peak SoLEXS counts) ──")
        day_max = (solexs_all.groupby("day")["counts"]
                   .max()
                   .sort_values(ascending=False))
        for rank, (day, peak) in enumerate(day_max.head(10).items(), 1):
            print(f"    {rank:>2}. May {day}:  {peak:>10.1f} c/s peak")

    # ── Data availability table ──
    if avail is not None:
        print(f"\n  ── Data availability per day ──")
        cols = ["day", "solexs_span_h", "solexs_gti_valid_h",
                "cdte_span_h", "czt_span_h"]
        available_cols = [c for c in cols if c in avail.columns]
        print(avail[available_cols].to_string(index=False))

    elapsed = time.time() - t0
    print(f"\n  ── Completed in {elapsed:.1f} s ──")
    print(f"\n  Outputs:")
    print(f"    {plot_path}")
    print(f"    {avail_path}")


if __name__ == "__main__":
    main()
