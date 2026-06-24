# TEJAS Data Viewer
# =================
# Visualize SoLEXS and HEL1OS data for a single day to build intuition
# about what solar flare data looks like.
#
# Usage: python scripts/view_solexs_hel1os_data.py
#
# Change TARGET_DATE below to view a different day.
# The script produces an interactive matplotlib window with multiple panels.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from astropy.io import fits
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# CONFIGURATION - Change this to view a different day
# ============================================================
TARGET_DATE = "2026-05-07"  # YYYY-MM-DD format (pick a day with data)
# ============================================================

# Paths
PROJECT_DIR = Path(r"C:\Users\aadit\Documents\Projects\TEJAS")
SOLEXS_DIR = PROJECT_DIR / "data" / "raw" / "solexs"
HEL1OS_DIR = PROJECT_DIR / "data" / "raw" / "hel1os"


def parse_date(date_str):
    """Parse YYYY-MM-DD into components."""
    parts = date_str.split("-")
    return parts[0], parts[1], parts[2]


def unix_to_datetime(unix_times):
    """Convert Unix timestamps to datetime objects."""
    return pd.to_datetime(unix_times, unit="s", origin="unix", utc=True)


# ------------------------------------------------------------------
# SoLEXS Readers
# ------------------------------------------------------------------
def read_solexs_lc(date_str):
    """Read SoLEXS light curve file for a given date."""
    year, month, day = parse_date(date_str)
    date_compact = f"{year}{month}{day}"
    lc_path = (
        SOLEXS_DIR / year / month / day
        / f"AL1_SOLEXS_{date_compact}_SDD2_L1.lc.gz"
    )

    if not lc_path.exists():
        print(f"  SoLEXS LC not found: {lc_path}")
        return None

    with fits.open(lc_path) as hdul:
        data = hdul["RATE"].data
        df = pd.DataFrame({
            "time_unix": data["TIME"],
            "counts": data["COUNTS"],
        })
        df["datetime"] = unix_to_datetime(df["time_unix"])

    print(f"  SoLEXS LC: {len(df)} rows, "
          f"{df['datetime'].min().strftime('%H:%M:%S')} to "
          f"{df['datetime'].max().strftime('%H:%M:%S')} UTC")
    return df


def read_solexs_pi(date_str):
    """Read SoLEXS PI (spectral) file and extract multi-band counts."""
    year, month, day = parse_date(date_str)
    date_compact = f"{year}{month}{day}"
    pi_path = (
        SOLEXS_DIR / year / month / day
        / f"AL1_SOLEXS_{date_compact}_SDD2_L1.pi.gz"
    )

    if not pi_path.exists():
        print(f"  SoLEXS PI not found: {pi_path}")
        return None

    with fits.open(pi_path) as hdul:
        spec_data = hdul["SPECTRUM"].data
        tstart = spec_data["TSTART"]
        counts = spec_data["COUNTS"]  # shape: (N_seconds, 340)

    # Multi-band extraction (from data_used.md)
    # Channels are 0-indexed in numpy, but 1-indexed in the manual
    ch_low = slice(58, 105)    # ~2.8-5 keV
    ch_mid = slice(105, 189)   # ~5-10 keV
    ch_high = slice(189, 340)  # ~10-22 keV

    df = pd.DataFrame({
        "time_unix": tstart,
        "counts_low": np.sum(counts[:, ch_low], axis=1),
        "counts_mid": np.sum(counts[:, ch_mid], axis=1),
        "counts_high": np.sum(counts[:, ch_high], axis=1),
    })
    df["hardness_ratio"] = df["counts_high"] / df["counts_low"].replace(0, np.nan)
    df["datetime"] = unix_to_datetime(df["time_unix"])

    print(f"  SoLEXS PI: {len(df)} spectra, "
          f"shape of counts array: {counts.shape}")
    return df


def read_solexs_gti(date_str):
    """Read SoLEXS GTI file."""
    year, month, day = parse_date(date_str)
    date_compact = f"{year}{month}{day}"
    gti_path = (
        SOLEXS_DIR / year / month / day
        / f"AL1_SOLEXS_{date_compact}_SDD2_L1.gti.gz"
    )

    if not gti_path.exists():
        print(f"  SoLEXS GTI not found: {gti_path}")
        return None

    with fits.open(gti_path) as hdul:
        data = hdul["GTI"].data
        df = pd.DataFrame({
            "start": data["START"],
            "stop": data["STOP"],
        })
        df["start_dt"] = unix_to_datetime(df["start"])
        df["stop_dt"] = unix_to_datetime(df["stop"])

    total_valid = (df["stop"] - df["start"]).sum()
    print(f"  SoLEXS GTI: {len(df)} interval(s), "
          f"{total_valid/3600:.1f}h valid data")
    return df


# ------------------------------------------------------------------
# HEL1OS Readers
# ------------------------------------------------------------------
def read_hel1os_lightcurves(date_str, detector_type):
    """Read HEL1OS light curve files for a given date and detector type.

    detector_type: 'cdte' or 'czt'
    Returns combined (averaged) light curve from both detectors.
    """
    year, month, day = parse_date(date_str)
    day_dir = HEL1OS_DIR / year / month / day

    if not day_dir.exists():
        print(f"  HEL1OS {detector_type} dir not found: {day_dir}")
        return None

    # Find all dump directories for this day
    dump_dirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])

    all_dfs = []
    for dump_dir in dump_dirs:
        for det_num in [1, 2]:
            lc_file = dump_dir / f"lightcurve_{detector_type}{det_num}.fits"
            if not lc_file.exists():
                continue

            with fits.open(lc_file) as hdul:
                # Try to find the data extension
                ext = hdul[1]
                col_names = [c.name.upper() for c in ext.columns]
                data = ext.data

                # Get time column
                time_col = None
                for candidate in ["TIME", "MJD"]:
                    if candidate in col_names:
                        time_col = candidate
                        break

                if time_col is None:
                    print(f"    WARNING: No TIME column in {lc_file.name}")
                    continue

                times = data[time_col]

                # Get count rate column
                rate_col = None
                for candidate in ["COUNT_RATE", "RATE", "COUNTS"]:
                    if candidate in col_names:
                        rate_col = candidate
                        break

                if rate_col is None:
                    print(f"    WARNING: No count rate column in {lc_file.name}")
                    print(f"    Available columns: {col_names}")
                    continue

                rates = data[rate_col]

                # Get error column if available
                err_col = None
                for candidate in ["STAT_ERR", "ERROR", "ERR"]:
                    if candidate in col_names:
                        err_col = candidate
                        break

                # Handle multi-band light curves (HEL1OS has energy bands)
                # rates might be 2D: (N_times, N_bands)
                if rates.ndim == 2:
                    # Sum across all energy bands for total count rate
                    total_rate = np.sum(rates, axis=1)
                else:
                    total_rate = rates

                # Determine time format
                # If values are small (< 100000), likely MJD
                # If values are large (> 1e9), likely Unix time
                if np.median(times) < 100000:
                    # MJD - convert to Unix
                    # Unix = (MJD - 40587) * 86400
                    time_unix = (times - 40587.0) * 86400.0
                else:
                    time_unix = times

                df = pd.DataFrame({
                    "time_unix": time_unix,
                    "count_rate": total_rate.astype(float),
                    "detector": f"{detector_type}{det_num}",
                    "dump": dump_dir.name,
                })
                all_dfs.append(df)

    if not all_dfs:
        print(f"  HEL1OS {detector_type}: No data found")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["datetime"] = unix_to_datetime(combined["time_unix"])
    combined = combined.sort_values("time_unix").reset_index(drop=True)

    n_dumps = len(dump_dirs)
    det_counts = combined["detector"].nunique()
    print(f"  HEL1OS {detector_type}: {len(combined)} rows from "
          f"{n_dumps} dump(s), {det_counts} detector(s)")
    return combined


def read_hel1os_gti(date_str, detector_type):
    """Read HEL1OS GTI files for a given date and detector type."""
    year, month, day = parse_date(date_str)
    day_dir = HEL1OS_DIR / year / month / day

    if not day_dir.exists():
        return None

    dump_dirs = sorted([d for d in day_dir.iterdir() if d.is_dir()])
    all_gtis = []

    for dump_dir in dump_dirs:
        for det_num in [1, 2]:
            gti_file = dump_dir / f"gti{detector_type}{det_num}.fits"
            if not gti_file.exists():
                continue

            with fits.open(gti_file) as hdul:
                # GTI extension might be named differently
                gti_ext = None
                for ext in hdul:
                    if hasattr(ext, "columns") and ext.columns is not None:
                        col_names = [c.name.upper() for c in ext.columns]
                        if "START" in col_names and "STOP" in col_names:
                            gti_ext = ext
                            break

                if gti_ext is None:
                    continue

                data = gti_ext.data
                starts = data["START"]
                stops = data["STOP"]

                # Handle MJD vs Unix
                if np.median(starts) < 100000:
                    starts = (starts - 40587.0) * 86400.0
                    stops = (stops - 40587.0) * 86400.0

                for s, e in zip(starts, stops):
                    all_gtis.append({"start": s, "stop": e})

    if not all_gtis:
        return None

    df = pd.DataFrame(all_gtis)
    df["start_dt"] = unix_to_datetime(df["start"])
    df["stop_dt"] = unix_to_datetime(df["stop"])
    return df


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_data(date_str, solexs_lc, solexs_pi, solexs_gti,
              hel1os_cdte, hel1os_czt):
    """Create a multi-panel plot of all data for the day."""

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"TEJAS Data Viewer — {date_str}",
                 fontsize=16, fontweight="bold", y=0.98)

    time_fmt = mdates.DateFormatter("%H:%M")

    # --- Panel 1: SoLEXS Total Light Curve + GTI shading ---
    ax = axes[0]
    if solexs_lc is not None:
        ax.plot(solexs_lc["datetime"], solexs_lc["counts"],
                linewidth=0.3, color="#2196F3", alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("Counts/s")
        ax.set_title("SoLEXS SDD2 — Total Light Curve (2-22 keV)",
                      fontsize=10, loc="left")

        # Shade GTI intervals
        if solexs_gti is not None:
            ymin, ymax = ax.get_ylim()
            for _, row in solexs_gti.iterrows():
                ax.axvspan(row["start_dt"], row["stop_dt"],
                           alpha=0.08, color="green", zorder=0)
            ax.set_ylim(ymin, ymax)
    else:
        ax.text(0.5, 0.5, "No SoLEXS LC data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: SoLEXS Multi-band Light Curves ---
    ax = axes[1]
    if solexs_pi is not None:
        ax.plot(solexs_pi["datetime"], solexs_pi["counts_low"],
                linewidth=0.3, color="#4CAF50", alpha=0.8, label="2.8-5 keV (low)")
        ax.plot(solexs_pi["datetime"], solexs_pi["counts_mid"],
                linewidth=0.3, color="#FF9800", alpha=0.8, label="5-10 keV (mid)")
        ax.plot(solexs_pi["datetime"], solexs_pi["counts_high"],
                linewidth=0.3, color="#F44336", alpha=0.8, label="10-22 keV (high)")
        ax.set_yscale("log")
        ax.set_ylabel("Counts/s")
        ax.set_title("SoLEXS SDD2 — Multi-band Light Curves (from PI file)",
                      fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No SoLEXS PI data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Hardness Ratio ---
    ax = axes[2]
    if solexs_pi is not None:
        # Smooth for readability
        hr = solexs_pi["hardness_ratio"].rolling(window=60, center=True).median()
        ax.plot(solexs_pi["datetime"], hr,
                linewidth=0.5, color="#9C27B0", alpha=0.8)
        ax.set_ylabel("High/Low ratio")
        ax.set_title("SoLEXS — Hardness Ratio (10-22 keV / 2.8-5 keV) "
                      "[60s median smoothed]",
                      fontsize=10, loc="left")
    else:
        ax.text(0.5, 0.5, "No hardness ratio data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: HEL1OS CdTe ---
    ax = axes[3]
    if hel1os_cdte is not None:
        for det_name, group in hel1os_cdte.groupby("detector"):
            ax.plot(group["datetime"], group["count_rate"],
                    linewidth=0.3, alpha=0.7, label=det_name)
        ax.set_yscale("log")
        ax.set_ylabel("Counts/s")
        ax.set_title("HEL1OS CdTe — Hard X-ray Light Curve (8-70 keV)",
                      fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No HEL1OS CdTe data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.grid(True, alpha=0.3)

    # --- Panel 5: HEL1OS CZT ---
    ax = axes[4]
    if hel1os_czt is not None:
        for det_name, group in hel1os_czt.groupby("detector"):
            ax.plot(group["datetime"], group["count_rate"],
                    linewidth=0.3, alpha=0.7, label=det_name)
        ax.set_yscale("log")
        ax.set_ylabel("Counts/s")
        ax.set_title("HEL1OS CZT — High-Energy X-ray Light Curve (20-150 keV)",
                      fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No HEL1OS CZT data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (UTC)")

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(time_fmt)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"TEJAS Data Viewer")
    print(f"{'='*50}")
    print(f"Target date: {TARGET_DATE}")
    print(f"Loading data...\n")

    # Load all data
    solexs_lc = read_solexs_lc(TARGET_DATE)
    solexs_pi = read_solexs_pi(TARGET_DATE)
    solexs_gti = read_solexs_gti(TARGET_DATE)
    hel1os_cdte = read_hel1os_lightcurves(TARGET_DATE, "cdte")
    hel1os_czt = read_hel1os_lightcurves(TARGET_DATE, "czt")

    print(f"\nGenerating plots...")
    plot_data(TARGET_DATE, solexs_lc, solexs_pi, solexs_gti,
              hel1os_cdte, hel1os_czt)


if __name__ == "__main__":
    main()
