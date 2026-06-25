"""
TEJAS EDA Script 3: Data Quality & Cross-Instrument Report
============================================================
Systematic assessment of data quality across the full month:
  - GTI coverage per day (SoLEXS + HEL1OS)
  - Noise / baseline characterisation (quiet-Sun count rates)
  - Cross-instrument consistency checks
  - Dump-to-dump continuity (HEL1OS)
  - Time alignment validation (MJD <-> Unix, ISOT cross-check)

Produces a comprehensive Markdown report and supporting CSVs.

Usage:  python scripts/eda_03_data_quality.py
        (set PYTHONIOENCODING=utf-8 on Windows)

Outputs:
  outputs/eda/data_quality_report.md   -- full report
  outputs/eda/gti_summary.csv          -- per-day GTI statistics
  outputs/eda/noise_levels.csv         -- quiet-Sun baseline stats
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eda_utils as utils

from astropy.io import fits
from pathlib import Path


# -- Configuration --------------------------------------------------------

# A "quiet" second is one where SoLEXS counts < this threshold
QUIET_THRESHOLD_CPS = 30


# -- GTI Analysis ----------------------------------------------------------

def analyze_gti_coverage():
    """Compute GTI coverage statistics per day for all instruments.

    Returns a DataFrame with one row per day.
    """
    solexs_days, hel1os_days = utils.get_available_days()
    all_days = sorted(set(solexs_days) | set(hel1os_days))

    rows = []
    for day in all_days:
        date_str = utils.day_to_date_str(day)
        row = {"date": date_str, "day": int(day)}

        # -- SoLEXS GTI --
        sgti = utils.read_solexs_gti(date_str)
        if sgti is not None and len(sgti) > 0:
            row["solexs_gti_n"] = len(sgti)
            row["solexs_gti_total_s"] = round(sgti["duration_s"].sum(), 1)
            row["solexs_gti_total_h"] = round(sgti["duration_s"].sum() / 3600, 2)
            row["solexs_gti_pct"] = round(sgti["duration_s"].sum() / 86400 * 100, 1)
            row["solexs_gti_max_gap_s"] = 0
            if len(sgti) > 1:
                gaps = sgti["start"].iloc[1:].values - sgti["stop"].iloc[:-1].values
                row["solexs_gti_max_gap_s"] = round(float(gaps.max()), 1) if len(gaps) > 0 else 0
            row["solexs_gti_fragmented"] = len(sgti) > 3
        else:
            row.update(solexs_gti_n=0, solexs_gti_total_s=0,
                       solexs_gti_total_h=0, solexs_gti_pct=0,
                       solexs_gti_max_gap_s=0, solexs_gti_fragmented=False)

        # -- HEL1OS CdTe GTI --
        cgti = utils.read_hel1os_gti(date_str, "cdte")
        if cgti is not None and len(cgti) > 0:
            row["cdte_gti_n"] = len(cgti)
            row["cdte_gti_total_h"] = round(cgti["duration_s"].sum() / 3600, 2)
            row["cdte_gti_pct"] = round(cgti["duration_s"].sum() / 86400 * 100, 1)
        else:
            row.update(cdte_gti_n=0, cdte_gti_total_h=0, cdte_gti_pct=0)

        # -- HEL1OS CZT GTI --
        zgti = utils.read_hel1os_gti(date_str, "czt")
        if zgti is not None and len(zgti) > 0:
            row["czt_gti_n"] = len(zgti)
            row["czt_gti_total_h"] = round(zgti["duration_s"].sum() / 3600, 2)
            row["czt_gti_pct"] = round(zgti["duration_s"].sum() / 86400 * 100, 1)
        else:
            row.update(czt_gti_n=0, czt_gti_total_h=0, czt_gti_pct=0)

        rows.append(row)

    return pd.DataFrame(rows)


# -- Noise / Baseline Characterisation -------------------------------------

def analyze_noise_levels():
    """Compute quiet-Sun baseline statistics per day for SoLEXS.

    Uses only rows where counts < QUIET_THRESHOLD_CPS as 'quiet'.
    Also computes basic statistics for HEL1OS CdTe and CZT.

    Returns a DataFrame with one row per day.
    """
    solexs_days, hel1os_days = utils.get_available_days()
    all_days = sorted(set(solexs_days) | set(hel1os_days))

    rows = []
    for day in all_days:
        date_str = utils.day_to_date_str(day)
        row = {"date": date_str, "day": int(day)}

        # -- SoLEXS baseline --
        lc = utils.read_solexs_lc(date_str)
        if lc is not None and len(lc) > 0:
            quiet = lc[lc["counts"] < QUIET_THRESHOLD_CPS]["counts"]
            row["solexs_all_median"] = round(float(lc["counts"].median()), 2)
            row["solexs_all_mean"] = round(float(lc["counts"].mean()), 2)
            row["solexs_all_std"] = round(float(lc["counts"].std()), 2)
            row["solexs_all_max"] = round(float(lc["counts"].max()), 2)
            row["solexs_quiet_n"] = len(quiet)
            row["solexs_quiet_pct"] = round(len(quiet) / len(lc) * 100, 1)
            if len(quiet) > 0:
                row["solexs_quiet_median"] = round(float(quiet.median()), 2)
                row["solexs_quiet_mean"] = round(float(quiet.mean()), 2)
                row["solexs_quiet_std"] = round(float(quiet.std()), 2)
            else:
                row.update(solexs_quiet_median=np.nan,
                           solexs_quiet_mean=np.nan,
                           solexs_quiet_std=np.nan)

            # Zero-count fraction
            row["solexs_zero_pct"] = round(
                (lc["counts"] == 0).sum() / len(lc) * 100, 1)
        else:
            for k in ["solexs_all_median", "solexs_all_mean", "solexs_all_std",
                       "solexs_all_max", "solexs_quiet_n", "solexs_quiet_pct",
                       "solexs_quiet_median", "solexs_quiet_mean",
                       "solexs_quiet_std", "solexs_zero_pct"]:
                row[k] = 0

        # -- HEL1OS CdTe baseline --
        cdte = utils.read_hel1os_full_range(date_str, "cdte")
        if cdte is not None and len(cdte) > 0:
            row["cdte_median"] = round(float(cdte["count_rate"].median()), 4)
            row["cdte_mean"] = round(float(cdte["count_rate"].mean()), 4)
            row["cdte_std"] = round(float(cdte["count_rate"].std()), 4)
            row["cdte_max"] = round(float(cdte["count_rate"].max()), 4)
            row["cdte_zero_pct"] = round(
                (cdte["count_rate"] == 0).sum() / len(cdte) * 100, 1)
        else:
            row.update(cdte_median=0, cdte_mean=0, cdte_std=0,
                       cdte_max=0, cdte_zero_pct=0)

        # -- HEL1OS CZT baseline --
        czt = utils.read_hel1os_full_range(date_str, "czt")
        if czt is not None and len(czt) > 0:
            row["czt_median"] = round(float(czt["count_rate"].median()), 4)
            row["czt_mean"] = round(float(czt["count_rate"].mean()), 4)
            row["czt_std"] = round(float(czt["count_rate"].std()), 4)
            row["czt_max"] = round(float(czt["count_rate"].max()), 4)
            row["czt_zero_pct"] = round(
                (czt["count_rate"] == 0).sum() / len(czt) * 100, 1)
        else:
            row.update(czt_median=0, czt_mean=0, czt_std=0,
                       czt_max=0, czt_zero_pct=0)

        print(f"  Day {day}: SoLEXS median={row.get('solexs_all_median',0):.1f}  "
              f"CdTe zero%={row.get('cdte_zero_pct',0):.0f}%  "
              f"CZT zero%={row.get('czt_zero_pct',0):.0f}%")
        rows.append(row)

    return pd.DataFrame(rows)


# -- Cross-Instrument Consistency -----------------------------------------

def check_solexs_lc_vs_pi():
    """Compare SoLEXS LC total counts vs PI band sum for consistency.

    For each day, load both LC and PI, compare total counts.
    Returns a DataFrame with correlation statistics.
    """
    solexs_days, _ = utils.get_available_days()
    rows = []

    for day in solexs_days:
        date_str = utils.day_to_date_str(day)
        lc = utils.read_solexs_lc(date_str)
        pi = utils.read_solexs_pi(date_str)

        if lc is None or pi is None:
            continue

        row = {"date": date_str, "day": int(day)}
        row["lc_rows"] = len(lc)
        row["pi_rows"] = len(pi)
        row["row_match"] = len(lc) == len(pi)

        # Compare on matching timestamps (merge on rounded time)
        lc_r = lc.copy()
        pi_r = pi.copy()
        lc_r["t"] = np.round(lc_r["time_unix"])
        pi_r["t"] = np.round(pi_r["time_unix"])

        merged = pd.merge(lc_r[["t", "counts"]], pi_r[["t", "counts_total_pi"]],
                          on="t", how="inner")

        row["merged_rows"] = len(merged)
        if len(merged) > 100:
            corr = merged["counts"].corr(merged["counts_total_pi"])
            ratio = (merged["counts"] / merged["counts_total_pi"].replace(0, np.nan)).median()
            row["correlation"] = round(corr, 6)
            row["lc_to_pi_ratio_median"] = round(float(ratio), 4) if not np.isnan(ratio) else np.nan
        else:
            row["correlation"] = np.nan
            row["lc_to_pi_ratio_median"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def check_detector_consistency():
    """Compare CdTe1 vs CdTe2 and CZT1 vs CZT2 count rates.

    For a sample of days, compute correlation between paired detectors.
    """
    sample_days = ["01", "07", "10", "15", "20", "25", "30"]
    _, hel1os_days = utils.get_available_days()
    sample_days = [d for d in sample_days if d in hel1os_days]

    rows = []
    for day in sample_days:
        date_str = utils.day_to_date_str(day)

        for det_type in ["cdte", "czt"]:
            df = utils.read_hel1os_full_range(date_str, det_type)
            if df is None:
                continue

            det1 = df[df["detector"] == f"{det_type}1"]
            det2 = df[df["detector"] == f"{det_type}2"]

            if len(det1) == 0 or len(det2) == 0:
                continue

            # Merge on rounded time
            d1 = det1.copy()
            d2 = det2.copy()
            d1["t"] = np.round(d1["time_unix"])
            d2["t"] = np.round(d2["time_unix"])

            merged = pd.merge(
                d1[["t", "count_rate"]].rename(columns={"count_rate": "det1"}),
                d2[["t", "count_rate"]].rename(columns={"count_rate": "det2"}),
                on="t", how="inner",
            )

            if len(merged) < 100:
                continue

            corr = merged["det1"].corr(merged["det2"])
            ratio = (merged["det1"] / merged["det2"].replace(0, np.nan)).median()

            rows.append({
                "date": date_str,
                "detector_type": det_type,
                "merged_rows": len(merged),
                "correlation": round(corr, 6),
                "det1_to_det2_ratio": round(float(ratio), 4) if not np.isnan(ratio) else np.nan,
                "det1_mean": round(float(merged["det1"].mean()), 4),
                "det2_mean": round(float(merged["det2"].mean()), 4),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# -- Dump Continuity -------------------------------------------------------

def check_dump_continuity():
    """Check for gaps or overlaps between consecutive HEL1OS dumps.

    Returns a list of findings (date, gap/overlap, duration).
    """
    _, hel1os_days = utils.get_available_days()
    findings = []

    for day in hel1os_days:
        date_str = utils.day_to_date_str(day)
        year, month, d = date_str.split("-")
        day_dir = utils.HEL1OS_DIR / year / month / d

        if not day_dir.exists():
            continue

        dump_dirs = sorted([dd for dd in day_dir.iterdir() if dd.is_dir()])
        if len(dump_dirs) < 2:
            continue

        # Get time ranges per dump from CdTe1 (representative)
        dump_ranges = []
        for dump_dir in dump_dirs:
            lc_file = dump_dir / "lightcurve_cdte1.fits"
            if not lc_file.exists():
                continue
            try:
                with fits.open(lc_file) as hdul:
                    mjd = hdul[-1].data["MJD"].astype(np.float64)
                    t_start = utils.mjd_to_unix(mjd.min())
                    t_end = utils.mjd_to_unix(mjd.max())
                    dump_ranges.append({
                        "dump": dump_dir.name,
                        "start": t_start,
                        "end": t_end,
                    })
            except Exception:
                pass

        # Check consecutive dump pairs
        for i in range(len(dump_ranges) - 1):
            gap_s = dump_ranges[i + 1]["start"] - dump_ranges[i]["end"]
            findings.append({
                "date": date_str,
                "dump_a": dump_ranges[i]["dump"],
                "dump_b": dump_ranges[i + 1]["dump"],
                "gap_seconds": round(gap_s, 1),
                "type": "gap" if gap_s > 1 else ("overlap" if gap_s < -1 else "continuous"),
            })

    return pd.DataFrame(findings) if findings else pd.DataFrame()


# -- Time Alignment Validation ---------------------------------------------

def validate_time_alignment():
    """Verify MJD-to-Unix conversion by cross-checking with ISOT strings.

    For a few sample files, compare the computed datetime from MJD with
    the ISOT column directly.
    """
    sample_days = ["01", "10", "20", "30"]
    _, hel1os_days = utils.get_available_days()
    sample_days = [d for d in sample_days if d in hel1os_days]

    results = []
    for day in sample_days:
        date_str = utils.day_to_date_str(day)
        year, month, d = date_str.split("-")
        day_dir = utils.HEL1OS_DIR / year / month / d

        dump_dirs = sorted([dd for dd in day_dir.iterdir() if dd.is_dir()])
        if not dump_dirs:
            continue

        # Check first dump, CdTe1, first band
        dump_dir = dump_dirs[0]
        lc_file = dump_dir / "lightcurve_cdte1.fits"
        if not lc_file.exists():
            continue

        try:
            with fits.open(lc_file) as hdul:
                ext = hdul[1]  # first band
                mjd = ext.data["MJD"][:5].astype(np.float64)
                isot = ext.data["ISOT"][:5]

                for m, iso_str in zip(mjd, isot):
                    unix_computed = utils.mjd_to_unix(m)
                    dt_computed = pd.Timestamp(unix_computed, unit="s", tz="UTC")

                    # Parse ISOT string
                    iso_clean = iso_str.strip()
                    if isinstance(iso_clean, bytes):
                        iso_clean = iso_clean.decode("utf-8")
                    dt_isot = pd.Timestamp(iso_clean)
                    if dt_isot.tzinfo is None:
                        dt_isot = dt_isot.tz_localize("UTC")

                    diff_ms = abs((dt_computed - dt_isot).total_seconds() * 1000)

                    results.append({
                        "date": date_str,
                        "dump": dump_dir.name,
                        "mjd": round(m, 8),
                        "isot": iso_clean,
                        "computed_utc": str(dt_computed),
                        "diff_ms": round(diff_ms, 3),
                    })
        except Exception as e:
            print(f"  Warning: time validation failed for {lc_file}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


# -- Report Generation ----------------------------------------------------

def generate_report(gti_df, noise_df, lc_vs_pi_df, det_consistency_df,
                    dump_continuity_df, time_alignment_df):
    """Generate the Markdown data quality report."""

    lines = []
    lines.append("# TEJAS - Data Quality Report")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"\nDataset: May 2026 (30 days, day 22 missing)")
    lines.append("")

    # -- Section 1: GTI Coverage --
    lines.append("---")
    lines.append("## 1. GTI Coverage")
    lines.append("")

    if gti_df is not None and len(gti_df) > 0:
        avg_pct = gti_df["solexs_gti_pct"].mean()
        min_day = gti_df.loc[gti_df["solexs_gti_pct"].idxmin()]
        lines.append(f"**SoLEXS GTI:** Average coverage = {avg_pct:.1f}% of day")
        lines.append("")

        # Flag days with < 80% coverage
        low_coverage = gti_df[gti_df["solexs_gti_pct"] < 80]
        if len(low_coverage) > 0:
            lines.append("**Days with reduced SoLEXS GTI coverage (<80%):**")
            lines.append("")
            lines.append("| Day | GTI Hours | Coverage % | Intervals | Max Gap (s) |")
            lines.append("|-----|-----------|-----------|-----------|-------------|")
            for _, r in low_coverage.iterrows():
                lines.append(
                    f"| May {r['day']:02d} | {r['solexs_gti_total_h']:.1f} | "
                    f"{r['solexs_gti_pct']:.1f}% | {r['solexs_gti_n']} | "
                    f"{r['solexs_gti_max_gap_s']:.0f} |"
                )
            lines.append("")
        else:
            lines.append("All days have >= 80% SoLEXS GTI coverage.")
            lines.append("")

        # HEL1OS GTI summary
        if "cdte_gti_pct" in gti_df.columns:
            avg_cdte = gti_df["cdte_gti_pct"].mean()
            avg_czt = gti_df["czt_gti_pct"].mean()
            lines.append(f"**HEL1OS GTI:** CdTe avg = {avg_cdte:.1f}%, "
                         f"CZT avg = {avg_czt:.1f}%")
            lines.append("")

    # -- Section 2: Noise / Baseline --
    lines.append("---")
    lines.append("## 2. Noise & Baseline Characterisation")
    lines.append("")

    if noise_df is not None and len(noise_df) > 0:
        overall_median = noise_df["solexs_all_median"].median()
        overall_quiet_median = noise_df["solexs_quiet_median"].median()
        overall_quiet_std = noise_df["solexs_quiet_std"].median()

        lines.append(f"**SoLEXS quiet-Sun baseline** (counts < {QUIET_THRESHOLD_CPS} c/s):")
        lines.append(f"  - Overall median: {overall_median:.1f} c/s")
        lines.append(f"  - Quiet median: {overall_quiet_median:.1f} c/s")
        lines.append(f"  - Quiet std: {overall_quiet_std:.1f} c/s")
        lines.append("")

        # Zero-count percentages
        solexs_zero = noise_df["solexs_zero_pct"].mean()
        cdte_zero = noise_df["cdte_zero_pct"].mean()
        czt_zero = noise_df["czt_zero_pct"].mean()

        lines.append("**Zero-count percentages (average across month):**")
        lines.append(f"  - SoLEXS: {solexs_zero:.1f}%")
        lines.append(f"  - CdTe: {cdte_zero:.1f}%")
        lines.append(f"  - CZT: {czt_zero:.1f}%")
        lines.append("")

        lines.append("> **Note:** HEL1OS data uses integer count rates. A median of 0.0 "
                     "indicates that most 1-second bins have zero photon detections, which "
                     "is expected for these small-area detectors outside of flare periods.")
        lines.append("")

    # -- Section 3: LC vs PI Consistency --
    lines.append("---")
    lines.append("## 3. SoLEXS LC vs PI Consistency")
    lines.append("")

    if lc_vs_pi_df is not None and len(lc_vs_pi_df) > 0:
        avg_corr = lc_vs_pi_df["correlation"].mean()
        avg_ratio = lc_vs_pi_df["lc_to_pi_ratio_median"].median()
        all_match = lc_vs_pi_df["row_match"].all()

        lines.append(f"- Row count match (LC == PI): {'YES' if all_match else 'NO'}")
        lines.append(f"- Average correlation (LC total vs PI band sum): {avg_corr:.6f}")
        lines.append(f"- Median LC-to-PI ratio: {avg_ratio:.4f}")
        lines.append("")

        if abs(avg_corr) < 0.9:
            lines.append("> **WARNING:** LC and PI counts show low correlation. "
                         "The LC broadband counts may include channels outside the "
                         "PI extraction range (channels 59-340).")
        elif avg_ratio and (avg_ratio < 0.8 or avg_ratio > 1.2):
            lines.append("> **NOTE:** LC total differs from PI band sum. Expected: "
                         "LC includes all 340 channels while PI extraction uses "
                         "channels 59-340 (skipping low-energy channels with "
                         "uncertain response).")
        else:
            lines.append("> LC and PI counts are consistent.")
        lines.append("")

    # -- Section 4: Detector Consistency --
    lines.append("---")
    lines.append("## 4. Detector Consistency (CdTe1 vs CdTe2, CZT1 vs CZT2)")
    lines.append("")

    if det_consistency_df is not None and len(det_consistency_df) > 0:
        lines.append("| Date | Detector | Correlation | Det1/Det2 Ratio | Det1 Mean | Det2 Mean |")
        lines.append("|------|----------|-------------|-----------------|-----------|-----------|")
        for _, r in det_consistency_df.iterrows():
            lines.append(
                f"| {r['date']} | {r['detector_type']} | "
                f"{r['correlation']:.4f} | {r['det1_to_det2_ratio']:.4f} | "
                f"{r['det1_mean']:.4f} | {r['det2_mean']:.4f} |"
            )
        lines.append("")
    else:
        lines.append("No detector consistency data available.")
        lines.append("")

    # -- Section 5: Dump Continuity --
    lines.append("---")
    lines.append("## 5. HEL1OS Dump-to-Dump Continuity")
    lines.append("")

    if dump_continuity_df is not None and len(dump_continuity_df) > 0:
        gaps = dump_continuity_df[dump_continuity_df["type"] == "gap"]
        overlaps = dump_continuity_df[dump_continuity_df["type"] == "overlap"]
        continuous = dump_continuity_df[dump_continuity_df["type"] == "continuous"]

        lines.append(f"- Continuous transitions: {len(continuous)}")
        lines.append(f"- Gaps (>1s): {len(gaps)}")
        lines.append(f"- Overlaps (<-1s): {len(overlaps)}")
        lines.append("")

        if len(gaps) > 0:
            lines.append("**Gaps between dumps:**")
            lines.append("")
            lines.append("| Date | Dump A | Dump B | Gap (seconds) |")
            lines.append("|------|--------|--------|---------------|")
            for _, r in gaps.iterrows():
                lines.append(
                    f"| {r['date']} | {r['dump_a']} | {r['dump_b']} | "
                    f"{r['gap_seconds']:.1f} |"
                )
            lines.append("")

        if len(overlaps) > 0:
            lines.append("**Overlaps between dumps:**")
            lines.append("")
            lines.append("| Date | Dump A | Dump B | Overlap (seconds) |")
            lines.append("|------|--------|--------|-------------------|")
            for _, r in overlaps.iterrows():
                lines.append(
                    f"| {r['date']} | {r['dump_a']} | {r['dump_b']} | "
                    f"{abs(r['gap_seconds']):.1f} |"
                )
            lines.append("")

        avg_gap = dump_continuity_df["gap_seconds"].mean()
        max_gap = dump_continuity_df["gap_seconds"].max()
        lines.append(f"Average inter-dump gap: {avg_gap:.1f}s, "
                     f"Maximum: {max_gap:.1f}s")
        lines.append("")
    else:
        lines.append("No dump continuity data available.")
        lines.append("")

    # -- Section 6: Time Alignment --
    lines.append("---")
    lines.append("## 6. Time Alignment Validation (MJD <-> ISOT)")
    lines.append("")

    if time_alignment_df is not None and len(time_alignment_df) > 0:
        max_diff = time_alignment_df["diff_ms"].max()
        avg_diff = time_alignment_df["diff_ms"].mean()

        lines.append(f"Compared MJD->Unix conversion against ISOT strings "
                     f"({len(time_alignment_df)} samples):")
        lines.append(f"  - Average discrepancy: {avg_diff:.3f} ms")
        lines.append(f"  - Maximum discrepancy: {max_diff:.3f} ms")
        lines.append("")

        if max_diff < 100:
            lines.append("> Time alignment is excellent (< 100 ms discrepancy).")
        elif max_diff < 1000:
            lines.append("> Time alignment is acceptable (< 1s discrepancy).")
        else:
            lines.append("> **WARNING:** Time alignment shows >1s discrepancy. "
                         "Investigate MJD conversion formula.")
        lines.append("")

        lines.append("**Sample comparisons:**")
        lines.append("")
        lines.append("| Date | MJD | ISOT | Computed UTC | Diff (ms) |")
        lines.append("|------|-----|------|-------------|-----------|")
        for _, r in time_alignment_df.head(10).iterrows():
            lines.append(
                f"| {r['date']} | {r['mjd']:.6f} | {r['isot']} | "
                f"{r['computed_utc'][:23]} | {r['diff_ms']:.3f} |"
            )
        lines.append("")

    # -- Section 7: Summary --
    lines.append("---")
    lines.append("## 7. Summary & Recommendations for Phase 2")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    if gti_df is not None:
        low = gti_df[gti_df["solexs_gti_pct"] < 80]
        if len(low) > 0:
            days_str = ", ".join([f"May {r['day']:02d}" for _, r in low.iterrows()])
            lines.append(f"1. **GTI gaps:** Days with <80% coverage: {days_str}. "
                         "These days need careful gap handling in the data pipeline.")
        else:
            lines.append("1. **GTI coverage:** Excellent across all days (>80%).")

    lines.append("2. **Day 22 is missing** from both instruments.")

    if noise_df is not None:
        lines.append(f"3. **HEL1OS zero-count fraction:** CdTe = {cdte_zero:.0f}%, "
                     f"CZT = {czt_zero:.0f}%. Integer count rates with many zeros "
                     "require special handling (avoid log(0), use appropriate "
                     "smoothing windows).")

    lines.append("")
    lines.append("### Recommendations for Data Ingestion (Phase 2)")
    lines.append("")
    lines.append("- Apply GTI filtering before any analysis")
    lines.append("- Handle zero counts: use `np.maximum(counts, small_value)` "
                 "before log transforms")
    lines.append("- Resample HEL1OS to 10s or 30s bins for cleaner signals")
    lines.append("- Mark inter-dump gaps as NaN, do not interpolate across them")
    lines.append("- Use the LC-to-PI ratio to validate pipeline correctness")
    lines.append("")

    return "\n".join(lines)


# -- Main ------------------------------------------------------------------

def main():
    t0 = time.time()
    utils.ensure_output_dir()

    print("=" * 72)
    print("  TEJAS EDA Script 3 - Data Quality & Cross-Instrument Report")
    print("=" * 72)

    # -- 1. GTI Coverage --
    print("\n-- 1. Analysing GTI coverage --")
    gti_df = analyze_gti_coverage()
    gti_path = utils.OUTPUT_DIR / "gti_summary.csv"
    gti_df.to_csv(gti_path, index=False)
    print(f"  Saved -> {gti_path}")

    # -- 2. Noise / Baseline --
    print("\n-- 2. Analysing noise levels & baselines --")
    noise_df = analyze_noise_levels()
    noise_path = utils.OUTPUT_DIR / "noise_levels.csv"
    noise_df.to_csv(noise_path, index=False)
    print(f"  Saved -> {noise_path}")

    # -- 3. LC vs PI Consistency --
    print("\n-- 3. Checking SoLEXS LC vs PI consistency --")
    lc_vs_pi_df = check_solexs_lc_vs_pi()
    if len(lc_vs_pi_df) > 0:
        avg_corr = lc_vs_pi_df["correlation"].mean()
        print(f"  Average LC-PI correlation: {avg_corr:.6f}")
    else:
        print("  No LC-PI comparison data.")

    # -- 4. Detector Consistency --
    print("\n-- 4. Checking detector consistency (det1 vs det2) --")
    det_df = check_detector_consistency()
    if len(det_df) > 0:
        for _, r in det_df.iterrows():
            print(f"  {r['date']} {r['detector_type']}: "
                  f"corr={r['correlation']:.4f}  "
                  f"ratio={r['det1_to_det2_ratio']:.4f}")
    else:
        print("  No detector consistency data.")

    # -- 5. Dump Continuity --
    print("\n-- 5. Checking dump-to-dump continuity --")
    dump_df = check_dump_continuity()
    if len(dump_df) > 0:
        gaps = dump_df[dump_df["type"] == "gap"]
        overlaps = dump_df[dump_df["type"] == "overlap"]
        print(f"  {len(dump_df)} transitions checked: "
              f"{len(gaps)} gaps, {len(overlaps)} overlaps")
    else:
        print("  No dump continuity data.")

    # -- 6. Time Alignment --
    print("\n-- 6. Validating time alignment (MJD <-> ISOT) --")
    time_df = validate_time_alignment()
    if len(time_df) > 0:
        print(f"  Max discrepancy: {time_df['diff_ms'].max():.3f} ms")
        print(f"  Avg discrepancy: {time_df['diff_ms'].mean():.3f} ms")
    else:
        print("  No time alignment data.")

    # -- 7. Generate Report --
    print("\n-- 7. Generating Markdown report --")
    report = generate_report(
        gti_df, noise_df, lc_vs_pi_df, det_df, dump_df, time_df
    )
    report_path = utils.OUTPUT_DIR / "data_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved -> {report_path}")

    elapsed = time.time() - t0
    print(f"\n-- Completed in {elapsed:.1f}s --")

    print(f"\n  Outputs:")
    print(f"    {gti_path}")
    print(f"    {noise_path}")
    print(f"    {report_path}")


if __name__ == "__main__":
    main()
