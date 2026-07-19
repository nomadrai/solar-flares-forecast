"""
TEJAS ↔ GOES Flare Cross-Matching (external confirmation)
=========================================================
Confirms TEJAS-detected flares against NOAA's public GOES flare list
(fetched by ``scripts/fetch_goes_flares.py``).  Produces:

  * a per-event ``goes_matched`` flag + matched GOES class, used by the
    credible-event filter (Phase B), and
  * a validation report answering the two operator questions:
      - which GOES flares did TEJAS miss? (detection recall vs external truth)
      - which TEJAS events have no GOES counterpart? (TEJAS-only / spurious)

Matching is peak-time nearest-neighbour within
``GOES_VALIDATION_TOLERANCE_S``.  All times are treated as UTC (TEJAS
catalogue times are tz-naive UTC wall-clock; GOES times are UTC).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tejas.config import (
    GOES_REFERENCE_DIR,
    GOES_FLARE_LIST_NAME,
    GOES_VALIDATION_TOLERANCE_S,
)


def load_goes_flare_list(directory=None):
    """Load the distilled GOES flare list as a time-sorted DataFrame.

    Returns an empty DataFrame (with the expected columns) if the list has
    not been fetched yet.
    """
    if directory is None:
        directory = GOES_REFERENCE_DIR
    path = Path(directory) / GOES_FLARE_LIST_NAME
    cols = ["flare_id", "begin_time", "peak_time", "end_time",
            "goes_class", "peak_flux_wm2"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, parse_dates=["begin_time", "peak_time", "end_time"])
    return df.sort_values("peak_time").reset_index(drop=True)


def _to_unix(series):
    """tz-naive/aware datetime Series → float64 unix seconds (UTC)."""
    dt = pd.to_datetime(series, utc=True).dt.tz_localize(None)
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().values


def cross_match_events(catalogue, goes_df, tolerance_s=None):
    """Match each TEJAS event to the nearest GOES flare by peak time.

    Args:
        catalogue: TEJAS master catalogue (needs 'peak_time').
        goes_df: GOES flare list from load_goes_flare_list().
        tolerance_s: max peak-to-peak offset; default
            GOES_VALIDATION_TOLERANCE_S.

    Returns:
        catalogue copy with added columns:
          goes_matched (bool), goes_match_class (str/NaN),
          goes_match_dt_s (float, signed TEJAS−GOES seconds / NaN).
    """
    if tolerance_s is None:
        tolerance_s = GOES_VALIDATION_TOLERANCE_S

    out = catalogue.copy()
    out["goes_matched"] = False
    out["goes_match_class"] = pd.Series([np.nan] * len(out), dtype=object,
                                        index=out.index)
    out["goes_match_dt_s"] = np.nan
    if goes_df is None or goes_df.empty or out.empty:
        return out

    tejas_t = _to_unix(out["peak_time"])
    goes_t = _to_unix(goes_df["peak_time"])
    goes_cls = goes_df["goes_class"].values

    order = np.argsort(goes_t)
    goes_t_sorted = goes_t[order]

    for i, t in enumerate(tejas_t):
        if np.isnan(t):
            continue
        idx = np.searchsorted(goes_t_sorted, t)
        cand = [j for j in (idx - 1, idx) if 0 <= j < len(goes_t_sorted)]
        if not cand:
            continue
        best = min(cand, key=lambda j: abs(goes_t_sorted[j] - t))
        dt = t - goes_t_sorted[best]
        if abs(dt) <= tolerance_s:
            g = order[best]
            out.iat[i, out.columns.get_loc("goes_matched")] = True
            out.iat[i, out.columns.get_loc("goes_match_class")] = goes_cls[g]
            out.iat[i, out.columns.get_loc("goes_match_dt_s")] = float(dt)

    return out


def find_missed_goes_flares(catalogue, goes_df, tolerance_s=None):
    """GOES flares with no TEJAS event within tolerance (external misses)."""
    if tolerance_s is None:
        tolerance_s = GOES_VALIDATION_TOLERANCE_S
    if goes_df is None or goes_df.empty:
        return goes_df
    if catalogue is None or catalogue.empty:
        return goes_df.copy()

    tejas_t = np.sort(_to_unix(catalogue["peak_time"]))
    goes_t = _to_unix(goes_df["peak_time"])

    missed = []
    for k, t in enumerate(goes_t):
        idx = np.searchsorted(tejas_t, t)
        cand = [j for j in (idx - 1, idx) if 0 <= j < len(tejas_t)]
        nearest = min((abs(tejas_t[j] - t) for j in cand), default=np.inf)
        if nearest > tolerance_s:
            missed.append(k)
    return goes_df.iloc[missed].reset_index(drop=True)


def _class_letter(series):
    """First letter (A/B/C/M/X) of a GOES class string Series."""
    return series.dropna().astype(str).str[0]


def build_validation_report(catalogue, goes_df, tolerance_s=None,
                            report_path=None):
    """Cross-match, summarise, and write the Markdown validation report.

    Returns the matched catalogue (from cross_match_events) so callers can
    reuse the goes_matched flags for the credible filter.
    """
    from tejas.config import GOES_VALIDATION_REPORT
    if tolerance_s is None:
        tolerance_s = GOES_VALIDATION_TOLERANCE_S
    if report_path is None:
        report_path = GOES_VALIDATION_REPORT

    matched = cross_match_events(catalogue, goes_df, tolerance_s)
    missed = find_missed_goes_flares(catalogue, goes_df, tolerance_s)

    n_goes = 0 if goes_df is None else len(goes_df)
    n_tejas = len(catalogue)
    n_soft = int(catalogue.get("has_soft", pd.Series(dtype=bool)).sum())
    n_matched = int(matched["goes_matched"].sum())
    n_missed = len(missed)

    lines = []
    lines.append("# GOES External Validation Report\n")
    lines.append(f"Peak-to-peak match tolerance: ±{tolerance_s} s "
                 f"({tolerance_s/60:.0f} min).\n")
    lines.append("## Summary\n")
    lines.append(f"- GOES flares (external truth): **{n_goes}**")
    lines.append(f"- TEJAS catalogue events: **{n_tejas}** "
                 f"({n_soft} with a soft/SoLEXS detection)")
    lines.append(f"- TEJAS events matched to a GOES flare: **{n_matched}**")
    if n_goes:
        det_recall = (n_goes - n_missed) / n_goes
        lines.append(f"- GOES flares detected by TEJAS: "
                     f"**{n_goes - n_missed}/{n_goes}** "
                     f"(detection recall vs external truth = "
                     f"**{det_recall:.1%}**)")
    lines.append(f"- GOES flares TEJAS missed: **{n_missed}**\n")

    if n_goes:
        lines.append("## GOES class distribution (external)\n")
        vc = _class_letter(goes_df["goes_class"]).value_counts()
        for letter in ["X", "M", "C", "B", "A"]:
            if letter in vc.index:
                lines.append(f"- {letter}-class: {int(vc[letter])}")
        lines.append("")

        # Recall by class — do we miss the big ones?
        lines.append("## Detection recall by GOES class\n")
        lines.append("| Class | GOES total | Missed by TEJAS | Recall |")
        lines.append("|---|---|---|---|")
        miss_by = _class_letter(missed["goes_class"]).value_counts() \
            if n_missed else pd.Series(dtype=int)
        for letter in ["X", "M", "C", "B", "A"]:
            tot = int(vc.get(letter, 0))
            if tot == 0:
                continue
            m = int(miss_by.get(letter, 0))
            lines.append(f"| {letter} | {tot} | {m} | "
                         f"{(tot - m) / tot:.1%} |")
        lines.append("")

    if n_missed:
        lines.append("## Missed GOES flares (sample, up to 25)\n")
        lines.append("| Peak time (UTC) | Class |")
        lines.append("|---|---|")
        for _, r in missed.head(25).iterrows():
            lines.append(f"| {r['peak_time']} | {r.get('goes_class', '')} |")
        lines.append("")

    report_path = Path(report_path)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return matched


def apply_goes_flags(catalogue, directory=None, tolerance_s=None):
    """Convenience: load the GOES list and return catalogue + goes_matched.

    If no GOES list is present, returns the catalogue with goes_matched=False
    so downstream credible-filtering still works (SoLEXS-only fallback).
    """
    goes_df = load_goes_flare_list(directory)
    return cross_match_events(catalogue, goes_df, tolerance_s)

