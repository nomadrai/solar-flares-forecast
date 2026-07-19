"""
TEJAS — GOES Flare Event List Fetcher (external ground truth)
=============================================================
Downloads NOAA's official GOES flare-summary product (``xrsf-l2-flsum``)
for the TEJAS study month and distils it into one flat flare list:

    data/reference/goes/YYYY/MM/goes_flare_events.csv

Each flsum netCDF holds one row per state transition
(EVENT_START / EVENT_PEAK / EVENT_END / POST_EVENT); rows that belong to
the same physical flare share a ``flare_id``.  We group by ``flare_id``
and emit one row per flare with begin/peak/end (UTC) and GOES class.

This is the trusted, publicly-available reference used to confirm which
TEJAS-detected events are real solar flares.

Usage:
    python scripts/fetch_goes_flares.py               # GOES-18, full month
    python scripts/fetch_goes_flares.py --sat goes19  # fallback satellite
    python scripts/fetch_goes_flares.py --dry-run     # print URLs only

Dependencies: requests, xarray, netCDF4 (installed in the project venv).
"""

import sys
import argparse
from pathlib import Path
from datetime import date, timedelta

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from tejas.config import YEAR, MONTH, MISSING_DAYS, GOES_REFERENCE_DIR

NCEI_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
VERSION_SUFFIXES = ["v2-2-1", "v2-2-0", "v2-1-0", "v2-0-0", "v1-0-0"]


def build_urls(sat, day):
    """Return candidate flsum URLs (varying version suffix) for one day."""
    yyyy = f"{day.year:04d}"
    mm = f"{day.month:02d}"
    ymd = day.strftime("%Y%m%d")
    sat_short = "g" + sat.replace("goes", "")  # goes18 -> g18
    return [
        f"{NCEI_BASE}/{sat}/l2/data/xrsf-l2-flsum_science/{yyyy}/{mm}/"
        f"sci_xrsf-l2-flsum_{sat_short}_d{ymd}_{v}.nc"
        for v in VERSION_SUFFIXES
    ]


def _download(url, timeout=60):
    """Fetch bytes from a URL; return content or None on any failure."""
    import requests
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        print(f"  [WARN] network error: {e}")
        return None
    if r.status_code != 200:
        return None
    return r.content


def _open_dataset(content):
    """Open netCDF bytes as an xarray Dataset (h5netcdf, then netcdf4)."""
    import io
    import xarray as xr
    try:
        return xr.open_dataset(io.BytesIO(content), engine="h5netcdf")
    except Exception:
        import os
        import tempfile
        cache_dir = project_root / "data" / "goes_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False,
                                         dir=str(cache_dir)) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return xr.open_dataset(tmp_path, engine="netcdf4").load()
        except Exception as e:
            print(f"  [WARN] failed to parse netCDF: {e}")
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def parse_flsum(ds):
    """Group flsum transition rows into one row per physical flare.

    Returns a DataFrame with columns
    [flare_id, begin_time, peak_time, end_time, goes_class,
     peak_flux_wm2, background_flux_wm2] (UTC, tz-naive).
    """
    n = ds.sizes.get("time", 0)
    if n == 0:
        return pd.DataFrame()

    raw = pd.DataFrame({
        "time": pd.to_datetime(ds["time"].values),
        "status": ds["status"].values.astype(str),
        "flare_class": ds["flare_class"].values.astype(str),
        "flare_id": ds["flare_id"].values,
        "xrsb_flux": ds["xrsb_flux"].values
        if "xrsb_flux" in ds.variables else np.nan,
        "background_flux": ds["background_flux"].values
        if "background_flux" in ds.variables else np.nan,
    })
    raw = raw.dropna(subset=["flare_id"])

    rows = []
    for fid, g in raw.groupby("flare_id"):
        starts = g.loc[g["status"] == "EVENT_START", "time"]
        peaks = g[g["status"] == "EVENT_PEAK"]
        ends = g.loc[g["status"] == "EVENT_END", "time"]

        cls = "nan"
        peak_time = pd.NaT
        peak_flux = np.nan
        if len(peaks):
            peak_time = peaks["time"].min()
            cls = str(peaks["flare_class"].iloc[0])
            peak_flux = float(peaks["xrsb_flux"].iloc[0])

        rows.append({
            "flare_id": int(fid),
            "begin_time": starts.min() if len(starts) else pd.NaT,
            "peak_time": peak_time,
            "end_time": ends.max() if len(ends) else pd.NaT,
            "goes_class": cls if cls not in ("nan", "") else np.nan,
            "peak_flux_wm2": peak_flux,
            "background_flux_wm2": float(g["background_flux"].dropna().iloc[0])
            if g["background_flux"].notna().any() else np.nan,
        })

    out = pd.DataFrame(rows)
    # A valid flare needs at least a begin and a peak.
    out = out.dropna(subset=["begin_time", "peak_time"]).reset_index(drop=True)
    return out


def fetch_day(sat, day):
    """Download + parse one day's flsum. Returns flare DataFrame or None."""
    for url in build_urls(sat, day):
        content = _download(url)
        if content is None:
            continue
        ds = _open_dataset(content)
        if ds is None:
            continue
        try:
            flares = parse_flsum(ds)
        finally:
            ds.close()
        return flares
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GOES flare event lists for the TEJAS study period.")
    parser.add_argument("--sat", default="goes18",
                        choices=["goes16", "goes17", "goes18", "goes19"],
                        help="GOES satellite id (default: goes18)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show candidate URLs, download nothing.")
    args = parser.parse_args()

    year, month = int(YEAR), int(MONTH)
    output_dir = Path(GOES_REFERENCE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    d = date(year, month, 1)
    end = (date(year, month + 1, 1) if month < 12
           else date(year + 1, 1, 1)) - timedelta(days=1)

    print(f"\n{'='*60}\nGOES flare-list fetch — {args.sat.upper()} — {YEAR}-{MONTH}"
          f"\n  Output: {output_dir}\n{'='*60}\n")

    if args.dry_run:
        while d <= end:
            print(f"  {build_urls(args.sat, d)[0]}")
            d += timedelta(days=1)
        return 0

    all_flares = []
    n_days_ok, n_days_fail = 0, 0
    while d <= end:
        dd = f"{d.day:02d}"
        if dd in MISSING_DAYS:
            print(f"  {d}: SKIP (missing day in TEJAS dataset)")
            d += timedelta(days=1)
            continue

        print(f"  {d}: fetching...", end=" ", flush=True)
        flares = fetch_day(args.sat, d)
        if flares is None:
            print("FAILED")
            n_days_fail += 1
        else:
            print(f"OK ({len(flares)} flares)")
            all_flares.append(flares)
            n_days_ok += 1
        d += timedelta(days=1)

    if all_flares:
        combined = pd.concat(all_flares, ignore_index=True)
        combined = combined.sort_values("peak_time").reset_index(drop=True)
        combined["satellite"] = args.sat
        out_path = output_dir / "goes_flare_events.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nWrote {len(combined)} flares -> {out_path}")
        by_letter = combined["goes_class"].dropna().str[0].value_counts()
        print(f"Class breakdown: {by_letter.to_dict()}")
    else:
        print("\nNo flares fetched.")

    print(f"\nDone.  {n_days_ok} days OK, {n_days_fail} failed.")
    return 0 if n_days_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

