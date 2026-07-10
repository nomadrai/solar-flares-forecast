"""
TEJAS — Optional GOES XRS Data Fetcher
=========================================
Downloads GOES-16 / GOES-18 XRS 1-second flux data (1-8 Å channel) from
NOAA NCEI archive for the May 2026 period and saves per-day CSVs to
data/reference/goes/YYYY/MM/YYYY-MM-DD.csv.

This is an OPTIONAL step.  The core TEJAS classification pipeline works
without GOES data — it just skips the goes_* columns.  Only run this if
you want empirical count→flux calibration and GOES-equivalent labels.

Usage:
    python scripts/fetch_goes_data.py                # fetch full month
    python scripts/fetch_goes_data.py --sat goes16   # use GOES-16
    python scripts/fetch_goes_data.py --dry-run      # show URLs, no download

Dependencies:
    requests, xarray, netCDF4  (all pip-installable)
"""

import sys
import argparse
import io
from pathlib import Path
from datetime import date, timedelta

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from tejas.config import YEAR, MONTH, MISSING_DAYS, GOES_REFERENCE_DIR


NCEI_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
# Known version suffixes used by NCEI netCDF files.  Tried in order.
VERSION_SUFFIXES = ["v2-2-1", "v2-2-0", "v2-1-0", "v2-0-0", "v1-0-0"]


def build_url(sat, day):
    """Return a list of candidate URLs (varying version suffix)."""
    yyyy = f"{day.year:04d}"
    mm = f"{day.month:02d}"
    ymd = day.strftime("%Y%m%d")
    sat_short = "g" + sat.replace("goes", "")  # goes18 -> g18
    return [
        f"{NCEI_BASE}/{sat}/l2/data/xrsf-l2-flx1s_science/{yyyy}/{mm}/"
        f"sci_xrsf-l2-flx1s_{sat_short}_d{ymd}_{v}.nc"
        for v in VERSION_SUFFIXES
    ]


def fetch_day(sat, day, timeout=60):
    """Download and parse one day of GOES XRS data.

    Args:
        sat: Satellite id string, e.g. 'goes18'.
        day: datetime.date instance.
        timeout: HTTP timeout (seconds).

    Returns:
        DataFrame with columns [time_unix, flux_wm2] or None on failure.
    """
    import requests

    try:
        import xarray as xr
    except ImportError:
        print(
            "  [ERROR] xarray + netCDF4 required.  Install with:\n"
            "    pip install xarray netCDF4"
        )
        return None

    urls = build_url(sat, day)
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
        except requests.RequestException as e:
            print(f"  [WARN] Network error for {url}: {e}")
            continue
        if r.status_code != 200:
            continue

        try:
            ds = xr.open_dataset(io.BytesIO(r.content), engine="h5netcdf")
        except Exception:
            try:
                # Fall back to netCDF4 engine (via a temp file).
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    tmp.write(r.content)
                    tmp_path = tmp.name
                ds = xr.open_dataset(tmp_path, engine="netcdf4")
            except Exception as e:
                print(f"  [WARN] Failed to parse netCDF: {e}")
                continue

        # xrsb_flux (1-8 Å) is the classification-relevant channel.
        # Time coordinate is 'time' as datetime64[ns].
        if "xrsb_flux" not in ds.variables:
            print(f"  [WARN] xrsb_flux missing in {url}")
            ds.close()
            continue

        flux = ds["xrsb_flux"].values
        times = pd.to_datetime(ds["time"].values, utc=True).tz_localize(None)
        time_unix = times.astype("int64") // 10**9

        # NCEI marks bad data with a quality flag and/or negative fill values.
        if "xrsb_flag" in ds.variables:
            flag = ds["xrsb_flag"].values
            flux = np.where(flag == 0, flux, np.nan)
        flux = np.where(flux > 0, flux, np.nan)

        ds.close()
        return pd.DataFrame({"time_unix": time_unix, "flux_wm2": flux}).dropna()

    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch GOES XRS data for the TEJAS study period.")
    parser.add_argument("--sat", default="goes18", choices=["goes16", "goes17", "goes18"],
                        help="GOES satellite id (default: goes18)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show candidate URLs, download nothing.")
    args = parser.parse_args()

    year = int(YEAR)
    month = int(MONTH)
    output_dir = Path(GOES_REFERENCE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all days in the target month.
    d = date(year, month, 1)
    end = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)

    print(f"\n{'='*60}")
    print(f"GOES XRS fetch — {args.sat.upper()} — {YEAR}-{MONTH}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        while d <= end:
            for url in build_url(args.sat, d):
                print(f"  {url}")
            d += timedelta(days=1)
        return 0

    n_ok, n_fail = 0, 0
    while d <= end:
        dd = f"{d.day:02d}"
        if dd in MISSING_DAYS:
            print(f"  {d}: SKIP (missing day in TEJAS dataset)")
            d += timedelta(days=1)
            continue

        out_path = output_dir / f"{d.isoformat()}.csv"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  {d}: cached ({out_path.name})")
            n_ok += 1
            d += timedelta(days=1)
            continue

        print(f"  {d}: fetching...", end=" ", flush=True)
        df = fetch_day(args.sat, d)
        if df is None or df.empty:
            print("FAILED")
            n_fail += 1
        else:
            df.to_csv(out_path, index=False)
            print(f"OK ({len(df):,} rows)")
            n_ok += 1
        d += timedelta(days=1)

    print(f"\nDone.  {n_ok} days OK, {n_fail} failed.")
    if n_fail:
        print("Failures are usually because NCEI has not yet published these dates,")
        print("or the version suffix differs.  Re-run later or edit VERSION_SUFFIXES.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
