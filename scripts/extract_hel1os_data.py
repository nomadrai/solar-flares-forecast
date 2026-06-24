# HEL1OS Data Extraction Script
# ==============================
# Selectively extracts the needed FITS files from HEL1OS dump-wise zip archives
# and places them into the TEJAS project's data directory.
#
# From each dump, we extract ONLY:
#   - lightcurve_cdte1.fits, lightcurve_cdte2.fits  (from cdte/)
#   - lightcurve_czt1.fits, lightcurve_czt2.fits    (from czt/)
#   - gticdte1.fits, gticdte2.fits                   (from aux/)
#   - gticzt1.fits, gticzt2.fits                     (from aux/)
#
# We DISCARD:
#   - Spectral files (hel1os_*_spectra_*.fits) -- TEJAS uses 1-sec LC, not 20-sec spectra
#   - Event list (evt.fits) -- redundant, lightcurves are derived from this
#   - Housekeeping (hk.fits) -- not relevant to flare science
#   - CZT noisy pixel lists (czt*dispix.txt) -- already accounted for in LC generation
#
# HEL1OS data is dump-wise (~12 hours per dump), not day-wise. Multiple dumps
# can belong to the same calendar day. We use the internal zip path (YYYY/MM/DD/)
# to determine the correct day folder, and each dump gets its own subdirectory
# to avoid filename collisions between dumps of the same day.
#
# Target structure:
#   data/raw/hel1os/2026/05/DD/dump_HHMMSS/
#       lightcurve_cdte1.fits
#       lightcurve_cdte2.fits
#       lightcurve_czt1.fits
#       lightcurve_czt2.fits
#       gticdte1.fits
#       gticdte2.fits
#       gticzt1.fits
#       gticzt2.fits

import zipfile
import os
import re
import shutil
from pathlib import Path


# --- Configuration ---
SOURCE_DIR = Path(r"C:\Users\aadit\Desktop\TEJAS data\HEL1OS\Data")
DEST_DIR = Path(r"C:\Users\aadit\Documents\Projects\TEJAS\data\raw\hel1os")

# Files we want (basename matching)
WANTED_FILES = {
    "lightcurve_cdte1.fits",
    "lightcurve_cdte2.fits",
    "lightcurve_czt1.fits",
    "lightcurve_czt2.fits",
    "gticdte1.fits",
    "gticdte2.fits",
    "gticzt1.fits",
    "gticzt2.fits",
}


def parse_zip_name(zip_name: str) -> dict | None:
    """Parse HEL1OS zip filename to extract metadata.
    
    Example: HLS_20260501_000009_43176sec_lev1_V111.zip
    Returns: {'date': '20260501', 'time': '000009', 'duration_sec': 43176}
    """
    match = re.match(r"HLS_(\d{8})_(\d{6})_(\d+)sec_", zip_name)
    if match:
        return {
            "date": match.group(1),
            "time": match.group(2),
            "duration_sec": int(match.group(3)),
        }
    return None


def is_wanted_file(member_name: str) -> bool:
    """Check if a zip member is a file we want."""
    basename = os.path.basename(member_name)
    return basename in WANTED_FILES


def get_day_from_zip_path(member_name: str) -> tuple[str, str, str] | None:
    """Extract YYYY, MM, DD from the internal zip path.
    
    Internal paths look like: 2026/05/01/HLS_.../aux/gticdte1.fits
    The internal path reflects the catalogued day, which may differ from
    the dump start date (e.g. a dump starting at 23:59 on May 2 may be
    catalogued under May 3).
    """
    match = re.match(r"(\d{4})/(\d{2})/(\d{2})/", member_name)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def main():
    print("=" * 70)
    print("HEL1OS Data Extraction Script")
    print("=" * 70)
    print(f"Source:      {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print()

    # Find all zip files recursively
    zip_files = sorted(SOURCE_DIR.rglob("*.zip"))

    if not zip_files:
        print("ERROR: No .zip files found in source directory!")
        return

    print(f"Found {len(zip_files)} zip file(s)\n")

    # Tracking
    total_files_extracted = 0
    dumps_processed = 0
    dumps_skipped = []
    errors = []

    for zip_path in zip_files:
        zip_name = zip_path.name
        meta = parse_zip_name(zip_name)

        if not meta:
            errors.append(f"Could not parse: {zip_name}")
            continue

        dump_time = meta["time"]
        duration_h = meta["duration_sec"] / 3600

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find wanted files
                wanted_members = [m for m in zf.namelist() if is_wanted_file(m)]

                if not wanted_members:
                    dumps_skipped.append(zip_name)
                    print(f"  SKIP {zip_name}: No wanted files found")
                    continue

                # Determine the day from the internal zip path
                day_info = get_day_from_zip_path(wanted_members[0])
                if not day_info:
                    errors.append(f"Could not determine day from internal path: {zip_name}")
                    continue

                year, month, day = day_info

                # Each dump gets its own subdirectory named by dump start time
                # to avoid filename collisions (e.g. two dumps on same day both
                # have lightcurve_cdte1.fits)
                dump_dir = DEST_DIR / year / month / day / f"dump_{dump_time}"
                dump_dir.mkdir(parents=True, exist_ok=True)

                # Extract each wanted file
                files_this_dump = 0
                extracted_names = []
                for member in wanted_members:
                    basename = os.path.basename(member)
                    dest_file = dump_dir / basename

                    with zf.open(member) as src, open(dest_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    files_this_dump += 1
                    total_files_extracted += 1
                    extracted_names.append(basename)

                dumps_processed += 1
                date_str = f"{year}-{month}-{day}"
                print(f"  OK   {date_str} dump_{dump_time} ({duration_h:.1f}h): "
                      f"{files_this_dump} files")

        except zipfile.BadZipFile:
            errors.append(f"Bad zip file: {zip_name}")
            print(f"  ERR  {zip_name}: Bad zip file")
        except Exception as e:
            errors.append(f"{zip_name}: {e}")
            print(f"  ERR  {zip_name}: {e}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Dumps processed:        {dumps_processed}")
    print(f"  Dumps skipped:          {len(dumps_skipped)}")
    print(f"  Errors:                 {len(errors)}")
    print(f"  Total files extracted:  {total_files_extracted}")
    print(f"  Expected per dump:      {len(WANTED_FILES)} files")
    print(f"  Destination:            {DEST_DIR}")
    print()

    if dumps_skipped:
        print("  Skipped dumps:")
        for d in dumps_skipped:
            print(f"    - {d}")
        print()

    if errors:
        print("  Errors:")
        for e in errors:
            print(f"    - {e}")
        print()

    # Verify the output - show day-level summary
    print("  Output structure (day-level):")
    if DEST_DIR.exists():
        for year_dir in sorted(DEST_DIR.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    dumps = sorted([d for d in day_dir.iterdir() if d.is_dir()])
                    total_size = sum(
                        f.stat().st_size
                        for dump in dumps
                        for f in dump.iterdir()
                        if f.is_file()
                    )
                    size_mb = total_size / 1024 / 1024
                    rel = day_dir.relative_to(DEST_DIR)
                    print(f"    {rel}: {len(dumps)} dump(s), {size_mb:.1f} MB")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
