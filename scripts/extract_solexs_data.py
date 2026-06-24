# SoLEXS Data Extraction Script
# ==============================
# Selectively extracts SDD2 files (.lc.gz, .pi.gz, .gti.gz) from the downloaded
# SoLEXS zip archives and places them into the TEJAS project's data directory
# following the structure: data/raw/solexs/YYYY/MM/DD/
#
# Source: C:/Users/aadit/Desktop/TEJAS data/SoLEXS/Data/
# Destination: C:/Users/aadit/Documents/Projects/TEJAS/data/raw/solexs/
#
# What it does:
#   1. Iterates over all .zip files in the source directory
#   2. Extracts ONLY SDD2 files (skips SDD1 entirely)
#   3. Places the .gz files into data/raw/solexs/YYYY/MM/DD/
#   4. Reports a summary of what was processed
#
# What it does NOT do:
#   - Does not decompress .gz files (astropy reads them natively)
#   - Does not modify or delete the source zip files

import zipfile
import os
import re
import shutil
from pathlib import Path


# --- Configuration ---
SOURCE_DIR = Path(r"C:\Users\aadit\Desktop\TEJAS data\SoLEXS\Data")
DEST_DIR = Path(r"C:\Users\aadit\Documents\Projects\TEJAS\data\raw\solexs")

# Files we want from SDD2
WANTED_EXTENSIONS = {".lc.gz", ".pi.gz", ".gti.gz"}


def extract_date_from_zip(zip_name: str) -> tuple[str, str, str] | None:
    """Extract YYYY, MM, DD from zip filename like AL1_SLX_L1_20260501_v1.0.zip"""
    match = re.search(r"(\d{4})(\d{2})(\d{2})", zip_name)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def is_wanted_sdd2_file(name: str) -> bool:
    """Check if a zip member is a SDD2 file we want."""
    # Must be in SDD2 directory
    if "SDD2" not in name:
        return False
    # Must not be a directory entry
    if name.endswith("/"):
        return False
    # Must end with one of our wanted extensions
    return any(name.endswith(ext) for ext in WANTED_EXTENSIONS)


def main():
    print("=" * 70)
    print("SoLEXS Data Extraction Script")
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
    days_processed = []
    days_skipped = []
    errors = []

    for zip_path in zip_files:
        zip_name = zip_path.name
        date_parts = extract_date_from_zip(zip_name)

        if not date_parts:
            errors.append(f"Could not parse date from: {zip_name}")
            continue

        year, month, day = date_parts
        day_dir = DEST_DIR / year / month / day
        date_str = f"{year}-{month}-{day}"

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find SDD2 files we want
                sdd2_files = [m for m in zf.namelist() if is_wanted_sdd2_file(m)]

                if not sdd2_files:
                    days_skipped.append(date_str)
                    print(f"  SKIP {date_str}: No SDD2 data files found in {zip_name}")
                    continue

                # Create destination directory
                day_dir.mkdir(parents=True, exist_ok=True)

                # Extract each SDD2 file
                files_this_day = 0
                for member in sdd2_files:
                    # Extract to a temp location first, then move just the file
                    basename = os.path.basename(member)
                    dest_file = day_dir / basename

                    # Read from zip and write directly to destination
                    with zf.open(member) as src, open(dest_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    files_this_day += 1
                    total_files_extracted += 1

                days_processed.append(date_str)
                file_types = [os.path.basename(f).split("_L1.")[-1] for f in sdd2_files]
                print(f"  OK   {date_str}: {files_this_day} file(s) → {day_dir.relative_to(DEST_DIR.parent.parent.parent)}  [{', '.join(file_types)}]")

        except zipfile.BadZipFile:
            errors.append(f"Bad zip file: {zip_name}")
            print(f"  ERR  {date_str}: Bad zip file - {zip_name}")
        except Exception as e:
            errors.append(f"{zip_name}: {e}")
            print(f"  ERR  {date_str}: {e}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Days processed successfully: {len(days_processed)}")
    print(f"  Days skipped (no SDD2 data): {len(days_skipped)}")
    print(f"  Errors:                      {len(errors)}")
    print(f"  Total files extracted:       {total_files_extracted}")
    print(f"  Destination:                 {DEST_DIR}")
    print()

    if days_skipped:
        print("  Skipped days:")
        for d in days_skipped:
            print(f"    - {d}")
        print()

    if errors:
        print("  Errors:")
        for e in errors:
            print(f"    - {e}")
        print()

    # Verify the output
    print("  Output structure:")
    for day_path in sorted(DEST_DIR.rglob("*")):
        if day_path.is_file():
            rel = day_path.relative_to(DEST_DIR)
            size_kb = day_path.stat().st_size / 1024
            print(f"    {rel}  ({size_kb:.0f} KB)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
