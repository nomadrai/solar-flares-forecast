# TEJAS — Data Reference Document

This document describes every data file used in the TEJAS project, its origin, its internal structure, what is extracted or derived from it, and exactly how it contributes to the project pipeline.

---

## Data Source

All data is obtained from **ISRO's PRADAN portal** ([pradan.issdc.gov.in](https://pradan.issdc.gov.in/pradan/al1)), which hosts publicly available Aditya-L1 mission data. Files are downloaded in day-wise (SoLEXS) or dump-wise (HEL1OS) zip archives, unzipped locally, and organized into the `data/raw/` directory.

All files are in **FITS (Flexible Image Transport System)** format, readable in Python using the `astropy.io.fits` module.

---

## Directory Structure

```
data/
├── raw/
│   ├── solexs/
│   │   └── 2026/
│   │       └── 05/
│   │           ├── 01/
│   │           │   ├── AL1_SOLEXS_20260501_SDD2_L1.lc.gz
│   │           │   ├── AL1_SOLEXS_20260501_SDD2_L1.pi.gz
│   │           │   └── AL1_SOLEXS_20260501_SDD2_L1.gti.gz
│   │           ├── 02/
│   │           │   └── ...
│   │           └── 31/
│   └── hel1os/
│       └── 2026/
│           └── 05/
│               ├── 01/
│               │   ├── lightcurve_cdte1.fits
│               │   ├── lightcurve_cdte2.fits
│               │   ├── lightcurve_czt1.fits
│               │   ├── lightcurve_czt2.fits
│               │   ├── gticdte1.fits
│               │   ├── gticdte2.fits
│               │   ├── gticzt1.fits
│               │   └── gticzt2.fits
│               ├── 02/
│               │   └── ...
│               └── 31/
├── processed/
│   ├── unified/                          ← Phase 2 output
│   │   └── 2026/
│   │       └── 05/
│   │           ├── 2026-05-01.parquet
│   │           ├── ...
│   │           └── ingestion_summary.csv
│   └── clean/                            ← Phase 3 output
│       └── 2026/
│           └── 05/
│               ├── 2026-05-01.parquet
│               ├── ...
│               └── processing_summary.csv
└── features/                             ← Phase 4 output
    └── 2026/
        └── 05/
            ├── 2026-05-01.parquet
            ├── ...
            └── feature_summary.csv
```

All directories follow **type-first, then chronological** (`type/year/month/`). Raw data additionally nests by day. This mirrors the processing pipeline stages and scales cleanly when new months or years are added — just change `YEAR` / `MONTH` in `tejas/config.py`.

---

## Instruments Used

| Instrument | Detector Used | Energy Range | Role in TEJAS |
|---|---|---|---|
| **SoLEXS** | SDD2 only | 2–22 keV (soft X-ray) | Thermal emission monitoring, flare gradual phase, spectral evolution, classification |
| **HEL1OS** | CdTe1, CdTe2 | 8–70 keV (hard X-ray) | Thermal + low non-thermal emission, flare transition phase |
| **HEL1OS** | CZT1, CZT2 | 20–150 keV (hard X-ray) | Non-thermal bremsstrahlung, flare impulsive phase, high-energy flare indicator |

### Why SDD2 Only (SoLEXS)

SoLEXS has two Silicon Drift Detectors: SDD1 (large aperture, 7.1 mm²) and SDD2 (small aperture, 0.1 mm²). During solar maximum, SDD1 frequently saturates due to high photon flux. For most days in the downloaded dataset, SDD1 does not even have LC or PI files generated. SDD2 remained operational without saturation and is recommended by ISRO for spectral analysis. TEJAS uses SDD2 exclusively.

### Why All Four HEL1OS Detectors

- **CdTe1 + CdTe2** (combined area 0.5 cm²): Small area means negligible background contamination. Clean hard X-ray signal in 8–70 keV. Primary hard X-ray channel for TEJAS.
- **CZT1 + CZT2** (combined area 32 cm²): Large area captures hard X-rays up to 150 keV. Has significant background from cosmic rays, SEPs, and diffuse X-ray sources. However, during flares the flare signal dominates over background, and the ML model can learn to distinguish signal from noise. Excluding CZT would discard the 70–150 keV range entirely — a region where only the most energetic flares produce emission, making it a strong discriminator for flare severity.

---

## Combined Energy Coverage

```
SoLEXS SDD2:  |████ 2 ──────── 22 keV ████|
HEL1OS CdTe:            |████ 8 ──────────────── 70 keV ████|
HEL1OS CZT:                    |████ 20 ─────────────────────── 150 keV ████|
                ─────────────────────────────────────────────────────────────────
Combined:       |████ 2 keV ─────────────────────────────────────── 150 keV ████|
```

This provides continuous coverage from 2 keV to 150 keV — spanning the full thermal-to-non-thermal X-ray spectrum of solar flares.

---

## File-by-File Reference

---

### 1. SoLEXS Light Curve — `.lc.gz`

**Filename pattern:** `AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc.gz`

**What it is:**
A gzip-compressed FITS file containing a time series of the total X-ray count rate detected by SDD2 in the energy range 2–22 keV, at a cadence of one second, for the entire day.

**Internal structure:**
The FITS file contains a `RATE` extension with two columns:

| Column | Type | Description |
|---|---|---|
| `TIME` | float | Unix timestamp (seconds since 1970-01-01 00:00:00 UTC) |
| `COUNTS` | float | Total counts per second in 2–22 keV |

**How to read it:**
```python
from astropy.io import fits
import pandas as pd

with fits.open('AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc.gz') as hdul:
    data = hdul['RATE'].data
    df = pd.DataFrame({
        'time_unix': data['TIME'],
        'counts': data['COUNTS']
    })
    df['datetime'] = pd.to_datetime(df['time_unix'], unit='s', origin='unix')
```

**What is derived from it:**
- Raw soft X-ray flux time series (the primary monitoring signal).
- Rolling averages, rolling variance, flux gradients, first/second derivatives — all computed from this single counts column during feature engineering.

**Contribution to TEJAS:**
- **EDA Phase:** Plot daily light curves, visually identify flares, understand baseline behaviour.
- **Data Ingestion:** Primary soft X-ray time series fed into the unified dataset.
- **Flare Detection (Nowcasting):** Threshold-based or algorithm-based detection of flare start, peak, and end from the counts time series.
- **Flare Classification:** Peak count rate is directly related to flare class (B, C, M, X).
- **Dashboard:** Live soft X-ray monitoring graph.

---

### 2. SoLEXS Pulse Invariant Spectral File — `.pi.gz`

**Filename pattern:** `AL1_SOLEXS_YYYYMMDD_SDD2_L1.pi.gz`

**What it is:**
A gzip-compressed FITS file containing a Type-II Pulse Invariant spectrum — energy-resolved X-ray counts recorded at one-second cadence for the entire day. Each row is a full 340-channel energy spectrum captured in one second.

**Internal structure:**
The FITS file contains a `SPECTRUM` extension with the following columns:

| Column | Type | Description |
|---|---|---|
| `TSTART` | float | Unix timestamp for the start of the 1-second integration |
| `TELAPSE` | float | Integration time (always 1 second) |
| `SPEC_NUM` | int | Unique spectrum identifier |
| `CHANNEL` | int array | Energy channel indices (1–340) |
| `COUNTS` | int array | Counts in each energy channel during the 1-second interval |
| `EXPOSURE` | float | Effective integration time after deadtime correction (currently equals TELAPSE) |

**Channel-to-energy mapping (from the SoLEXS manual):**
- Channels 1–168: each channel spans ~47.6 eV (full resolution)
- Channels 169–340: each channel spans ~95.2 eV (paired channels, double width)
- Approximate energy of channel `n`:
  - If n ≤ 168: Energy ≈ n × 0.0476 keV
  - If n > 168: Energy ≈ 8.0 + (n − 168) × 0.0952 keV

> **Note:** The instrument response below 2.8 keV is currently uncertain (per the SoLEXS manual). Data below 2.8 keV (~channel 59) should be treated with caution.

**How to read it:**
```python
from astropy.io import fits
import pandas as pd
import numpy as np

with fits.open('AL1_SOLEXS_YYYYMMDD_SDD2_L1.pi.gz') as hdul:
    spec_data = hdul['SPECTRUM'].data
    tstart = spec_data['TSTART']       # 1D array of Unix timestamps
    counts = spec_data['COUNTS']       # 2D array: shape (N_seconds, 340)
```

**What is derived from it (manually, during data processing):**

The PI file is **not** used for spectral fitting (no XSPEC, no response matrices, no plasma temperature modelling). Instead, it is used to extract **multi-band light curves** by summing counts across selected channel ranges:

| Derived Feature | Channel Range (approx.) | Energy Band | Physical Meaning |
|---|---|---|---|
| `counts_low` | Channels 59–105 | 2.8–5 keV | Cool thermal plasma emission |
| `counts_mid` | Channels 105–189 | 5–10 keV | Hotter plasma, transition |
| `counts_high` | Channels 189–340 | 10–22 keV | Hottest thermal plasma, early non-thermal |
| `hardness_ratio` | Derived: `counts_high / counts_low` | — | Spectral shape indicator |

```python
# Extracting multi-band light curves from PI data
ch_low = slice(58, 105)     # 2.8 - 5 keV
ch_mid = slice(105, 189)    # 5 - 10 keV
ch_high = slice(189, 340)   # 10 - 22 keV

df = pd.DataFrame({
    'time_unix': tstart,
    'counts_low':  np.sum(counts[:, ch_low], axis=1),
    'counts_mid':  np.sum(counts[:, ch_mid], axis=1),
    'counts_high': np.sum(counts[:, ch_high], axis=1),
})
df['hardness_ratio'] = df['counts_high'] / df['counts_low'].replace(0, np.nan)
```

**Contribution to TEJAS:**
- **Feature Engineering:** The hardness ratio and multi-band counts are powerful engineered features. Spectral hardening (hardness ratio increasing) is a known **precursor signature** that often appears minutes before flare onset — this is one of the most valuable features for the forecasting model.
- **Forecasting:** The ML model uses the temporal evolution of these multi-band features to predict future flare probability. A model trained only on total counts (from the LC file) would miss spectral evolution entirely.
- **Flare Classification:** Different flare classes have distinct spectral profiles. Multi-band ratios help the classifier distinguish between B, C, M, and X-class events.
- **Explainability:** When the system predicts a flare, it can point to "hardness ratio increased by X% over the last Y minutes" as an interpretable reason.

---

### 3. SoLEXS Good Time Interval File — `.gti.gz`

**Filename pattern:** `AL1_SOLEXS_YYYYMMDD_SDD2_L1.gti.gz`

**What it is:**
A gzip-compressed FITS file listing the time intervals during which SDD2 was collecting valid data. Data outside these intervals may be corrupted, absent, or recorded while the instrument was not in a valid observing state.

**Criteria for a valid interval (from the SoLEXS manual):**
1. The Sun was within the field of view of the instrument.
2. SDD2 was operational.
3. Health parameters were within recommended ranges.

**Internal structure:**
The FITS file contains a `GTI` extension with two columns:

| Column | Type | Description |
|---|---|---|
| `START` | float | Unix timestamp of the start of a good interval |
| `STOP` | float | Unix timestamp of the end of a good interval |

**How to read and apply it:**
```python
from astropy.io import fits
import pandas as pd
import numpy as np

def read_gti(filepath):
    with fits.open(filepath) as hdul:
        data = hdul['GTI'].data
        return pd.DataFrame({
            'start': data['START'],
            'stop': data['STOP']
        })

def apply_gti(lc_df, gti_df):
    """Keep only rows whose time_unix falls within a good time interval."""
    mask = np.zeros(len(lc_df), dtype=bool)
    for _, row in gti_df.iterrows():
        mask |= (lc_df['time_unix'] >= row['start']) & (lc_df['time_unix'] <= row['stop'])
    return lc_df[mask].reset_index(drop=True)
```

**What is derived from it:**
- A boolean mask applied to both the LC and PI data to **discard invalid observations**.
- Gap durations between valid intervals — useful for understanding data completeness.

**Contribution to TEJAS:**
- **Data Processing:** Filtering out bad data prevents the algorithm from training on or responding to spurious signals (instrument noise, off-pointing periods, etc.).
- **Data Quality Assessment:** The fraction of valid time per day helps assess data reliability.
- **Gap Handling:** Gaps between GTI intervals must be handled explicitly (interpolation, NaN-filling, or segmenting) so the ML model does not confuse a data gap with a quiet Sun.

---

### 4. HEL1OS CdTe Light Curves — `lightcurve_cdte1.fits`, `lightcurve_cdte2.fits`

**Filename pattern:** `lightcurve_cdte1.fits`, `lightcurve_cdte2.fits`

**What they are:**
FITS files containing the count rate time series from the two CdTe (Cadmium Telluride) detectors, at 1-second cadence, for the duration of the data dump. CdTe detectors cover the energy range ~8–70 keV.

**Why two files:**
HEL1OS has two independent CdTe detectors (CdTe1 and CdTe2), each with 0.25 cm² area. They observe the same Sun simultaneously, providing redundancy and improved statistics when combined.

**Internal structure:**
Each FITS file contains a data extension with (at minimum) the following columns:

| Column | Type | Description |
|---|---|---|
| `TIME` | float | Timestamp (format to be confirmed on first read — likely MET or Unix) |
| `COUNT_RATE` | float | Counts per second across the CdTe energy range |
| `STAT_ERR` | float | Statistical (Poissonian) error on the count rate |

> **Important:** On first read, inspect the file with `hdul.info()` and `print(hdul[1].columns.names)` to confirm the exact extension index and column names.

**How to read it:**
```python
from astropy.io import fits
import pandas as pd

with fits.open('lightcurve_cdte1.fits') as hdul:
    # Inspect structure on first read:
    # hdul.info()
    # print(hdul[1].columns.names)
    data = hdul[1].data
    df = pd.DataFrame({
        'time_unix': data['TIME'],
        'count_rate': data['COUNT_RATE'],
        'stat_err': data['STAT_ERR']
    })
    df['datetime'] = pd.to_datetime(df['time_unix'], unit='s', origin='unix')
```

**What is derived from it:**
- Combined CdTe count rate: average or sum of CdTe1 and CdTe2 for improved signal-to-noise.
- Hard X-ray flux time series at 1-second cadence.
- Hard X-ray gradients, rolling averages, impulsive phase indicators.

**Contribution to TEJAS:**
- **EDA:** Visualize hard X-ray activity, identify impulsive flare phases.
- **Data Ingestion:** Primary hard X-ray channel in the unified time series.
- **Flare Detection:** Hard X-ray bursts mark the **impulsive phase** of flares — often the earliest and sharpest signal of a flare's onset in the non-thermal regime.
- **Feature Engineering:** Soft-to-hard X-ray ratio (SoLEXS counts / CdTe counts), hard X-ray derivatives, and temporal lag between soft and hard X-ray peaks (related to the **Neupert effect**).
- **Forecasting:** The relationship between soft and hard X-ray evolution is a key physical signature the model must learn.
- **Dashboard:** Live hard X-ray monitoring graph.

**Why CdTe background is negligible:**
CdTe's combined area is only 0.5 cm² — tiny compared to CZT's 32 cm². Non-solar background sources (cosmic rays, diffuse X-rays) contribute negligibly at this area, giving CdTe an inherently clean signal.

---

### 5. HEL1OS CZT Light Curves — `lightcurve_czt1.fits`, `lightcurve_czt2.fits`

**Filename pattern:** `lightcurve_czt1.fits`, `lightcurve_czt2.fits`

**What they are:**
FITS files containing the count rate time series from the two CZT (Cadmium Zinc Telluride) detectors, at 1-second cadence. CZT detectors cover the energy range ~20–150 keV — the highest energy band available in the TEJAS dataset.

**Why two files:**
HEL1OS has two pixelated CZT detectors (256 pixels each), providing a combined geometric area of 32 cm².

**Internal structure:**
Same column format as CdTe light curves:

| Column | Type | Description |
|---|---|---|
| `TIME` | float | Timestamp |
| `COUNT_RATE` | float | Counts per second across the CZT energy range |
| `STAT_ERR` | float | Statistical error on the count rate |

**How to read it:**
```python
# Identical to CdTe light curve reading
with fits.open('lightcurve_czt1.fits') as hdul:
    data = hdul[1].data
    df = pd.DataFrame({
        'time_unix': data['TIME'],
        'count_rate': data['COUNT_RATE'],
        'stat_err': data['STAT_ERR']
    })
```

**Background considerations:**
CZT's large area (32 cm²) means it is sensitive to **non-solar background** sources:
- Cosmic diffuse X-ray background
- Charged particles from Solar Energetic Particles (SEPs) and Galactic Cosmic Rays

During **quiet periods**, CZT data will show an elevated, noisy baseline. During **flares** (especially M-class and above), the solar flare signal dominates over background. The ML model is expected to learn to distinguish these regimes.

**What is derived from it:**
- High-energy hard X-ray flux time series.
- Combined CZT count rate (CZT1 + CZT2 averaged).
- Presence/absence of high-energy emission as a binary or continuous feature.

**Contribution to TEJAS:**
- **Flare Detection:** Emission above 70 keV is a hallmark of the most energetic flares involving significant particle acceleration. If CZT sees a clear signal above its background, a strong flare is in progress.
- **Flare Classification:** CZT signal strength helps discriminate between moderate flares (C/M-class, little to no CZT emission) and severe flares (strong M/X-class, clear CZT emission). This is information that CdTe alone cannot provide.
- **Feature Engineering:** CZT-to-CdTe ratio, CZT-to-SoLEXS ratio, and whether CZT count rate exceeds a background threshold — all useful features.
- **Forecasting:** The model learns that certain pre-flare patterns lead to flares with high-energy emission (and thus higher severity).

---

### 6. HEL1OS CdTe GTI Files — `gticdte1.fits`, `gticdte2.fits`

**Filename pattern:** `gticdte1.fits`, `gticdte2.fits`

**What they are:**
Good Time Interval files for the CdTe1 and CdTe2 detectors. They list the time ranges during which each CdTe detector was collecting valid scientific data.

**Internal structure:**

| Column | Type | Description |
|---|---|---|
| `START` | float | Start time of a valid interval |
| `STOP` | float | End time of a valid interval |

**How to read and apply:** Same pattern as SoLEXS GTI (see section 3 above).

**Contribution to TEJAS:**
Applied to CdTe light curves to discard invalid data before ingestion. Each detector has its own GTI because CdTe1 and CdTe2 may have been independently switched on/off.

---

### 7. HEL1OS CZT GTI Files — `gticzt1.fits`, `gticzt2.fits`

**Filename pattern:** `gticzt1.fits`, `gticzt2.fits`

**What they are:**
Good Time Interval files for the CZT1 and CZT2 detectors.

**Internal structure and usage:** Identical to CdTe GTI files.

**Contribution to TEJAS:**
Applied to CZT light curves to discard invalid data before ingestion.

---

## Files NOT Used (and Why)

The downloaded archives contain additional files that are **not used** in TEJAS:

### SoLEXS

| File | Why Excluded |
|---|---|
| SDD1 files (`.lc.gz`, `.pi.gz`, `.gti.gz`) | SDD1 frequently saturates during solar maximum. Most days in the dataset have no SDD1 LC or PI files generated. |
| Calibration database (ARF, RMF) | These are instrument response files for spectral fitting with XSPEC/Sherpa. TEJAS does not perform spectral fitting — it extracts multi-band counts directly from PI channel ranges. |

### HEL1OS

| File | Why Excluded |
|---|---|
| `hel1os_cdte_spectra_cdte1.fits` / `cdte2.fits` | Type-II spectral files at 20-second cadence. Too coarse for real-time flare detection (TEJAS operates at 1-second cadence). Energy-resolved information for the soft X-ray range is already obtained from SoLEXS PI files. |
| `hel1os_czt_spectra_czt1.fits` / `czt2.fits` | Same reason as CdTe spectral files. |
| `evt.fits` | Raw event list (photon-by-photon, 10 ms cadence). The light curve files are already derived from this. Processing raw events would be redundant and computationally expensive. |
| `hk.fits` | Housekeeping telemetry (instrument temperatures, voltages, etc.). Not relevant to flare science. |
| CalDB response files (ARF, SRF) | For spectral fitting with OSPEX. Not used in TEJAS. |
| `czt*dispix.txt` | Noisy CZT pixel lists. Already accounted for in the light curve generation pipeline at POC. |

---

## Summary: Data Flow Into TEJAS

```
  SoLEXS SDD2                      HEL1OS CdTe                    HEL1OS CZT
  ───────────                       ───────────                    ──────────
  .lc.gz ──→ total soft X-ray      lightcurve_cdte1 ──┐           lightcurve_czt1 ──┐
             counts (2-22 keV)      lightcurve_cdte2 ──┤→ combined lightcurve_czt2 ──┤→ combined
                                    hard X-ray counts  │           high-energy hard   │
  .pi.gz ──→ multi-band counts     (8-70 keV)         │           X-ray counts       │
             - low  (2.8-5 keV)                        │           (20-150 keV)       │
             - mid  (5-10 keV)     gticdte1 ───────────┘                              │
             - high (10-22 keV)    gticdte2  (quality filter)     gticzt1 ────────────┘
             - hardness ratio                                     gticzt2  (quality filter)
                    │
  .gti.gz ──→ quality filter
                    │
                    ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                    UNIFIED TIME-SERIES DATASET (1-sec cadence)              │
  │                                                                            │
  │  timestamp | soft_total | soft_low | soft_mid | soft_high | hardness_ratio │
  │            | cdte_rate  | czt_rate | soft_hard_ratio | ...derived features │
  └──────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
          Feature Engineering → Flare Detection → Classification
                → Forecasting → Alerts → Dashboard
```

---

## Time Format Reference

| Instrument | Time Format | Description |
|---|---|---|
| SoLEXS | Unix time | Seconds since 1970-01-01 00:00:00 UTC |
| HEL1OS | To be confirmed on first read | Likely Unix time or Mission Elapsed Time (MET). Inspect the `TIME` column and FITS header keywords (`MJDREFI`, `TIMEZERO`, etc.) on first read. |

Both must be converted to a common time axis (UTC datetime) during data ingestion for synchronization.
