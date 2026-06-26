# TEJAS: Temporal Evaluation and Joint Analysis of Solar-flares

Welcome to **TEJAS**, an AI-powered solar-flare nowcasting and forecasting platform built using combined SoLEXS and HEL1OS X-ray observations from the Aditya-L1 mission. 

This README provides a complete guide for setting up the environment, getting the required data, and contributing to the project.

For a deeper dive into the scientific background, strict integrity requirements, and architecture of this project, please read [CONTEXT.md](CONTEXT.md). For detailed information about the dataset and FITS files used, refer to [data_used.md](data_used.md).

## Project Overview
The objective of TEJAS is to:
1. Detect solar flares as they occur (nowcasting).
2. Forecast the probability of future solar flares before they occur.
3. Generate an automated flare catalogue.
4. Provide explainable alerts.
5. Visualize X-ray activity through a web-based operational dashboard.

**Core Philosophy:** This project must never cheat and use future data for inference or playback. It behaves exactly like a real-time operational monitoring system.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

- **Python 3.8+**: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).
- **Git**: Ensure Git is installed for version control.

### 1. Clone the Repository

```bash
git clone https://github.com/nomadrai/solar-flares-forecast
cd solar-flares-forecast
```

### 2. Set Up the Python Virtual Environment

We use a Python `venv` to keep dependencies isolated from your system packages.

**For Windows:**
```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

**For macOS / Linux:**
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

*(Note: Your terminal prompt should now show `(.venv)` indicating the environment is active.)*

### 3. Install Dependencies

With the virtual environment active, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install packages like `pandas`, `astropy` (for FITS file parsing), `scikit-learn`, `xgboost`, `streamlit`, and others.

### 4. Fetch the Data and Models

Because the raw FITS files and generated model outputs are very large, they are not stored directly in this Git repository. Instead, we host them on a shared cloud storage folder (e.g., Google Drive).

We have included `gdown` in the requirements to make downloading easy.

**Important Note on What to Download:**
- **If you are only working on the Frontend / Dashboard:** You **DO NOT** need to download the massive `data/` folder. You only need to download `outputs/` (which contains the pre-trained models and catalogues) so the UI has something to display.
- **If you are working on Data Processing, Feature Engineering, or ML Training:** You **MUST** download the `data/` folder. The TEJAS pipeline parses these raw `.lc.gz` and `.fits` files directly to build the unified datasets. Without it, you cannot run any of the processing or training scripts.

**To get the required directories:**
1. The project maintainer will provide a shared Google Drive Folder Link or File IDs containing the `data/` and `outputs/` folder.
2. You can download them manually and extract them into the root of this project, or use `gdown` from the terminal (once the `venv` is active):

```bash
# Run this for downloading the outputs/ folder
gdown --folder 1F6k36Tu1Ws6FLzkYbVTgt_N0Tdi9bXbg

# ONLY run this if you are doing ML/Data Science work!
gdown --folder 1RNQfZRFI5C-0ZmhETwfW_az8UXxxumi9

# These commands will download the required data folders in your workspace
```

**Expected Directory Structure after extraction:**
```
TEJAS/
├── data/
│   └── raw/
│       ├── solexs/
│       └── hel1os/
├── outputs/
│   ├── models/
│   └── catalogues/
├── CONTEXT.md
├── data_used.md
├── requirements.txt
└── ...
```

*(Refer to [data_used.md](data_used.md) for an exact breakdown of what these folders contain.)*

### 5. Running the Dashboard (Coming Soon / Example)

Once the data is in place, you can launch the Streamlit operational dashboard:

```bash
streamlit run app/main.py  # (Update this path based on the actual entry point of the UI)
```

---

## 🤝 Contributing

1. Ensure your local `venv` is active before writing code.
2. **Never** push the `data/` or `outputs/` directories to Git (they are already included in `.gitignore`).
3. If you add new pip packages, remember to update the requirements file:
   ```bash
   pip freeze > requirements.txt
   ```
4. Read the `CONTEXT.md` again. The #1 rule is **no future leakage**. Your algorithms must only use historical observations available at the current simulated timestamp.

---
*For any questions regarding access to the Google Drive dataset or architecture decisions, reach out to the project maintainer.*
