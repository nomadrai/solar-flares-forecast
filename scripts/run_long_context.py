"""
v2 — Long-Context Feature Augmentation (in place)
==================================================
Adds the 5 long-window context features (bg_pctrank_1h/3h/6h,
rise_consistency_300s, time_elevated_s) to the Phase-5 detection parquets
without re-running Phases 4-5, preserving all labels (including the v2
credible relabel).  Mirrors the run_ar_proxy.py augmentation pattern.

Run:  python -m scripts.run_long_context
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tejas.config import DETECTION_DIR, YEAR, MONTH, MISSING_DAYS
from tejas.features.long_context import compute_long_context_features


def main():
    days = [f"{d:02d}" for d in range(1, 32) if f"{d:02d}" not in MISSING_DAYS]
    t0 = time.time()
    n_done = 0
    for dd in days:
        date_str = f"{YEAR}-{MONTH}-{dd}"
        pq = Path(DETECTION_DIR) / f"{date_str}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        result = compute_long_context_features(df)
        out = result["df"]
        out.to_parquet(pq, index=False, engine="pyarrow")
        n_done += 1
        print(f"  {date_str}: +{len(result['columns_added'])} cols "
              f"({', '.join(result['columns_added'])})")
    print(f"Done: {n_done} days in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
