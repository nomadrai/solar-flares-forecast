"""Quick preview of a unified Parquet file."""
import pandas as pd

df = pd.read_parquet("data/processed/unified/2026-05-10.parquet")

print("=== Unified Dataset: May 10, 2026 ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

cols = ["datetime", "solexs_counts", "solexs_hardness_ratio",
        "cdte_count_rate", "czt_count_rate", "solexs_valid", "hel1os_valid"]
print("--- First 10 rows ---")
print(df[cols].head(10).to_string())
print()

print("--- Summary stats ---")
print(df[["solexs_counts", "cdte_count_rate", "czt_count_rate"]].describe().round(2).to_string())
print()

n = len(df)
sv = df["solexs_valid"].sum()
hv = df["hel1os_valid"].sum()
print(f"SoLEXS valid: {sv}/{n} ({sv/n*100:.1f}%)")
print(f"HEL1OS valid: {hv}/{n} ({hv/n*100:.1f}%)")
