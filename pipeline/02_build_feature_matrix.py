import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── 1. pivot integrated OFI → wide (one row per ticker-day) ──────────
folder = Path("data/processed/integrated_ofi")
files  = sorted(folder.glob("*.csv"))

dfs = []
for file in tqdm(files, desc="Pivoting integrated OFI"):
    df = pd.read_csv(file)
    df = df.sort_values(["source", "time"]).reset_index(drop=True)
    df["minute"] = df.groupby("source").cumcount() + 1
    df_wide = df.pivot(index="source", columns="minute", values="IOFI")
    df_wide.columns = [f"minute_{int(col)}" for col in df_wide.columns]
    df_wide = df_wide.reset_index().rename(columns={"source": "date"})
    df_wide["ticker"] = file.stem
    dfs.append(df_wide)

combined = pd.concat(dfs, ignore_index=True)
combined["date"] = pd.to_datetime(combined["date"])

# ── 2. merge OPCL returns (wide → long) ──────────────────────────────
ret_wide = pd.read_csv("data/processed/return_matrix_unbalanced.csv")
ret_wide["date"] = pd.to_datetime(ret_wide["date"])
ret_long = ret_wide.melt(id_vars="date", var_name="ticker", value_name="OPCL")
ret_long = ret_long.dropna(subset=["OPCL"])

combined = combined.merge(ret_long, on=["ticker", "date"], how="left")

# ── 3. merge sector ───────────────────────────────────────────────────
sectors = pd.read_csv("data/processed/ticker_sector_used.csv")
combined = combined.merge(sectors, on="ticker", how="left")

combined.to_csv("data/processed/feature_table.csv", index=False)
print(f"Wrote feature_table.csv: {combined.shape[0]:,} rows × {combined.shape[1]} cols")