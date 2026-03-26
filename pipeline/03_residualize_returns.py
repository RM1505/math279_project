import numpy as np
import pandas as pd
from pathlib import Path

WINDOW     = 60
INPUT      = Path("data/processed/feature_table.csv")
OUTPUT     = Path("data/processed/feature_table_with_residuals_10level.csv")
CHUNK_SIZE = 50_000
TMP_RESID  = Path("data/processed/_residuals_tmp.csv")

# ── step 1: load OPCL from the tiny return matrix (19MB, not 12GB) ───
print("Loading OPCL columns...", flush=True)
ret_wide = pd.read_csv("data/processed/return_matrix_unbalanced.csv", index_col=0)
# ret_wide: rows=dates, cols=tickers → melt to (date, ticker, OPCL)
ret_df = ret_wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="OPCL")
ret_df = ret_df.dropna(subset=["OPCL"])
del ret_wide

spy_opcl = pd.read_csv("data/processed/spy_opcl.csv").set_index("date")["SPY"]
ret_df["spy_ret"] = ret_df["date"].map(spy_opcl)

# ── step 2: rolling beta per ticker — write directly to disk ─────────
print("Computing rolling betas (writing to temp file)...", flush=True)
first_ticker = True
for ticker, grp in ret_df.groupby("ticker", sort=False):
    grp = grp.sort_values("date").copy()
    cov = grp["OPCL"].rolling(WINDOW).cov(grp["spy_ret"])
    var = grp["spy_ret"].rolling(WINDOW).var()
    grp["beta"]         = cov / var
    grp["residual_ret"] = grp["OPCL"] - grp["beta"] * grp["spy_ret"]
    out = grp[["date", "ticker", "spy_ret", "beta", "residual_ret"]].dropna(subset=["beta"])
    out.to_csv(TMP_RESID, mode="w" if first_ticker else "a",
               header=first_ticker, index=False)
    first_ticker = False

del ret_df
print(f"  Temp residuals written: {TMP_RESID.stat().st_size / 1e6:.1f} MB", flush=True)

# ── step 3: read back residuals with memory diet ──────────────────────
print("Loading residuals (memory diet mode)...", flush=True)
dtypes = {"ticker": "category", "spy_ret": "float32",
          "beta": "float32", "residual_ret": "float32"}
residuals = pd.read_csv(TMP_RESID, dtype=dtypes)
print(f"  {len(residuals):,} rows loaded.", flush=True)

# ── step 4: stream feature_table in chunks, merge residuals ──────────
print("Writing output in chunks...", flush=True)
first_chunk = True
for chunk in pd.read_csv(INPUT, chunksize=CHUNK_SIZE, low_memory=False):
    chunk = chunk[chunk["ticker"] != "SPY"]
    chunk = pd.merge(chunk, residuals, on=["date", "ticker"], how="inner")
    chunk.to_csv(OUTPUT, mode="w" if first_chunk else "a",
                 header=first_chunk, index=False)
    first_chunk = False

if TMP_RESID.exists():
    TMP_RESID.unlink()
print(f"Done. Saved to {OUTPUT}", flush=True)
