import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("data/processed")

MODEL_NAME = "dense"

W_PATH = BASE_DIR / f"adjacency_matrix_{MODEL_NAME}_best_lambda.csv"
SECTOR_PATH = BASE_DIR / "ticker_sector_used.csv"

# -----------------------------
# load data
# -----------------------------

W = pd.read_csv(W_PATH, index_col=0)

ticker_sector = pd.read_csv(SECTOR_PATH)
ticker_to_sector = ticker_sector.set_index("ticker")["sector"]

common = [t for t in W.index if t in ticker_to_sector.index]
W = W.loc[common, common]
ticker_to_sector = ticker_to_sector.loc[common]

# -----------------------------
# threshold matrix
# -----------------------------

W_abs = np.abs(W)

threshold = np.percentile(W_abs.values, 99)   # keep top 1%

W_thr = W.copy()
W_thr[W_abs < threshold] = 0

print("Threshold:", threshold)
print("Nonzero fraction:", (W_thr != 0).sum().sum() / W_thr.size)

# -----------------------------
# sector influence scores
# -----------------------------

sectors = ticker_to_sector.unique()

influencer = {}
influencee = {}

for s in sectors:

    tickers = ticker_to_sector[ticker_to_sector == s].index

    influencer[s] = np.abs(W_thr.loc[:, tickers]).mean().mean()
    influencee[s] = np.abs(W_thr.loc[tickers, :]).mean().mean()

influencer = pd.Series(influencer).sort_values()
influencee = pd.Series(influencee).sort_values()

# -----------------------------
# plots
# -----------------------------

fig, axes = plt.subplots(1,2, figsize=(14,6))

axes[0].barh(influencer.index, influencer.values)
axes[0].set_title("Sector Influence (Top 1% Edges)")
axes[0].set_xlabel("Average |W_ij|")

axes[1].barh(influencee.index, influencee.values)
axes[1].set_title("Sector Predictability (Top 1% Edges)")
axes[1].set_xlabel("Average |W_ij|")

plt.tight_layout()
plt.show()