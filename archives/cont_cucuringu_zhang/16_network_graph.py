import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# --------------------------------------------------
# paths
# --------------------------------------------------
BASE_DIR = Path("data/processed")

MODEL_NAME = "dense"
W_PATH = BASE_DIR / f"adjacency_matrix_{MODEL_NAME}_best_lambda.csv"
SECTOR_PATH = BASE_DIR / "ticker_sector_used.csv"
OUT_PATH = BASE_DIR / f"sector_network_{MODEL_NAME}_top1pct.png"

# --------------------------------------------------
# settings
# --------------------------------------------------
ASSET_EDGE_PERCENTILE = 99.0   # keep top 1% strongest asset-level edges
MIN_SECTOR_EDGE_PERCENTILE = 75.0  # after aggregation, keep top 25% sector edges
REMOVE_SELF_LOOPS = False

# --------------------------------------------------
# load data
# --------------------------------------------------
W = pd.read_csv(W_PATH, index_col=0)

ticker_sector = pd.read_csv(SECTOR_PATH)
ticker_sector.columns = [c.strip() for c in ticker_sector.columns]
ticker_to_sector = ticker_sector.set_index("ticker")["sector"]

# align tickers
common = [t for t in W.index if t in ticker_to_sector.index and t in W.columns]
W = W.loc[common, common]
ticker_to_sector = ticker_to_sector.loc[common]

# --------------------------------------------------
# threshold asset-level matrix
# --------------------------------------------------
W_abs = np.abs(W.to_numpy(dtype=float))
asset_thresh = np.percentile(W_abs, ASSET_EDGE_PERCENTILE)

W_thr = W.copy()
W_thr[np.abs(W_thr) < asset_thresh] = 0.0

print(f"Asset-level threshold: {asset_thresh:.6f}")
print(f"Fraction of nonzero asset edges kept: {(W_thr.to_numpy()!=0).mean():.4%}")

# --------------------------------------------------
# aggregate to sector-level network
# sector edge j -> i = average |W_ij| over surviving edges
# rows = predicted sector i
# cols = predictor sector j
# --------------------------------------------------
sectors = sorted(ticker_to_sector.unique())

sector_mat = pd.DataFrame(0.0, index=sectors, columns=sectors)

for pred_sector in sectors:
    pred_tickers = ticker_to_sector[ticker_to_sector == pred_sector].index.tolist()

    for src_sector in sectors:
        src_tickers = ticker_to_sector[ticker_to_sector == src_sector].index.tolist()

        block = W_thr.loc[pred_tickers, src_tickers].to_numpy(dtype=float)
        nonzero = np.abs(block[block != 0])

        # average surviving edge strength; 0 if no strong edges survive
        sector_mat.loc[pred_sector, src_sector] = nonzero.mean() if nonzero.size > 0 else 0.0

# optionally remove diagonal
if REMOVE_SELF_LOOPS:
    np.fill_diagonal(sector_mat.values, 0.0)

# --------------------------------------------------
# threshold sector-level edges again for readability
# --------------------------------------------------
positive_vals = sector_mat.to_numpy(dtype=float)
positive_vals = positive_vals[positive_vals > 0]

if positive_vals.size == 0:
    raise ValueError("No sector-level edges survived. Lower the thresholds.")

sector_thresh = np.percentile(positive_vals, MIN_SECTOR_EDGE_PERCENTILE)
sector_mat_plot = sector_mat.copy()
sector_mat_plot[sector_mat_plot < sector_thresh] = 0.0

print(f"Sector-level threshold: {sector_thresh:.6f}")
print(f"Nonzero sector edges kept: {(sector_mat_plot.to_numpy()>0).sum()}")

# --------------------------------------------------
# build directed graph
# edge: src_sector -> pred_sector
# --------------------------------------------------
G = nx.DiGraph()

# node sizes: total influence + influencee strength
out_strength = sector_mat_plot.sum(axis=0)  # source/predictor strength
in_strength = sector_mat_plot.sum(axis=1)   # predicted/influencee strength

for s in sectors:
    node_strength = out_strength.get(s, 0.0) + in_strength.get(s, 0.0)
    G.add_node(s, strength=node_strength)

for pred_sector in sectors:
    for src_sector in sectors:
        weight = sector_mat_plot.loc[pred_sector, src_sector]
        if weight > 0:
            G.add_edge(src_sector, pred_sector, weight=weight)

# --------------------------------------------------
# layout
# --------------------------------------------------
pos = nx.spring_layout(G, seed=42, k=1.2)

# node sizes
node_strengths = np.array([G.nodes[n]["strength"] for n in G.nodes()])
if node_strengths.max() > node_strengths.min():
    node_sizes = 1500 + 4000 * (node_strengths - node_strengths.min()) / (node_strengths.max() - node_strengths.min())
else:
    node_sizes = np.full(len(node_strengths), 2500.0)

# edge widths
edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
if edge_weights.size > 0:
    if edge_weights.max() > edge_weights.min():
        edge_widths = 1.5 + 6.0 * (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
    else:
        edge_widths = np.full(edge_weights.shape, 3.0)
else:
    edge_widths = []

# --------------------------------------------------
# draw
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    ax=ax,
)

nx.draw_networkx_labels(
    G, pos,
    font_size=11,
    ax=ax,
)

nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    arrows=True,
    arrowsize=20,
    arrowstyle="-|>",
    alpha=0.8,
    connectionstyle="arc3,rad=0.08",
    ax=ax,
)

ax.set_title(
    f"Thresholded Sector Order-Flow Influence Network ({MODEL_NAME})\n"
    f"Top {100-ASSET_EDGE_PERCENTILE:.0f}% asset edges, top {100-MIN_SECTOR_EDGE_PERCENTILE:.0f}% sector edges",
    fontsize=14,
    pad=15,
)

ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved network plot to:", OUT_PATH.resolve())
print("\nSector-level matrix used for the graph:")
print(sector_mat_plot.round(5))
print("\nOut-strength (influencer score):")
print(out_strength.sort_values(ascending=False).round(5))
print("\nIn-strength (influencee score):")
print(in_strength.sort_values(ascending=False).round(5))