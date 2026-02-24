from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px

# -----------------------
# Config
# -----------------------
MEAN_PATH = Path("data/processed/mean_pnl.npy")
SR_PATH   = Path("data/processed/sr_annualized.npy")
INDEX_CSV = Path("data/index.csv")

OUT_DIR = Path("data/processed/viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If you have too many tickers, labels get messy.
# Recommended: hide labels and rely on hover.
SHOW_AXIS_TICK_LABELS = False
SHOW_EVERY_K_LABELS = 25  # only used if SHOW_AXIS_TICK_LABELS=True

# Sharpe visualization clipping (helps avoid a few huge values ruining contrast)
CLIP_SR_FOR_DISPLAY = True
SR_CLIP_LO, SR_CLIP_HI = -10, 10

# -----------------------
# Load
# -----------------------
mean_pnl = np.load(MEAN_PATH)          # (N,N)
sr_ann   = np.load(SR_PATH)            # (N,N)

idx = pd.read_csv(INDEX_CSV, index_col=0)
tickers = idx["ticker"].astype(str).tolist()
N = len(tickers)

if mean_pnl.shape != (N, N):
    raise ValueError(f"mean_pnl shape {mean_pnl.shape} != ({N},{N})")
if sr_ann.shape != (N, N):
    raise ValueError(f"sr_ann shape {sr_ann.shape} != ({N},{N})")

mean_df = pd.DataFrame(mean_pnl, index=tickers, columns=tickers)
sr_df   = pd.DataFrame(sr_ann,   index=tickers, columns=tickers)

sr_plot = sr_df.copy()
if CLIP_SR_FOR_DISPLAY:
    sr_plot = sr_plot.clip(lower=SR_CLIP_LO, upper=SR_CLIP_HI)

# -----------------------
# Axis label handling
# -----------------------
def axis_kwargs(title: str):
    if not SHOW_AXIS_TICK_LABELS:
        return dict(title=title, showticklabels=False)
    # show sparse labels
    tickvals = list(range(0, N, SHOW_EVERY_K_LABELS))
    ticktext = [tickers[i] for i in tickvals]
    return dict(title=title, tickmode="array", tickvals=tickvals, ticktext=ticktext)

# -----------------------
# Plot 1: Mean PnL heatmap
# -----------------------
fig_mean = px.imshow(
    mean_df,
    origin="lower",
    aspect="auto",
    title="Mean PnL Matrix (row=follower i, col=leader j)",
    labels=dict(x="Leader (j)", y="Follower (i)", color="Mean PnL"),
)

fig_mean.update_layout(
    width=950,
    height=850,
    xaxis=axis_kwargs("Leader (j)"),
    yaxis=axis_kwargs("Follower (i)"),
)

# -----------------------
# Plot 2: Annualized Sharpe heatmap
# -----------------------
sr_title = "Annualized Sharpe Matrix (row=follower i, col=leader j)"
if CLIP_SR_FOR_DISPLAY:
    sr_title += f" (clipped to [{SR_CLIP_LO}, {SR_CLIP_HI}] for display)"

fig_sr = px.imshow(
    sr_plot,
    origin="lower",
    aspect="auto",
    title=sr_title,
    labels=dict(x="Leader (j)", y="Follower (i)", color="Annualized Sharpe"),
)

fig_sr.update_layout(
    width=950,
    height=850,
    xaxis=axis_kwargs("Leader (j)"),
    yaxis=axis_kwargs("Follower (i)"),
)

# -----------------------
# Save HTML
# -----------------------
mean_html = OUT_DIR / "mean_pnl_heatmap.html"
sr_html   = OUT_DIR / "sr_annualized_heatmap.html"

fig_mean.write_html(mean_html, include_plotlyjs="cdn")
fig_sr.write_html(sr_html, include_plotlyjs="cdn")

print("Saved:")
print(" -", mean_html)
print(" -", sr_html)