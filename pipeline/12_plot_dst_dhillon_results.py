#!/usr/bin/env python3
"""12_plot_dst_dhillon_results.py

Figures for scripts 04 (DST) and 05 (Dhillon) for the write-up.

DST figures  (4):
  dst_01_sharpe_bars       — SN Sharpe bar chart, rn vs raw, all half-lives
  dst_02_cumulative_sn     — cumulative SN P&L, top 5 rn models
  dst_03_rolling_ic        — 60-day rolling IC, top 4 rn models
  dst_04_hl_sensitivity    — SN Sharpe vs half-life, rn vs raw

Dhillon figures  (4):
  dh_01_sharpe_bars        — grouped SN Sharpe: ws vs no-ws, rn models
  dh_02_cumulative_sn      — cumulative SN P&L, top 5 models
  dh_03_rolling_ic         — 60-day rolling IC, top 4 models
  dh_04_hl_sensitivity     — SN Sharpe vs half-life, ws vs no-ws

Combined figure  (1):
  combined_01_comparison   — SN Sharpe: best DST / Dhillon / Ridge side-by-side

Outputs → results/figures_dst_dhillon/

Run from repo root:
    python pipeline/12_plot_dst_dhillon_results.py
"""
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# config
# ============================================================
INPUT_PATH = Path("data/processed/feature_table_with_residuals_10level.csv")
DST_DIR    = Path("results/rolling_adjacency_dst")
DH_DIR     = Path("results/dhillon_adapted")
RIDGE_DIR  = Path("results/rolling_adjacency_ridge")
OUT        = Path("results/figures_dst_dhillon")
OUT.mkdir(parents=True, exist_ok=True)

Q          = 0.10
ROLL_WIN   = 60
DPI        = 160
MIN_SN     = 4
HLS        = [15, 20, 25, 30, 35, 40, 45, 60, 90]

BLUE  = "#2980b9"
TEAL  = "#16a085"
AMBER = "#e67e22"
RED   = "#c0392b"
GREY  = "#7f8c8d"
DARK  = "#2c3e50"

PALETTE = ["#1a5276", "#2980b9", "#27ae60", "#e67e22", "#8e44ad",
           "#c0392b", "#16a085", "#f39c12", "#2c3e50"]


# ============================================================
# helpers
# ============================================================
def annualize_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return np.nan
    return float(s.mean() / s.std() * np.sqrt(252))


def cumulative(s: pd.Series) -> pd.Series:
    return s.dropna().cumsum()


def rolling_ic(ic: pd.Series, w: int = ROLL_WIN) -> pd.Series:
    return ic.dropna().rolling(w, min_periods=w // 2).mean()


def save(name: str):
    p = OUT / name
    plt.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}")


# ============================================================
# load returns + sector for DST spread recomputation
# ============================================================
print("Loading returns for DST spread computation...", flush=True)
ret_df = pd.read_csv(INPUT_PATH, usecols=["date", "ticker", "residual_ret", "sector"])
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df = ret_df[ret_df["ticker"] != "GOOGL"]
ret_wide = ret_df.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
sec_map  = ret_df[["ticker","sector"]].drop_duplicates().set_index("ticker")["sector"]


def spread_sn_series(pred_df: pd.DataFrame) -> pd.Series:
    """Row-wise sector-neutral spread from a predicted-returns DataFrame."""
    common_c = pred_df.columns.intersection(ret_wide.columns)
    pred_df  = pred_df[common_c]
    ret_sub  = ret_wide.reindex(index=pred_df.index, columns=common_c)
    sec_sub  = sec_map.reindex(common_c)

    rows = {}
    for date, pred_row in pred_df.iterrows():
        if date not in ret_sub.index:
            continue
        ret_row = ret_sub.loc[date]
        tmp = pd.DataFrame({"s": pred_row, "r": ret_row, "sec": sec_sub}).dropna()
        spreads = []
        for _, g in tmp.groupby("sec"):
            n = len(g)
            if n < MIN_SN:
                continue
            k = max(1, int(n * Q))
            if 2 * k > n:
                continue
            g = g.sort_values("s")
            spreads.append(g.iloc[-k:]["r"].mean() - g.iloc[:k]["r"].mean())
        rows[date] = float(np.mean(spreads)) if spreads else np.nan
    return pd.Series(rows).dropna()


def ic_series_from_pred(pred_df: pd.DataFrame) -> pd.Series:
    common_c = pred_df.columns.intersection(ret_wide.columns)
    pred_df  = pred_df[common_c]
    ret_sub  = ret_wide.reindex(index=pred_df.index, columns=common_c)
    rows = {}
    for date, pred_row in pred_df.iterrows():
        if date not in ret_sub.index:
            continue
        ic = pred_row.corr(ret_sub.loc[date], method="spearman")
        rows[date] = ic
    return pd.Series(rows).dropna()


# ============================================================
# load DST data
# ============================================================
print("Loading DST results...", flush=True)
dst_summary = pd.read_csv(DST_DIR / "rolling_summary_all_models.csv")

# compute SN spread + IC series for rn models only (to keep runtime short)
dst_sn: dict[str, pd.Series] = {}
dst_ic: dict[str, pd.Series] = {}
for hl in HLS:
    model = f"dst_hl{hl}_rn"
    f = DST_DIR / f"predicted_returns_{model}_rolling.csv"
    if not f.exists():
        continue
    pred = pd.read_csv(f, index_col=0, parse_dates=True)
    dst_sn[model] = spread_sn_series(pred)
    dst_ic[model] = ic_series_from_pred(pred)
    print(f"  {model}: {len(dst_sn[model])} days", flush=True)


# ============================================================
# load Dhillon data  (daily backtest CSVs already have spread_sn)
# ============================================================
print("Loading Dhillon results...", flush=True)
dh_summary = pd.read_csv(DH_DIR / "summary_all.csv")

dh_sn: dict[str, pd.Series] = {}
dh_ic: dict[str, pd.Series] = {}
for f in sorted(DH_DIR.glob("backtest_dh_hl*_rn*.csv")):
    model = f.stem.replace("backtest_", "")
    bt = pd.read_csv(f, index_col=0, parse_dates=True)
    if "spread_sn" in bt.columns:
        dh_sn[model] = bt["spread_sn"].dropna()
    if "ic" in bt.columns:
        dh_ic[model] = bt["ic"].dropna()

print(f"  Loaded {len(dh_sn)} Dhillon models", flush=True)

# short label helpers
def dst_label(m: str) -> str:
    # "dst_hl30_rn" -> "hl30 rn"
    return m.replace("dst_", "").replace("_rn", " rn").replace("_", " ")

def dh_label(m: str) -> str:
    # "dh_hl60_rn_k2_ws" -> "hl60 ws"  /  "dh_hl60_rn_k2" -> "hl60"
    s = m.replace("dh_", "").replace("_rn_k2_ws", " ws").replace("_rn_k2", "").replace("_k2", "")
    return s


# ============================================================
# ── DST figures ──────────────────────────────────────────────
# ============================================================

# DST 01 — SN Sharpe bars, rn vs raw
print("\nGenerating DST figures...", flush=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(HLS))
w = 0.38

rn_vals  = [dst_summary.loc[dst_summary["model"] == f"dst_hl{h}_rn",
             "annualized_spread_sharpe_sn"].values[0]
            if f"dst_hl{h}_rn" in dst_summary["model"].values else np.nan
            for h in HLS]
raw_vals = [dst_summary.loc[dst_summary["model"] == f"dst_hl{h}",
             "annualized_spread_sharpe_sn"].values[0]
            if f"dst_hl{h}" in dst_summary["model"].values else np.nan
            for h in HLS]

ax.bar(x - w/2, rn_vals,  w, label="Rank-norm OFI", color=BLUE,  alpha=0.88)
ax.bar(x + w/2, raw_vals, w, label="Raw OFI",        color=GREY, alpha=0.72)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"hl{h}" for h in HLS], fontsize=9)
ax.set_ylabel("Ann. SN Sharpe", fontsize=10)
ax.set_title("DST: SN Sharpe by Half-Life", fontsize=11)
ax.legend(fontsize=9, framealpha=0.7)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("dst_01_sharpe_bars.png")


# DST 02 — cumulative SN P&L, top 5 rn models
top5_dst = (dst_summary[dst_summary["model"].str.endswith("_rn")]
            .sort_values("annualized_spread_sharpe_sn", ascending=False)
            .head(5)["model"].tolist())

fig, ax = plt.subplots(figsize=(8, 4))
for i, m in enumerate(top5_dst):
    if m not in dst_sn:
        continue
    s = cumulative(dst_sn[m])
    lbl = dst_label(m)
    ax.plot(s.index, s.values * 100, color=PALETTE[i], lw=1.6, label=lbl)

ax.axhline(0, color="black", lw=0.7, ls="--")
ax.set_ylabel("Cumulative SN Return (%)", fontsize=10)
ax.set_title("DST: Cumulative Sector-Neutral P&L — Top 5 Models", fontsize=11)
ax.legend(fontsize=9, framealpha=0.7, ncol=2)
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(alpha=0.25)
plt.tight_layout()
save("dst_02_cumulative_sn.png")


# DST 03 — rolling 60-day IC, top 4 rn models
top4_dst = top5_dst[:4]

fig, ax = plt.subplots(figsize=(8, 3.5))
for i, m in enumerate(top4_dst):
    if m not in dst_ic:
        continue
    s = rolling_ic(dst_ic[m])
    ax.plot(s.index, s.values, color=PALETTE[i], lw=1.4, label=dst_label(m), alpha=0.9)

ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel(f"Rolling {ROLL_WIN}-day IC", fontsize=10)
ax.set_title("DST: Rolling Information Coefficient — Top 4 Models", fontsize=11)
ax.legend(fontsize=9, framealpha=0.7, ncol=2)
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(alpha=0.25)
plt.tight_layout()
save("dst_03_rolling_ic.png")


# DST 04 — SN Sharpe vs half-life line chart
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(HLS, rn_vals,  "o-", color=BLUE, lw=2,   ms=6, label="Rank-norm OFI")
ax.plot(HLS, raw_vals, "s--", color=GREY, lw=1.5, ms=5, label="Raw OFI")
ax.axhline(0, color="black", lw=0.7, ls=":")
ax.set_xlabel("OFI Half-Life (minutes)", fontsize=10)
ax.set_ylabel("Ann. SN Sharpe", fontsize=10)
ax.set_title("DST: SN Sharpe vs OFI Half-Life", fontsize=11)
ax.set_xticks(HLS)
ax.legend(fontsize=9, framealpha=0.7)
ax.grid(alpha=0.25)
plt.tight_layout()
save("dst_04_hl_sensitivity.png")


# ============================================================
# ── Dhillon figures ───────────────────────────────────────────
# ============================================================
print("\nGenerating Dhillon figures...", flush=True)

# collect SN Sharpe by (hl, ws) for rn models
dh_rn_nows = {}   # hl -> SN SR (no within-sector)
dh_rn_ws   = {}   # hl -> SN SR (within-sector)
for hl in HLS:
    m_nows = f"dh_hl{hl}_rn_k2"
    m_ws   = f"dh_hl{hl}_rn_k2_ws"
    row_nows = dh_summary[dh_summary["name"] == m_nows]
    row_ws   = dh_summary[dh_summary["name"] == m_ws]
    dh_rn_nows[hl] = float(row_nows["annualized_sn_spread_sharpe"].values[0]) \
                     if len(row_nows) else np.nan
    dh_rn_ws[hl]   = float(row_ws["annualized_sn_spread_sharpe"].values[0]) \
                     if len(row_ws) else np.nan

nows_vals = [dh_rn_nows.get(h, np.nan) for h in HLS]
ws_vals   = [dh_rn_ws.get(h, np.nan)   for h in HLS]


# DH 01 — grouped SN Sharpe bars: ws vs no-ws
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(HLS))
w = 0.38

ax.bar(x - w/2, ws_vals,   w, label="Within-sector", color=TEAL,  alpha=0.88)
ax.bar(x + w/2, nows_vals, w, label="Full graph",     color=AMBER, alpha=0.80)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"hl{h}" for h in HLS], fontsize=9)
ax.set_ylabel("Ann. SN Sharpe", fontsize=10)
ax.set_title("Dhillon: SN Sharpe — Within-Sector vs Full Graph (Rank-Norm OFI)", fontsize=10.5)
ax.legend(fontsize=9, framealpha=0.7)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("dh_01_sharpe_bars.png")


# DH 02 — cumulative SN P&L, top 5 models
top5_dh = (dh_summary.sort_values("annualized_sn_spread_sharpe", ascending=False)
           .head(6)["name"].tolist())
# filter to rn only
top5_dh = [m for m in top5_dh if "_rn" in m][:5]

fig, ax = plt.subplots(figsize=(8, 4))
for i, m in enumerate(top5_dh):
    if m not in dh_sn:
        continue
    s = cumulative(dh_sn[m])
    ax.plot(s.index, s.values * 100, color=PALETTE[i], lw=1.6, label=dh_label(m))

ax.axhline(0, color="black", lw=0.7, ls="--")
ax.set_ylabel("Cumulative SN Return (%)", fontsize=10)
ax.set_title("Dhillon: Cumulative Sector-Neutral P&L — Top 5 Models", fontsize=11)
ax.legend(fontsize=9, framealpha=0.7, ncol=2)
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(alpha=0.25)
plt.tight_layout()
save("dh_02_cumulative_sn.png")


# DH 03 — rolling 60-day IC, top 4 models
top4_dh = top5_dh[:4]

fig, ax = plt.subplots(figsize=(8, 3.5))
for i, m in enumerate(top4_dh):
    if m not in dh_ic:
        continue
    s = rolling_ic(dh_ic[m])
    ax.plot(s.index, s.values, color=PALETTE[i], lw=1.4, label=dh_label(m), alpha=0.9)

ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel(f"Rolling {ROLL_WIN}-day IC", fontsize=10)
ax.set_title("Dhillon: Rolling Information Coefficient — Top 4 Models", fontsize=11)
ax.legend(fontsize=9, framealpha=0.7, ncol=2)
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(alpha=0.25)
plt.tight_layout()
save("dh_03_rolling_ic.png")


# DH 04 — SN Sharpe vs half-life: ws vs no-ws
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(HLS, ws_vals,   "o-",  color=TEAL,  lw=2,   ms=6, label="Within-sector")
ax.plot(HLS, nows_vals, "s--", color=AMBER, lw=1.5, ms=5, label="Full graph")
ax.axhline(0, color="black", lw=0.7, ls=":")
ax.set_xlabel("OFI Half-Life (minutes)", fontsize=10)
ax.set_ylabel("Ann. SN Sharpe", fontsize=10)
ax.set_title("Dhillon: SN Sharpe vs Half-Life (Rank-Norm OFI)", fontsize=11)
ax.set_xticks(HLS)
ax.legend(fontsize=9, framealpha=0.7)
ax.grid(alpha=0.25)
plt.tight_layout()
save("dh_04_hl_sensitivity.png")


# ============================================================
# ── Combined comparison figure ────────────────────────────────
# ============================================================
print("\nGenerating combined figure...", flush=True)

ridge_summary = pd.read_csv(RIDGE_DIR / "rolling_summary_all_models.csv")

# Best SN SR per method
best_dst   = float(dst_summary["annualized_spread_sharpe_sn"].max())
best_dh    = float(dh_summary["annualized_sn_spread_sharpe"].max())
best_ridge = float(ridge_summary["annualized_spread_sharpe_sector_neutral"].max())

# Per-half-life SN SR for each method (rn models only)
def ridge_sn(hl):
    m = f"ridge_sb_gd_hl{hl}_rn" if hl != 45 else "ridge_sb_gd"
    row = ridge_summary[ridge_summary["model"] == m]
    if len(row):
        return float(row["annualized_spread_sharpe_sector_neutral"].values[0])
    # fall back to non-rn
    row = ridge_summary[ridge_summary["model"] == f"ridge_sb_gd_hl{hl}"]
    return float(row["annualized_spread_sharpe_sector_neutral"].values[0]) if len(row) else np.nan

ridge_vals = [ridge_sn(h) for h in HLS]
dst_hl_vals = rn_vals   # already computed above
dh_hl_vals  = ws_vals   # ws rn is the best Dhillon variant

fig, ax = plt.subplots(figsize=(8.5, 4.5))
x = np.arange(len(HLS))
w = 0.26

ax.bar(x - w,   ridge_vals,   w, label="Ridge + GD",        color="#1a5276", alpha=0.90)
ax.bar(x,       dh_hl_vals,   w, label="Dhillon (WS)",       color=TEAL,     alpha=0.88)
ax.bar(x + w,   dst_hl_vals,  w, label="DST",                color=AMBER,    alpha=0.82)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"hl{h}" for h in HLS], fontsize=9)
ax.set_ylabel("Ann. SN Sharpe (Rank-Norm OFI)", fontsize=10)
ax.set_title("Method Comparison: Ridge vs Dhillon vs DST by Half-Life", fontsize=11)
ax.legend(fontsize=9, framealpha=0.8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("combined_01_method_comparison.png")

print(f"\nAll figures saved to {OUT.resolve()}")
print(f"Total: {len(list(OUT.glob('*.png')))} figures")
