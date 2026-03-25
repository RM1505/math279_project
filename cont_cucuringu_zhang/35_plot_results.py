#!/usr/bin/env python3
"""35_plot_results.py

Comprehensive figures for the Math 279 write-up.
Compiles results from:
  27  —  next-day OPCL rolling ridge
  31  —  overnight (close→open) rolling ridge
  33  —  next-day + transaction costs
  34  —  overnight + transaction costs

Outputs all figures to results/figures/.
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
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

# ============================================================
# config
# ============================================================
INPUT_PATH      = Path("data/processed/feature_table_with_residuals_10level.csv")
OVERNIGHT_CACHE = Path("data/processed/overnight_returns_residualized.csv")
DIR_27 = Path("results/rolling_adjacency_ridge")
DIR_31 = Path("results/rolling_adjacency_ridge_overnight")
DIR_33 = Path("results/rolling_adjacency_ridge_tc")
DIR_34 = Path("results/rolling_adjacency_ridge_overnight_tc")
OUT    = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

Q            = 0.10
DPI          = 200          # 200 dpi is crisp; use 300 for print-ready
TC_SCENARIOS = [0, 5, 10, 20, 30]   # bps round-trip (0 = gross)
ROLL_WIN     = 60           # days for rolling IC / Sharpe

# color palettes
BLUE  = ["#1a5276", "#2980b9", "#5dade2", "#aed6f1", "#d6eaf8"]
GREEN = ["#1e8449", "#27ae60", "#58d68d", "#a9dfbf", "#d5f5e3"]
TC_COL = {0: "#2c3e50", 5: "#27ae60", 10: "#e67e22", 20: "#e74c3c", 30: "#922b21"}
AMBER = "#f39c12"

# readable short labels
ND_LABEL = {
    "ridge_sb":             "No GD (hl45)",
    "ridge_sb_gd":          "GD hl45",
    "ridge_sb_gd_hl15":     "GD hl15",
    "ridge_sb_gd_hl20":     "GD hl20",
    "ridge_sb_gd_hl25":     "GD hl25",
    "ridge_sb_gd_hl30":     "GD hl30",
    "ridge_sb_gd_hl35":     "GD hl35",
    "ridge_sb_gd_hl40":     "GD hl40",
    "ridge_sb_gd_hl60":     "GD hl60",
    "ridge_sb_gd_hl90":     "GD hl90",
    "ridge_sb_gd_hl30_rn":  "GD hl30 RN",
    "ridge_sb_gd_hl90_rn":  "GD hl90 RN",
    "ridge_sb_gd_ms3090":   "GD ms30+90",
    "ridge_sb_gd_lag1":     "GD lag1",
}
ON_LABEL = {k.replace("ridge_sb", "ridge_on"): v for k, v in ND_LABEL.items()}

# key models to feature in curve / IC plots
FOCUS_ND = ["ridge_sb_gd_hl30_rn", "ridge_sb_gd", "ridge_sb_gd_hl40",
            "ridge_sb_gd_hl60", "ridge_sb_gd_hl15"]
FOCUS_ON = ["ridge_on_sb_gd_hl60", "ridge_on_sb_gd_hl90_rn", "ridge_on_sb_gd_hl90",
            "ridge_on_sb_gd_hl30_rn", "ridge_on_sb_gd"]

# matplotlib style
plt.rcParams.update({
    "figure.dpi": DPI,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.8,
})


# ============================================================
# helpers
# ============================================================
def _pct(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"

def _bps(x: float, decimals: int = 1) -> str:
    return f"{x*10000:.{decimals}f}bps"

def load_pred(result_dir: Path, model: str) -> pd.DataFrame:
    p = result_dir / f"predicted_returns_{model}_rolling.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else pd.DataFrame()

def load_turnover(result_dir: Path, model: str) -> pd.DataFrame:
    p = result_dir / f"turnover_{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    return df

def gross_spread_series(pred_df: pd.DataFrame, ret_df: pd.DataFrame, q: float = Q) -> pd.Series:
    cd = pred_df.index.intersection(ret_df.index)
    ct = pred_df.columns.intersection(ret_df.columns)
    if not len(cd) or not len(ct):
        return pd.Series(dtype=float)
    P = pred_df.loc[cd, ct].values
    R = ret_df.loc[cd, ct].values
    out = np.full(len(cd), np.nan)
    for i in range(len(cd)):
        p, r = P[i], R[i]
        mask = ~np.isnan(p) & ~np.isnan(r)
        pm, rm = p[mask], r[mask]
        n = len(pm)
        k = max(1, int(n * q))
        if 2 * k > n:
            continue
        idx = np.argsort(pm)
        out[i] = rm[idx[-k:]].mean() - rm[idx[:k]].mean()
    return pd.Series(out, index=cd)

def ic_series(pred_df: pd.DataFrame, ret_df: pd.DataFrame) -> pd.Series:
    cd = pred_df.index.intersection(ret_df.index)
    ct = pred_df.columns.intersection(ret_df.columns)
    if not len(cd):
        return pd.Series(dtype=float)
    P = pred_df.loc[cd, ct].values
    R = ret_df.loc[cd, ct].values
    out = np.full(len(cd), np.nan)
    for i in range(len(cd)):
        p, r = P[i], R[i]
        mask = ~np.isnan(p) & ~np.isnan(r)
        if mask.sum() < 5:
            continue
        out[i], _ = spearmanr(p[mask], r[mask])
    return pd.Series(out, index=cd)

def ann_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.std(ddof=1) == 0:
        return np.nan
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252))

def style_table(ax, col_labels, rows, col_widths=None,
                hdr="#1a5276", alt=("#eaf2ff", "white"),
                green_cols=None, fontsize=9):
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    nc = len(col_labels)
    nr = len(rows)
    for j in range(nc):
        c = tbl[0, j]
        c.set_facecolor(hdr)
        c.set_text_props(color="white", fontweight="bold")
    for i in range(1, nr + 1):
        bg = alt[(i - 1) % 2]
        for j in range(nc):
            tbl[i, j].set_facecolor(bg)
            tbl[i, j].set_edgecolor("#cccccc")
    if green_cols:
        for j in green_cols:
            for i in range(1, nr + 1):
                try:
                    v = float(str(rows[i-1][j]).replace("+","").replace("bps",""))
                    if v > 0.5:
                        tbl[i, j].set_facecolor("#d5f5e3")
                    elif v > 0:
                        tbl[i, j].set_facecolor("#eafaf1")
                    elif v > -0.5:
                        tbl[i, j].set_facecolor("#fef9e7")
                    else:
                        tbl[i, j].set_facecolor("#fadbd8")
                except Exception:
                    pass
    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(nr + 1):
                tbl[i, j].set_width(w)
    tbl.scale(1, 1.8)
    return tbl

def save(name: str):
    p = OUT / name
    plt.savefig(p, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  Saved {name}", flush=True)


# ============================================================
# load data
# ============================================================
print("Loading returns...", flush=True)
df_feat = pd.read_csv(INPUT_PATH, usecols=["date", "ticker", "sector", "residual_ret"])
df_feat["date"] = pd.to_datetime(df_feat["date"])
df_feat = df_feat[df_feat["ticker"] != "GOOGL"].copy()
ticker_sector = df_feat[["ticker","sector"]].drop_duplicates().set_index("ticker")["sector"]
ret_nd = df_feat.pivot(index="date", columns="ticker", values="residual_ret")
ret_nd.index = pd.to_datetime(ret_nd.index)

ret_on = pd.read_csv(OVERNIGHT_CACHE, index_col=0, parse_dates=True)

print("Loading summaries...", flush=True)
sum27 = pd.read_csv(DIR_27 / "rolling_summary_all_models.csv")
sum31 = pd.read_csv(DIR_31 / "rolling_summary_all_models.csv")
sum33 = pd.read_csv(DIR_33 / "rolling_summary_all_models.csv")
sum34 = pd.read_csv(DIR_34 / "rolling_summary_all_models.csv")

for s, lmap in [(sum27, ND_LABEL), (sum33, ND_LABEL),
                (sum31, ON_LABEL), (sum34, ON_LABEL)]:
    s["label"] = s["model"].map(lmap).fillna(s["model"])

print("Computing spread / IC series...", flush=True)
spread_nd, ic_nd = {}, {}
for m in FOCUS_ND:
    pred = load_pred(DIR_27, m)
    if pred.empty:
        pred = load_pred(DIR_33, m)
    if pred.empty:
        continue
    spread_nd[m] = gross_spread_series(pred, ret_nd)
    ic_nd[m]     = ic_series(pred, ret_nd)
    print(f"  ND {m}: {len(spread_nd[m])} days", flush=True)

spread_on, ic_on = {}, {}
for m in FOCUS_ON:
    pred = load_pred(DIR_31, m)
    if pred.empty:
        pred = load_pred(DIR_34, m)
    if pred.empty:
        continue
    spread_on[m] = gross_spread_series(pred, ret_on)
    ic_on[m]     = ic_series(pred, ret_on)
    print(f"  ON {m}: {len(spread_on[m])} days", flush=True)


# ============================================================
# Figure 01 — Table: Next-day results (all models, sorted by SN Sharpe)
# ============================================================
print("\nGenerating figures...", flush=True)

def _tbl_nd(df):
    rows = []
    for _, r in df.sort_values("annualized_spread_sharpe_sector_neutral", ascending=False).iterrows():
        rows.append([
            ND_LABEL.get(r["model"], r["model"]),
            f"{r['annualized_spread_sharpe']:.2f}",
            f"{r['annualized_spread_sharpe_sector_neutral']:.2f}",
            f"{r['mean_daily_ic']*100:.2f}%",
            str(int(r["half_life_minutes"])) if not pd.isna(r["half_life_minutes"]) else "—",
            "Yes" if r["use_gd"] else "No",
        ])
    return rows

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Next-Day Strategy — All Model Results  (MSA=4, GOOG only)", fontsize=13, fontweight="bold", y=1.01)
cols = ["Model", "Gross Sharpe", "SN Sharpe", "Mean IC", "HL (min)", "GD"]
rows = _tbl_nd(sum27)
style_table(ax, cols, rows, green_cols=[1, 2], fontsize=9)
save("01_table_nextday_all_models.png")

# ============================================================
# Figure 02 — Table: Overnight results (all models, sorted by gross Sharpe)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Overnight Strategy — All Model Results  (MSA=4, GOOG only)", fontsize=13, fontweight="bold", y=1.01)
cols = ["Model", "Gross Sharpe", "SN Sharpe", "Mean IC", "HL (min)", "GD"]
rows = []
for _, r in sum31.sort_values("annualized_spread_sharpe", ascending=False).iterrows():
    rows.append([
        ON_LABEL.get(r["model"], r["model"]),
        f"{r['annualized_spread_sharpe']:.2f}",
        f"{r['annualized_spread_sharpe_sector_neutral']:.2f}",
        f"{r['mean_daily_ic']*100:.2f}%",
        str(int(r["half_life_minutes"])) if not pd.isna(r["half_life_minutes"]) else "—",
        "Yes" if r["use_gd"] else "No",
    ])
style_table(ax, cols, rows, hdr="#1e6b3a", green_cols=[1, 2], fontsize=9)
save("02_table_overnight_all_models.png")

# ============================================================
# Figure 03 — Table: TC results next-day (33)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle("Next-Day + Transaction Costs — All Models  (Script 33)", fontsize=13, fontweight="bold", y=1.01)
tc_cols = ["Model", "Gross Sharpe", "SN Sharpe", "τ daily", "Ann τ", "BE (gross)", "BE (SN)",
           "Net SN @5", "Net SN @10", "Net SN @20", "Net SN @30"]
rows = []
for _, r in sum33.sort_values("annualized_spread_sharpe_sn", ascending=False).iterrows():
    rows.append([
        ND_LABEL.get(r["model"], r["model"]),
        f"{r['annualized_spread_sharpe']:.2f}",
        f"{r['annualized_spread_sharpe_sn']:.2f}",
        f"{r['mean_daily_turnover_sn']:.3f}",
        f"{r['annualized_turnover_sn']:.0f}x",
        f"{r['breakeven_rt_bps_gross']:.1f}",
        f"{r['breakeven_rt_bps_sn']:.1f}",
        f"{r['net_sharpe_sn_5bps']:.2f}",
        f"{r['net_sharpe_sn_10bps']:.2f}",
        f"{r['net_sharpe_sn_20bps']:.2f}",
        f"{r['net_sharpe_sn_30bps']:.2f}",
    ])
style_table(ax, tc_cols, rows, green_cols=[7, 8, 9, 10], fontsize=8)
save("03_table_tc_nextday.png")

# ============================================================
# Figure 04 — Table: TC results overnight (34)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle("Overnight + Transaction Costs — All Models  (Script 34)", fontsize=13, fontweight="bold", y=1.01)
rows = []
for _, r in sum34.sort_values("annualized_spread_sharpe_sn", ascending=False).iterrows():
    rows.append([
        ON_LABEL.get(r["model"], r["model"]),
        f"{r['annualized_spread_sharpe']:.2f}",
        f"{r['annualized_spread_sharpe_sn']:.2f}",
        f"{r['mean_daily_turnover_sn']:.3f}",
        f"{r['annualized_turnover_sn']:.0f}x",
        f"{r['breakeven_rt_bps_gross']:.1f}",
        f"{r['breakeven_rt_bps_sn']:.1f}",
        f"{r['net_sharpe_sn_5bps']:.2f}",
        f"{r['net_sharpe_sn_10bps']:.2f}",
        f"{r['net_sharpe_sn_20bps']:.2f}",
        f"{r['net_sharpe_sn_30bps']:.2f}",
    ])
style_table(ax, tc_cols, rows, hdr="#1e6b3a", green_cols=[7, 8, 9, 10], fontsize=8)
save("04_table_tc_overnight.png")

# ============================================================
# Figure 05 — Table: Cross-strategy comparison (best 5 from each)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 5.5))
fig.suptitle("Cross-Strategy Comparison — Best Models  (MSA=4, GOOG only)", fontsize=13, fontweight="bold", y=1.01)
cross_cols = ["Strategy", "Model", "HL", "Gross Sharpe", "SN Sharpe", "IC",
              "τ SN (daily)", "Ann τ SN", "BE SN (bps)", "Net SN @5bps", "Net SN @10bps"]
rows = []
# best 5 next-day by SN Sharpe
for _, r in sum33.sort_values("annualized_spread_sharpe_sn", ascending=False).head(5).iterrows():
    rows.append([
        "Next-Day",
        ND_LABEL.get(r["model"], r["model"]),
        str(int(r["half_life_minutes"])) if not pd.isna(r["half_life_minutes"]) else "—",
        f"{r['annualized_spread_sharpe']:.2f}",
        f"{r['annualized_spread_sharpe_sn']:.2f}",
        f"{r['mean_daily_ic']*100:.2f}%",
        f"{r['mean_daily_turnover_sn']:.3f}",
        f"{r['annualized_turnover_sn']:.0f}x",
        f"{r['breakeven_rt_bps_sn']:.1f}",
        f"{r['net_sharpe_sn_5bps']:.2f}",
        f"{r['net_sharpe_sn_10bps']:.2f}",
    ])
# best 5 overnight by SN Sharpe
for _, r in sum34.sort_values("annualized_spread_sharpe_sn", ascending=False).head(5).iterrows():
    rows.append([
        "Overnight",
        ON_LABEL.get(r["model"], r["model"]),
        str(int(r["half_life_minutes"])) if not pd.isna(r["half_life_minutes"]) else "—",
        f"{r['annualized_spread_sharpe']:.2f}",
        f"{r['annualized_spread_sharpe_sn']:.2f}",
        f"{r['mean_daily_ic']*100:.2f}%",
        f"{r['mean_daily_turnover_sn']:.3f}",
        f"{r['annualized_turnover_sn']:.0f}x",
        f"{r['breakeven_rt_bps_sn']:.1f}",
        f"{r['net_sharpe_sn_5bps']:.2f}",
        f"{r['net_sharpe_sn_10bps']:.2f}",
    ])
style_table(ax, cross_cols, rows, green_cols=[3, 4, 9, 10], fontsize=8)
save("05_table_cross_strategy.png")

# ============================================================
# Figure 06 — Cumulative P&L: next-day top models
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle("Cumulative Gross Spread — Next-Day Strategy  (top models, Q=10%)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.set_title("Gross portfolio")
for i, m in enumerate(FOCUS_ND):
    if m not in spread_nd:
        continue
    s = spread_nd[m]
    cum = s.cumsum() * 10_000  # in bps
    ax.plot(cum.index, cum.values, color=BLUE[i], label=ND_LABEL.get(m, m), linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}bps"))
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative gross spread (bps)")
ax.legend(loc="upper left")

ax = axes[1]
ax.set_title("Rolling 252-day Sharpe")
for i, m in enumerate(FOCUS_ND):
    if m not in spread_nd:
        continue
    rs = spread_nd[m].rolling(252).apply(ann_sharpe, raw=False).dropna()
    ax.plot(rs.index, rs.values, color=BLUE[i], label=ND_LABEL.get(m, m), linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Annualised Sharpe (252-day rolling)")
ax.legend(loc="upper left")

plt.tight_layout()
save("06_cumulative_pnl_nextday.png")

# ============================================================
# Figure 07 — Cumulative P&L: overnight top models
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle("Cumulative Gross Spread — Overnight Strategy  (top models, Q=10%)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.set_title("Gross portfolio")
for i, m in enumerate(FOCUS_ON):
    if m not in spread_on:
        continue
    s = spread_on[m]
    cum = s.cumsum() * 10_000
    ax.plot(cum.index, cum.values, color=GREEN[i], label=ON_LABEL.get(m, m), linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}bps"))
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative gross spread (bps)")
ax.legend(loc="upper left")

ax = axes[1]
ax.set_title("Rolling 252-day Sharpe")
for i, m in enumerate(FOCUS_ON):
    if m not in spread_on:
        continue
    rs = spread_on[m].rolling(252).apply(ann_sharpe, raw=False).dropna()
    ax.plot(rs.index, rs.values, color=GREEN[i], label=ON_LABEL.get(m, m), linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Annualised Sharpe (252-day rolling)")
ax.legend(loc="upper left")

plt.tight_layout()
save("07_cumulative_pnl_overnight.png")

# ============================================================
# Figure 08 — Cross-strategy cumulative P&L on one chart
# ============================================================
best_nd = FOCUS_ND[0]   # hl30_rn
best_on = FOCUS_ON[0]   # hl60

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title("Cross-Strategy Cumulative Gross Spread  (best model per strategy)", fontsize=13)
if best_nd in spread_nd:
    cum = spread_nd[best_nd].cumsum() * 10_000
    ax.plot(cum.index, cum.values, color=BLUE[0], linewidth=2,
            label=f"Next-Day: {ND_LABEL.get(best_nd)}")
if best_on in spread_on:
    cum = spread_on[best_on].cumsum() * 10_000
    ax.plot(cum.index, cum.values, color=GREEN[0], linewidth=2,
            label=f"Overnight: {ON_LABEL.get(best_on)}")
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}bps"))
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative gross spread (bps)")
ax.legend()
plt.tight_layout()
save("08_cumulative_pnl_best_comparison.png")

# ============================================================
# Figure 09 — Rolling IC: next-day
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_title(f"Rolling {ROLL_WIN}-day Information Coefficient — Next-Day Strategy", fontsize=13)
for i, m in enumerate(FOCUS_ND[:4]):
    if m not in ic_nd:
        continue
    rIC = ic_nd[m].rolling(ROLL_WIN).mean()
    ax.plot(rIC.index, rIC.values, color=BLUE[i], label=ND_LABEL.get(m, m), linewidth=1.4)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel(f"Rolling {ROLL_WIN}-day mean IC (Spearman)")
ax.legend()
plt.tight_layout()
save("09_rolling_ic_nextday.png")

# ============================================================
# Figure 10 — Rolling IC: overnight
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_title(f"Rolling {ROLL_WIN}-day Information Coefficient — Overnight Strategy", fontsize=13)
for i, m in enumerate(FOCUS_ON[:4]):
    if m not in ic_on:
        continue
    rIC = ic_on[m].rolling(ROLL_WIN).mean()
    ax.plot(rIC.index, rIC.values, color=GREEN[i], label=ON_LABEL.get(m, m), linewidth=1.4)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel(f"Rolling {ROLL_WIN}-day mean IC (Spearman)")
ax.legend()
plt.tight_layout()
save("10_rolling_ic_overnight.png")

# ============================================================
# Figure 11 — Half-life sensitivity: Sharpe vs HL for both strategies
# ============================================================
HL_VALS = [15, 20, 25, 30, 35, 40, 45, 60, 90]

def hl_rows(df, gross_col, sn_col):
    out = {}
    for _, r in df.iterrows():
        hl = r.get("half_life_minutes")
        if pd.isna(hl) or hl not in HL_VALS:
            continue
        sig = str(r.get("sig_cols", ""))
        if "|" in sig or "rn" in sig:  # skip multi-sig and rank-norm in sweep
            continue
        out[int(hl)] = (r[gross_col], r[sn_col])
    hls = sorted(out.keys())
    return hls, [out[h][0] for h in hls], [out[h][1] for h in hls]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Half-Life Sensitivity: Annualised Sharpe vs OFI Decay Half-Life", fontsize=13, fontweight="bold")

ax = axes[0]
ax.set_title("Next-Day Strategy")
hls, gross, sn = hl_rows(sum27, "annualized_spread_sharpe", "annualized_spread_sharpe_sector_neutral")
ax.plot(hls, gross, "o--", color=BLUE[0], label="Gross", linewidth=1.8, markersize=7)
ax.plot(hls, sn,    "s-",  color=BLUE[2], label="Sector-Neutral", linewidth=1.8, markersize=7)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Half-life (minutes)")
ax.set_ylabel("Annualised Sharpe")
ax.set_xticks(hls)
ax.legend()

ax = axes[1]
ax.set_title("Overnight Strategy")
hls, gross, sn = hl_rows(sum31, "annualized_spread_sharpe", "annualized_spread_sharpe_sector_neutral")
ax.plot(hls, gross, "o--", color=GREEN[0], label="Gross", linewidth=1.8, markersize=7)
ax.plot(hls, sn,    "s-",  color=GREEN[2], label="Sector-Neutral", linewidth=1.8, markersize=7)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Half-life (minutes)")
ax.set_ylabel("Annualised Sharpe")
ax.set_xticks(hls)
ax.legend()

plt.tight_layout()
save("11_halflife_sensitivity.png")

# ============================================================
# Figure 12 — Sharpe bar chart: all models, both strategies side-by-side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle("Annualised Sharpe — All Models  (left: gross, right: sector-neutral)", fontsize=13, fontweight="bold")

for ax, df, lmap, col_g, col_sn, palette, title in [
    (axes[0], sum27, ND_LABEL, "annualized_spread_sharpe", "annualized_spread_sharpe_sector_neutral", BLUE, "Next-Day"),
    (axes[1], sum31, ON_LABEL, "annualized_spread_sharpe", "annualized_spread_sharpe_sector_neutral", GREEN, "Overnight"),
]:
    df2 = df.sort_values(col_sn, ascending=True).copy()
    labels = [lmap.get(m, m) for m in df2["model"]]
    y = np.arange(len(labels))
    ax.barh(y - 0.2, df2[col_g].values,  height=0.35, color=palette[0], label="Gross",          alpha=0.85)
    ax.barh(y + 0.2, df2[col_sn].values, height=0.35, color=palette[2], label="Sector-Neutral",  alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Annualised Sharpe")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
save("12_sharpe_bars_all_models.png")

# ============================================================
# Figure 13 — IC bar chart: mean daily IC all models
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mean Daily IC (Spearman) — All Models", fontsize=13, fontweight="bold")

for ax, df, lmap, palette, title in [
    (axes[0], sum27, ND_LABEL, BLUE,  "Next-Day"),
    (axes[1], sum31, ON_LABEL, GREEN, "Overnight"),
]:
    df2 = df.sort_values("mean_daily_ic", ascending=True).copy()
    labels = [lmap.get(m, m) for m in df2["model"]]
    y = np.arange(len(labels))
    colors = [palette[0] if v >= 0 else "#e74c3c" for v in df2["mean_daily_ic"].values]
    ax.barh(y, df2["mean_daily_ic"].values * 100, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Mean daily IC (%)")
    ax.set_title(title)

plt.tight_layout()
save("13_ic_bars_all_models.png")

# ============================================================
# Figure 14 — IC histogram: distribution of daily IC
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Distribution of Daily IC (Spearman)  —  best model per strategy", fontsize=13, fontweight="bold")

for ax, ic_dict, best, lmap, palette, title in [
    (axes[0], ic_nd, best_nd, ND_LABEL, BLUE,  "Next-Day"),
    (axes[1], ic_on, best_on, ON_LABEL, GREEN, "Overnight"),
]:
    if best not in ic_dict:
        continue
    vals = ic_dict[best].dropna().values
    ax.hist(vals, bins=40, color=palette[1], edgecolor="white", alpha=0.85, density=True)
    ax.axvline(0,           color="k",   linewidth=1,   linestyle="--")
    ax.axvline(vals.mean(), color=palette[0], linewidth=1.5, linestyle="-", label=f"Mean={vals.mean()*100:.2f}%")
    ax.set_xlabel("Daily IC")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}: {lmap.get(best, best)}")
    ax.legend()

plt.tight_layout()
save("14_ic_histogram.png")

# ============================================================
# Figure 15 — TC breakeven bars (scripts 33 & 34)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Breakeven Round-Trip Cost (bps)  —  Gross & Sector-Neutral", fontsize=13, fontweight="bold")

for ax, df, lmap, palette, title in [
    (axes[0], sum33, ND_LABEL, BLUE,  "Next-Day"),
    (axes[1], sum34, ON_LABEL, GREEN, "Overnight"),
]:
    df2 = df.sort_values("breakeven_rt_bps_sn", ascending=True).copy()
    labels = [lmap.get(m, m) for m in df2["model"]]
    y = np.arange(len(labels))
    ax.barh(y - 0.2, df2["breakeven_rt_bps_gross"].values, height=0.35,
            color=palette[0], label="Gross",          alpha=0.85)
    ax.barh(y + 0.2, df2["breakeven_rt_bps_sn"].values,   height=0.35,
            color=palette[2], label="Sector-Neutral",  alpha=0.85)
    for xv in [5, 10, 20]:
        ax.axvline(xv, color="gray", linewidth=0.6, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Breakeven round-trip cost (bps)")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
save("15_tc_breakeven_bars.png")

# ============================================================
# Figure 16 — Net SN Sharpe at TC scenarios: top models
# ============================================================
top_nd_models = sum33.sort_values("annualized_spread_sharpe_sn", ascending=False).head(6)["model"].tolist()
top_on_models = sum34.sort_values("annualized_spread_sharpe_sn", ascending=False).head(6)["model"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle("Net Sector-Neutral Sharpe at TC Scenarios  (top 6 models per strategy)", fontsize=13, fontweight="bold")

for ax, df, models, lmap, palette, title in [
    (axes[0], sum33, top_nd_models, ND_LABEL, BLUE,  "Next-Day"),
    (axes[1], sum34, top_on_models, ON_LABEL, GREEN, "Overnight"),
]:
    sharpe_cols = {
        0:  "annualized_spread_sharpe_sn",
        5:  "net_sharpe_sn_5bps",
        10: "net_sharpe_sn_10bps",
        20: "net_sharpe_sn_20bps",
        30: "net_sharpe_sn_30bps",
    }
    n = len(models)
    x = np.arange(n)
    width = 0.14
    offsets = np.linspace(-2, 2, len(TC_SCENARIOS)) * width
    for off, bps in zip(offsets, TC_SCENARIOS):
        col = sharpe_cols[bps]
        vals = [df[df["model"] == m][col].values[0] if len(df[df["model"] == m]) else np.nan
                for m in models]
        ax.bar(x + off, vals, width=width * 0.9, color=TC_COL[bps],
               label=f"{bps}bps RT" if bps > 0 else "Gross", alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lmap.get(m, m) for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Annualised net SN Sharpe")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
save("16_net_sharpe_tc_scenarios.png")

# ============================================================
# Figure 17 — Cumulative net P&L at TC levels (best model, next-day)
# ============================================================
best_tc_nd = "ridge_sb_gd_hl30_rn"
best_tc_on = "ridge_on_sb_gd_hl60"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cumulative Net P&L at Transaction Cost Scenarios  (sector-neutral weights)", fontsize=13, fontweight="bold")

for ax, model, tc_dir, spread_dict, ret_df, lmap, title in [
    (axes[0], best_tc_nd, DIR_33, spread_nd, ret_nd, ND_LABEL, "Next-Day"),
    (axes[1], best_tc_on, DIR_34, spread_on, ret_on, ON_LABEL, "Overnight"),
]:
    if model not in spread_dict:
        ax.text(0.5, 0.5, "data not available", ha="center", va="center", transform=ax.transAxes)
        continue
    gross = spread_dict[model]
    tau_df = load_turnover(tc_dir, model)
    if tau_df.empty:
        ax.text(0.5, 0.5, "turnover not available", ha="center", va="center", transform=ax.transAxes)
        continue
    tau = tau_df["turnover_sn"].reindex(gross.index)
    for bps, lbl in [(0, "Gross (0 bps)"), (5, "5 bps RT"), (10, "10 bps RT"), (20, "20 bps RT")]:
        net = gross - (bps / 10_000) * tau.fillna(0)
        cum = net.cumsum() * 10_000
        ax.plot(cum.index, cum.values, color=TC_COL[bps], linewidth=1.6, label=lbl)
    ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}bps"))
    ax.set_title(f"{title}: {lmap.get(model, model)}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net P&L (bps)")
    ax.legend()

plt.tight_layout()
save("17_cumulative_net_pnl_tc.png")

# ============================================================
# Figure 18 — Turnover distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Daily Sector-Neutral Turnover Distribution  (best models)", fontsize=13, fontweight="bold")

for ax, model, tc_dir, palette, title in [
    (axes[0], best_tc_nd, DIR_33, BLUE,  "Next-Day"),
    (axes[1], best_tc_on, DIR_34, GREEN, "Overnight"),
]:
    tau_df = load_turnover(tc_dir, model)
    if tau_df.empty:
        continue
    vals_g  = tau_df["turnover_gross"].dropna().values
    vals_sn = tau_df["turnover_sn"].dropna().values
    ax.hist(vals_g,  bins=35, color=palette[0], alpha=0.6, density=True, label=f"Gross  (mean={vals_g.mean():.3f})")
    ax.hist(vals_sn, bins=35, color=palette[2], alpha=0.6, density=True, label=f"SN     (mean={vals_sn.mean():.3f})")
    ax.axvline(vals_g.mean(),  color=palette[0], linewidth=1.5, linestyle="--")
    ax.axvline(vals_sn.mean(), color=palette[2], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Daily one-way turnover τ")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
save("18_turnover_distribution.png")

# ============================================================
# Figure 19 — Sharpe heatmap: half-life × strategy
# ============================================================
hl_sweep = [15, 20, 25, 30, 35, 40, 45, 60, 90]
strategies = ["ND Gross", "ND SN", "ON Gross", "ON SN"]
matrix = np.full((len(hl_sweep), 4), np.nan)

for i, hl in enumerate(hl_sweep):
    for df, col_g, col_sn, j_g, j_sn in [
        (sum27, "annualized_spread_sharpe", "annualized_spread_sharpe_sector_neutral", 0, 1),
        (sum31, "annualized_spread_sharpe", "annualized_spread_sharpe_sector_neutral", 2, 3),
    ]:
        mask = (df["half_life_minutes"] == hl) & (~df["sig_cols"].astype(str).str.contains(r"\|"))
        mask &= ~df["sig_cols"].astype(str).str.contains("_rn")
        sub = df[mask]
        if len(sub):
            matrix[i, j_g] = sub[col_g].values[0]
            matrix[i, j_sn] = sub[col_sn].values[0]

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Sharpe Heatmap: Half-Life × Strategy  (GD, no RN, no multi-sig)", fontsize=13, fontweight="bold")
vmax = np.nanmax(np.abs(matrix))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
plt.colorbar(im, ax=ax, label="Annualised Sharpe")
ax.set_xticks(range(4))
ax.set_xticklabels(strategies, fontsize=10)
ax.set_yticks(range(len(hl_sweep)))
ax.set_yticklabels([f"hl{h}" for h in hl_sweep])
ax.set_xlabel("Strategy")
ax.set_ylabel("Half-life (minutes)")
for i in range(len(hl_sweep)):
    for j in range(4):
        v = matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > vmax * 0.6 else "black")
plt.tight_layout()
save("19_sharpe_heatmap.png")

# ============================================================
# Figure 20 — Rolling 60-day IC side by side (next-day vs overnight, best model)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_title(f"Rolling {ROLL_WIN}-day IC — Next-Day vs Overnight  (best model each)", fontsize=13)
if best_nd in ic_nd:
    rIC = ic_nd[best_nd].rolling(ROLL_WIN).mean()
    ax.plot(rIC.index, rIC.values, color=BLUE[0], linewidth=1.6,
            label=f"Next-Day: {ND_LABEL.get(best_nd)}")
if best_on in ic_on:
    rIC = ic_on[best_on].rolling(ROLL_WIN).mean()
    ax.plot(rIC.index, rIC.values, color=GREEN[0], linewidth=1.6,
            label=f"Overnight: {ON_LABEL.get(best_on)}")
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel(f"Rolling {ROLL_WIN}-day IC (Spearman)")
ax.legend()
plt.tight_layout()
save("20_rolling_ic_comparison.png")

# ============================================================
# Figure 21 — Scatter: IC vs Sharpe across all models & both strategies
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title("Mean IC vs Annualised SN Sharpe  (all models, both strategies)", fontsize=13)
for df, lmap, col, marker, palette, label in [
    (sum27, ND_LABEL, "annualized_spread_sharpe_sector_neutral", "o", BLUE,  "Next-Day"),
    (sum31, ON_LABEL, "annualized_spread_sharpe_sector_neutral", "s", GREEN, "Overnight"),
]:
    ax.scatter(df["mean_daily_ic"] * 100, df[col], s=60, c=palette[0],
               marker=marker, alpha=0.8, label=label)
    for _, r in df.iterrows():
        ax.annotate(lmap.get(r["model"], r["model"]),
                    xy=(r["mean_daily_ic"] * 100, r[col]),
                    fontsize=6.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", alpha=0.8)
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.axvline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_xlabel("Mean daily IC (%)")
ax.set_ylabel("Annualised SN Sharpe")
ax.legend()
plt.tight_layout()
save("21_ic_vs_sharpe_scatter.png")

# ============================================================
# Figure 22 — Lambda history: chosen regularisation over time
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Chosen Ridge λ Over Time  (walk-forward refit blocks, best models)", fontsize=13, fontweight="bold")

for ax, result_dir, model, lmap, palette, title in [
    (axes[0], DIR_27, best_nd, ND_LABEL, BLUE,  "Next-Day"),
    (axes[1], DIR_31, best_on, ON_LABEL, GREEN, "Overnight"),
]:
    p = result_dir / f"rolling_lambda_history_{model}.csv"
    if not p.exists():
        continue
    lh = pd.read_csv(p, parse_dates=["pred_start"])
    ax.step(lh["pred_start"], lh["lambda"], where="post", color=palette[0], linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Prediction start date")
    ax.set_ylabel("λ (log scale)")
    ax.set_title(f"{title}: {lmap.get(model, model)}")

plt.tight_layout()
save("22_lambda_history.png")

# ============================================================
# done
# ============================================================
print(f"\nAll figures saved to {OUT.resolve()}")
print(f"Total figures: 22")
