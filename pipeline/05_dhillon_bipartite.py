#!/usr/bin/env python3
"""05_dhillon_bipartite.py

Dhillon bipartite spectral co-clustering adapted to match the data pipeline
and configuration of scripts 27 / 33 / 37.

Compared to script 29:
  - GOOGL filtered out (same as 27/33/37)
  - MIN_SECTOR_ASSETS=4 gate on within-sector masking
  - Half-life sweep [15,20,25,30,35,40,45,60,90] — identical to script 27
  - Both plain rank-norm (_rn, same as 27) and signed rank-norm (_srn = rank_pct-0.5)
  - Stripped to core configs only (no training-window sweep, no k=3/4 variants)
  - Results comparable to DST (script 37) and ridge (script 27)

Method
------
The pairwise edge-stat matrix  A[i,j] = mean(R_{t,i} * P_{t-1,j}) / std(...)
is treated as a bipartite graph.  Dhillon (2001) co-clustering:
  1. D_r^{-1/2} A D_c^{-1/2}  normalised by row/col degrees
  2. Randomised SVD; embed rows and columns into shared spectral space
  3. KMeans on the stacked embedding → simultaneous row/col cluster labels
  4. Select the source-cluster → target-cluster block maximising mean edge score

Daily prediction: score_i = Σ_{j in S} w_j * P_{t-1,j}  (weighted average of
source OFI), scaled by target loadings.  Evaluated identically to scripts 27/37.

Outputs → results/dhillon_adapted/

Run from repo root:
    python pipeline/05_dhillon_bipartite.py
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


# ============================================================
# config
# ============================================================
INPUT_PATH       = Path("data/processed/feature_table_with_residuals_10level.csv")
OUT_DIR          = Path("results/dhillon_adapted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS = 6
MAX_ASSETS       = 400
HALF_LIFE_SWEEP  = [15, 20, 25, 30, 35, 40, 45, 60, 90]
MIN_TICKER_OBS   = 252
Q                = 0.10
MIN_NAMES_SN     = 4
MIN_SECTOR_ASSETS = 4       # same as scripts 27/33 — sectors < 4 stocks excluded
RANDOM_STATE     = 0

TRAIN_DAYS    = 750
RECLUSTER     = 21
MIN_COUNT     = 125         # ~TRAIN_DAYS/6; same logic as script 29
FDR_Q         = 0.10
COST_BPS      = 5.0

# Core sweep — plain OFI + rank-norm, with/without within-sector masking
SWEEP_CONFIGS = []
for hl in HALF_LIFE_SWEEP:
    # raw OFI
    SWEEP_CONFIGS.append({
        "name": f"dh_hl{hl}_k2",
        "sig":  f"ofi_hl{hl}",
        "k": 2, "ws": False,
    })
    # rank-norm OFI  (plain pct-rank, same as script 27)
    SWEEP_CONFIGS.append({
        "name": f"dh_hl{hl}_rn_k2",
        "sig":  f"ofi_hl{hl}_rn",
        "k": 2, "ws": False,
    })
    # within-sector + rank-norm  (BH-FDR usable at smaller test count)
    SWEEP_CONFIGS.append({
        "name": f"dh_hl{hl}_rn_k2_ws",
        "sig":  f"ofi_hl{hl}_rn",
        "k": 2, "ws": True,
    })


# ============================================================
# signal helpers
# ============================================================
def get_minute_cols(columns):
    return sorted([c for c in columns if c.startswith("minute_")],
                  key=lambda x: int(x.split("_")[1]))


def build_exp_decay_kernel(n_minutes: int, half_life: float) -> np.ndarray:
    lam = np.log(2) / half_life
    k = np.arange(n_minutes, 0, -1, dtype=float)
    w = np.exp(-lam * k)
    return w / w.sum()


# ============================================================
# edge statistics  (from script 29)
# ============================================================
@dataclass
class EdgeStats:
    mu:           np.ndarray
    sigma:        np.ndarray
    sr:           np.ndarray
    tstat:        np.ndarray
    pval:         np.ndarray
    count:        np.ndarray
    screened_mask: np.ndarray


def _bh_mask(pvals: np.ndarray, q: float) -> np.ndarray:
    flat  = pvals.ravel()
    valid = np.isfinite(flat)
    out   = np.zeros_like(flat, dtype=bool)
    if not np.any(valid):
        return out.reshape(pvals.shape)
    pv    = flat[valid]
    m     = pv.size
    order = np.argsort(pv)
    pv_s  = pv[order]
    keep  = pv_s <= q * (np.arange(1, m + 1) / m)
    if not np.any(keep):
        return out.reshape(pvals.shape)
    cutoff = pv_s[np.max(np.where(keep)[0])]
    out[valid] = pv <= cutoff
    return out.reshape(pvals.shape)


def compute_edge_stats(
    signals_train: pd.DataFrame,
    returns_train: pd.DataFrame,
    min_count: int,
    fdr_q: float,
) -> EdgeStats:
    P_lag = signals_train.shift(1)
    R = returns_train.to_numpy(float)
    P = P_lag.to_numpy(float)

    vr = np.isfinite(R);  vp = np.isfinite(P)
    R0 = np.where(vr, R, 0.);  P0 = np.where(vp, P, 0.)

    count    = vr.astype(np.int64).T @ vp.astype(np.int64)
    prod_sum = R0.T @ P0
    prod_ss  = (R0 * R0).T @ (P0 * P0)

    mu = np.full_like(prod_sum, np.nan)
    ok = count > 0
    mu[ok] = prod_sum[ok] / count[ok]

    sig2 = np.full_like(prod_sum, np.nan)
    ok2  = count >= 2
    sig2[ok2] = (prod_ss[ok2] - count[ok2] * mu[ok2] ** 2) / (count[ok2] - 1)
    sig2 = np.where(np.isfinite(sig2), np.maximum(sig2, 0.), np.nan)
    sig  = np.sqrt(sig2)

    sr = np.full_like(prod_sum, np.nan)
    ok_sr = ok2 & (sig > 0)
    sr[ok_sr] = mu[ok_sr] / sig[ok_sr]

    tstat = np.full_like(prod_sum, np.nan)
    tstat[ok_sr] = mu[ok_sr] / (sig[ok_sr] / np.sqrt(count[ok_sr]))

    pval = np.full_like(prod_sum, np.nan)
    if np.any(ok_sr):
        df = np.maximum(count[ok_sr] - 1, 1)
        pval[ok_sr] = 2. * sp_stats.t.sf(np.abs(tstat[ok_sr]), df=df)

    screened = (count >= min_count) & _bh_mask(pval, fdr_q) & np.isfinite(sr)
    return EdgeStats(mu=mu, sigma=sig, sr=sr, tstat=tstat,
                     pval=pval, count=count.astype(np.int64),
                     screened_mask=screened)


# ============================================================
# Dhillon co-clustering  (from script 29)
# ============================================================
@dataclass
class CoClusterResult:
    row_labels: np.ndarray
    col_labels: np.ndarray
    singular_values: np.ndarray
    row_degrees: np.ndarray
    col_degrees: np.ndarray


def dhillon_coclustering(W: np.ndarray, n_clusters: int,
                         random_state: int = 0) -> CoClusterResult:
    W  = np.asarray(W, dtype=float)
    nr, nc = W.shape
    rd = W.sum(1);  cd = W.sum(0)
    eps = 1e-12
    isr = 1. / np.sqrt(np.maximum(rd, eps))
    isc = 1. / np.sqrt(np.maximum(cd, eps))
    Wn  = (isr[:, None] * W) * isc[None, :]

    g   = int(math.ceil(math.log2(max(n_clusters, 2))))
    nc_ = min(g + 1, min(nr, nc))

    if nc_ < 2:
        Z  = np.zeros((nr + nc, 1))
        km = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
        labels = km.fit_predict(Z)
        return CoClusterResult(row_labels=labels[:nr].astype(int),
                               col_labels=labels[nr:].astype(int),
                               singular_values=np.array([]),
                               row_degrees=rd, col_degrees=cd)

    U, s, Vt = randomized_svd(Wn, n_components=nc_, n_iter=7,
                               random_state=random_state)
    U_use = U[:, 1:nc_]
    V_use = Vt.T[:, 1:nc_]
    row_emb = isr[:, None] * U_use
    col_emb = isc[:, None] * V_use
    Z = np.vstack([row_emb, col_emb])

    n_uniq = np.unique(np.round(Z, 12), axis=0).shape[0]
    k_eff  = min(n_clusters, max(2, n_uniq))
    km     = KMeans(n_clusters=k_eff, n_init=50, random_state=random_state)
    labels = km.fit_predict(Z)

    return CoClusterResult(row_labels=labels[:nr].astype(int),
                           col_labels=labels[nr:].astype(int),
                           singular_values=s,
                           row_degrees=rd, col_degrees=cd)


# ============================================================
# block selection  (from script 29)
# ============================================================
@dataclass
class BlockSelection:
    row_cluster:     int
    col_cluster:     int
    row_idx:         np.ndarray
    col_idx:         np.ndarray
    block_score:     float
    source_weights:  np.ndarray
    target_loadings: np.ndarray
    n_edges_used:    int
    mask_name:       str


def _safe_l1(x: np.ndarray) -> np.ndarray:
    d = np.sum(np.abs(x))
    return x / d if (np.isfinite(d) and d > 0) else np.zeros_like(x)


def select_best_block(coclust: CoClusterResult, est: EdgeStats,
                      usable_mask: np.ndarray, mask_name: str) -> BlockSelection:
    K = int(max(coclust.row_labels.max(initial=0),
                coclust.col_labels.max(initial=0)) + 1)
    S = np.where(np.isfinite(est.sr), est.sr, np.nan)
    best: Optional[BlockSelection] = None

    for rc in range(K):
        rows = np.where(coclust.row_labels == rc)[0]
        if rows.size == 0:
            continue
        for cc in range(K):
            cols = np.where(coclust.col_labels == cc)[0]
            if cols.size == 0:
                continue
            good = usable_mask[np.ix_(rows, cols)] & np.isfinite(S[np.ix_(rows, cols)])
            if not np.any(good):
                continue
            sub_S = S[np.ix_(rows, cols)]
            sub_c = est.count[np.ix_(rows, cols)].astype(float)
            w     = np.where(good, sub_c, 0.)
            denom = w.sum()
            if denom <= 0:
                continue
            score = float(np.sum(np.where(good, sub_S * w, 0.)) / denom)

            col_n = np.nansum(np.where(good, sub_S, np.nan), axis=0)
            col_d = np.sum(good, axis=0)
            row_n = np.nansum(np.where(good, sub_S, np.nan), axis=1)
            row_d = np.sum(good, axis=1)
            sw = _safe_l1(np.divide(col_n, col_d, out=np.zeros_like(col_n), where=col_d > 0))
            tl = _safe_l1(np.divide(row_n, row_d, out=np.zeros_like(row_n), where=row_d > 0))

            if sw.sum() == 0 or tl.sum() == 0:
                continue
            cand = BlockSelection(row_cluster=rc, col_cluster=cc,
                                  row_idx=rows.astype(int), col_idx=cols.astype(int),
                                  block_score=score, source_weights=sw,
                                  target_loadings=tl, n_edges_used=int(good.sum()),
                                  mask_name=mask_name)
            if best is None or (np.isfinite(score) and score > best.block_score):
                best = cand

    if best is None:
        raise RuntimeError("No valid block found.")
    return best


def choose_block_with_fallbacks(est: EdgeStats, n_clusters: int,
                                 min_count: int, random_state: int) -> BlockSelection:
    base = np.where(np.isfinite(est.tstat), np.abs(est.tstat), 0.)
    masks = [
        ("screened",   est.screened_mask),
        ("count_only", (est.count >= min_count) & np.isfinite(base) & (base > 0)),
        ("finite_any", (base > 0)),
    ]
    last: Optional[Exception] = None
    for mname, mask in masks:
        if not np.any(mask):
            continue
        W_clust = np.where(mask, base, 0.)
        if not np.any(W_clust > 0):
            continue
        try:
            cc = dhillon_coclustering(W_clust, n_clusters, random_state)
            return select_best_block(cc, est, mask, mname)
        except Exception as e:
            last = e
    raise (last or RuntimeError("No usable mask."))


# ============================================================
# daily position construction  (from script 29)
# ============================================================
def make_daily_positions(sig_today: np.ndarray, n_assets: int,
                          block: BlockSelection) -> np.ndarray:
    pos = np.zeros(n_assets, dtype=float)
    src = sig_today[block.col_idx]
    good = np.isfinite(src)
    if not np.any(good):
        return pos
    sw = _safe_l1(block.source_weights[good])
    z  = float(np.dot(sw, src[good]))
    raw = block.target_loadings * z
    gross = np.sum(np.abs(raw))
    if not np.isfinite(gross) or gross <= 0:
        return pos
    pos[block.row_idx] = raw / gross
    return pos


# ============================================================
# evaluation helpers  (identical to scripts 27/37)
# ============================================================
def daily_spread(scores: np.ndarray, rets: np.ndarray, q: float = 0.10) -> float:
    mask = np.isfinite(scores) & np.isfinite(rets)
    if mask.sum() < 2:
        return np.nan
    s, r = scores[mask], rets[mask]
    n = len(s);  k = max(1, int(n * q))
    if 2 * k > n:
        return np.nan
    idx = np.argsort(s)
    return float(r[idx[-k:]].mean() - r[idx[:k]].mean())


def daily_spread_sn(scores: np.ndarray, rets: np.ndarray,
                    sectors: np.ndarray, q: float = 0.10,
                    min_per_sector: int = 4) -> float:
    spreads = []
    for sec in np.unique(sectors):
        m = sectors == sec
        si, ri = scores[m], rets[m]
        vm = np.isfinite(si) & np.isfinite(ri)
        if vm.sum() < min_per_sector:
            continue
        si, ri = si[vm], ri[vm]
        n = len(si);  k = max(1, int(n * q))
        if 2 * k > n:
            continue
        idx = np.argsort(si)
        spreads.append(float(ri[idx[-k:]].mean() - ri[idx[:k]].mean()))
    return float(np.mean(spreads)) if spreads else np.nan


def annualized_sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    s = float(np.std(x, ddof=1))
    return float(np.sqrt(252) * x.mean() / s) if s > 0 else np.nan


# ============================================================
# walk-forward backtest
# ============================================================
def run_walkforward(
    signals_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    train_days: int,
    recluster_days: int,
    n_clusters: int,
    fdr_q: float,
    min_count: int,
    cost_bps: float,
    random_state: int,
    name: str,
    within_sector: bool = False,
) -> pd.DataFrame:
    cost_rate = cost_bps / 1e4
    dates     = signals_df.index
    n_assets  = signals_df.shape[1]
    tickers   = signals_df.columns.tolist()
    sec_arr   = ticker_to_sector.reindex(tickers).to_numpy()

    sig_np  = signals_df.to_numpy(float)
    ret_np  = returns_df.to_numpy(float)
    sig_lag = np.vstack([np.full((1, n_assets), np.nan), sig_np[:-1]])

    records = []
    prev_pos = np.zeros(n_assets, dtype=float)
    block: Optional[BlockSelection] = None

    for t in range(train_days, len(dates)):
        need = (block is None) or ((t - train_days) % recluster_days == 0)

        if need:
            sl  = slice(t - train_days, t)
            est = compute_edge_stats(signals_df.iloc[sl], returns_df.iloc[sl],
                                     min_count=min_count, fdr_q=fdr_q)

            if within_sector:
                sec_vec = ticker_to_sector.reindex(tickers).to_numpy()
                same    = sec_vec[:, None] == sec_vec[None, :]

                # within-sector: also enforce MIN_SECTOR_ASSETS=4
                sec_sizes = pd.Series(sec_vec).value_counts()
                small_sec = set(sec_sizes[sec_sizes < MIN_SECTOR_ASSETS].index)
                small_mask = np.array([s in small_sec for s in sec_vec])
                # zero out rows/cols belonging to small sectors
                same = same & ~small_mask[:, None] & ~small_mask[None, :]

                est.mu            = np.where(same, est.mu,    np.nan)
                est.sigma         = np.where(same, est.sigma, np.nan)
                est.sr            = np.where(same, est.sr,    np.nan)
                est.tstat         = np.where(same, est.tstat, np.nan)
                est.pval          = np.where(same, est.pval,  np.nan)
                est.count         = np.where(same, est.count, 0)
                est.screened_mask = est.screened_mask & same

            try:
                block = choose_block_with_fallbacks(
                    est, n_clusters=n_clusters,
                    min_count=min_count, random_state=random_state)
            except Exception as e:
                print(f"  [{name}] recluster error at t={t}: {e}", flush=True)
                block = None

        if block is None:
            prev_pos = np.zeros(n_assets, dtype=float)
            continue

        pos   = make_daily_positions(sig_lag[t], n_assets, block)
        ret_t = ret_np[t]
        gross = float(np.nansum(np.where(np.isfinite(ret_t), pos * ret_t, 0.)))
        turn  = float(np.sum(np.abs(pos - prev_pos)))
        net   = gross - cost_rate * turn
        prev_pos = pos

        ic    = float(pd.Series(pos).corr(pd.Series(ret_t), method="spearman")) \
                if np.isfinite(ret_t).sum() > 2 else np.nan
        sp    = daily_spread(pos, ret_t, Q)
        sp_sn = daily_spread_sn(pos, ret_t, sec_arr, Q, MIN_NAMES_SN)

        records.append({
            "date": str(dates[t].date()),
            "gross_pnl": gross, "net_pnl": net, "turnover": turn,
            "ic": ic, "spread": sp, "spread_sn": sp_sn,
            "n_targets": int(block.row_idx.size),
            "n_sources": int(block.col_idx.size),
            "block_score": float(block.block_score),
            "mask_name": block.mask_name,
        })

        if (t - train_days + 1) % 200 == 0:
            print(f"  [{name}] day {t - train_days + 1}/{len(dates) - train_days} "
                  f"| {dates[t].date()} | mask={block.mask_name} "
                  f"| tgt={block.row_idx.size} src={block.col_idx.size}", flush=True)

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ============================================================
# data loading  (mirrors script 27 / 37)
# ============================================================
print("Reading data...", flush=True)
header   = pd.read_csv(INPUT_PATH, nrows=0)
min_cols = get_minute_cols(list(header.columns))
df = pd.read_csv(INPUT_PATH, usecols=["date", "ticker", "residual_ret", "sector"] + min_cols)
df["date"] = pd.to_datetime(df["date"])

if USE_LAST_N_YEARS is not None:
    cutoff = df["date"].max() - pd.DateOffset(years=USE_LAST_N_YEARS)
    df = df[df["date"] >= cutoff].copy()
    print(f"  Using data from {cutoff.date()} onward", flush=True)

df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
df[min_cols] = df[min_cols].fillna(0.0)
df = df.dropna(subset=["ticker", "date", "residual_ret", "sector"]).copy()
df = df[df["ticker"] != "GOOGL"].copy()   # drop GOOGL duplicate

sc_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
df = df[df["ticker"].isin(sc_counts[sc_counts == 1].index)].copy()
ticker_to_sector_full = df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

n_min  = len(min_cols)
min_np = df[min_cols].to_numpy(float)
print("Building exp-decay OFI signals:", flush=True)
for hl in HALF_LIFE_SWEEP:
    k = build_exp_decay_kernel(n_min, hl)
    df[f"ofi_hl{hl}"]    = min_np @ k
    df[f"ofi_hl{hl}_rn"] = df.groupby("date")[f"ofi_hl{hl}"].rank(pct=True)
    print(f"  hl={hl:3d}min", flush=True)
del min_np

# coverage filter (hl=45 proxy, same as 27)
sig_proxy = df.pivot(index="date", columns="ticker", values="ofi_hl45")
ret_proxy = df.pivot(index="date", columns="ticker", values="residual_ret")
cd = sig_proxy.index.intersection(ret_proxy.index)
ct = sig_proxy.columns.intersection(ret_proxy.columns)
ticker_obs = (sig_proxy.loc[cd, ct].notna() & ret_proxy.loc[cd, ct].notna()).sum(axis=0)
keep_tickers = (
    ticker_obs[ticker_obs >= MIN_TICKER_OBS]
    .sort_values(ascending=False)
    .index[:MAX_ASSETS].tolist()
)
df = df[df["ticker"].isin(keep_tickers)].copy()
ticker_to_sector = ticker_to_sector_full.loc[keep_tickers].copy()
print(f"  Using {len(keep_tickers)} assets.", flush=True)


# ============================================================
# main sweep
# ============================================================
all_summaries = []

for cfg in SWEEP_CONFIGS:
    name    = cfg["name"]
    sig_col = cfg["sig"]
    k       = cfg["k"]
    ws      = cfg["ws"]

    print(f"\n{'='*70}", flush=True)
    print(f"CONFIG: {name}  |  sig={sig_col}  k={k}  ws={ws}", flush=True)
    print(f"{'='*70}", flush=True)

    sig_wide = df.pivot(index="date", columns="ticker", values=sig_col).sort_index()
    ret_wide = df.pivot(index="date", columns="ticker", values="residual_ret").sort_index()

    cd2 = sig_wide.index.intersection(ret_wide.index)
    ct2 = sig_wide.columns.intersection(ret_wide.columns)
    sig_wide = sig_wide.loc[cd2, ct2]
    ret_wide = ret_wide.loc[cd2, ct2]
    ts_use   = ticker_to_sector.reindex(ct2).dropna()
    ct2      = ts_use.index
    sig_wide = sig_wide[ct2].fillna(0.0)
    ret_wide = ret_wide[ct2]

    if len(cd2) <= TRAIN_DAYS:
        print(f"  Skipping: not enough dates.", flush=True)
        continue

    t0 = time.time()
    bt = run_walkforward(
        signals_df=sig_wide,
        returns_df=ret_wide,
        ticker_to_sector=ts_use,
        train_days=TRAIN_DAYS,
        recluster_days=RECLUSTER,
        n_clusters=k,
        fdr_q=FDR_Q,
        min_count=MIN_COUNT,
        cost_bps=COST_BPS,
        random_state=RANDOM_STATE,
        name=name,
        within_sector=ws,
    )
    elapsed = time.time() - t0

    if bt.empty:
        print(f"  [{name}] No output.", flush=True)
        continue

    gross_sr  = annualized_sharpe(bt["gross_pnl"].to_numpy())
    sn_sr     = annualized_sharpe(bt["spread_sn"].dropna().to_numpy())
    spread_sr = annualized_sharpe(bt["spread"].dropna().to_numpy())
    mean_ic   = float(bt["ic"].dropna().mean())

    print(f"  [{name}] gross SR={gross_sr:.3f}  SN SR={sn_sr:.3f}  "
          f"spread SR={spread_sr:.3f}  IC={mean_ic:.4f}  ({elapsed:.0f}s)", flush=True)

    all_summaries.append({
        "name": name, "sig_col": sig_col, "within_sector": ws, "n_clusters": k,
        "annualized_gross_sharpe":     gross_sr,
        "annualized_spread_sharpe":    spread_sr,
        "annualized_sn_spread_sharpe": sn_sr,
        "mean_ic":     mean_ic,
        "n_eval_days": len(bt),
        "elapsed_s":   round(elapsed, 1),
    })
    bt.to_csv(OUT_DIR / f"backtest_{name}.csv")

# ============================================================
# combined summary
# ============================================================
if not all_summaries:
    raise RuntimeError("No configs produced output.")

summary_df = (pd.DataFrame(all_summaries)
              .sort_values("annualized_sn_spread_sharpe", ascending=False)
              .reset_index(drop=True))
summary_df.to_csv(OUT_DIR / "summary_all.csv", index=False)

print(f"\n{'='*70}", flush=True)
print("ALL CONFIG SUMMARY (sorted by SN spread Sharpe)", flush=True)
print(f"{'='*70}", flush=True)
cols = ["name", "annualized_gross_sharpe", "annualized_spread_sharpe",
        "annualized_sn_spread_sharpe", "mean_ic", "within_sector"]
print(summary_df[cols].to_string(index=False), flush=True)
print(f"\nSaved to: {OUT_DIR.resolve()}", flush=True)
