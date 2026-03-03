#!/usr/bin/env python3
"""
18_dhillon_bipartite_source_target.py

Dhillon-style bipartite spectral co-clustering for a directed source-target
OFI -> next-day-return matrix, wrapped in a walk-forward backtest.

Main idea
---------
For each training window:
1. Build pairwise directed edge statistics from lagged OFI signals and next-day returns.
2. Screen edges by min_count + BH-FDR on two-sided t-tests.
3. Convert the screened matrix into a NONNEGATIVE bipartite weight matrix
   (default: abs(t-stat)), because Dhillon (2001) is a bipartite spectral graph method.
4. Run spectral co-clustering on the row/column sides jointly.
5. Score row-cluster/column-cluster blocks using the SIGNED training score matrix.
6. Pick the best source-target block and trade a block-factor portfolio out of sample.

Expected input
--------------
Two wide matrices with the same daily index and the same asset columns:
- signals: daily OFI-type signal matrix P, shape [T x N]
- returns: daily next-day tradable return matrix R, shape [T x N]

Missing values are allowed. Supported formats: .csv, .parquet, .pkl, .npy, .npz

Example
-------
python3 scripts/18_dhillon_bipartite_source_target.py \
  --signals data/processed/P.npy \
  --returns data/processed/R.npy \
  --outdir results/18_dhillon_bipartite \
  --train-days 252 \
  --rebalance-days 5 \
  --n-clusters 6 \
  --fdr-q 0.10 \
  --min-count 120 \
  --cluster-weight-stat abs_t \
  --block-score-stat sr \
  --cost-bps 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _read_matrix(path: str) -> pd.DataFrame:
    """Read a wide T x N matrix with date-like index and asset columns."""
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".csv":
        df = pd.read_csv(p, index_col=0)
    elif suf in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    elif suf in {".pkl", ".pickle"}:
        df = pd.read_pickle(p)
    elif suf == ".npy":
        arr = np.load(p, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype.names is None:
            # For .npy files, create a DatetimeIndex starting from 2007-06-27 (daily)
            T, N = arr.shape
            date_index = pd.date_range(start="2007-06-27", periods=T, freq="D")
            df = pd.DataFrame(arr, index=date_index)
        else:
            raise ValueError(".npy input must be a plain 2D array if used directly.")
    elif suf == ".npz":
        z = np.load(p, allow_pickle=True)
        if "data" in z:
            data = z["data"]
            index = z["index"] if "index" in z else np.arange(data.shape[0])
            columns = z["columns"] if "columns" in z else np.arange(data.shape[1])
            df = pd.DataFrame(data, index=index, columns=columns)
        else:
            raise ValueError(".npz input must contain key 'data' and optionally 'index','columns'.")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    # Only convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Force numeric; bad parses become NaN.
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_index()


def align_signal_and_returns(signals: pd.DataFrame, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = signals.index.intersection(returns.index)
    common_cols = signals.columns.intersection(returns.columns)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between signals and returns.")
    if len(common_cols) == 0:
        raise ValueError("No overlapping asset columns between signals and returns.")
    s = signals.loc[common_dates, common_cols].copy()
    r = returns.loc[common_dates, common_cols].copy()
    return s, r


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------

def benjamini_hochberg_mask(pvals: NDArray[np.float64], q: float) -> NDArray[np.bool_]:
    """BH mask on a 2D p-value array; NaNs are always False."""
    flat = pvals.ravel()
    valid = np.isfinite(flat)
    out = np.zeros_like(flat, dtype=bool)
    if not np.any(valid):
        return out.reshape(pvals.shape)

    pv = flat[valid]
    m = pv.size
    order = np.argsort(pv)
    pv_sorted = pv[order]
    thresh = q * (np.arange(1, m + 1) / m)
    keep = pv_sorted <= thresh
    if not np.any(keep):
        return out.reshape(pvals.shape)

    k = np.max(np.where(keep)[0])
    cutoff = pv_sorted[k]
    out[valid] = pv <= cutoff
    return out.reshape(pvals.shape)


@dataclass
class EdgeStats:
    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    sr: NDArray[np.float64]
    tstat: NDArray[np.float64]
    pval: NDArray[np.float64]
    count: NDArray[np.int64]
    keep_mask: NDArray[np.bool_]


def compute_edge_stats(signals_train: pd.DataFrame, returns_train: pd.DataFrame, min_count: int, fdr_q: float) -> EdgeStats:
    """
    Compute pairwise statistics for x_{t,i,j} = R_{t,i} * P_{t-1,j}.

    Efficient formulas:
      count_{ij} = sum_t 1{R_{ti} observed} 1{P_{t-1,j} observed}
      sum_{ij}   = (R0.T @ P0)_{ij}
      sumsq_{ij} = ((R0^2).T @ (P0^2))_{ij}
    where missing entries are filled with 0 in R0 and P0.
    """
    # Lag signals by one day inside the training window.
    P_lag = signals_train.shift(1)

    R = returns_train.to_numpy(dtype=float)
    P = P_lag.to_numpy(dtype=float)

    valid_r = np.isfinite(R)
    valid_p = np.isfinite(P)

    R0 = np.where(valid_r, R, 0.0)
    P0 = np.where(valid_p, P, 0.0)

    count = valid_r.astype(np.int64).T @ valid_p.astype(np.int64)  # [target i, source j]
    prod_sum = R0.T @ P0
    prod_sumsq = (R0 * R0).T @ (P0 * P0)

    mu = np.full_like(prod_sum, np.nan, dtype=float)
    ok_mean = count > 0
    mu[ok_mean] = prod_sum[ok_mean] / count[ok_mean]

    sigma2 = np.full_like(prod_sum, np.nan, dtype=float)
    ok_var = count >= 2
    sigma2[ok_var] = (prod_sumsq[ok_var] - count[ok_var] * mu[ok_var] ** 2) / (count[ok_var] - 1)
    sigma2 = np.where(np.isfinite(sigma2), np.maximum(sigma2, 0.0), np.nan)
    sigma = np.sqrt(sigma2)

    sr = np.full_like(prod_sum, np.nan, dtype=float)
    ok_sr = ok_var & (sigma > 0)
    sr[ok_sr] = mu[ok_sr] / sigma[ok_sr]

    tstat = np.full_like(prod_sum, np.nan, dtype=float)
    tstat[ok_sr] = mu[ok_sr] / (sigma[ok_sr] / np.sqrt(count[ok_sr]))

    pval = np.full_like(prod_sum, np.nan, dtype=float)
    if np.any(ok_sr):
        df = np.maximum(count[ok_sr] - 1, 1)
        pval[ok_sr] = 2.0 * stats.t.sf(np.abs(tstat[ok_sr]), df=df)

    enough_data = count >= min_count
    bh_mask = benjamini_hochberg_mask(pval, q=fdr_q)
    keep_mask = enough_data & bh_mask & np.isfinite(sr)

    return EdgeStats(
        mu=mu,
        sigma=sigma,
        sr=sr,
        tstat=tstat,
        pval=pval,
        count=count.astype(np.int64),
        keep_mask=keep_mask,
    )


# -----------------------------------------------------------------------------
# Dhillon-style co-clustering
# -----------------------------------------------------------------------------

def build_cluster_weight_matrix(stats_obj: EdgeStats, mode: str) -> NDArray[np.float64]:
    """
    Convert screened directed statistics to the NONNEGATIVE weight matrix W used
    by Dhillon-style bipartite spectral co-clustering.
    """
    if mode == "abs_t":
        base = np.abs(stats_obj.tstat)
    elif mode == "abs_sr":
        base = np.abs(stats_obj.sr)
    elif mode == "positive_sr":
        base = np.maximum(stats_obj.sr, 0.0)
    elif mode == "positive_mu":
        base = np.maximum(stats_obj.mu, 0.0)
    else:
        raise ValueError(f"Unknown cluster-weight mode: {mode}")

    W = np.where(stats_obj.keep_mask & np.isfinite(base), base, 0.0)
    return W


@dataclass
class CoClusterResult:
    row_labels: NDArray[np.int64]
    col_labels: NDArray[np.int64]
    row_embedding: NDArray[np.float64]
    col_embedding: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    row_degrees: NDArray[np.float64]
    col_degrees: NDArray[np.float64]


def dhillon_coclustering(W: NDArray[np.float64], n_clusters: int, random_state: int = 0) -> CoClusterResult:
    """
    Multipartition(k) from Dhillon (2001), adapted to a general nonnegative matrix W.

    We use:
      Wn = D_r^{-1/2} W D_c^{-1/2}
      ell = ceil(log2(k))
    then compute the top ell+1 singular vectors, drop the trivial first one, form
    the stacked embedding
      Z = [D_r^{-1/2} U_{2:ell+1}; D_c^{-1/2} V_{2:ell+1}]
    and run k-means jointly on rows+columns.
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be 2D.")

    n_rows, n_cols = W.shape
    row_deg = W.sum(axis=1)
    col_deg = W.sum(axis=0)

    eps = 1e-12
    inv_sqrt_row = 1.0 / np.sqrt(np.maximum(row_deg, eps))
    inv_sqrt_col = 1.0 / np.sqrt(np.maximum(col_deg, eps))

    Wn = (inv_sqrt_row[:, None] * W) * inv_sqrt_col[None, :]

    ell = int(math.ceil(math.log2(n_clusters)))
    n_components = min(ell + 1, min(n_rows, n_cols))
    if n_components < 2:
        # Extremely degenerate case.
        row_emb = np.zeros((n_rows, 1), dtype=float)
        col_emb = np.zeros((n_cols, 1), dtype=float)
        Z = np.vstack([row_emb, col_emb])
        km = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
        labels_all = km.fit_predict(Z)
        return CoClusterResult(
            row_labels=labels_all[:n_rows],
            col_labels=labels_all[n_rows:],
            row_embedding=row_emb,
            col_embedding=col_emb,
            singular_values=np.array([]),
            row_degrees=row_deg,
            col_degrees=col_deg,
        )

    U, s, Vt = randomized_svd(
        Wn,
        n_components=n_components,
        n_iter=7,
        random_state=random_state,
    )

    # Singular values already descending for randomized_svd.
    U_use = U[:, 1:n_components]
    V_use = Vt.T[:, 1:n_components]

    row_emb = inv_sqrt_row[:, None] * U_use
    col_emb = inv_sqrt_col[:, None] * V_use
    Z = np.vstack([row_emb, col_emb])

    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
    labels_all = km.fit_predict(Z)
    row_labels = labels_all[:n_rows].astype(int)
    col_labels = labels_all[n_rows:].astype(int)

    return CoClusterResult(
        row_labels=row_labels,
        col_labels=col_labels,
        row_embedding=row_emb,
        col_embedding=col_emb,
        singular_values=s,
        row_degrees=row_deg,
        col_degrees=col_deg,
    )


# -----------------------------------------------------------------------------
# Block scoring + portfolio construction
# -----------------------------------------------------------------------------

def choose_signed_score_matrix(stats_obj: EdgeStats, mode: str) -> NDArray[np.float64]:
    if mode == "sr":
        S = stats_obj.sr
    elif mode == "t":
        S = stats_obj.tstat
    elif mode == "mu":
        S = stats_obj.mu
    else:
        raise ValueError(f"Unknown block-score-stat mode: {mode}")
    return np.where(np.isfinite(S), S, np.nan)


@dataclass
class BlockSelection:
    row_cluster: int
    col_cluster: int
    row_idx: NDArray[np.int64]
    col_idx: NDArray[np.int64]
    block_score: float
    source_weights: NDArray[np.float64]
    target_loadings: NDArray[np.float64]
    block_edge_count: int


def _safe_l1_normalize(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=float)
    denom = np.sum(np.abs(x))
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(x)
    return x / denom


def select_best_block(
    coclust: CoClusterResult,
    stats_obj: EdgeStats,
    block_score_stat: str = "sr",
    weight_by_count: bool = True,
    allow_same_cluster_pair: bool = True,
) -> BlockSelection:
    """
    Pick the best row-cluster / column-cluster block using the SIGNED training matrix.

    The score is a counts-weighted mean of the chosen signed stat across screened edges.
    The resulting per-source and per-target weights are signed row/column means inside the block.
    """
    S = choose_signed_score_matrix(stats_obj, block_score_stat)
    K = int(max(coclust.row_labels.max(initial=0), coclust.col_labels.max(initial=0)) + 1)

    best: Optional[BlockSelection] = None

    for rc in range(K):
        rows = np.where(coclust.row_labels == rc)[0]
        if rows.size == 0:
            continue
        for cc in range(K):
            if (not allow_same_cluster_pair) and (rc == cc):
                continue
            cols = np.where(coclust.col_labels == cc)[0]
            if cols.size == 0:
                continue

            sub_keep = stats_obj.keep_mask[np.ix_(rows, cols)]
            if not np.any(sub_keep):
                continue

            sub_S = S[np.ix_(rows, cols)]
            sub_counts = stats_obj.count[np.ix_(rows, cols)].astype(float)
            good = sub_keep & np.isfinite(sub_S)
            if not np.any(good):
                continue

            if weight_by_count:
                w = np.where(good, sub_counts, 0.0)
                denom = w.sum()
                block_score = float(np.sum(np.where(good, sub_S * w, 0.0)) / denom) if denom > 0 else np.nan
            else:
                block_score = float(np.nanmean(np.where(good, sub_S, np.nan)))

            # Source weights = signed average strength of each source column within this block.
            # Target loadings = signed average strength of each target row within this block.
            col_num = np.nansum(np.where(good, sub_S, np.nan), axis=0)
            col_den = np.sum(good, axis=0)
            row_num = np.nansum(np.where(good, sub_S, np.nan), axis=1)
            row_den = np.sum(good, axis=1)

            source_raw = np.divide(col_num, col_den, out=np.zeros_like(col_num), where=col_den > 0)
            target_raw = np.divide(row_num, row_den, out=np.zeros_like(row_num), where=row_den > 0)

            source_w = _safe_l1_normalize(source_raw)
            target_b = _safe_l1_normalize(target_raw)

            if np.sum(np.abs(source_w)) == 0 or np.sum(np.abs(target_b)) == 0:
                continue

            cand = BlockSelection(
                row_cluster=rc,
                col_cluster=cc,
                row_idx=rows.astype(int),
                col_idx=cols.astype(int),
                block_score=block_score,
                source_weights=source_w,
                target_loadings=target_b,
                block_edge_count=int(np.sum(good)),
            )
            if best is None or (np.isfinite(cand.block_score) and cand.block_score > best.block_score):
                best = cand

    if best is None:
        raise RuntimeError("No valid source-target block was found. Try loosening q/min_count or changing stats.")
    return best


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------

@dataclass
class RebalanceRecord:
    rebalance_date: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    row_cluster: int
    col_cluster: int
    n_targets: int
    n_sources: int
    block_score: float
    block_edge_count: int


def make_daily_positions(
    signals_lag_day: pd.Series,
    n_assets: int,
    block: BlockSelection,
) -> NDArray[np.float64]:
    """
    Position rule:
      z_t = sum_{j in S} w_j * P_{t-1,j}
      p_raw_i = b_i * z_t for i in T
      then normalize to unit gross leverage.
    """
    p = np.zeros(n_assets, dtype=float)
    src_signal = signals_lag_day.iloc[block.col_idx].to_numpy(dtype=float)
    if np.any(~np.isfinite(src_signal)):
        # Missing sources on this day -> no trade.
        return p

    z_t = float(np.dot(block.source_weights, src_signal))
    raw = block.target_loadings * z_t
    gross = np.sum(np.abs(raw))
    if not np.isfinite(gross) or gross <= 0:
        return p
    p[block.row_idx] = raw / gross
    return p


def summarize_performance(bt: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in ["gross_pnl", "net_pnl"]:
        x = bt[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out[f"{col}_mean"] = np.nan
            out[f"{col}_vol"] = np.nan
            out[f"{col}_sharpe"] = np.nan
            out[f"{col}_cum"] = np.nan
            continue
        mu = float(np.mean(x))
        vol = float(np.std(x, ddof=1)) if x.size >= 2 else np.nan
        sharpe = float(np.sqrt(252.0) * mu / vol) if np.isfinite(vol) and vol > 0 else np.nan
        cum = float(np.sum(x))
        out[f"{col}_mean"] = mu
        out[f"{col}_vol"] = vol
        out[f"{col}_sharpe"] = sharpe
        out[f"{col}_cum"] = cum

    active = bt["gross_exposure"] > 0
    x = bt.loc[active, "net_pnl"].to_numpy(dtype=float)
    if x.size > 0:
        mu = float(np.mean(x))
        vol = float(np.std(x, ddof=1)) if x.size >= 2 else np.nan
        sharpe = float(np.sqrt(252.0) * mu / vol) if np.isfinite(vol) and vol > 0 else np.nan
    else:
        sharpe = np.nan
    out["active_only_net_sharpe"] = sharpe
    out["active_days"] = int(active.sum())
    out["total_days"] = int(bt.shape[0])
    out["active_fraction"] = float(active.mean()) if bt.shape[0] > 0 else np.nan
    out["avg_turnover"] = float(bt["turnover"].mean()) if bt.shape[0] > 0 else np.nan
    out["avg_gross_exposure"] = float(bt["gross_exposure"].mean()) if bt.shape[0] > 0 else np.nan
    return out


def run_walk_forward(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    train_days: int,
    rebalance_days: int,
    n_clusters: int,
    fdr_q: float,
    min_count: int,
    cluster_weight_stat: str,
    block_score_stat: str,
    cost_bps: float,
    allow_same_cluster_pair: bool,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:

    dates = signals.index
    n_assets = signals.shape[1]
    tickers = list(signals.columns)

    records: List[Dict[str, float]] = []
    rebalance_records: List[RebalanceRecord] = []

    prev_pos = np.zeros(n_assets, dtype=float)
    cost_rate = cost_bps / 1e4

    # Need at least train_days + 1 because signals are lagged by 1 internally.
    start_idx = train_days
    t0 = start_idx
    while t0 < len(dates):
        train_slice = slice(t0 - train_days, t0)
        test_slice = slice(t0, min(t0 + rebalance_days, len(dates)))

        s_train = signals.iloc[train_slice]
        r_train = returns.iloc[train_slice]
        stats_obj = compute_edge_stats(s_train, r_train, min_count=min_count, fdr_q=fdr_q)

        W = build_cluster_weight_matrix(stats_obj, mode=cluster_weight_stat)
        coclust = dhillon_coclustering(W, n_clusters=n_clusters, random_state=random_state)
        block = select_best_block(
            coclust,
            stats_obj,
            block_score_stat=block_score_stat,
            weight_by_count=True,
            allow_same_cluster_pair=allow_same_cluster_pair,
        )

        rebalance_records.append(
            RebalanceRecord(
                rebalance_date=str(dates[t0].date()),
                train_start=str(dates[t0 - train_days].date()),
                train_end=str(dates[t0 - 1].date()),
                test_start=str(dates[test_slice.start].date()),
                test_end=str(dates[test_slice.stop - 1].date()),
                row_cluster=int(block.row_cluster),
                col_cluster=int(block.col_cluster),
                n_targets=int(block.row_idx.size),
                n_sources=int(block.col_idx.size),
                block_score=float(block.block_score),
                block_edge_count=int(block.block_edge_count),
            )
        )

        for tt in range(test_slice.start, test_slice.stop):
            d = dates[tt]
            sig_lag = signals.shift(1).iloc[tt]
            pos = make_daily_positions(sig_lag, n_assets, block)
            ret = returns.iloc[tt].to_numpy(dtype=float)

            gross_pnl = float(np.nansum(np.where(np.isfinite(ret), pos * ret, 0.0)))
            turnover = float(np.sum(np.abs(pos - prev_pos)))
            net_pnl = gross_pnl - cost_rate * turnover
            gross_exposure = float(np.sum(np.abs(pos)))

            records.append(
                {
                    "date": d,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "turnover": turnover,
                    "gross_exposure": gross_exposure,
                    "row_cluster": int(block.row_cluster),
                    "col_cluster": int(block.col_cluster),
                    "n_targets": int(block.row_idx.size),
                    "n_sources": int(block.col_idx.size),
                    "block_score": float(block.block_score),
                }
            )
            prev_pos = pos

        t0 += rebalance_days

    bt = pd.DataFrame(records).set_index("date")
    rebal = pd.DataFrame([asdict(x) for x in rebalance_records])
    summary = summarize_performance(bt)
    summary["n_clusters"] = int(n_clusters)
    summary["train_days"] = int(train_days)
    summary["rebalance_days"] = int(rebalance_days)
    summary["fdr_q"] = float(fdr_q)
    summary["min_count"] = int(min_count)
    summary["cost_bps"] = float(cost_bps)
    summary["cluster_weight_stat"] = cluster_weight_stat
    summary["block_score_stat"] = block_score_stat
    summary["n_assets"] = int(n_assets)
    summary["n_backtest_days"] = int(bt.shape[0])
    return bt, rebal, summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dhillon-style bipartite spectral co-clustering for source-target OFI networks")
    p.add_argument("--signals", default="data/processed/P.npy", help="Wide daily signal matrix P (dates x assets)")
    p.add_argument("--returns", default="data/processed/R.npy", help="Wide daily return matrix R (dates x assets)")
    p.add_argument("--outdir", required=True, help="Directory for outputs")

    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--rebalance-days", type=int, default=5)
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--fdr-q", type=float, default=0.10)
    p.add_argument("--min-count", type=int, default=120)
    p.add_argument("--cluster-weight-stat", choices=["abs_t", "abs_sr", "positive_sr", "positive_mu"], default="abs_t")
    p.add_argument("--block-score-stat", choices=["sr", "t", "mu"], default="sr")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--disallow-same-cluster-pair", action="store_true")
    p.add_argument("--random-state", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = _read_matrix(args.signals)
    returns = _read_matrix(args.returns)
    signals, returns = align_signal_and_returns(signals, returns)

    bt, rebal, summary = run_walk_forward(
        signals=signals,
        returns=returns,
        train_days=args.train_days,
        rebalance_days=args.rebalance_days,
        n_clusters=args.n_clusters,
        fdr_q=args.fdr_q,
        min_count=args.min_count,
        cluster_weight_stat=args.cluster_weight_stat,
        block_score_stat=args.block_score_stat,
        cost_bps=args.cost_bps,
        allow_same_cluster_pair=not args.disallow_same_cluster_pair,
        random_state=args.random_state,
    )

    bt.to_csv(outdir / "daily_backtest.csv")
    rebal.to_csv(outdir / "rebalance_history.csv", index=False)

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save cumulative PnL series for convenience.
    cum = pd.DataFrame(index=bt.index)
    cum["cum_gross_pnl"] = bt["gross_pnl"].cumsum()
    cum["cum_net_pnl"] = bt["net_pnl"].cumsum()
    cum.to_csv(outdir / "cumulative_pnl.csv")

    print("Saved outputs to:", outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
