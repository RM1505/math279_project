#!/usr/bin/env python3
"""
18_dhillon_bipartite_source_target.py

Dhillon-style bipartite spectral co-clustering for a directed source-target
OFI -> next-day residual-return matrix, with DAILY trading but slower
reclustering / block reselection.

Main design
-----------
- Trade every day using lagged daily signals.
- Recompute edge stats, re-cluster, and reselect the best source-target block
  only every RECLUSTER_DAYS.
- Default to N_CLUSTERS = 2, which is much more stable and much closer to the
  original Dhillon bipartition idea.
- Keep the same data paths:
    data/processed/signal_matrix_unbalanced.csv
    data/processed/return_matrix_unbalanced.csv
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


# =============================================================================
# Paths and default parameters
# =============================================================================

SIGNALS_PATH = Path("data/processed/signal_matrix_unbalanced.csv")
RETURNS_PATH = Path("data/processed/return_matrix_unbalanced.csv")
OUT_DIR = Path("data/processed/dhillon_bipartite_daily")

TRAIN_DAYS = 252
RECLUSTER_DAYS = 20
N_CLUSTERS = 2

FDR_Q = 0.10
MIN_COUNT = 120

CLUSTER_WEIGHT_STAT = "abs_t"   # {"abs_t", "abs_sr", "positive_sr", "positive_mu"}
BLOCK_SCORE_STAT = "sr"         # {"sr", "t", "mu"}

COST_BPS = 5.0
ALLOW_SAME_CLUSTER_PAIR = True
RANDOM_STATE = 0


# =============================================================================
# IO
# =============================================================================

def read_matrix(path: Path) -> pd.DataFrame:
    """Read a wide [dates x assets] matrix."""
    suf = path.suffix.lower()

    if suf == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif suf in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suf in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suf == ".npy":
        arr = np.load(path, allow_pickle=True)
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"{path} must be a plain 2D array.")
        df = pd.DataFrame(arr)
    elif suf == ".npz":
        z = np.load(path, allow_pickle=True)
        if "data" not in z:
            raise ValueError(f"{path} must contain key 'data'.")
        data = z["data"]
        index = z["index"] if "index" in z else np.arange(data.shape[0])
        columns = z["columns"] if "columns" in z else np.arange(data.shape[1])
        df = pd.DataFrame(data, index=index, columns=columns)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            parsed = pd.to_datetime(df.index, errors="raise")
            df.index = parsed
    except Exception:
        pass

    df = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_index()


def align_signal_and_returns(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = signals.index.intersection(returns.index)
    common_cols = signals.columns.intersection(returns.columns)

    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between signals and returns.")
    if len(common_cols) == 0:
        raise ValueError("No overlapping asset columns between signals and returns.")

    s = signals.loc[common_dates, common_cols].copy()
    r = returns.loc[common_dates, common_cols].copy()

    s = s.sort_index().sort_index(axis=1)
    r = r.sort_index().sort_index(axis=1)
    return s, r


# =============================================================================
# Statistics helpers
# =============================================================================

def benjamini_hochberg_mask(
    pvals: NDArray[np.float64],
    q: float,
) -> NDArray[np.bool_]:
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
    screened_mask: NDArray[np.bool_]


def compute_edge_stats(
    signals_train: pd.DataFrame,
    returns_train: pd.DataFrame,
    min_count: int,
    fdr_q: float,
) -> EdgeStats:
    """
    Pairwise source-target payoff statistics for:

        x_{t, i, j} = R_{t, i} * P_{t-1, j}

    where:
        i = target asset
        j = source asset

    Returns matrices of shape [n_targets x n_sources].
    """
    P_lag = signals_train.shift(1)

    R = returns_train.to_numpy(dtype=float)
    P = P_lag.to_numpy(dtype=float)

    valid_r = np.isfinite(R)
    valid_p = np.isfinite(P)

    R0 = np.where(valid_r, R, 0.0)
    P0 = np.where(valid_p, P, 0.0)

    count = valid_r.astype(np.int64).T @ valid_p.astype(np.int64)
    prod_sum = R0.T @ P0
    prod_sumsq = (R0 * R0).T @ (P0 * P0)

    mu = np.full_like(prod_sum, np.nan, dtype=float)
    ok_mean = count > 0
    mu[ok_mean] = prod_sum[ok_mean] / count[ok_mean]

    sigma2 = np.full_like(prod_sum, np.nan, dtype=float)
    ok_var = count >= 2
    sigma2[ok_var] = (
        prod_sumsq[ok_var] - count[ok_var] * mu[ok_var] ** 2
    ) / (count[ok_var] - 1)
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
    bh = benjamini_hochberg_mask(pval, q=fdr_q)
    screened_mask = enough_data & bh & np.isfinite(sr)

    return EdgeStats(
        mu=mu,
        sigma=sigma,
        sr=sr,
        tstat=tstat,
        pval=pval,
        count=count.astype(np.int64),
        screened_mask=screened_mask,
    )


def choose_signed_score_matrix(
    stats_obj: EdgeStats,
    mode: str,
) -> NDArray[np.float64]:
    if mode == "sr":
        S = stats_obj.sr
    elif mode == "t":
        S = stats_obj.tstat
    elif mode == "mu":
        S = stats_obj.mu
    else:
        raise ValueError(f"Unknown BLOCK_SCORE_STAT: {mode}")
    return np.where(np.isfinite(S), S, np.nan)


def choose_nonnegative_cluster_matrix(
    stats_obj: EdgeStats,
    mode: str,
) -> NDArray[np.float64]:
    if mode == "abs_t":
        A = np.abs(stats_obj.tstat)
    elif mode == "abs_sr":
        A = np.abs(stats_obj.sr)
    elif mode == "positive_sr":
        A = np.maximum(stats_obj.sr, 0.0)
    elif mode == "positive_mu":
        A = np.maximum(stats_obj.mu, 0.0)
    else:
        raise ValueError(f"Unknown CLUSTER_WEIGHT_STAT: {mode}")
    return np.where(np.isfinite(A), A, 0.0)


# =============================================================================
# Dhillon co-clustering
# =============================================================================

@dataclass
class CoClusterResult:
    row_labels: NDArray[np.int64]
    col_labels: NDArray[np.int64]
    row_embedding: NDArray[np.float64]
    col_embedding: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    row_degrees: NDArray[np.float64]
    col_degrees: NDArray[np.float64]


def dhillon_coclustering(
    W: NDArray[np.float64],
    n_clusters: int,
    random_state: int = 0,
) -> CoClusterResult:
    """
    Dhillon multipartition:

        Wn = D_r^{-1/2} W D_c^{-1/2}
        g  = ceil(log2(k))

    Use the top g+1 singular vectors of Wn, drop the trivial first one,
    build the stacked embedding, and run k-means jointly on rows + columns.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be 2D.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    n_rows, n_cols = W.shape
    row_deg = W.sum(axis=1)
    col_deg = W.sum(axis=0)

    eps = 1e-12
    inv_sqrt_row = 1.0 / np.sqrt(np.maximum(row_deg, eps))
    inv_sqrt_col = 1.0 / np.sqrt(np.maximum(col_deg, eps))

    Wn = (inv_sqrt_row[:, None] * W) * inv_sqrt_col[None, :]

    g = int(math.ceil(math.log2(n_clusters)))
    n_components = min(g + 1, min(n_rows, n_cols))

    if n_components < 2:
        row_emb = np.zeros((n_rows, 1), dtype=float)
        col_emb = np.zeros((n_cols, 1), dtype=float)
        Z = np.vstack([row_emb, col_emb])
        km = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
        labels_all = km.fit_predict(Z)
        return CoClusterResult(
            row_labels=labels_all[:n_rows].astype(int),
            col_labels=labels_all[n_rows:].astype(int),
            row_embedding=row_emb,
            col_embedding=col_emb,
            singular_values=np.array([], dtype=float),
            row_degrees=row_deg,
            col_degrees=col_deg,
        )

    U, s, Vt = randomized_svd(
        Wn,
        n_components=n_components,
        n_iter=7,
        random_state=random_state,
    )

    U_use = U[:, 1:n_components]
    V_use = Vt.T[:, 1:n_components]

    row_emb = inv_sqrt_row[:, None] * U_use
    col_emb = inv_sqrt_col[:, None] * V_use
    Z = np.vstack([row_emb, col_emb])

    # Reduce silly warnings by capping k at number of unique embedding rows if needed.
    n_unique = np.unique(np.round(Z, 12), axis=0).shape[0]
    k_eff = min(n_clusters, max(2, n_unique))

    km = KMeans(n_clusters=k_eff, n_init=50, random_state=random_state)
    labels_all = km.fit_predict(Z)

    return CoClusterResult(
        row_labels=labels_all[:n_rows].astype(int),
        col_labels=labels_all[n_rows:].astype(int),
        row_embedding=row_emb,
        col_embedding=col_emb,
        singular_values=s,
        row_degrees=row_deg,
        col_degrees=col_deg,
    )


# =============================================================================
# Block selection and portfolio construction
# =============================================================================

@dataclass
class BlockSelection:
    row_cluster: int
    col_cluster: int
    row_idx: NDArray[np.int64]      # targets
    col_idx: NDArray[np.int64]      # sources
    block_score: float
    source_weights: NDArray[np.float64]
    target_loadings: NDArray[np.float64]
    n_edges_used: int
    mask_name: str


def safe_l1_normalize(x: NDArray[np.float64]) -> NDArray[np.float64]:
    denom = np.sum(np.abs(x))
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(x)
    return x / denom


def select_best_block(
    coclust: CoClusterResult,
    stats_obj: EdgeStats,
    usable_mask: NDArray[np.bool_],
    mask_name: str,
    block_score_stat: str,
    allow_same_cluster_pair: bool = True,
    weight_by_count: bool = True,
) -> BlockSelection:
    """
    Score row-cluster / column-cluster blocks on the SIGNED training matrix.

    Rows = targets
    Cols = sources
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

            good = usable_mask[np.ix_(rows, cols)] & np.isfinite(S[np.ix_(rows, cols)])
            if not np.any(good):
                continue

            sub_S = S[np.ix_(rows, cols)]
            sub_count = stats_obj.count[np.ix_(rows, cols)].astype(float)

            if weight_by_count:
                w = np.where(good, sub_count, 0.0)
                denom = w.sum()
                if denom <= 0:
                    continue
                block_score = float(np.sum(np.where(good, sub_S * w, 0.0)) / denom)
            else:
                block_score = float(np.nanmean(np.where(good, sub_S, np.nan)))

            col_num = np.nansum(np.where(good, sub_S, np.nan), axis=0)
            col_den = np.sum(good, axis=0)
            row_num = np.nansum(np.where(good, sub_S, np.nan), axis=1)
            row_den = np.sum(good, axis=1)

            source_raw = np.divide(col_num, col_den, out=np.zeros_like(col_num), where=col_den > 0)
            target_raw = np.divide(row_num, row_den, out=np.zeros_like(row_num), where=row_den > 0)

            source_weights = safe_l1_normalize(source_raw)
            target_loadings = safe_l1_normalize(target_raw)

            if np.sum(np.abs(source_weights)) == 0 or np.sum(np.abs(target_loadings)) == 0:
                continue

            cand = BlockSelection(
                row_cluster=rc,
                col_cluster=cc,
                row_idx=rows.astype(int),
                col_idx=cols.astype(int),
                block_score=block_score,
                source_weights=source_weights,
                target_loadings=target_loadings,
                n_edges_used=int(np.sum(good)),
                mask_name=mask_name,
            )

            if best is None or (np.isfinite(cand.block_score) and cand.block_score > best.block_score):
                best = cand

    if best is None:
        raise RuntimeError("No valid block found under this mask.")
    return best


def choose_block_with_fallbacks(
    stats_obj: EdgeStats,
    n_clusters: int,
    cluster_weight_stat: str,
    block_score_stat: str,
    allow_same_cluster_pair: bool,
    random_state: int,
    min_count: int,
) -> Tuple[BlockSelection, CoClusterResult]:
    """
    Try progressively looser masks so the strategy stays active:

    1) screened_mask = min_count + BH-FDR
    2) count_only    = min_count only
    3) finite_any    = any finite positive edge
    """
    base_nonneg = choose_nonnegative_cluster_matrix(stats_obj, cluster_weight_stat)

    masks = [
        ("screened", stats_obj.screened_mask),
        ("count_only", (stats_obj.count >= min_count) & np.isfinite(base_nonneg)),
        ("finite_any", np.isfinite(base_nonneg) & (base_nonneg > 0)),
    ]

    last_err: Optional[Exception] = None

    for mask_name, usable_mask in masks:
        if not np.any(usable_mask):
            continue

        W = np.where(usable_mask, base_nonneg, 0.0)
        if not np.any(W > 0):
            continue

        try:
            coclust = dhillon_coclustering(W, n_clusters=n_clusters, random_state=random_state)
            block = select_best_block(
                coclust=coclust,
                stats_obj=stats_obj,
                usable_mask=usable_mask,
                mask_name=mask_name,
                block_score_stat=block_score_stat,
                allow_same_cluster_pair=allow_same_cluster_pair,
                weight_by_count=True,
            )
            return block, coclust
        except Exception as err:
            last_err = err

    if last_err is not None:
        raise last_err
    raise RuntimeError("No usable mask produced a valid block.")


def make_daily_positions(
    lagged_signal_today: pd.Series,
    n_assets: int,
    block: BlockSelection,
) -> NDArray[np.float64]:
    """
    Daily trade using the CURRENT block:

        z_t = sum_{j in S} w_j * P_{t-1,j}
        p_raw_i = b_i * z_t for i in T
        normalize to unit gross leverage

    If some sources are missing today, renormalize over available sources.
    """
    pos = np.zeros(n_assets, dtype=float)

    src = lagged_signal_today.iloc[block.col_idx].to_numpy(dtype=float)
    good = np.isfinite(src)
    if not np.any(good):
        return pos

    src_w = safe_l1_normalize(block.source_weights[good])
    if np.sum(np.abs(src_w)) == 0:
        return pos

    z_t = float(np.dot(src_w, src[good]))
    raw = block.target_loadings * z_t

    gross = np.sum(np.abs(raw))
    if not np.isfinite(gross) or gross <= 0:
        return pos

    pos[block.row_idx] = raw / gross
    return pos


# =============================================================================
# Backtest
# =============================================================================

@dataclass
class DailyRecord:
    date: str
    gross_pnl: float
    net_pnl: float
    turnover: float
    gross_exposure: float
    row_cluster: int
    col_cluster: int
    n_targets: int
    n_sources: int
    block_score: float
    mask_name: str


@dataclass
class BlockHistoryRecord:
    rebalance_date: str
    train_start: str
    train_end: str
    row_cluster: int
    col_cluster: int
    n_targets: int
    n_sources: int
    block_score: float
    n_edges_used: int
    mask_name: str


def summarize_performance(bt: pd.DataFrame) -> dict:
    out: dict = {}

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
        out[f"{col}_mean"] = mu
        out[f"{col}_vol"] = vol
        out[f"{col}_sharpe"] = sharpe
        out[f"{col}_cum"] = float(np.sum(x))

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
    recluster_days: int,
    n_clusters: int,
    fdr_q: float,
    min_count: int,
    cluster_weight_stat: str,
    block_score_stat: str,
    cost_bps: float,
    allow_same_cluster_pair: bool,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Daily trading, but only recompute the block every `recluster_days`.
    """
    dates = signals.index
    n_assets = signals.shape[1]
    lagged_signals = signals.shift(1)

    daily_records: list[DailyRecord] = []
    block_history: list[BlockHistoryRecord] = []

    prev_pos = np.zeros(n_assets, dtype=float)
    cost_rate = cost_bps / 1e4

    current_block: Optional[BlockSelection] = None

    for t in range(train_days, len(dates)):
        need_recluster = (current_block is None) or ((t - train_days) % recluster_days == 0)

        if need_recluster:
            train_slice = slice(t - train_days, t)
            s_train = signals.iloc[train_slice]
            r_train = returns.iloc[train_slice]

            stats_obj = compute_edge_stats(
                signals_train=s_train,
                returns_train=r_train,
                min_count=min_count,
                fdr_q=fdr_q,
            )

            current_block, _ = choose_block_with_fallbacks(
                stats_obj=stats_obj,
                n_clusters=n_clusters,
                cluster_weight_stat=cluster_weight_stat,
                block_score_stat=block_score_stat,
                allow_same_cluster_pair=allow_same_cluster_pair,
                random_state=random_state,
                min_count=min_count,
            )

            block_history.append(
                BlockHistoryRecord(
                    rebalance_date=str(dates[t].date()) if isinstance(dates[t], pd.Timestamp) else str(dates[t]),
                    train_start=str(dates[t - train_days].date()) if isinstance(dates[t - train_days], pd.Timestamp) else str(dates[t - train_days]),
                    train_end=str(dates[t - 1].date()) if isinstance(dates[t - 1], pd.Timestamp) else str(dates[t - 1]),
                    row_cluster=int(current_block.row_cluster),
                    col_cluster=int(current_block.col_cluster),
                    n_targets=int(current_block.row_idx.size),
                    n_sources=int(current_block.col_idx.size),
                    block_score=float(current_block.block_score),
                    n_edges_used=int(current_block.n_edges_used),
                    mask_name=current_block.mask_name,
                )
            )

        assert current_block is not None

        pos = make_daily_positions(
            lagged_signal_today=lagged_signals.iloc[t],
            n_assets=n_assets,
            block=current_block,
        )

        ret = returns.iloc[t].to_numpy(dtype=float)
        gross_pnl = float(np.nansum(np.where(np.isfinite(ret), pos * ret, 0.0)))
        turnover = float(np.sum(np.abs(pos - prev_pos)))
        net_pnl = gross_pnl - cost_rate * turnover
        gross_exposure = float(np.sum(np.abs(pos)))

        daily_records.append(
            DailyRecord(
                date=str(dates[t].date()) if isinstance(dates[t], pd.Timestamp) else str(dates[t]),
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                turnover=turnover,
                gross_exposure=gross_exposure,
                row_cluster=int(current_block.row_cluster),
                col_cluster=int(current_block.col_cluster),
                n_targets=int(current_block.row_idx.size),
                n_sources=int(current_block.col_idx.size),
                block_score=float(current_block.block_score),
                mask_name=current_block.mask_name,
            )
        )

        prev_pos = pos

        if (t - train_days + 1) % 50 == 0:
            print(
                f"Processed {t - train_days + 1}/{len(dates) - train_days} trade days | "
                f"date={daily_records[-1].date} | "
                f"mask={current_block.mask_name} | "
                f"score={current_block.block_score:.6f}"
            )

    bt = pd.DataFrame([asdict(x) for x in daily_records]).set_index("date")
    bh = pd.DataFrame([asdict(x) for x in block_history])

    summary = summarize_performance(bt)
    summary["train_days"] = int(train_days)
    summary["recluster_days"] = int(recluster_days)
    summary["n_clusters"] = int(n_clusters)
    summary["fdr_q"] = float(fdr_q)
    summary["min_count"] = int(min_count)
    summary["cluster_weight_stat"] = cluster_weight_stat
    summary["block_score_stat"] = block_score_stat
    summary["cost_bps"] = float(cost_bps)
    summary["n_assets"] = int(n_assets)
    summary["n_backtest_days"] = int(bt.shape[0])

    return bt, bh, summary


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dhillon bipartite source-target backtest with daily trading and slower reclustering")
    p.add_argument("--signals", default=str(SIGNALS_PATH))
    p.add_argument("--returns", default=str(RETURNS_PATH))
    p.add_argument("--outdir", default=str(OUT_DIR))

    p.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    p.add_argument("--recluster-days", type=int, default=RECLUSTER_DAYS)
    p.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    p.add_argument("--fdr-q", type=float, default=FDR_Q)
    p.add_argument("--min-count", type=int, default=MIN_COUNT)
    p.add_argument("--cluster-weight-stat", choices=["abs_t", "abs_sr", "positive_sr", "positive_mu"], default=CLUSTER_WEIGHT_STAT)
    p.add_argument("--block-score-stat", choices=["sr", "t", "mu"], default=BLOCK_SCORE_STAT)
    p.add_argument("--cost-bps", type=float, default=COST_BPS)
    p.add_argument("--disallow-same-cluster-pair", action="store_true")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    signals_path = Path(args.signals)
    returns_path = Path(args.returns)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = read_matrix(signals_path)
    returns = read_matrix(returns_path)
    signals, returns = align_signal_and_returns(signals, returns)

    if len(signals) <= args.train_days:
        raise ValueError(
            f"Not enough rows for train_days={args.train_days}. "
            f"Need more than {args.train_days}, got {len(signals)}."
        )

    print("Signals shape:", signals.shape)
    print("Returns shape:", returns.shape)
    print("Start date:", signals.index[0])
    print("End date:", signals.index[-1])
    print("Training window:", args.train_days)
    print("Recluster every:", args.recluster_days, "days")
    print("Clusters:", args.n_clusters)
    print("Trading every day:", True)

    bt, bh, summary = run_walk_forward(
        signals=signals,
        returns=returns,
        train_days=int(args.train_days),
        recluster_days=int(args.recluster_days),
        n_clusters=int(args.n_clusters),
        fdr_q=float(args.fdr_q),
        min_count=int(args.min_count),
        cluster_weight_stat=args.cluster_weight_stat,
        block_score_stat=args.block_score_stat,
        cost_bps=float(args.cost_bps),
        allow_same_cluster_pair=not args.disallow_same_cluster_pair,
        random_state=int(args.random_state),
    )

    bt.to_csv(out_dir / "daily_backtest.csv")
    bh.to_csv(out_dir / "daily_block_history.csv", index=False)

    cum = pd.DataFrame(index=bt.index)
    cum["cum_gross_pnl"] = bt["gross_pnl"].cumsum()
    cum["cum_net_pnl"] = bt["net_pnl"].cumsum()
    cum.to_csv(out_dir / "cumulative_pnl.csv")

    summary["signals_path"] = str(signals_path)
    summary["returns_path"] = str(returns_path)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs to:", out_dir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()