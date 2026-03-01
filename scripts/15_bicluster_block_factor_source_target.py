from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# =========================================================
# Basic I/O
# =========================================================

def load_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix at {path}, got shape={arr.shape}")
    return arr


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


# =========================================================
# Edge estimation from R and P
# =========================================================

@dataclass
class EdgeStats:
    n_eff: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    A: np.ndarray
    tstat: np.ndarray
    pval: np.ndarray


def compute_edge_stats_from_RP(
    R: np.ndarray,
    P: np.ndarray,
    *,
    lag: int = 1,
    min_count_for_std: int = 2,
    dtype=np.float64,
    eps: float = 1e-12,
) -> EdgeStats:
    """
    For each ordered pair (i,j), define
        X_t(i,j) = R[t,i] * P[t-lag,j]
    and compute pairwise statistics over the training slice.
    """
    R = np.asarray(R, dtype=dtype)
    P = np.asarray(P, dtype=dtype)

    if R.shape != P.shape:
        raise ValueError(f"R and P must have same shape. Got R={R.shape}, P={P.shape}")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    T, N = R.shape
    if T - lag <= 0:
        raise ValueError(f"No usable observations: T={T}, lag={lag}")

    R1 = R[lag:, :]
    P0 = P[: T - lag, :]

    MR = np.isfinite(R1)
    MP = np.isfinite(P0)

    R1z = np.where(MR, R1, 0.0)
    P0z = np.where(MP, P0, 0.0)

    n_eff = MR.astype(dtype).T @ MP.astype(dtype)
    sum_x = R1z.T @ P0z
    sum_x2 = (R1z * R1z).T @ (P0z * P0z)

    mu = np.zeros((N, N), dtype=dtype)
    ok_mu = n_eff > 0
    mu[ok_mu] = sum_x[ok_mu] / n_eff[ok_mu]

    var = np.zeros((N, N), dtype=dtype)
    ok_var = n_eff >= min_count_for_std
    if np.any(ok_var):
        numer = sum_x2[ok_var] - (sum_x[ok_var] ** 2) / n_eff[ok_var]
        numer = np.maximum(numer, 0.0)
        var[ok_var] = numer / np.maximum(n_eff[ok_var] - 1.0, 1.0)

    sigma = np.sqrt(np.maximum(var, 0.0))

    A = np.zeros((N, N), dtype=dtype)
    tstat = np.zeros((N, N), dtype=dtype)

    good_sigma = sigma > eps
    good_A = ok_mu & good_sigma
    A[good_A] = mu[good_A] / sigma[good_A]

    good_t = ok_var & good_sigma
    tstat[good_t] = mu[good_t] / (sigma[good_t] / np.sqrt(n_eff[good_t]))

    pval = np.ones((N, N), dtype=dtype)
    if np.any(good_t):
        z = np.abs(tstat[good_t]) / math.sqrt(2.0)
        pval[good_t] = np.vectorize(math.erfc)(z)

    np.fill_diagonal(n_eff, 0.0)
    np.fill_diagonal(mu, 0.0)
    np.fill_diagonal(sigma, 0.0)
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(tstat, 0.0)
    np.fill_diagonal(pval, 1.0)

    return EdgeStats(n_eff=n_eff, mu=mu, sigma=sigma, A=A, tstat=tstat, pval=pval)


# =========================================================
# Multiple-testing control (Benjamini-Hochberg)
# =========================================================

def benjamini_hochberg_mask(pvals: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    m = flat.size
    order = np.argsort(flat)
    ps = flat[order]

    thresh = q * (np.arange(1, m + 1) / m)
    keep_sorted = ps <= thresh

    mask_flat = np.zeros(m, dtype=bool)
    if np.any(keep_sorted):
        k = np.max(np.where(keep_sorted)[0])
        mask_flat[order[: k + 1]] = True

    return mask_flat.reshape(p.shape)


# =========================================================
# Sparse rectangular biclustering helpers
# =========================================================

def winsorize_nonzero(W: np.ndarray, upper_q: float = 0.99) -> np.ndarray:
    W = np.asarray(W, dtype=float).copy()
    pos = W[W > 0]
    if pos.size == 0:
        return W
    cap = float(np.quantile(pos, upper_q))
    W[W > cap] = cap
    return W


def top_k_indices(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, x.size)
    idx = np.argpartition(-x, k - 1)[:k]
    idx = idx[np.argsort(-x[idx])]
    return idx.astype(int)


def safe_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = float(np.linalg.norm(x))
    if nrm <= eps:
        return np.zeros_like(x)
    return x / nrm


def safe_l1_normalize_signed(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(np.abs(x)))
    if s <= eps:
        return np.zeros_like(x)
    return x / s


def truncated_positive_l2(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.maximum(x, 0.0)
    if np.count_nonzero(y) == 0:
        return np.zeros_like(y)
    idx = top_k_indices(y, k)
    out = np.zeros_like(y)
    out[idx] = y[idx]
    return safe_l2_normalize(out)


def apply_bipartite_scaling(W: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_sum = W.sum(axis=1)
    col_sum = W.sum(axis=0)

    row_scale = np.zeros_like(row_sum)
    col_scale = np.zeros_like(col_sum)

    row_ok = row_sum > eps
    col_ok = col_sum > eps

    row_scale[row_ok] = 1.0 / np.sqrt(row_sum[row_ok])
    col_scale[col_ok] = 1.0 / np.sqrt(col_sum[col_ok])

    W_scaled = row_scale[:, None] * W * col_scale[None, :]
    return W_scaled, row_scale, col_scale


@dataclass
class SparseRectResult:
    sources: np.ndarray
    targets: np.ndarray
    source_scores: np.ndarray
    target_scores: np.ndarray
    u: np.ndarray
    v: np.ndarray
    objective_value: float
    iterations: int
    converged: bool
    scaled_objective_value: float


def sparse_rectangular_topk(
    W: np.ndarray,
    *,
    ks: int,
    kt: int,
    scale_rows_cols: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    init: str = "degree",
) -> SparseRectResult:
    """
    Sparse rectangular rank-1 selection by alternating truncated bilinear updates.
    Rows correspond to targets, columns to sources.
    The supports of u and v define the target/source sets.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D matrix")

    n_rows, n_cols = W.shape
    if np.count_nonzero(W) == 0:
        zr = np.zeros(n_rows, dtype=float)
        zc = np.zeros(n_cols, dtype=float)
        return SparseRectResult(
            sources=np.array([], dtype=int),
            targets=np.array([], dtype=int),
            source_scores=zc.copy(),
            target_scores=zr.copy(),
            u=zr.copy(),
            v=zc.copy(),
            objective_value=0.0,
            iterations=0,
            converged=True,
            scaled_objective_value=0.0,
        )

    if scale_rows_cols:
        W_work, _, _ = apply_bipartite_scaling(W)
    else:
        W_work = W.copy()

    col_scores_init = W_work.sum(axis=0)
    row_scores_init = W_work.sum(axis=1)

    if init == "svd":
        try:
            _, _, vh = np.linalg.svd(W_work, full_matrices=False)
            v0 = np.abs(vh[0])
        except np.linalg.LinAlgError:
            v0 = np.maximum(col_scores_init, 0.0)
    elif init == "row_degree":
        u0 = np.maximum(row_scores_init, 0.0)
        u = truncated_positive_l2(u0, kt)
        v0 = W_work.T @ u
    else:
        v0 = np.maximum(col_scores_init, 0.0)

    v = truncated_positive_l2(v0, ks)
    if np.count_nonzero(v) == 0:
        zr = np.zeros(n_rows, dtype=float)
        zc = np.zeros(n_cols, dtype=float)
        return SparseRectResult(
            sources=np.array([], dtype=int),
            targets=np.array([], dtype=int),
            source_scores=zc.copy(),
            target_scores=zr.copy(),
            u=zr.copy(),
            v=zc.copy(),
            objective_value=0.0,
            iterations=0,
            converged=True,
            scaled_objective_value=0.0,
        )

    prev_sources = None
    prev_targets = None
    converged = False
    scaled_objective_value = 0.0

    for it in range(1, max_iter + 1):
        u_raw = W_work @ v
        u = truncated_positive_l2(u_raw, kt)

        v_raw = W_work.T @ u
        v_new = truncated_positive_l2(v_raw, ks)

        targets = top_k_indices(u, kt)
        sources = top_k_indices(v_new, ks)
        scaled_objective_value = float(u @ (W_work @ v_new))

        support_stable = (
            prev_sources is not None
            and np.array_equal(sources, prev_sources)
            and np.array_equal(targets, prev_targets)
        )
        vector_stable = float(np.linalg.norm(v_new - v)) <= tol

        v = v_new
        prev_sources = sources.copy()
        prev_targets = targets.copy()

        if support_stable or vector_stable:
            converged = True
            break

    target_scores = W[:, sources].sum(axis=1) if sources.size else np.zeros(n_rows, dtype=float)
    source_scores = W[targets, :].sum(axis=0) if targets.size else np.zeros(n_cols, dtype=float)
    objective_value = float(W[np.ix_(targets, sources)].sum()) if (targets.size and sources.size) else 0.0

    u_final = np.zeros(n_rows, dtype=float)
    v_final = np.zeros(n_cols, dtype=float)
    if targets.size:
        u_final[targets] = target_scores[targets]
        u_final = safe_l2_normalize(u_final)
    if sources.size:
        v_final[sources] = source_scores[sources]
        v_final = safe_l2_normalize(v_final)

    return SparseRectResult(
        sources=sources,
        targets=targets,
        source_scores=source_scores,
        target_scores=target_scores,
        u=u_final,
        v=v_final,
        objective_value=objective_value,
        iterations=it,
        converged=converged,
        scaled_objective_value=scaled_objective_value,
    )


# =========================================================
# Source-target block factor model
# =========================================================

def build_keep_mask(tstat: np.ndarray, n_eff: np.ndarray, *, q: float, min_count: int) -> np.ndarray:
    pval = np.ones_like(tstat, dtype=float)
    good = np.isfinite(tstat)
    z = np.abs(tstat[good]) / math.sqrt(2.0)
    pval[good] = np.vectorize(math.erfc)(z)

    bh_mask = benjamini_hochberg_mask(pval, q=q)
    reliable = n_eff >= float(min_count)
    keep_mask = bh_mask & reliable
    np.fill_diagonal(keep_mask, False)
    return keep_mask


def build_selection_matrix(
    A: np.ndarray,
    tstat: np.ndarray,
    keep_mask: np.ndarray,
    *,
    selection_mode: str,
    winsor_q: float | None,
) -> np.ndarray:
    keep = keep_mask.astype(float)
    if selection_mode == "abs_tstat":
        W = np.abs(tstat) * keep
    elif selection_mode == "abs_A":
        W = np.abs(A) * keep
    elif selection_mode == "positive_tstat":
        W = np.maximum(tstat, 0.0) * keep
    elif selection_mode == "positive_A":
        W = np.maximum(A, 0.0) * keep
    else:
        raise ValueError("selection_mode must be one of: abs_tstat, abs_A, positive_tstat, positive_A")

    np.fill_diagonal(W, 0.0)
    if winsor_q is not None:
        W = winsorize_nonzero(W, upper_q=winsor_q)
    return W


@dataclass
class FactorBlock:
    sources: np.ndarray
    targets: np.ndarray
    source_weights: np.ndarray   # full length N, nonzero only on sources
    target_loadings: np.ndarray  # full length N, nonzero only on targets
    block_matrix: np.ndarray     # signed selected block used for factor fit
    singular_value: float
    target_scores: np.ndarray
    source_scores: np.ndarray


@dataclass
class SelectionResult:
    selection_mode: str
    keep_mask: np.ndarray
    W: np.ndarray
    bicluster: SparseRectResult
    factor_block: FactorBlock


def fit_block_factor_from_selection(
    A: np.ndarray,
    keep_mask: np.ndarray,
    bicluster: SparseRectResult,
    *,
    factor_mode: str = "signed_svd",
) -> FactorBlock:
    A = np.asarray(A, dtype=float)
    N_rows, N_cols = A.shape
    targets = bicluster.targets
    sources = bicluster.sources

    source_weights = np.zeros(N_cols, dtype=float)
    target_loadings = np.zeros(N_rows, dtype=float)

    if targets.size == 0 or sources.size == 0:
        return FactorBlock(
            sources=sources.astype(int),
            targets=targets.astype(int),
            source_weights=source_weights,
            target_loadings=target_loadings,
            block_matrix=np.zeros((targets.size, sources.size), dtype=float),
            singular_value=0.0,
            target_scores=bicluster.target_scores,
            source_scores=bicluster.source_scores,
        )

    block = A[np.ix_(targets, sources)]
    block_keep = keep_mask[np.ix_(targets, sources)]
    block = np.where(block_keep, block, 0.0)

    if np.count_nonzero(block) == 0:
        # Fallback: use support scores with sign from row sums on the signed block.
        w_local = np.ones(sources.size, dtype=float)
        w_local = safe_l1_normalize_signed(w_local)
        b_local = block @ w_local
        if np.sum(np.abs(b_local)) <= 1e-12:
            b_local = np.ones(targets.size, dtype=float)
        b_local = safe_l1_normalize_signed(b_local)
        source_weights[sources] = w_local
        target_loadings[targets] = b_local
        return FactorBlock(
            sources=sources,
            targets=targets,
            source_weights=source_weights,
            target_loadings=target_loadings,
            block_matrix=block,
            singular_value=0.0,
            target_scores=bicluster.target_scores,
            source_scores=bicluster.source_scores,
        )

    if factor_mode == "signed_svd":
        try:
            U, s, Vt = np.linalg.svd(block, full_matrices=False)
            b_local = U[:, 0]
            w_local = Vt[0, :]
            singular_value = float(s[0])
        except np.linalg.LinAlgError:
            w_local = np.sign(block.sum(axis=0)) * np.maximum(np.abs(block).sum(axis=0), 1e-12)
            w_local = safe_l1_normalize_signed(w_local)
            b_local = block @ w_local
            singular_value = float(np.linalg.norm(block, ord="fro"))
    elif factor_mode == "signed_rowcol_sums":
        w_local = np.sign(block.sum(axis=0)) * np.maximum(np.abs(block).sum(axis=0), 1e-12)
        w_local = safe_l1_normalize_signed(w_local)
        b_local = block @ w_local
        singular_value = float(np.linalg.norm(block, ord="fro"))
    else:
        raise ValueError("factor_mode must be one of: signed_svd, signed_rowcol_sums")

    # Global sign flip is irrelevant; orient to make source weights sum nonnegative for consistency.
    if np.sum(w_local) < 0:
        w_local = -w_local
        b_local = -b_local

    w_local = safe_l1_normalize_signed(w_local)
    if np.sum(np.abs(b_local)) <= 1e-12:
        b_local = np.sign(block.sum(axis=1)) * np.maximum(np.abs(block).sum(axis=1), 1e-12)
    b_local = safe_l1_normalize_signed(b_local)

    source_weights[sources] = w_local
    target_loadings[targets] = b_local

    return FactorBlock(
        sources=sources,
        targets=targets,
        source_weights=source_weights,
        target_loadings=target_loadings,
        block_matrix=block,
        singular_value=singular_value,
        target_scores=bicluster.target_scores,
        source_scores=bicluster.source_scores,
    )


def select_block_factor_strategy(
    A: np.ndarray,
    tstat: np.ndarray,
    n_eff: np.ndarray,
    *,
    ks: int,
    kt: int,
    q: float = 0.10,
    min_count: int = 60,
    selection_mode: str = "abs_tstat",
    winsor_q: float | None = 0.99,
    scale_rows_cols: bool = True,
    init: str = "degree",
    factor_mode: str = "signed_svd",
    max_iter: int = 100,
    tol: float = 1e-8,
) -> SelectionResult:
    keep_mask = build_keep_mask(tstat, n_eff, q=q, min_count=min_count)
    W = build_selection_matrix(A, tstat, keep_mask, selection_mode=selection_mode, winsor_q=winsor_q)

    bicluster = sparse_rectangular_topk(
        W,
        ks=ks,
        kt=kt,
        scale_rows_cols=scale_rows_cols,
        max_iter=max_iter,
        tol=tol,
        init=init,
    )

    factor_block = fit_block_factor_from_selection(
        A,
        keep_mask,
        bicluster,
        factor_mode=factor_mode,
    )

    return SelectionResult(
        selection_mode=selection_mode,
        keep_mask=keep_mask,
        W=W,
        bicluster=bicluster,
        factor_block=factor_block,
    )


# =========================================================
# Backtest helpers
# =========================================================

def l1_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(np.abs(x)))
    if s <= eps:
        return np.zeros_like(x)
    return x / s


def build_raw_factor_portfolio(
    lagged_signal: np.ndarray,
    factor_block: FactorBlock,
) -> tuple[np.ndarray, float]:
    """
    Given lagged signal P[t-lag, :], compute:
        z_t = sum_{j in S} w_j P_{t-lag,j}
        raw_p[i] = b_i z_t for i in T
    """
    lagged_signal = np.asarray(lagged_signal, dtype=float)
    source_weights = factor_block.source_weights
    target_loadings = factor_block.target_loadings

    z_t = float(np.dot(source_weights, lagged_signal))
    raw_p = target_loadings * z_t
    return raw_p, z_t


@dataclass
class BacktestResult:
    pnl_gross: np.ndarray
    pnl_net: np.ndarray
    turnover: np.ndarray
    active: np.ndarray
    metrics: dict


def max_drawdown_from_pnl(pnl: np.ndarray) -> float:
    vals = pnl[np.isfinite(pnl)]
    if vals.size == 0:
        return 0.0
    eq = np.cumsum(vals)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq - peak))


def summarize_series(x: np.ndarray, periods_per_year: int = 252) -> dict:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ann_mean": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "cum_pnl": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "n_obs": 0,
        }
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
    ann_mean = mean * periods_per_year
    ann_vol = std * math.sqrt(periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 1e-12 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ann_mean": ann_mean,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "cum_pnl": float(np.sum(vals)),
        "hit_rate": float(np.mean(vals > 0)),
        "max_drawdown": max_drawdown_from_pnl(vals),
        "n_obs": int(vals.size),
    }


def compute_metrics(
    pnl_gross: np.ndarray,
    pnl_net: np.ndarray,
    turnover: np.ndarray,
    active: np.ndarray,
    periods_per_year: int = 252,
) -> dict:
    active = np.asarray(active, dtype=bool)
    tv = turnover[np.isfinite(turnover)]

    return {
        "active_obs": int(np.sum(active)),
        "active_fraction": float(np.mean(active)) if active.size else 0.0,
        "gross": summarize_series(pnl_gross, periods_per_year=periods_per_year),
        "net": summarize_series(pnl_net, periods_per_year=periods_per_year),
        "gross_active_only": summarize_series(pnl_gross[active], periods_per_year=periods_per_year),
        "net_active_only": summarize_series(pnl_net[active], periods_per_year=periods_per_year),
        "turnover": {
            "mean": float(np.mean(tv)) if tv.size else 0.0,
            "median": float(np.median(tv)) if tv.size else 0.0,
            "n_obs": int(tv.size),
        },
    }


def make_walkforward_windows(
    T: int,
    *,
    train_len: int,
    test_len: int,
    step: int,
    lag: int,
    start: int = 0,
    end: int | None = None,
):
    if end is None:
        end = T
    windows = []
    te0 = max(start + train_len, lag)
    while te0 < end:
        tr1 = te0
        tr0 = tr1 - train_len
        te1 = min(te0 + test_len, end)
        if tr0 < start or te1 <= te0:
            break
        windows.append((tr0, tr1, te0, te1))
        te0 += step
    return windows


def run_walkforward(
    R: np.ndarray,
    P: np.ndarray,
    windows: list[tuple[int, int, int, int]],
    *,
    lag: int,
    ks: int,
    kt: int,
    q: float,
    min_count: int,
    selection_mode: str,
    winsor_q: float | None,
    scale_rows_cols: bool,
    init: str,
    factor_mode: str,
    tc_bps: float,
    periods_per_year: int,
    dollar_neutral: bool,
    rebalance_every: int,
    max_iter: int = 100,
    tol: float = 1e-8,
    dtype=np.float64,
) -> tuple[BacktestResult, list[dict]]:
    R = np.asarray(R, dtype=dtype)
    P = np.asarray(P, dtype=dtype)
    T_total, N = R.shape

    pnl_gross = np.full(T_total, np.nan, dtype=float)
    pnl_net = np.full(T_total, np.nan, dtype=float)
    turnover = np.full(T_total, np.nan, dtype=float)
    active = np.zeros(T_total, dtype=bool)

    prev_p = None
    window_reports: list[dict] = []

    for w_idx, (tr0, tr1, te0, te1) in enumerate(windows):
        stats = compute_edge_stats_from_RP(R[tr0:tr1], P[tr0:tr1], lag=lag, dtype=dtype)
        sel = select_block_factor_strategy(
            stats.A,
            stats.tstat,
            stats.n_eff,
            ks=ks,
            kt=kt,
            q=q,
            min_count=min_count,
            selection_mode=selection_mode,
            winsor_q=winsor_q,
            scale_rows_cols=scale_rows_cols,
            init=init,
            factor_mode=factor_mode,
            max_iter=max_iter,
            tol=tol,
        )
        factor_block = sel.factor_block

        start_t = max(te0, lag)
        current_target = None
        current_z = 0.0

        for t in range(start_t, te1):
            if current_target is None or ((t - start_t) % rebalance_every == 0):
                sig = np.nan_to_num(P[t - lag], nan=0.0, posinf=0.0, neginf=0.0)
                raw_p, current_z = build_raw_factor_portfolio(sig, factor_block)
                if dollar_neutral:
                    raw_p = raw_p - float(np.mean(raw_p))
                current_target = l1_normalize(raw_p)

            p = current_target.copy()
            ret = np.nan_to_num(R[t], nan=0.0, posinf=0.0, neginf=0.0)
            gross = float(np.dot(p, ret))

            if prev_p is None:
                turn = float(np.sum(np.abs(p)))
            else:
                turn = float(np.sum(np.abs(p - prev_p)))

            cost = (tc_bps / 1e4) * turn
            net = gross - cost

            pnl_gross[t] = gross
            pnl_net[t] = net
            turnover[t] = turn
            active[t] = bool(np.sum(np.abs(p)) > 1e-12)
            prev_p = p

        S = factor_block.sources
        T = factor_block.targets
        forward_mass = float(sel.W[np.ix_(T, S)].sum()) if (T.size and S.size) else 0.0
        reverse_mass = float(sel.W[np.ix_(S, T)].sum()) if (S.size and T.size and sel.W.shape[0] == sel.W.shape[1]) else 0.0

        window_reports.append({
            "window_index": w_idx,
            "train": [int(tr0), int(tr1)],
            "test": [int(te0), int(te1)],
            "ks": int(ks),
            "kt": int(kt),
            "q": float(q),
            "min_count": int(min_count),
            "selection_mode": selection_mode,
            "winsor_q": None if winsor_q is None else float(winsor_q),
            "scale_rows_cols": int(scale_rows_cols),
            "init": init,
            "factor_mode": factor_mode,
            "sources": S.tolist(),
            "targets": T.tolist(),
            "source_weights": factor_block.source_weights[S].tolist() if S.size else [],
            "target_loadings": factor_block.target_loadings[T].tolist() if T.size else [],
            "objective_value": float(sel.bicluster.objective_value),
            "scaled_objective_value": float(sel.bicluster.scaled_objective_value),
            "iterations": int(sel.bicluster.iterations),
            "converged": int(sel.bicluster.converged),
            "n_kept_edges": int(np.count_nonzero(sel.keep_mask)),
            "n_block_edges": int(np.count_nonzero(sel.keep_mask[np.ix_(T, S)])) if (T.size and S.size) else 0,
            "forward_mass": forward_mass,
            "reverse_mass": reverse_mass,
            "direction_gap": forward_mass - reverse_mass,
            "singular_value": float(factor_block.singular_value),
            "source_weight_l1": float(np.sum(np.abs(factor_block.source_weights))),
            "target_loading_l1": float(np.sum(np.abs(factor_block.target_loadings))),
            "mean_abs_source_weight": float(np.mean(np.abs(factor_block.source_weights[S]))) if S.size else 0.0,
            "mean_abs_target_loading": float(np.mean(np.abs(factor_block.target_loadings[T]))) if T.size else 0.0,
            "last_rebalance_z": float(current_z),
        })

    metrics = compute_metrics(pnl_gross, pnl_net, turnover, active, periods_per_year=periods_per_year)
    return BacktestResult(pnl_gross, pnl_net, turnover, active, metrics), window_reports


# =========================================================
# Main CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(
        description="Biclustered block-factor directed source-target pipeline for OFI -> return"
    )
    ap.add_argument("--R", type=str, required=True, help="Path to R.npy")
    ap.add_argument("--P", type=str, required=True, help="Path to P.npy")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")

    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--ks", type=int, default=25, help="number of source assets")
    ap.add_argument("--kt", type=int, default=25, help="number of target assets")
    ap.add_argument("--q", type=float, default=0.10, help="BH-FDR level")
    ap.add_argument("--min_count", type=int, default=60, help="minimum n_eff per edge")
    ap.add_argument(
        "--selection_mode",
        type=str,
        default="abs_tstat",
        choices=["abs_tstat", "abs_A", "positive_tstat", "positive_A"],
        help="how to build the nonnegative biclustering matrix",
    )
    ap.add_argument(
        "--factor_mode",
        type=str,
        default="signed_svd",
        choices=["signed_svd", "signed_rowcol_sums"],
        help="how to fit signed source weights and target loadings inside the selected block",
    )
    ap.add_argument("--winsor_q", type=float, default=0.99, help="winsorization quantile for positive weights; set <=0 to disable")
    ap.add_argument("--no_scale_rows_cols", action="store_true", help="disable row/column scaling before sparse rectangular updates")
    ap.add_argument("--init", type=str, default="degree", choices=["degree", "row_degree", "svd"])
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-8)

    ap.add_argument("--train_len", type=int, default=1260)
    ap.add_argument("--test_len", type=int, default=63)
    ap.add_argument("--step", type=int, default=63)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)

    ap.add_argument("--tc_bps", type=float, default=0.0)
    ap.add_argument("--periods_per_year", type=int, default=252)
    ap.add_argument("--dollar_neutral", action="store_true")
    ap.add_argument("--rebalance_every", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])

    args = ap.parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64
    winsor_q = None if args.winsor_q <= 0 else float(args.winsor_q)

    R = load_matrix(args.R)
    P = load_matrix(args.P)
    if R.shape != P.shape:
        raise ValueError(f"R and P shape mismatch: R={R.shape}, P={P.shape}")

    T, _ = R.shape
    windows = make_walkforward_windows(
        T,
        train_len=args.train_len,
        test_len=args.test_len,
        step=args.step,
        lag=args.lag,
        start=args.start,
        end=args.end,
    )
    if not windows:
        raise ValueError("No walk-forward windows created. Check train/test lengths.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result, window_reports = run_walkforward(
        R,
        P,
        windows,
        lag=args.lag,
        ks=args.ks,
        kt=args.kt,
        q=args.q,
        min_count=args.min_count,
        selection_mode=args.selection_mode,
        winsor_q=winsor_q,
        scale_rows_cols=not args.no_scale_rows_cols,
        init=args.init,
        factor_mode=args.factor_mode,
        tc_bps=args.tc_bps,
        periods_per_year=args.periods_per_year,
        dollar_neutral=args.dollar_neutral,
        rebalance_every=args.rebalance_every,
        max_iter=args.max_iter,
        tol=args.tol,
        dtype=dtype,
    )

    save_npy(outdir / "pnl_gross.npy", result.pnl_gross)
    save_npy(outdir / "pnl_net.npy", result.pnl_net)
    save_npy(outdir / "turnover.npy", result.turnover)
    save_npy(outdir / "active.npy", result.active.astype(np.uint8))

    (outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    (outdir / "window_reports.json").write_text(json.dumps(window_reports, indent=2))

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "active_obs",
                "active_fraction",
                "gross_sharpe",
                "gross_ann_mean",
                "gross_ann_vol",
                "gross_cum_pnl",
                "gross_active_only_sharpe",
                "net_sharpe",
                "net_ann_mean",
                "net_ann_vol",
                "net_cum_pnl",
                "net_active_only_sharpe",
                "turnover_mean",
                "turnover_median",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "active_obs": result.metrics["active_obs"],
                "active_fraction": result.metrics["active_fraction"],
                "gross_sharpe": result.metrics["gross"]["sharpe"],
                "gross_ann_mean": result.metrics["gross"]["ann_mean"],
                "gross_ann_vol": result.metrics["gross"]["ann_vol"],
                "gross_cum_pnl": result.metrics["gross"]["cum_pnl"],
                "gross_active_only_sharpe": result.metrics["gross_active_only"]["sharpe"],
                "net_sharpe": result.metrics["net"]["sharpe"],
                "net_ann_mean": result.metrics["net"]["ann_mean"],
                "net_ann_vol": result.metrics["net"]["ann_vol"],
                "net_cum_pnl": result.metrics["net"]["cum_pnl"],
                "net_active_only_sharpe": result.metrics["net_active_only"]["sharpe"],
                "turnover_mean": result.metrics["turnover"]["mean"],
                "turnover_median": result.metrics["turnover"]["median"],
            }
        )

    print(f"Saved outputs to: {outdir}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
