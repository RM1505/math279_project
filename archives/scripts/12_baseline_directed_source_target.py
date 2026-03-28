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
    n_eff: np.ndarray   # (N,N)
    mu: np.ndarray      # mean of X_t(i,j)=R[t,i]*P[t-lag,j]
    sigma: np.ndarray   # std of X_t(i,j)
    A: np.ndarray       # Sharpe-like edge score = mu/sigma
    tstat: np.ndarray   # t-stat = mu / (sigma / sqrt(n_eff))
    pval: np.ndarray    # approximate two-sided p-value from tstat


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
    For each ordered pair (i,j), define:
        X_t(i,j) = R[t,i] * P[t-lag,j]

    Then compute pairwise finite-data statistics on the training slice only.
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

    # Align so row s in these arrays corresponds to original time t = s + lag
    R1 = R[lag:, :]     # returns at time t
    P0 = P[:T-lag, :]   # signals at time t-lag

    MR = np.isfinite(R1)
    MP = np.isfinite(P0)

    R1z = np.where(MR, R1, 0.0)
    P0z = np.where(MP, P0, 0.0)

    # Pairwise valid sample counts
    n_eff = MR.astype(dtype).T @ MP.astype(dtype)   # (N,N)

    # Pairwise sums of X and X^2
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

    # Approximate two-sided p-values using a normal tail.
    # This is simple and usually reasonable once n_eff is not tiny.
    pval = np.ones((N, N), dtype=dtype)
    if np.any(good_t):
        z = np.abs(tstat[good_t]) / math.sqrt(2.0)
        pval[good_t] = np.vectorize(math.erfc)(z)

    # Remove self-edges.
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
    """
    BH-FDR mask on a matrix of p-values.
    """
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
        mask_flat[order[:k + 1]] = True

    return mask_flat.reshape(p.shape)


# =========================================================
# Directed source-target selection
# =========================================================

def winsorize_nonzero(W: np.ndarray, upper_q: float = 0.99) -> np.ndarray:
    """
    Clamp only the positive entries of W to reduce domination by a few edges.
    """
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


@dataclass
class SelectionResult:
    sources: np.ndarray   # column indices S
    targets: np.ndarray   # row indices T
    source_scores: np.ndarray
    target_scores: np.ndarray
    objective_value: float
    iterations: int
    W: np.ndarray         # nonnegative selection matrix
    keep_mask: np.ndarray # significance / reliability mask


def select_sources_targets(
    A: np.ndarray,
    tstat: np.ndarray,
    n_eff: np.ndarray,
    *,
    ks: int,
    kt: int,
    q: float = 0.10,
    min_count: int = 60,
    weight_mode: str = "abs_tstat",
    winsor_q: float | None = 0.99,
    max_iter: int = 50,
    use_bh: bool = True,
    tstat_min: float = 0.0,
) -> SelectionResult:
    """
    Very simple directed bicluster heuristic.

    Edge filtering (two modes):
      use_bh=True  (default): BH-FDR at level q, plus n_eff >= min_count, plus |t| >= tstat_min.
                              With N=514 assets (263K edges), BH requires |t|≈5 for any edge to
                              survive — very aggressive. Use tstat_min as an additional floor.
      use_bh=False           : Skip BH entirely. Keep edges where |t| >= tstat_min AND
                              n_eff >= min_count. At tstat_min=4.0 with 263K tests, expected
                              false positives ≈ 17 (gaussian tail). Much more liberal than BH.

    Then:
    1. Build a NON-SYMMETRIC nonnegative weight matrix W from surviving edges.
    2. Alternating top-k updates:
          S <- best ks columns given current T
          T <- best kt rows given current S
       until convergence (approximately maximises sum_{i in T, j in S} W[i,j]).
    """
    A = np.asarray(A, dtype=float)
    tstat = np.asarray(tstat, dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)
    N = A.shape[0]

    if A.shape != (N, N) or tstat.shape != (N, N) or n_eff.shape != (N, N):
        raise ValueError("A, tstat, n_eff must all be square (N,N) arrays")

    reliable = n_eff >= float(min_count)
    tstat_hard = np.abs(tstat) >= tstat_min

    if use_bh:
        # BH-FDR on p-values derived from the normal approximation to t-stat
        pval = np.ones_like(tstat, dtype=float)
        good = np.isfinite(tstat)
        z = np.abs(tstat[good]) / math.sqrt(2.0)
        pval[good] = np.vectorize(math.erfc)(z)
        bh_mask = benjamini_hochberg_mask(pval, q=q)
        keep_mask = bh_mask & reliable & tstat_hard
    else:
        # Direct t-stat threshold — no BH penalty for multiple testing.
        # Expectation of false positives under H0 with N*(N-1) tests:
        #   FP ≈ N*(N-1) * erfc(tstat_min / sqrt(2))
        keep_mask = reliable & tstat_hard

    np.fill_diagonal(keep_mask, False)

    if weight_mode == "abs_tstat":
        W = np.abs(tstat) * keep_mask
    elif weight_mode == "abs_A":
        W = np.abs(A) * keep_mask
    elif weight_mode == "positive_A":
        W = np.maximum(A, 0.0) * keep_mask
    else:
        raise ValueError("weight_mode must be one of: abs_tstat, abs_A, positive_A")

    np.fill_diagonal(W, 0.0)

    if winsor_q is not None:
        W = winsorize_nonzero(W, upper_q=winsor_q)

    # No signal survived.
    if np.count_nonzero(W) == 0:
        sources = np.arange(min(ks, N), dtype=int)
        targets = np.arange(min(kt, N), dtype=int)
        return SelectionResult(
            sources=sources,
            targets=targets,
            source_scores=np.zeros(N),
            target_scores=np.zeros(N),
            objective_value=0.0,
            iterations=0,
            W=W,
            keep_mask=keep_mask,
        )

    # Initialize with strongest columns.
    col_scores = W.sum(axis=0)
    sources = top_k_indices(col_scores, ks)
    prev_sources = None
    prev_targets = None

    for it in range(1, max_iter + 1):
        # Best targets given current sources: rows with largest incoming mass from S.
        target_scores = W[:, sources].sum(axis=1)
        targets = top_k_indices(target_scores, kt)

        # Best sources given current targets: columns with largest outgoing mass into T.
        source_scores = W[targets, :].sum(axis=0)
        sources = top_k_indices(source_scores, ks)

        if prev_sources is not None and np.array_equal(sources, prev_sources) and np.array_equal(targets, prev_targets):
            break
        prev_sources = sources.copy()
        prev_targets = targets.copy()

    objective_value = float(W[np.ix_(targets, sources)].sum())

    return SelectionResult(
        sources=sources,
        targets=targets,
        source_scores=source_scores,
        target_scores=target_scores,
        objective_value=objective_value,
        iterations=it,
        W=W,
        keep_mask=keep_mask,
    )


# =========================================================
# Trading matrix and backtest
# =========================================================

def build_trading_matrix(A: np.ndarray, sources: np.ndarray, targets: np.ndarray, keep_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Build signed trading matrix M.
    Only rows in targets and columns in sources are kept.
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    M = np.zeros((N, N), dtype=float)
    M[np.ix_(targets, sources)] = A[np.ix_(targets, sources)]
    if keep_mask is not None:
        M = np.where(keep_mask, M, 0.0)
    np.fill_diagonal(M, 0.0)
    return M


def l1_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(np.abs(x)))
    if s <= eps:
        return np.zeros_like(x)
    return x / s


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


def compute_metrics(pnl_gross: np.ndarray, pnl_net: np.ndarray, turnover: np.ndarray, active: np.ndarray, periods_per_year: int = 252) -> dict:
    def summarize(x: np.ndarray) -> dict:
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

    tv = turnover[np.isfinite(turnover)]
    return {
        "active_obs": int(np.sum(active)),
        "gross": summarize(pnl_gross),
        "net": summarize(pnl_net),
        "turnover": {
            "mean": float(np.mean(tv)) if tv.size else 0.0,
            "median": float(np.median(tv)) if tv.size else 0.0,
            "n_obs": int(tv.size),
        },
    }


def make_walkforward_windows(T: int, *, train_len: int, test_len: int, step: int, lag: int, start: int = 0, end: int | None = None):
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
    weight_mode: str,
    tc_bps: float,
    periods_per_year: int,
    dollar_neutral: bool,
    rebalance_every: int,
    use_bh: bool = True,
    tstat_min: float = 0.0,
    dtype=np.float64,
) -> tuple[BacktestResult, list[dict]]:
    R = np.asarray(R, dtype=dtype)
    P = np.asarray(P, dtype=dtype)
    T, N = R.shape

    pnl_gross = np.full(T, np.nan, dtype=float)
    pnl_net = np.full(T, np.nan, dtype=float)
    turnover = np.full(T, np.nan, dtype=float)
    active = np.zeros(T, dtype=bool)

    prev_p = None
    window_reports: list[dict] = []

    for w_idx, (tr0, tr1, te0, te1) in enumerate(windows):
        stats = compute_edge_stats_from_RP(R[tr0:tr1], P[tr0:tr1], lag=lag, dtype=dtype)
        sel = select_sources_targets(
            stats.A,
            stats.tstat,
            stats.n_eff,
            ks=ks,
            kt=kt,
            q=q,
            min_count=min_count,
            weight_mode=weight_mode,
            use_bh=use_bh,
            tstat_min=tstat_min,
        )
        M = build_trading_matrix(stats.A, sel.sources, sel.targets, keep_mask=sel.keep_mask)

        start_t = max(te0, lag)
        current_target = None

        for t in range(start_t, te1):
            if current_target is None or ((t - start_t) % rebalance_every == 0):
                sig = np.nan_to_num(P[t - lag], nan=0.0, posinf=0.0, neginf=0.0)
                raw_p = M @ sig
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
            active[t] = True
            prev_p = p

        reverse_mass = float(sel.W[np.ix_(sel.sources, sel.targets)].sum())
        forward_mass = float(sel.W[np.ix_(sel.targets, sel.sources)].sum())
        window_reports.append({
            "window_index": w_idx,
            "train": [int(tr0), int(tr1)],
            "test": [int(te0), int(te1)],
            "ks": int(ks),
            "kt": int(kt),
            "q": float(q),
            "min_count": int(min_count),
            "weight_mode": weight_mode,
            "sources": sel.sources.tolist(),
            "targets": sel.targets.tolist(),
            "objective_value": float(sel.objective_value),
            "iterations": int(sel.iterations),
            "n_kept_edges": int(np.count_nonzero(sel.keep_mask)),
            "n_strategy_edges": int(np.count_nonzero(M)),
            "forward_mass": forward_mass,
            "reverse_mass": reverse_mass,
            "direction_gap": forward_mass - reverse_mass,
        })

    metrics = compute_metrics(pnl_gross, pnl_net, turnover, active, periods_per_year=periods_per_year)
    return BacktestResult(pnl_gross, pnl_net, turnover, active, metrics), window_reports


# =========================================================
# Main CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Directed source-target lead-lag pipeline for OFI -> return")
    ap.add_argument("--R", type=str, required=True, help="Path to R.npy")
    ap.add_argument("--P", type=str, required=True, help="Path to P.npy")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")

    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--ks", type=int, default=25, help="number of source assets")
    ap.add_argument("--kt", type=int, default=25, help="number of target assets")
    ap.add_argument("--q", type=float, default=0.10, help="BH-FDR level")
    ap.add_argument("--min_count", type=int, default=60, help="minimum n_eff per edge")
    ap.add_argument("--weight_mode", type=str, default="abs_tstat", choices=["abs_tstat", "abs_A", "positive_A"])

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

    R = load_matrix(args.R)
    P = load_matrix(args.P)
    if R.shape != P.shape:
        raise ValueError(f"R and P shape mismatch: R={R.shape}, P={P.shape}")

    T, N = R.shape
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
        weight_mode=args.weight_mode,
        tc_bps=args.tc_bps,
        periods_per_year=args.periods_per_year,
        dollar_neutral=args.dollar_neutral,
        rebalance_every=args.rebalance_every,
        dtype=dtype,
    )

    save_npy(outdir / "pnl_gross.npy", result.pnl_gross)
    save_npy(outdir / "pnl_net.npy", result.pnl_net)
    save_npy(outdir / "turnover.npy", result.turnover)
    save_npy(outdir / "active.npy", result.active.astype(np.uint8))

    (outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    (outdir / "window_reports.json").write_text(json.dumps(window_reports, indent=2))

    # Small summary CSV
    csv_path = outdir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "active_obs",
            "gross_sharpe",
            "gross_ann_mean",
            "gross_ann_vol",
            "gross_cum_pnl",
            "net_sharpe",
            "net_ann_mean",
            "net_ann_vol",
            "net_cum_pnl",
            "turnover_mean",
            "turnover_median",
        ])
        writer.writeheader()
        writer.writerow({
            "active_obs": result.metrics["active_obs"],
            "gross_sharpe": result.metrics["gross"]["sharpe"],
            "gross_ann_mean": result.metrics["gross"]["ann_mean"],
            "gross_ann_vol": result.metrics["gross"]["ann_vol"],
            "gross_cum_pnl": result.metrics["gross"]["cum_pnl"],
            "net_sharpe": result.metrics["net"]["sharpe"],
            "net_ann_mean": result.metrics["net"]["ann_mean"],
            "net_ann_vol": result.metrics["net"]["ann_vol"],
            "net_cum_pnl": result.metrics["net"]["cum_pnl"],
            "turnover_mean": result.metrics["turnover"]["mean"],
            "turnover_median": result.metrics["turnover"]["median"],
        })

    print(f"Saved outputs to: {outdir}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()


# from __future__ import annotations

# import argparse
# import csv
# import json
# import math
# from dataclasses import dataclass
# from pathlib import Path

# import numpy as np


# # =========================================================
# # Basic I/O
# # =========================================================

# def load_matrix(path: str | Path) -> np.ndarray:
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"Missing file: {path}")
#     arr = np.load(path)
#     if arr.ndim != 2:
#         raise ValueError(f"Expected a 2D matrix at {path}, got shape={arr.shape}")
#     return arr


# def save_npy(path: str | Path, arr: np.ndarray) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     np.save(path, arr)


# # =========================================================
# # Edge estimation from R and P
# # =========================================================

# @dataclass
# class EdgeStats:
#     n_eff: np.ndarray   # (N,N)
#     mu: np.ndarray      # mean of X_t(i,j)=R[t,i]*P[t-lag,j]
#     sigma: np.ndarray   # std of X_t(i,j)
#     A: np.ndarray       # Sharpe-like edge score = mu/sigma
#     tstat: np.ndarray   # t-stat = mu / (sigma / sqrt(n_eff))
#     pval: np.ndarray    # approximate two-sided p-value from tstat


# def compute_edge_stats_from_RP(
#     R: np.ndarray,
#     P: np.ndarray,
#     *,
#     lag: int = 1,
#     min_count_for_std: int = 2,
#     dtype=np.float64,
#     eps: float = 1e-12,
# ) -> EdgeStats:
#     """
#     For each ordered pair (i,j), define:
#         X_t(i,j) = R[t,i] * P[t-lag,j]

#     Then compute pairwise finite-data statistics on the training slice only.
#     """
#     R = np.asarray(R, dtype=dtype)
#     P = np.asarray(P, dtype=dtype)

#     if R.shape != P.shape:
#         raise ValueError(f"R and P must have same shape. Got R={R.shape}, P={P.shape}")
#     if lag < 0:
#         raise ValueError("lag must be >= 0")

#     T, N = R.shape
#     if T - lag <= 0:
#         raise ValueError(f"No usable observations: T={T}, lag={lag}")

#     # Align so row s in these arrays corresponds to original time t = s + lag
#     R1 = R[lag:, :]     # returns at time t
#     P0 = P[:T-lag, :]   # signals at time t-lag

#     MR = np.isfinite(R1)
#     MP = np.isfinite(P0)

#     R1z = np.where(MR, R1, 0.0)
#     P0z = np.where(MP, P0, 0.0)

#     # Pairwise valid sample counts
#     n_eff = MR.astype(dtype).T @ MP.astype(dtype)   # (N,N)

#     # Pairwise sums of X and X^2
#     sum_x = R1z.T @ P0z
#     sum_x2 = (R1z * R1z).T @ (P0z * P0z)

#     mu = np.zeros((N, N), dtype=dtype)
#     ok_mu = n_eff > 0
#     mu[ok_mu] = sum_x[ok_mu] / n_eff[ok_mu]

#     var = np.zeros((N, N), dtype=dtype)
#     ok_var = n_eff >= min_count_for_std
#     if np.any(ok_var):
#         numer = sum_x2[ok_var] - (sum_x[ok_var] ** 2) / n_eff[ok_var]
#         numer = np.maximum(numer, 0.0)
#         var[ok_var] = numer / np.maximum(n_eff[ok_var] - 1.0, 1.0)

#     sigma = np.sqrt(np.maximum(var, 0.0))

#     A = np.zeros((N, N), dtype=dtype)
#     tstat = np.zeros((N, N), dtype=dtype)

#     good_sigma = sigma > eps
#     good_A = ok_mu & good_sigma
#     A[good_A] = mu[good_A] / sigma[good_A]

#     good_t = ok_var & good_sigma
#     tstat[good_t] = mu[good_t] / (sigma[good_t] / np.sqrt(n_eff[good_t]))

#     # Approximate two-sided p-values using a normal tail.
#     # This is simple and usually reasonable once n_eff is not tiny.
#     pval = np.ones((N, N), dtype=dtype)
#     if np.any(good_t):
#         z = np.abs(tstat[good_t]) / math.sqrt(2.0)
#         pval[good_t] = np.vectorize(math.erfc)(z)

#     # Remove self-edges.
#     np.fill_diagonal(n_eff, 0.0)
#     np.fill_diagonal(mu, 0.0)
#     np.fill_diagonal(sigma, 0.0)
#     np.fill_diagonal(A, 0.0)
#     np.fill_diagonal(tstat, 0.0)
#     np.fill_diagonal(pval, 1.0)

#     return EdgeStats(n_eff=n_eff, mu=mu, sigma=sigma, A=A, tstat=tstat, pval=pval)


# # =========================================================
# # Multiple-testing control (Benjamini-Hochberg)
# # =========================================================

# def benjamini_hochberg_mask(pvals: np.ndarray, q: float) -> np.ndarray:
#     """
#     BH-FDR mask on a matrix of p-values.
#     """
#     p = np.asarray(pvals, dtype=float)
#     flat = p.ravel()
#     m = flat.size
#     order = np.argsort(flat)
#     ps = flat[order]

#     thresh = q * (np.arange(1, m + 1) / m)
#     keep_sorted = ps <= thresh

#     mask_flat = np.zeros(m, dtype=bool)
#     if np.any(keep_sorted):
#         k = np.max(np.where(keep_sorted)[0])
#         mask_flat[order[:k + 1]] = True

#     return mask_flat.reshape(p.shape)


# # =========================================================
# # Directed source-target selection
# # =========================================================

# def winsorize_nonzero(W: np.ndarray, upper_q: float = 0.99) -> np.ndarray:
#     """
#     Clamp only the positive entries of W to reduce domination by a few edges.
#     """
#     W = np.asarray(W, dtype=float).copy()
#     pos = W[W > 0]
#     if pos.size == 0:
#         return W
#     cap = float(np.quantile(pos, upper_q))
#     W[W > cap] = cap
#     return W


# def top_k_indices(x: np.ndarray, k: int) -> np.ndarray:
#     if k <= 0:
#         return np.array([], dtype=int)
#     k = min(k, x.size)
#     idx = np.argpartition(-x, k - 1)[:k]
#     idx = idx[np.argsort(-x[idx])]
#     return idx.astype(int)


# @dataclass
# class SelectionResult:
#     sources: np.ndarray   # column indices S
#     targets: np.ndarray   # row indices T
#     source_scores: np.ndarray
#     target_scores: np.ndarray
#     objective_value: float
#     iterations: int
#     W: np.ndarray         # nonnegative selection matrix
#     keep_mask: np.ndarray # significance / reliability mask


# def select_sources_targets(
#     A: np.ndarray,
#     tstat: np.ndarray,
#     n_eff: np.ndarray,
#     *,
#     ks: int,
#     kt: int,
#     q: float = 0.10,
#     min_count: int = 60,
#     weight_mode: str = "abs_tstat",
#     winsor_q: float | None = 0.99,
#     max_iter: int = 50,
# ) -> SelectionResult:
#     """
#     Very simple directed bicluster heuristic.

#     1. Keep only statistically credible edges using:
#           n_eff >= min_count  and  BH-FDR on p-values from t-stats.
#     2. Build a NON-SYMMETRIC nonnegative weight matrix W.
#     3. Alternating top-k updates:
#           S <- best ks columns given current T
#           T <- best kt rows given current S
#        until convergence.

#     This approximately maximizes the total directed weight from sources S to targets T:
#           sum_{i in T, j in S} W[i,j]
#     """
#     A = np.asarray(A, dtype=float)
#     tstat = np.asarray(tstat, dtype=float)
#     n_eff = np.asarray(n_eff, dtype=float)
#     N = A.shape[0]

#     if A.shape != (N, N) or tstat.shape != (N, N) or n_eff.shape != (N, N):
#         raise ValueError("A, tstat, n_eff must all be square (N,N) arrays")

#     # p-values from t-stat, using same normal approximation as above
#     pval = np.ones_like(tstat, dtype=float)
#     good = np.isfinite(tstat)
#     z = np.abs(tstat[good]) / math.sqrt(2.0)
#     pval[good] = np.vectorize(math.erfc)(z)

#     bh_mask = benjamini_hochberg_mask(pval, q=q)
#     reliable = n_eff >= float(min_count)
#     keep_mask = bh_mask & reliable
#     np.fill_diagonal(keep_mask, False)

#     if weight_mode == "abs_tstat":
#         W = np.abs(tstat) * keep_mask
#     elif weight_mode == "abs_A":
#         W = np.abs(A) * keep_mask
#     elif weight_mode == "positive_A":
#         W = np.maximum(A, 0.0) * keep_mask
#     else:
#         raise ValueError("weight_mode must be one of: abs_tstat, abs_A, positive_A")

#     np.fill_diagonal(W, 0.0)

#     if winsor_q is not None:
#         W = winsorize_nonzero(W, upper_q=winsor_q)

#     # No signal survived.
#     if np.count_nonzero(W) == 0:
#         sources = np.arange(min(ks, N), dtype=int)
#         targets = np.arange(min(kt, N), dtype=int)
#         return SelectionResult(
#             sources=sources,
#             targets=targets,
#             source_scores=np.zeros(N),
#             target_scores=np.zeros(N),
#             objective_value=0.0,
#             iterations=0,
#             W=W,
#             keep_mask=keep_mask,
#         )

#     # Initialize with strongest columns.
#     col_scores = W.sum(axis=0)
#     sources = top_k_indices(col_scores, ks)
#     prev_sources = None
#     prev_targets = None

#     for it in range(1, max_iter + 1):
#         # Best targets given current sources: rows with largest incoming mass from S.
#         target_scores = W[:, sources].sum(axis=1)
#         targets = top_k_indices(target_scores, kt)

#         # Best sources given current targets: columns with largest outgoing mass into T.
#         source_scores = W[targets, :].sum(axis=0)
#         sources = top_k_indices(source_scores, ks)

#         if prev_sources is not None and np.array_equal(sources, prev_sources) and np.array_equal(targets, prev_targets):
#             break
#         prev_sources = sources.copy()
#         prev_targets = targets.copy()

#     objective_value = float(W[np.ix_(targets, sources)].sum())

#     return SelectionResult(
#         sources=sources,
#         targets=targets,
#         source_scores=source_scores,
#         target_scores=target_scores,
#         objective_value=objective_value,
#         iterations=it,
#         W=W,
#         keep_mask=keep_mask,
#     )


# # =========================================================
# # Trading matrix and backtest
# # =========================================================

# def build_trading_matrix(A: np.ndarray, sources: np.ndarray, targets: np.ndarray, keep_mask: np.ndarray | None = None) -> np.ndarray:
#     """
#     Build signed trading matrix M.
#     Only rows in targets and columns in sources are kept.
#     """
#     A = np.asarray(A, dtype=float)
#     N = A.shape[0]
#     M = np.zeros((N, N), dtype=float)
#     M[np.ix_(targets, sources)] = A[np.ix_(targets, sources)]
#     if keep_mask is not None:
#         M = np.where(keep_mask, M, 0.0)
#     np.fill_diagonal(M, 0.0)
#     return M


# def l1_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     s = float(np.sum(np.abs(x)))
#     if s <= eps:
#         return np.zeros_like(x)
#     return x / s


# @dataclass
# class BacktestResult:
#     pnl_gross: np.ndarray
#     pnl_net: np.ndarray
#     turnover: np.ndarray
#     active: np.ndarray
#     metrics: dict


# def max_drawdown_from_pnl(pnl: np.ndarray) -> float:
#     vals = pnl[np.isfinite(pnl)]
#     if vals.size == 0:
#         return 0.0
#     eq = np.cumsum(vals)
#     peak = np.maximum.accumulate(eq)
#     return float(np.min(eq - peak))


# def compute_metrics(pnl_gross: np.ndarray, pnl_net: np.ndarray, turnover: np.ndarray, active: np.ndarray, periods_per_year: int = 252) -> dict:
#     def summarize(x: np.ndarray) -> dict:
#         vals = x[np.isfinite(x)]
#         if vals.size == 0:
#             return {
#                 "mean": 0.0,
#                 "std": 0.0,
#                 "ann_mean": 0.0,
#                 "ann_vol": 0.0,
#                 "sharpe": 0.0,
#                 "cum_pnl": 0.0,
#                 "hit_rate": 0.0,
#                 "max_drawdown": 0.0,
#                 "n_obs": 0,
#             }
#         mean = float(np.mean(vals))
#         std = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
#         ann_mean = mean * periods_per_year
#         ann_vol = std * math.sqrt(periods_per_year)
#         sharpe = ann_mean / ann_vol if ann_vol > 1e-12 else 0.0
#         return {
#             "mean": mean,
#             "std": std,
#             "ann_mean": ann_mean,
#             "ann_vol": ann_vol,
#             "sharpe": sharpe,
#             "cum_pnl": float(np.sum(vals)),
#             "hit_rate": float(np.mean(vals > 0)),
#             "max_drawdown": max_drawdown_from_pnl(vals),
#             "n_obs": int(vals.size),
#         }

#     tv = turnover[np.isfinite(turnover)]
#     return {
#         "active_obs": int(np.sum(active)),
#         "gross": summarize(pnl_gross),
#         "net": summarize(pnl_net),
#         "turnover": {
#             "mean": float(np.mean(tv)) if tv.size else 0.0,
#             "median": float(np.median(tv)) if tv.size else 0.0,
#             "n_obs": int(tv.size),
#         },
#     }


# def make_walkforward_windows(T: int, *, train_len: int, test_len: int, step: int, lag: int, start: int = 0, end: int | None = None):
#     if end is None:
#         end = T
#     windows = []
#     te0 = max(start + train_len, lag)
#     while te0 < end:
#         tr1 = te0
#         tr0 = tr1 - train_len
#         te1 = min(te0 + test_len, end)
#         if tr0 < start or te1 <= te0:
#             break
#         windows.append((tr0, tr1, te0, te1))
#         te0 += step
#     return windows


# def run_walkforward(
#     R: np.ndarray,
#     P: np.ndarray,
#     windows: list[tuple[int, int, int, int]],
#     *,
#     lag: int,
#     ks: int,
#     kt: int,
#     q: float,
#     min_count: int,
#     weight_mode: str,
#     tc_bps: float,
#     periods_per_year: int,
#     dollar_neutral: bool,
#     rebalance_every: int,
#     dtype=np.float64,
# ) -> tuple[BacktestResult, list[dict]]:
#     R = np.asarray(R, dtype=dtype)
#     P = np.asarray(P, dtype=dtype)
#     T, N = R.shape

#     pnl_gross = np.full(T, np.nan, dtype=float)
#     pnl_net = np.full(T, np.nan, dtype=float)
#     turnover = np.full(T, np.nan, dtype=float)
#     active = np.zeros(T, dtype=bool)

#     prev_p = None
#     window_reports: list[dict] = []

#     for w_idx, (tr0, tr1, te0, te1) in enumerate(windows):
#         stats = compute_edge_stats_from_RP(R[tr0:tr1], P[tr0:tr1], lag=lag, dtype=dtype)
#         sel = select_sources_targets(
#             stats.A,
#             stats.tstat,
#             stats.n_eff,
#             ks=ks,
#             kt=kt,
#             q=q,
#             min_count=min_count,
#             weight_mode=weight_mode,
#         )
#         M = build_trading_matrix(stats.A, sel.sources, sel.targets, keep_mask=sel.keep_mask)

#         start_t = max(te0, lag)
#         current_target = None

#         for t in range(start_t, te1):
#             if current_target is None or ((t - start_t) % rebalance_every == 0):
#                 sig = np.nan_to_num(P[t - lag], nan=0.0, posinf=0.0, neginf=0.0)
#                 raw_p = M @ sig
#                 if dollar_neutral:
#                     raw_p = raw_p - float(np.mean(raw_p))
#                 current_target = l1_normalize(raw_p)

#             p = current_target.copy()
#             ret = np.nan_to_num(R[t], nan=0.0, posinf=0.0, neginf=0.0)
#             gross = float(np.dot(p, ret))

#             if prev_p is None:
#                 turn = float(np.sum(np.abs(p)))
#             else:
#                 turn = float(np.sum(np.abs(p - prev_p)))

#             cost = (tc_bps / 1e4) * turn
#             net = gross - cost

#             pnl_gross[t] = gross
#             pnl_net[t] = net
#             turnover[t] = turn
#             active[t] = True
#             prev_p = p

#         reverse_mass = float(sel.W[np.ix_(sel.sources, sel.targets)].sum())
#         forward_mass = float(sel.W[np.ix_(sel.targets, sel.sources)].sum())
#         window_reports.append({
#             "window_index": w_idx,
#             "train": [int(tr0), int(tr1)],
#             "test": [int(te0), int(te1)],
#             "ks": int(ks),
#             "kt": int(kt),
#             "q": float(q),
#             "min_count": int(min_count),
#             "weight_mode": weight_mode,
#             "sources": sel.sources.tolist(),
#             "targets": sel.targets.tolist(),
#             "objective_value": float(sel.objective_value),
#             "iterations": int(sel.iterations),
#             "n_kept_edges": int(np.count_nonzero(sel.keep_mask)),
#             "n_strategy_edges": int(np.count_nonzero(M)),
#             "forward_mass": forward_mass,
#             "reverse_mass": reverse_mass,
#             "direction_gap": forward_mass - reverse_mass,
#         })

#     metrics = compute_metrics(pnl_gross, pnl_net, turnover, active, periods_per_year=periods_per_year)
#     return BacktestResult(pnl_gross, pnl_net, turnover, active, metrics), window_reports


# # =========================================================
# # Main CLI
# # =========================================================

# def main():
#     ap = argparse.ArgumentParser(description="Directed source-target lead-lag pipeline for OFI -> return")
#     ap.add_argument("--R", type=str, required=True, help="Path to R.npy")
#     ap.add_argument("--P", type=str, required=True, help="Path to P.npy")
#     ap.add_argument("--outdir", type=str, required=True, help="Output directory")

#     ap.add_argument("--lag", type=int, default=1)
#     ap.add_argument("--ks", type=int, default=25, help="number of source assets")
#     ap.add_argument("--kt", type=int, default=25, help="number of target assets")
#     ap.add_argument("--q", type=float, default=0.10, help="BH-FDR level")
#     ap.add_argument("--min_count", type=int, default=60, help="minimum n_eff per edge")
#     ap.add_argument("--weight_mode", type=str, default="abs_tstat", choices=["abs_tstat", "abs_A", "positive_A"])

#     ap.add_argument("--train_len", type=int, default=1260)
#     ap.add_argument("--test_len", type=int, default=63)
#     ap.add_argument("--step", type=int, default=63)
#     ap.add_argument("--start", type=int, default=0)
#     ap.add_argument("--end", type=int, default=None)

#     ap.add_argument("--tc_bps", type=float, default=0.0)
#     ap.add_argument("--periods_per_year", type=int, default=252)
#     ap.add_argument("--dollar_neutral", action="store_true")
#     ap.add_argument("--rebalance_every", type=int, default=1)
#     ap.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])

#     args = ap.parse_args()
#     dtype = np.float32 if args.dtype == "float32" else np.float64

#     R = load_matrix(args.R)
#     P = load_matrix(args.P)
#     if R.shape != P.shape:
#         raise ValueError(f"R and P shape mismatch: R={R.shape}, P={P.shape}")

#     T, N = R.shape
#     windows = make_walkforward_windows(
#         T,
#         train_len=args.train_len,
#         test_len=args.test_len,
#         step=args.step,
#         lag=args.lag,
#         start=args.start,
#         end=args.end,
#     )
#     if not windows:
#         raise ValueError("No walk-forward windows created. Check train/test lengths.")

#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     result, window_reports = run_walkforward(
#         R,
#         P,
#         windows,
#         lag=args.lag,
#         ks=args.ks,
#         kt=args.kt,
#         q=args.q,
#         min_count=args.min_count,
#         weight_mode=args.weight_mode,
#         tc_bps=args.tc_bps,
#         periods_per_year=args.periods_per_year,
#         dollar_neutral=args.dollar_neutral,
#         rebalance_every=args.rebalance_every,
#         dtype=dtype,
#     )

#     save_npy(outdir / "pnl_gross.npy", result.pnl_gross)
#     save_npy(outdir / "pnl_net.npy", result.pnl_net)
#     save_npy(outdir / "turnover.npy", result.turnover)
#     save_npy(outdir / "active.npy", result.active.astype(np.uint8))

#     (outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
#     (outdir / "window_reports.json").write_text(json.dumps(window_reports, indent=2))

#     # Small summary CSV
#     csv_path = outdir / "summary.csv"
#     with csv_path.open("w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=[
#             "active_obs",
#             "gross_sharpe",
#             "gross_ann_mean",
#             "gross_ann_vol",
#             "gross_cum_pnl",
#             "net_sharpe",
#             "net_ann_mean",
#             "net_ann_vol",
#             "net_cum_pnl",
#             "turnover_mean",
#             "turnover_median",
#         ])
#         writer.writeheader()
#         writer.writerow({
#             "active_obs": result.metrics["active_obs"],
#             "gross_sharpe": result.metrics["gross"]["sharpe"],
#             "gross_ann_mean": result.metrics["gross"]["ann_mean"],
#             "gross_ann_vol": result.metrics["gross"]["ann_vol"],
#             "gross_cum_pnl": result.metrics["gross"]["cum_pnl"],
#             "net_sharpe": result.metrics["net"]["sharpe"],
#             "net_ann_mean": result.metrics["net"]["ann_mean"],
#             "net_ann_vol": result.metrics["net"]["ann_vol"],
#             "net_cum_pnl": result.metrics["net"]["cum_pnl"],
#             "turnover_mean": result.metrics["turnover"]["mean"],
#             "turnover_median": result.metrics["turnover"]["median"],
#         })

#     print(f"Saved outputs to: {outdir}")
#     print(json.dumps(result.metrics, indent=2))


# if __name__ == "__main__":
#     main()
