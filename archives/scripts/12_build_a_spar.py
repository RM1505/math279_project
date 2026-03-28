"""
Build A, tstat, and A_sparse exactly like the slide:

Given mean_pnl (mu_hat), std_pnl (sigma_hat), and pair_counts (effective T per pair),
compute:

A_ij      = mu_hat_ij / sigma_hat_ij
tstat_ij  = mu_hat_ij / (sigma_hat_ij / sqrt(T_ij - 1))  = A_ij * sqrt(T_ij - 1)
A_spar_ij = A_ij * 1{|tstat_ij| >= tau}

Notes:
- Use pairwise counts (T_ij) because your data has missing days per ticker.
- Avoid divide-by-zero and require a minimum count.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def build_A_tstat_sparse(
    mean_pnl: np.ndarray,
    sr_annualized: np.ndarray,
    pair_counts: np.ndarray,
    *,
    periods_per_year: float = 252.0,
    tau: float = 3.0,
    min_count: int = 50,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs
    ------
    mean_pnl: (N,N)  daily mean of pnl_t(i,j) = R[t,i]*P[t-1,j]
    sr_annualized: (N,N) annualized sharpe = sqrt(periods_per_year) * A
    pair_counts: (N,N) number of valid observations used for each (i,j)

    Returns
    -------
    A: (N,N)            non-annualized Sharpe-like score
    tstat: (N,N)        t-stat per slide
    A_sparse: (N,N)     sparsified A with |tstat| >= tau
    """
    if mean_pnl.shape != sr_annualized.shape or mean_pnl.shape != pair_counts.shape:
        raise ValueError("mean_pnl, sr_annualized, pair_counts must have same shape")

    # Recover A from annualized Sharpe:
    # sr_annualized = sqrt(periods_per_year) * A
    A = sr_annualized / np.sqrt(periods_per_year)

    # Recover sigma from mu and A:  A = mu / sigma  => sigma = mu / A
    # This can be numerically unstable when A ~ 0, so we guard with eps and masking.
    sigma = np.full_like(mean_pnl, np.nan, dtype=np.float64)
    safe = np.isfinite(mean_pnl) & np.isfinite(A) & (np.abs(A) > eps)
    sigma[safe] = mean_pnl[safe] / A[safe]

    # tstat = mu / (sigma / sqrt(T-1)) = (mu/sigma)*sqrt(T-1) = A * sqrt(T-1)
    T_eff = pair_counts.astype(np.float64)
    tstat = np.full_like(A, np.nan, dtype=np.float64)

    ok_t = np.isfinite(A) & (T_eff > 1)
    tstat[ok_t] = A[ok_t] * np.sqrt(T_eff[ok_t] - 1.0)

    # Sparsify
    A_sparse = np.zeros_like(A, dtype=np.float64)
    keep = ok_t & (T_eff >= min_count) & np.isfinite(tstat) & (np.abs(tstat) >= tau)
    A_sparse[keep] = A[keep]

    return A.astype(np.float32), tstat.astype(np.float32), A_sparse.astype(np.float32)


def main():
    mean_path = Path("data/processed/mean_pnl.npy")
    sr_path   = Path("data/processed/sr_annualized.npy")
    cnt_path  = Path("data/processed/pair_counts.npy")

    mean_pnl = np.load(mean_path)
    sr_ann   = np.load(sr_path)
    counts   = np.load(cnt_path)

    A, tstat, A_spar = build_A_tstat_sparse(
        mean_pnl,
        sr_ann,
        counts,
        periods_per_year=252.0,
        tau=2.0,         # change threshold here
        min_count=500,   # you were using 500; keep consistent
    )

    out_dir = Path("data/processed")
    np.save(out_dir / "A.npy", A)
    np.save(out_dir / "tstat.npy", tstat)
    np.save(out_dir / "A_sparse.npy", A_spar)

    # quick diagnostics
    nnz = np.count_nonzero(np.isfinite(A_spar) & (A_spar != 0))
    print("Saved A.npy, tstat.npy, A_sparse.npy")
    print("A_sparse nonzeros:", nnz, "out of", A_spar.size)
    if nnz > 0:
        vals = A_spar[A_spar != 0]
        print("A_sparse stats: min", vals.min(), "max", vals.max(), "mean", vals.mean())


if __name__ == "__main__":
    main()