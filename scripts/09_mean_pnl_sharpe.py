from __future__ import annotations

import numpy as np
from pathlib import Path


def lag_matrix(P: np.ndarray, lag: int = 1, fill: float = np.nan) -> np.ndarray:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if lag == 0:
        return P.copy()
    P_lag = np.roll(P, shift=lag, axis=0)
    P_lag[:lag, :] = fill
    return P_lag


def mean_and_annualized_sharpe_from_RP(
    R: np.ndarray,
    P: np.ndarray,
    *,
    lag: int = 1,
    periods_per_year: float = 252.0,
    ddof: int = 1,          # sample std: / (count-1)
    min_count: int = 2,     # need >=2 obs for std
    dtype_acc=np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes pairwise:
      pnl_t(i,j) = R[t,i] * P[t-lag,j]
      mean_ij    = mean_t pnl_t(i,j) over valid (finite) pairs
      sr_ij      = annualized Sharpe = sqrt(periods_per_year) * mean_ij / std_ij

    Valid at time t iff both R[t,i] and P_lag[t,j] are finite.
    """
    if R.ndim != 2 or P.ndim != 2:
        raise ValueError("R and P must be 2D arrays (T,N)")
    if R.shape != P.shape:
        raise ValueError(f"R and P must have same shape. Got {R.shape} vs {P.shape}")

    T, N = R.shape
    P_lag = lag_matrix(P, lag=lag, fill=np.nan)

    maskR = np.isfinite(R)
    maskP = np.isfinite(P_lag)

    R0 = np.where(maskR, R, 0.0).astype(dtype_acc, copy=False)
    P0 = np.where(maskP, P_lag, 0.0).astype(dtype_acc, copy=False)

    # counts per (i,j)
    count = (maskR.astype(dtype_acc).T @ maskP.astype(dtype_acc))  # (N,N)

    # sum and sumsq of pnl
    sum_ = (R0.T @ P0)                            # (N,N)
    sumsq = ((R0 * R0).T @ (P0 * P0))             # (N,N)

    # mean
    mean = np.full((N, N), np.nan, dtype=dtype_acc)
    ok_mean = count > 0
    mean[ok_mean] = sum_[ok_mean] / count[ok_mean]

    # variance with ddof
    denom = count - ddof
    var = np.full((N, N), np.nan, dtype=dtype_acc)
    ok_var = denom > 0

    numer = sumsq - (sum_ * sum_) / np.where(count > 0, count, 1.0)
    numer = np.maximum(numer, 0.0)  # guard tiny negatives from rounding
    var[ok_var] = numer[ok_var] / denom[ok_var]
    std = np.sqrt(var)

    # annualized sharpe
    ann = np.sqrt(periods_per_year)
    sr = np.full((N, N), np.nan, dtype=dtype_acc)
    ok_sr = (count >= min_count) & np.isfinite(std) & (std > 0)
    sr[ok_sr] = ann * mean[ok_sr] / std[ok_sr]

    return mean, sr, count


def main():
    R = np.load("data/processed/R.npy")  # (T,N)
    P = np.load("data/processed/P.npy")  # (T,N)

    mean_pnl, sr_ann, count = mean_and_annualized_sharpe_from_RP(
        R, P, lag=1, periods_per_year=252.0
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "mean_pnl.npy", mean_pnl.astype(np.float32))
    np.save(out_dir / "sr_annualized.npy", sr_ann.astype(np.float32))
    np.save(out_dir / "pair_counts.npy", count.astype(np.float32))

    print("Saved:")
    print(" - mean_pnl.npy")
    print(" - sr_annualized.npy")
    print(" - pair_counts.npy")
    print("Shapes:", mean_pnl.shape, sr_ann.shape, count.shape)
    print("Count stats: min", np.nanmin(count), "max", np.nanmax(count), "mean", np.nanmean(count))


if __name__ == "__main__":
    main()