from __future__ import annotations

import numpy as np
from pathlib import Path


def load_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix at {path}, got shape={arr.shape}")
    return arr


def build_W(
    R: np.ndarray,
    P: np.ndarray,
    *,
    lag: int = 1,
    fill_first: float = np.nan,
    dtype=np.float32,
    method: str = "broadcast", 
) -> np.ndarray:
    """
    Build W[t,i,j] = R[t,i] * P[t-lag,j].

    Parameters
    ----------
    R, P : np.ndarray
        Shape (T, N).
    lag : int
        Time lag applied to P (default 1 => P[t-1]).
    fill_first : float
        Value used for rows t < lag where P[t-lag] is undefined.
        Use np.nan or 0.0 depending on your convention.
    dtype : numpy dtype
        Output dtype (float32 recommended).
    method : str
        "broadcast" or "einsum".

    Returns
    -------
    W : np.ndarray
        Shape (T, N, N).
    """
    if R.shape != P.shape:
        raise ValueError(f"R and P must have same shape. Got R={R.shape}, P={P.shape}")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    T, N = R.shape

    Rv = np.asarray(R, dtype=dtype, order="C")
    Pv = np.asarray(P, dtype=dtype, order="C")

    # Build lagged P: P_lag[t] = P[t-lag]
    if lag == 0:
        P_lag = Pv
    else:
        P_lag = np.roll(Pv, shift=lag, axis=0)
        if lag > 0:
            P_lag[:lag, :] = fill_first

    # Construct W
    if method == "einsum":
        # W[t,i,j] = R[t,i] * P_lag[t,j]
        W = np.einsum("ti,tj->tij", Rv, P_lag, optimize=True)
    elif method == "broadcast":
        # (T,N,1) * (T,1,N) -> (T,N,N)
        W = Rv[:, :, None] * P_lag[:, None, :]
    else:
        raise ValueError('method must be "broadcast" or "einsum"')

    return W


def main():
    R_path = Path("data/processed/R.npy")  # returns
    P_path = Path("data/processed/P.npy")  # z-scores
    out_path = Path("data/processed/W.npy")

    # === Load ===
    R = load_matrix(R_path)
    P = load_matrix(P_path)

    # === Build W ===
    W = build_W(
        R,
        P,
        lag=1,
        fill_first=np.nan,   # or 0.0
        dtype=np.float32,
        method="broadcast",
    )

    T, N = R.shape
    assert W.shape == (T, N, N), f"Unexpected W shape: {W.shape}"

    t, i, j = 5, 0, 1
    expected = np.float32(R[t, i]) * np.float32(P[t - 1, j])
    got = W[t, i, j]
    print("Spot check:", got, "expected:", expected)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, W)
    print(f"Saved W to: {out_path} with shape {W.shape} dtype {W.dtype}")

    bytes_W = W.nbytes
    print(f"W size on disk (raw bytes): {bytes_W / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()