import pandas as pd
import numpy as np
from pathlib import Path

def calculate_integrated_ofi(ticker):
    folder = Path(f"data/processed/minutely_deep/{ticker}")

    dfs = []

    for file in folder.glob("*.csv"):
        df = pd.read_csv(file)
        df["source"] = file.stem
        df = df.drop(index=0)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    X = combined.filter(like="ofi_").to_numpy(dtype=float, copy=True)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X -= X.mean(axis=0)

    cov = (X.T @ X) / (X.shape[0] - 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        print(f"{ticker}: eigh failed; skipping")
        print("ticker:", ticker)
        print("X shape:", X.shape)
        print("NaNs:", np.isnan(X).sum())
        print("Infs:", np.isinf(X).sum())
        return None
    idx = np.argsort(eigenvalues)[::-1]
    eigvalues = eigenvalues[idx]
    eigvectors = eigenvectors[:, idx]

    IOFI = X @ eigvectors[:,0]

    output_path = Path(f"data/processed/integrated_ofi/{ticker}.csv")

    combined["IOFI"] = IOFI

    combined[["source", "time", "IOFI"]].to_csv(output_path, index=False)