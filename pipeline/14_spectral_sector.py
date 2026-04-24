#!/usr/bin/env python3
"""14_spectral_sector.py

Data-driven sector discovery for cross-impact matrix estimation.

Motivation
----------
The sector-block constraint in scripts 06–10 uses exogenous GICS sector labels.
GICS sectors are coarse (e.g., Technology contains 20+ stocks) and may not
reflect the actual granularity of cross-impact relationships.  This script
discovers groups endogenously from the pairwise OFI-to-return edge statistics,
then uses those groups as the block structure for ridge regression.

Method
------
At each refit window:
1. Compute the N×N directed edge-score matrix A[i,j] = mean(R_i * P_j) / std(...)
   over the training window (same as script 04).
2. Symmetrise: S = (|A| + |A|.T) / 2  (undirected co-predictability).
3. Normalise: D^{-1/2} S D^{-1/2}  (Laplacian normalisation).
4. Extract the top K eigenvectors of the normalised matrix.
5. K-means on the eigenvector rows → N cluster labels.
6. Use these K discovered clusters as the block structure for ridge regression
   (same as sector_block in script 06, but with data-driven blocks).
7. Apply Gavish-Donoho denoising.

The number of clusters K is treated as a hyperparameter; we sweep K ∈ {5, 8, 10}.

Comparison baseline
-------------------
We also run the GICS sector-block model (identical to script 06's best model)
within the same walk-forward framework so that the two are directly comparable.

Outputs → results/spectral_sector/

Run from repo root:
    python pipeline/14_spectral_sector.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

from pipeline_utils import (
    build_exp_decay_kernel,
    gavish_donoho_denoise,
    fit_ridge,
    daily_spread,
    daily_sector_neutral_spread,
    sharpe_from_series,
    annualize_sharpe,
    zscore_with_train_stats,
)

# ── config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("data/processed/feature_table_with_residuals_10level.csv")
OUT_DIR     = Path("results/spectral_sector")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS = 6
MAX_ASSETS       = 400
HALF_LIFE        = 30          # minutes — use best model half-life
RANK_NORM        = True        # use rank-normalised OFI (best config)
Q                = 0.10
MIN_TICKER_OBS   = 252
MIN_CLUSTER_SIZE = 3           # minimum cluster size to fit a block
MIN_NAMES_SN     = 4

INITIAL_TRAIN_DAYS = 750
REFIT_EVERY_DAYS   = 21
RIDGE_GRID         = [1.0, 10.0, 50.0, 250.0, 1000.0]

# K sweep for spectral clustering
K_SWEEP = [5, 8, 10]

RANDOM_STATE = 0


# ══════════════════════════════════════════════════════════════════════════════
# Edge stats (from script 04)
# ══════════════════════════════════════════════════════════════════════════════

def compute_edge_stats(
    R: np.ndarray,
    P: np.ndarray,
    lag: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the N×N directed edge-score matrix A[i,j] = mu_ij / sigma_ij.

    A[i,j] = Sharpe of the strategy: use j's lagged OFI to predict i's return.
    """
    T, N = R.shape
    R1 = R[lag:]
    P0 = P[:T - lag]

    MR = np.isfinite(R1)
    MP = np.isfinite(P0)
    R1z = np.where(MR, R1, 0.0)
    P0z = np.where(MP, P0, 0.0)

    n_eff  = MR.astype(float).T @ MP.astype(float)
    sum_x  = R1z.T @ P0z
    sum_x2 = (R1z * R1z).T @ (P0z * P0z)

    mu = np.where(n_eff > 0, sum_x / np.maximum(n_eff, 1), 0.0)

    var = np.zeros((N, N))
    ok2 = n_eff >= 2
    if np.any(ok2):
        numer = sum_x2[ok2] - sum_x[ok2] ** 2 / np.maximum(n_eff[ok2], 1)
        var[ok2] = np.maximum(numer, 0.0) / np.maximum(n_eff[ok2] - 1.0, 1.0)

    sigma = np.sqrt(np.maximum(var, 0.0))
    A = np.where((n_eff > 0) & (sigma > eps), mu / sigma, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


# ══════════════════════════════════════════════════════════════════════════════
# Spectral clustering on the co-predictability matrix
# ══════════════════════════════════════════════════════════════════════════════

def spectral_cluster(A: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """Discover k groups from the directed edge-score matrix A.

    Steps:
      1. Symmetrise: S = (|A| + |A|.T) / 2
      2. Normalise: L = D^{-1/2} S D^{-1/2}
      3. Top k eigenvectors of L → row embedding
      4. K-means on the embedding → cluster labels (N,)
    """
    N = A.shape[0]
    S = (np.abs(A) + np.abs(A).T) / 2.0        # symmetric co-predictability

    d = S.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L = np.outer(d_inv_sqrt, d_inv_sqrt) * S    # normalised affinity

    eigvals, eigvecs = np.linalg.eigh(L)        # ascending
    V = eigvecs[:, -k:]                          # top-k eigenvectors (N × k)

    # Row-normalise for k-means stability
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_norm = V / np.where(norms > 0, norms, 1.0)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(V_norm)
    return labels.astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# Ridge fitting with arbitrary cluster labels as block structure
# ══════════════════════════════════════════════════════════════════════════════

def fit_cluster_block(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    labels: np.ndarray,
    tickers: list[str],
    lam: float,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> pd.DataFrame:
    """Ridge regression with data-driven cluster blocks.

    Parameters
    ----------
    X_df, Y_df : training DataFrames (same column ordering as tickers).
    labels : (N,) integer cluster assignment per ticker.
    tickers : list of ticker strings matching the label ordering.
    lam : ridge penalty.
    """
    W = pd.DataFrame(0.0, index=X_df.columns, columns=Y_df.columns, dtype=float)
    ticker_label = dict(zip(tickers, labels))

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        cluster_tickers = [t for t, l in ticker_label.items() if l == lab]
        sec_y = [t for t in cluster_tickers if t in Y_df.columns]
        if len(sec_y) < min_cluster_size:
            continue
        sec_y_set = set(sec_y)
        sec_x = [c for c in X_df.columns if c in sec_y_set]
        if not sec_x:
            continue
        Xg = X_df[sec_x].to_numpy(dtype=float)
        Yg = Y_df[sec_y].to_numpy(dtype=float)
        W.loc[sec_x, sec_y] = fit_ridge(Xg, Yg, lam)

    return W


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

print("Reading data...", flush=True)
needed_cols = ["date", "ticker", "residual_ret", "sector"]
header_df   = pd.read_csv(INPUT_PATH, nrows=0)
minute_cols = sorted(
    [c for c in header_df.columns if c.startswith("minute_")],
    key=lambda x: int(x.split("_")[1]),
)
df = pd.read_csv(INPUT_PATH, usecols=needed_cols + minute_cols)
df["date"] = pd.to_datetime(df["date"])

if USE_LAST_N_YEARS is not None:
    cutoff = df["date"].max() - pd.DateOffset(years=USE_LAST_N_YEARS)
    df = df[df["date"] >= cutoff].copy()

df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
df[minute_cols] = df[minute_cols].fillna(0.0)
df = df.dropna(subset=["ticker", "date", "residual_ret", "sector"]).copy()
df = df[df["ticker"] != "GOOGL"].copy()

sector_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
df = df[df["ticker"].isin(sector_counts[sector_counts == 1].index)].copy()
ticker_to_sector_full = df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

# Build exp-decay OFI signal
n_minutes = len(minute_cols)
kernel    = build_exp_decay_kernel(n_minutes, HALF_LIFE)
minute_np = df[minute_cols].to_numpy(dtype=float)
sig_col   = "ofi"
df[sig_col] = minute_np @ kernel
del minute_np

if RANK_NORM:
    df[sig_col] = df.groupby("date")[sig_col].rank(pct=True)

all_dates = np.sort(df["date"].unique())

# Coverage filter
sig_proxy = df.pivot(index="date", columns="ticker", values=sig_col)
ret_proxy = df.pivot(index="date", columns="ticker", values="residual_ret")
cd = sig_proxy.index.intersection(ret_proxy.index)
ct = sig_proxy.columns.intersection(ret_proxy.columns)
ticker_obs = (sig_proxy.loc[cd, ct].notna() & ret_proxy.loc[cd, ct].notna()).sum(axis=0)
keep_tickers = (
    ticker_obs[ticker_obs >= MIN_TICKER_OBS]
    .sort_values(ascending=False)
    .index[:MAX_ASSETS]
    .tolist()
)
df = df[df["ticker"].isin(keep_tickers)].copy()
ticker_to_sector = ticker_to_sector_full.loc[keep_tickers].copy()
print(f"Using {len(keep_tickers)} assets after filtering.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward evaluation
# ══════════════════════════════════════════════════════════════════════════════

test_start_idx = INITIAL_TRAIN_DAYS
test_dates     = pd.Index(all_dates[test_start_idx:])
refit_points   = list(range(0, len(test_dates), REFIT_EVERY_DAYS))

# Model specs: one per K value plus the GICS baseline
model_specs = [{"name": f"spectral_k{k}", "k": k} for k in K_SWEEP]
model_specs.append({"name": "gics_baseline", "k": None})  # GICS sectors

all_summaries = []

for spec in model_specs:
    model_name = spec["name"]
    k_clusters = spec["k"]          # None → GICS baseline
    use_spectral = k_clusters is not None

    print(f"\n{'=' * 70}", flush=True)
    print(f"RUNNING: {model_name}  (k={k_clusters})", flush=True)
    print(f"{'=' * 70}", flush=True)

    pred_chunks = []

    for block_num, start_offset in enumerate(refit_points, start=1):
        block_test_dates = test_dates[start_offset: start_offset + REFIT_EVERY_DAYS]
        if len(block_test_dates) == 0:
            continue

        first_pred_date      = block_test_dates[0]
        eligible_train_dates = all_dates[all_dates < first_pred_date]
        if len(eligible_train_dates) < INITIAL_TRAIN_DAYS + 1:
            continue
        train_dates_window = eligible_train_dates[-INITIAL_TRAIN_DAYS:]

        df_win = df[df["date"].isin(
            np.concatenate([train_dates_window, block_test_dates.values])
        )].copy()

        ret_mat = df_win.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
        sig_mat = df_win.pivot(index="date", columns="ticker", values=sig_col).sort_index()

        cd_b = ret_mat.index.intersection(sig_mat.index)
        ct_b = ret_mat.columns.intersection(sig_mat.columns)
        ret_mat = ret_mat.loc[cd_b, ct_b]
        sig_mat = sig_mat.loc[cd_b, ct_b]

        obs_count = (ret_mat.notna() & sig_mat.notna()).sum(axis=0)
        ct_b = obs_count[obs_count >= 30].index
        if len(ct_b) == 0:
            continue
        ret_mat = ret_mat[ct_b]
        sig_mat = sig_mat[ct_b]
        tickers = ct_b.tolist()
        N = len(tickers)

        train_mask = sig_mat.index.isin(train_dates_window)
        eval_mask  = sig_mat.index.isin(block_test_dates)

        sig_train = sig_mat.loc[train_mask].fillna(0.0)
        ret_train = ret_mat.loc[train_mask]
        sig_eval  = sig_mat.loc[eval_mask].fillna(0.0)
        ret_eval  = ret_mat.loc[eval_mask]

        X_train_df = sig_train.iloc[:-1].copy()
        Y_train_df = ret_train.iloc[1:].copy()
        X_train_df.index = Y_train_df.index

        X_eval_df = sig_eval.iloc[:-1].copy()
        if len(X_eval_df) == 0:
            continue

        # Common tickers
        cc = X_train_df.columns.intersection(Y_train_df.columns)
        X_train_df = X_train_df[cc]; Y_train_df = Y_train_df[cc]
        X_eval_df  = X_eval_df[cc]
        tickers    = cc.tolist()

        if len(X_train_df) < 5 or len(tickers) == 0:
            continue

        X_train_z, X_eval_z, _, _ = zscore_with_train_stats(X_train_df, X_eval_df)
        Y_train_z, _, y_mean, y_std = zscore_with_train_stats(Y_train_df, Y_train_df)
        X_train_z = X_train_z.fillna(0.0)
        X_eval_z  = X_eval_z.fillna(0.0)
        Y_train_z = Y_train_z.fillna(0.0)

        if use_spectral:
            # ── Discover cluster structure from training data ─────────────
            R_tr = Y_train_df.fillna(0.0).to_numpy(float)
            P_tr = X_train_df.fillna(0.0).to_numpy(float)
            A = compute_edge_stats(R_tr, P_tr, lag=1)
            labels = spectral_cluster(A, k=k_clusters, random_state=RANDOM_STATE)

            # ── Select lambda by in-sample IC ─────────────────────────────
            best_lam, best_ic = RIDGE_GRID[0], -np.inf
            for lam in RIDGE_GRID:
                W_cand = fit_cluster_block(X_train_z, Y_train_z, labels, tickers, lam)
                cx = X_train_z.columns.intersection(W_cand.index)
                pred_cand = X_train_z[cx].to_numpy(float) @ W_cand.loc[cx].to_numpy(float)
                P_rank = pd.DataFrame(pred_cand, index=X_train_z.index, columns=W_cand.columns).rank(axis=1)
                R_rank = Y_train_z.rank(axis=1)
                ic_vals = P_rank.corrwith(R_rank, axis=1, method="pearson").dropna()
                ic_mean = float(ic_vals.mean()) if len(ic_vals) else -np.inf
                if np.isfinite(ic_mean) and ic_mean > best_ic:
                    best_ic  = ic_mean
                    best_lam = lam

            W = fit_cluster_block(X_train_z, Y_train_z, labels, tickers, best_lam)
        else:
            # ── GICS baseline: sector-block ridge ─────────────────────────
            ts_block = ticker_to_sector.reindex(tickers).dropna()
            cc_gics  = ts_block.index.tolist()
            X_train_z = X_train_z[cc_gics]
            X_eval_z  = X_eval_z[cc_gics]
            Y_train_z = Y_train_z[cc_gics]
            y_mean, y_std = y_mean[cc_gics], y_std[cc_gics]
            tickers = cc_gics

            best_lam, best_ic = RIDGE_GRID[0], -np.inf
            for lam in RIDGE_GRID:
                sector_to_t = (
                    ts_block.rename("sector")
                    .reset_index().rename(columns={"index": "ticker"})
                    .groupby("sector")["ticker"].apply(list).to_dict()
                )
                W_cand = pd.DataFrame(0.0, index=X_train_z.columns, columns=Y_train_z.columns)
                for sector, sy in sector_to_t.items():
                    sy = [t for t in sy if t in Y_train_z.columns]
                    if len(sy) < 3:
                        continue
                    Xg = X_train_z[sy].to_numpy(float)
                    Yg = Y_train_z[sy].to_numpy(float)
                    W_cand.loc[sy, sy] = fit_ridge(Xg, Yg, lam)
                cx = X_train_z.columns.intersection(W_cand.index)
                pred_cand = X_train_z[cx].to_numpy(float) @ W_cand.loc[cx].to_numpy(float)
                P_rank = pd.DataFrame(pred_cand, index=X_train_z.index, columns=W_cand.columns).rank(axis=1)
                R_rank = Y_train_z.rank(axis=1)
                ic_vals = P_rank.corrwith(R_rank, axis=1, method="pearson").dropna()
                ic_mean = float(ic_vals.mean()) if len(ic_vals) else -np.inf
                if np.isfinite(ic_mean) and ic_mean > best_ic:
                    best_ic  = ic_mean
                    best_lam = lam
                    W = W_cand.copy()

        # ── Gavish-Donoho denoising ───────────────────────────────────────
        Wmat = W.to_numpy(dtype=float) if isinstance(W, pd.DataFrame) else W
        W_gd, gd_rank, _ = gavish_donoho_denoise(Wmat)
        if isinstance(W, pd.DataFrame):
            W = pd.DataFrame(W_gd, index=W.index, columns=W.columns)
        else:
            W = W_gd

        # ── Predict on eval window ────────────────────────────────────────
        cx = X_eval_z.columns.intersection(W.index if isinstance(W, pd.DataFrame) else X_eval_z.columns)
        if isinstance(W, pd.DataFrame):
            pred_z = X_eval_z[cx].to_numpy(float) @ W.loc[cx].to_numpy(float)
            cols = W.columns
        else:
            pred_z = X_eval_z.to_numpy(float) @ W
            cols = X_eval_z.columns
        pred = pred_z * y_std.loc[cols].to_numpy(float) + y_mean.loc[cols].to_numpy(float)
        pred_df = pd.DataFrame(pred, index=X_eval_z.index, columns=cols)
        pred_chunks.append(pred_df)

        print(
            f"  [{model_name}] block {block_num}: "
            f"train {pd.Timestamp(train_dates_window[0]).date()} to "
            f"{pd.Timestamp(train_dates_window[-1]).date()} | "
            f"predict {pd.Timestamp(block_test_dates[0]).date()} to "
            f"{pd.Timestamp(block_test_dates[-1]).date()} | "
            f"N={len(tickers)}" + (f" | GD rank={gd_rank}" if gd_rank else ""),
            flush=True,
        )

    if not pred_chunks:
        print(f"[{model_name}] No predictions produced.", flush=True)
        continue

    pred_all = pd.concat(pred_chunks).sort_index()
    pred_all = pred_all[~pred_all.index.duplicated(keep="last")]
    pred_all.to_csv(OUT_DIR / f"predicted_returns_{model_name}.csv")

    ret_full = df.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
    cd = pred_all.index.intersection(ret_full.index)
    cc = pred_all.columns.intersection(ret_full.columns)
    pred_all = pred_all.loc[cd, cc]
    ret_eval  = ret_full.loc[cd, cc]
    ts_eval   = ticker_to_sector.reindex(cc).dropna()
    cc = ts_eval.index
    pred_all, ret_eval = pred_all[cc], ret_eval[cc]

    gross = pred_all.apply(
        lambda row: daily_spread(row, ret_eval.loc[row.name], q=Q), axis=1
    ).dropna()
    sn = pred_all.apply(
        lambda row: daily_sector_neutral_spread(
            row, ret_eval.loc[row.name], ts_eval, q=Q, min_names_per_sector=MIN_NAMES_SN
        ), axis=1
    ).dropna()
    ic = pred_all.apply(
        lambda row: row.corr(ret_eval.loc[row.name], method="spearman"), axis=1
    ).dropna()

    summary_row = pd.DataFrame([{
        "model":                       model_name,
        "k_clusters":                  k_clusters if k_clusters else "GICS",
        "n_eval_days":                 len(gross),
        "mean_daily_spread":           gross.mean(),
        "annualized_spread_sharpe":    annualize_sharpe(sharpe_from_series(gross)),
        "mean_daily_spread_sn":        sn.mean(),
        "annualized_spread_sharpe_sn": annualize_sharpe(sharpe_from_series(sn)),
        "mean_daily_ic":               ic.mean(),
    }])
    all_summaries.append(summary_row)
    summary_row.to_csv(OUT_DIR / f"summary_{model_name}.csv", index=False)
    print(f"\n[{model_name}] Results:", flush=True)
    print(summary_row.to_string(index=False), flush=True)

# ── Combined summary ──────────────────────────────────────────────────────────
if all_summaries:
    final = pd.concat(all_summaries, ignore_index=True).sort_values(
        "annualized_spread_sharpe_sn", ascending=False
    )
    final.to_csv(OUT_DIR / "spectral_sector_summary_all.csv", index=False)
    print("\n" + "=" * 70, flush=True)
    print("SPECTRAL SECTOR — ALL MODELS", flush=True)
    print("=" * 70, flush=True)
    print(final.to_string(index=False), flush=True)
    print(f"\nOutputs saved to: {OUT_DIR.resolve()}", flush=True)
