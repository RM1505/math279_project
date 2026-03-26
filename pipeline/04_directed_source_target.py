#!/usr/bin/env python3
"""04_directed_source_target.py

Directed source-target (DST) biclustering applied to the same exp-decay OFI
feature table used by the ridge scripts (06, 07, 09, 10).

Method
------
For each ordered pair (i, j), define:
    X_t(i,j) = R[t,i] * P[t-1,j]      (return of i, lagged OFI of j)
Estimate the "edge Sharpe":
    A[i,j] = mean(X_t) / std(X_t)
and its t-statistic over the training window.

Significance filtering: keep edges where |t| >= TSTAT_MIN and
n_eff >= MIN_COUNT.  (BH-FDR is skipped — with N=63 stocks there are
only ~3900 edges so BH at q=0.10 would demand |t|≥4.7, killing nearly
everything.  We use a fixed t-threshold of 2.0 instead.)

After filtering, select:
    Sources S — stocks whose lagged OFI best predicts targets
    Targets T — stocks most predicted by source OFI
via alternating top-k column/row updates on the surviving W matrix.

Prediction score for stock i at time t:
    score_i = Σ_{j in S} W[i,j] * P[t-1, j]      (W[i,j] = A[i,j] * keep)

Scores are evaluated identically to the ridge scripts (06/09):
  - gross top-q spread (top 10% minus bottom 10%)
  - sector-neutral spread (rank within sector)

Walk-forward cadence: 750-day train, 21-day step — identical to script 06.

Outputs → results/rolling_adjacency_dst/

Run from repo root:
    python pipeline/04_directed_source_target.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate as _sci_integrate, optimize as _sci_optimize


# ============================================================
# config
# ============================================================
INPUT_PATH = Path("data/processed/feature_table_with_residuals_10level.csv")
OUT_DIR    = Path("results/rolling_adjacency_dst")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS = 6
MAX_ASSETS       = 400
HALF_LIFE_SWEEP  = [15, 20, 25, 30, 35, 40, 45, 60, 90]   # minutes
Q                = 0.10
MIN_TICKER_OBS   = 252
MIN_SECTOR_ASSETS            = 4
MIN_NAMES_PER_SECTOR_NEUTRAL = 4

INITIAL_TRAIN_DAYS = 750
REFIT_EVERY_DAYS   = 21

# Source / target counts — with ~63 active stocks, 15 each is ~24%
KS = 15   # number of source stocks
KT = 15   # number of target stocks

TSTAT_MIN = 2.0    # minimum |t| for an edge to survive (no BH with N=63)
MIN_COUNT = 60     # minimum co-observations per edge
WEIGHT_MODE = "abs_tstat"  # edge weight: |t-stat|

# ============================================================
# exp-decay kernel
# ============================================================
def build_exp_decay_kernel(n_minutes: int, half_life: float) -> np.ndarray:
    lam = np.log(2) / half_life
    k = np.arange(n_minutes, 0, -1, dtype=float)
    weights = np.exp(-lam * k)
    weights /= weights.sum()
    return weights


def get_minute_cols(columns: list[str]) -> list[str]:
    return sorted(
        [c for c in columns if c.startswith("minute_")],
        key=lambda x: int(x.split("_")[1]),
    )


# ============================================================
# portfolio helpers (same as scripts 27/33)
# ============================================================
def daily_spread(scores: pd.Series, returns: pd.Series, q: float = 0.10) -> float:
    tmp = pd.concat([scores.rename("s"), returns.rename("r")], axis=1).dropna()
    n = len(tmp)
    if n < 2:
        return np.nan
    k = max(1, int(n * q))
    if 2 * k > n:
        return np.nan
    tmp = tmp.sort_values("s")
    return float(tmp.iloc[-k:]["r"].mean() - tmp.iloc[:k]["r"].mean())


def daily_sector_neutral_spread(
    scores: pd.Series,
    returns: pd.Series,
    ticker_to_sector: pd.Series,
    q: float = 0.10,
    min_names: int = 4,
) -> float:
    tmp = pd.concat([scores.rename("s"), returns.rename("r")], axis=1).dropna()
    if tmp.empty:
        return np.nan
    tmp = tmp.join(ticker_to_sector.rename("sec"), how="left").dropna(subset=["sec"])
    spreads = []
    for _, g in tmp.groupby("sec"):
        n = len(g)
        if n < min_names:
            continue
        k = max(1, int(n * q))
        if 2 * k > n:
            continue
        g = g.sort_values("s")
        spreads.append(float(g.iloc[-k:]["r"].mean() - g.iloc[:k]["r"].mean()))
    return float(np.mean(spreads)) if spreads else np.nan


def sharpe_from_series(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 and np.isfinite(s) else np.nan


def annualize_sharpe(daily_sharpe: float) -> float:
    return float(daily_sharpe * np.sqrt(252)) if np.isfinite(daily_sharpe) else np.nan


# ============================================================
# core directed source-target functions
# ============================================================
@dataclass
class EdgeStats:
    n_eff:  np.ndarray
    mu:     np.ndarray
    sigma:  np.ndarray
    A:      np.ndarray
    tstat:  np.ndarray


def compute_edge_stats(
    R: np.ndarray,
    P: np.ndarray,
    lag: int = 1,
    eps: float = 1e-12,
) -> EdgeStats:
    """For each pair (i,j): X_t(i,j) = R[t,i] * P[t-lag,j].  Compute stats."""
    T, N = R.shape
    R1 = R[lag:]
    P0 = P[:T - lag]

    MR = np.isfinite(R1)
    MP = np.isfinite(P0)
    R1z = np.where(MR, R1, 0.0)
    P0z = np.where(MP, P0, 0.0)

    n_eff  = MR.astype(float).T @ MP.astype(float)        # (N,N)
    sum_x  = R1z.T @ P0z
    sum_x2 = (R1z * R1z).T @ (P0z * P0z)

    mu = np.zeros((N, N))
    ok = n_eff > 0
    mu[ok] = sum_x[ok] / n_eff[ok]

    var = np.zeros((N, N))
    ok2 = n_eff >= 2
    if np.any(ok2):
        numer = sum_x2[ok2] - sum_x[ok2] ** 2 / n_eff[ok2]
        var[ok2] = np.maximum(numer, 0.0) / np.maximum(n_eff[ok2] - 1.0, 1.0)

    sigma = np.sqrt(np.maximum(var, 0.0))

    A     = np.zeros((N, N))
    tstat = np.zeros((N, N))
    good  = ok & (sigma > eps)
    A[good]     = mu[good] / sigma[good]
    good2 = ok2 & (sigma > eps)
    tstat[good2] = mu[good2] / (sigma[good2] / np.sqrt(n_eff[good2]))

    np.fill_diagonal(n_eff,  0.0)
    np.fill_diagonal(mu,     0.0)
    np.fill_diagonal(sigma,  0.0)
    np.fill_diagonal(A,      0.0)
    np.fill_diagonal(tstat,  0.0)

    return EdgeStats(n_eff=n_eff, mu=mu, sigma=sigma, A=A, tstat=tstat)


def _top_k(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or x.size == 0:
        return np.array([], dtype=int)
    k = min(k, x.size)
    idx = np.argpartition(-x, k - 1)[:k]
    return idx[np.argsort(-x[idx])].astype(int)


def select_sources_targets(
    stats: EdgeStats,
    ks: int,
    kt: int,
    tstat_min: float,
    min_count: int,
    weight_mode: str = "abs_tstat",
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating top-k selection.  Returns (sources, targets, W_sparse)."""
    keep = (np.abs(stats.tstat) >= tstat_min) & (stats.n_eff >= min_count)
    np.fill_diagonal(keep, False)

    if weight_mode == "abs_tstat":
        W = np.abs(stats.tstat) * keep
    elif weight_mode == "abs_A":
        W = np.abs(stats.A) * keep
    else:
        W = np.maximum(stats.A, 0.0) * keep

    if np.count_nonzero(W) == 0:
        N = W.shape[0]
        return np.arange(min(ks, N)), np.arange(min(kt, N)), W

    sources = _top_k(W.sum(axis=0), ks)
    prev_s, prev_t = None, None

    for _ in range(max_iter):
        targets = _top_k(W[:, sources].sum(axis=1), kt)
        sources = _top_k(W[targets, :].sum(axis=0), ks)
        if prev_s is not None and np.array_equal(sources, prev_s) and np.array_equal(targets, prev_t):
            break
        prev_s, prev_t = sources.copy(), targets.copy()

    return sources, targets, W


def build_score_matrix(
    stats: EdgeStats,
    sources: np.ndarray,
    targets: np.ndarray,
    W: np.ndarray,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Returns a DataFrame of the sparse M matrix indexed by (target, source) tickers,
    for inspection.  Not used in the main prediction loop.
    """
    N = len(tickers)
    M = np.zeros((N, N))
    M[np.ix_(targets, sources)] = stats.A[np.ix_(targets, sources)]
    keep = (np.abs(stats.tstat) >= TSTAT_MIN) & (stats.n_eff >= MIN_COUNT)
    M = np.where(keep, M, 0.0)
    np.fill_diagonal(M, 0.0)
    return pd.DataFrame(M, index=tickers, columns=tickers)


# ============================================================
# load and prepare data
# ============================================================
print("Reading data...", flush=True)

needed_base_cols = ["date", "ticker", "residual_ret", "sector"]
header_df = pd.read_csv(INPUT_PATH, nrows=0)
minute_cols = get_minute_cols(list(header_df.columns))
df = pd.read_csv(INPUT_PATH, usecols=needed_base_cols + minute_cols)
df["date"] = pd.to_datetime(df["date"])

if USE_LAST_N_YEARS is not None:
    cutoff = df["date"].max() - pd.DateOffset(years=USE_LAST_N_YEARS)
    df = df[df["date"] >= cutoff].copy()
    print(f"Using data from {cutoff.date()} onward", flush=True)

df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
df[minute_cols] = df[minute_cols].fillna(0.0)
df = df.dropna(subset=["ticker", "date", "residual_ret", "sector"]).copy()
df = df[df["ticker"] != "GOOGL"].copy()

sector_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
df = df[df["ticker"].isin(sector_counts[sector_counts == 1].index)].copy()
ticker_to_sector_full = df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

# build all exp-decay OFI signals
n_minutes = len(minute_cols)
minute_np = df[minute_cols].to_numpy(dtype=float)
print("Building exp-decay OFI signals:", flush=True)
for hl in HALF_LIFE_SWEEP:
    kernel = build_exp_decay_kernel(n_minutes, hl)
    df[f"ofi_hl{hl}"] = minute_np @ kernel
    print(f"  hl={hl:3d}min", flush=True)
del minute_np

# rank-normalized variants
for hl in HALF_LIFE_SWEEP:
    df[f"ofi_hl{hl}_rn"] = df.groupby("date")[f"ofi_hl{hl}"].rank(pct=True)

all_dates = np.sort(df["date"].unique())

# coverage filter (using hl=45 proxy, same as script 06)
sig_proxy = df.pivot(index="date", columns="ticker", values="ofi_hl45")
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

# ============================================================
# model specs — half-life variants (plain + rank-norm)
# ============================================================
MODEL_SPECS = (
    [{"name": f"dst_hl{hl}",    "sig_col": f"ofi_hl{hl}"}    for hl in HALF_LIFE_SWEEP]
  + [{"name": f"dst_hl{hl}_rn", "sig_col": f"ofi_hl{hl}_rn"} for hl in HALF_LIFE_SWEEP]
)

# ============================================================
# rolling walk-forward
# ============================================================
test_start_idx = INITIAL_TRAIN_DAYS
test_dates     = pd.Index(all_dates[test_start_idx:])
refit_points   = list(range(0, len(test_dates), REFIT_EVERY_DAYS))

all_summaries = []

for spec in MODEL_SPECS:
    model_name = spec["name"]
    sig_col    = spec["sig_col"]

    print("\n" + "=" * 70, flush=True)
    print(f"RUNNING: {model_name}  sig={sig_col}", flush=True)
    print("=" * 70, flush=True)

    pred_chunks: list[pd.DataFrame] = []
    edge_counts: list[int] = []

    for block_num, start_offset in enumerate(refit_points, start=1):
        block_test_dates = test_dates[start_offset:start_offset + REFIT_EVERY_DAYS]
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

        # pivot to (T × N) matrices
        ret_mat = df_win.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
        sig_mat = df_win.pivot(index="date", columns="ticker", values=sig_col).sort_index()

        cd_b = ret_mat.index.intersection(sig_mat.index)
        ct_b = ret_mat.columns.intersection(sig_mat.columns)
        ret_mat = ret_mat.loc[cd_b, ct_b]
        sig_mat = sig_mat.loc[cd_b, ct_b]

        # coverage filter
        obs_count = (ret_mat.notna() & sig_mat.notna()).sum(axis=0)
        ct_b = obs_count[obs_count >= 30].index
        if len(ct_b) == 0:
            continue
        ret_mat = ret_mat[ct_b]
        sig_mat = sig_mat[ct_b]
        ts_block = ticker_to_sector.reindex(ct_b).dropna()
        ct_b = ts_block.index
        ret_mat = ret_mat[ct_b]
        sig_mat = sig_mat[ct_b]

        tickers = ct_b.tolist()
        N = len(tickers)

        # split train / eval
        train_mask = sig_mat.index.isin(train_dates_window)
        eval_mask  = sig_mat.index.isin(block_test_dates)

        R_train = ret_mat.loc[train_mask].fillna(0.0).to_numpy(float)
        P_train = sig_mat.loc[train_mask].fillna(0.0).to_numpy(float)
        R_eval  = ret_mat.loc[eval_mask]
        P_eval  = sig_mat.loc[eval_mask].fillna(0.0).to_numpy(float)
        eval_dates = ret_mat.loc[eval_mask].index

        if R_train.shape[0] < INITIAL_TRAIN_DAYS // 2:
            continue

        print(
            f"  [{model_name}] block {block_num}: "
            f"train {pd.Timestamp(train_dates_window[0]).date()} → "
            f"{pd.Timestamp(train_dates_window[-1]).date()} | "
            f"predict {pd.Timestamp(block_test_dates[0]).date()} → "
            f"{pd.Timestamp(block_test_dates[-1]).date()} | N={N}",
            flush=True,
        )

        # ── fit edge stats on training window ────────────────
        stats = compute_edge_stats(R_train, P_train, lag=1)
        sources, targets, W_sparse = select_sources_targets(
            stats, KS, KT, TSTAT_MIN, MIN_COUNT, WEIGHT_MODE
        )
        edge_counts.append(int(np.count_nonzero(W_sparse)))

        # Build M: restricted to (targets × sources), weighted by A[i,j]
        M = np.zeros((N, N))
        if len(sources) > 0 and len(targets) > 0:
            keep = (np.abs(stats.tstat) >= TSTAT_MIN) & (stats.n_eff >= MIN_COUNT)
            np.fill_diagonal(keep, False)
            M[np.ix_(targets, sources)] = stats.A[np.ix_(targets, sources)]
            M = np.where(keep, M, 0.0)
            np.fill_diagonal(M, 0.0)

        # ── generate predictions for eval window ─────────────
        # prediction score for day t: M @ P[t-1]  (lag=1)
        # eval rows start at index 0 of eval window;
        # we need the last row of training as the lag-1 signal for the first eval day
        last_train_sig = P_train[-1]  # signal on last training day

        pred_rows = {}
        for t_idx, date in enumerate(eval_dates):
            if t_idx == 0:
                sig_lag = last_train_sig
            else:
                sig_lag = P_eval[t_idx - 1]

            scores = M @ sig_lag  # (N,) raw scores
            pred_rows[date] = pd.Series(scores, index=tickers)

        if pred_rows:
            pred_chunks.append(pd.DataFrame(pred_rows).T)

    if len(pred_chunks) == 0:
        print(f"[{model_name}] No prediction blocks produced.", flush=True)
        continue

    pred_all_df = pd.concat(pred_chunks, axis=0).sort_index()
    pred_all_df = pred_all_df[~pred_all_df.index.duplicated(keep="last")]

    ret_full = df.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
    ce_d = pred_all_df.index.intersection(ret_full.index)
    ce_c = pred_all_df.columns.intersection(ret_full.columns)
    pred_all_df = pred_all_df.loc[ce_d, ce_c]
    ret_eval_df = ret_full.loc[ce_d, ce_c]
    ts_eval     = ticker_to_sector.loc[ce_c]

    # ── evaluate ─────────────────────────────────────────────
    daily_spreads = pred_all_df.apply(
        lambda row: daily_spread(row, ret_eval_df.loc[row.name], q=Q), axis=1
    ).dropna()

    daily_spreads_sn = pred_all_df.apply(
        lambda row: daily_sector_neutral_spread(
            row, ret_eval_df.loc[row.name],
            ticker_to_sector=ts_eval, q=Q, min_names=MIN_NAMES_PER_SECTOR_NEUTRAL,
        ), axis=1
    ).dropna()

    daily_ic = pred_all_df.apply(
        lambda row: row.corr(ret_eval_df.loc[row.name], method="spearman"), axis=1
    ).dropna()

    mean_edges = float(np.mean(edge_counts)) if edge_counts else np.nan

    summary_row = pd.DataFrame([{
        "model":           model_name,
        "sig_col":         sig_col,
        "ks":              KS,
        "kt":              KT,
        "tstat_min":       TSTAT_MIN,
        "rolling_train_days": INITIAL_TRAIN_DAYS,
        "refit_every_days":   REFIT_EVERY_DAYS,
        "mean_active_edges":  mean_edges,
        "mean_daily_spread":  daily_spreads.mean(),
        "annualized_spread_sharpe": annualize_sharpe(sharpe_from_series(daily_spreads)),
        "mean_daily_spread_sn": daily_spreads_sn.mean(),
        "annualized_spread_sharpe_sn": annualize_sharpe(sharpe_from_series(daily_spreads_sn)),
        "mean_daily_ic":  daily_ic.mean(),
        "n_eval_days":    len(pred_all_df),
        "n_refits":       len(pred_chunks),
    }])

    all_summaries.append(summary_row)
    summary_row.to_csv(OUT_DIR / f"rolling_summary_{model_name}.csv", index=False)
    pred_all_df.to_csv(OUT_DIR / f"predicted_returns_{model_name}_rolling.csv")

    print(f"\n[{model_name}] Summary:", flush=True)
    print(summary_row.to_string(index=False), flush=True)

# ============================================================
# combined summary
# ============================================================
if not all_summaries:
    raise ValueError("No models produced outputs.")

summary_all = pd.concat(all_summaries, ignore_index=True)
summary_all = summary_all.sort_values(
    ["annualized_spread_sharpe_sn", "annualized_spread_sharpe"],
    ascending=False,
).reset_index(drop=True)

summary_all.to_csv(OUT_DIR / "rolling_summary_all_models.csv", index=False)

print("\n" + "=" * 70, flush=True)
print("ALL MODEL SUMMARY", flush=True)
print("=" * 70, flush=True)
print(summary_all.to_string(index=False), flush=True)
print("\nSaved outputs to:", OUT_DIR.resolve(), flush=True)
