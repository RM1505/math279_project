#!/usr/bin/env python3
"""36_ols_vs_ridge.py

OLS (no regularization) version of the rolling sector-block strategy from
script 27, with transaction costs included.  Purpose: sanity-check that ridge
regularization adds value over plain OLS.

Same design as 27 / 33:
  - 750-day training window, refit every 21 days
  - Sector-block W matrix (separate fit per GICS sector, MIN_SECTOR_ASSETS=4)
  - Exp-decay OFI signal, half-life sweep [15, 30, 45, 60, 90] min
  - Optional Gavish-Donoho SVD denoising of W
  - Transaction-cost analysis at 5/10/20/30 bps round-trip
  - Next-day OPCL prediction (identical timing to script 27)
  - GOOGL filtered out (same as 27/33)

OLS fit:  W = lstsq(X, Y)   — equivalent to ridge with λ→0
          No lambda selection step needed.

Outputs → results/rolling_adjacency_ols/

Run from repo root:
    python cont_cucuringu_zhang/36_ols_vs_ridge.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate as _sci_integrate, optimize as _sci_optimize
from scipy.stats import rankdata


# ============================================================
# config
# ============================================================
INPUT_PATH = Path("data/processed/feature_table_with_residuals_10level.csv")
OUT_DIR    = Path("results/rolling_adjacency_ols")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS = 6
MAX_ASSETS       = 400
HALF_LIFE_SWEEP  = [15, 30, 45, 60, 90]   # minutes — focused subset
Q                = 0.10
MIN_TICKER_OBS   = 252
MIN_SECTOR_ASSETS             = 4
MIN_NAMES_PER_SECTOR_NEUTRAL  = 4

INITIAL_TRAIN_DAYS = 750
REFIT_EVERY_DAYS   = 21

TC_SCENARIOS = [5, 10, 20, 30]   # bps round-trip

MODEL_SPECS = [
    # ── OLS, no GD ───────────────────────────────────────────
    {"name": "ols_sb_hl45",     "use_gd": False, "hl": 45, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_hl15",     "use_gd": False, "hl": 15, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_hl30",     "use_gd": False, "hl": 30, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_hl60",     "use_gd": False, "hl": 60, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_hl90",     "use_gd": False, "hl": 90, "sig_cols": None, "rank_norm": False},
    # ── OLS + Gavish-Donoho SVD truncation of W ──────────────
    {"name": "ols_sb_gd_hl45",  "use_gd": True,  "hl": 45, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_gd_hl15",  "use_gd": True,  "hl": 15, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_gd_hl30",  "use_gd": True,  "hl": 30, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_gd_hl60",  "use_gd": True,  "hl": 60, "sig_cols": None, "rank_norm": False},
    {"name": "ols_sb_gd_hl90",  "use_gd": True,  "hl": 90, "sig_cols": None, "rank_norm": False},
    # ── OLS + GD + rank-normalized OFI ───────────────────────
    {"name": "ols_sb_gd_hl30_rn", "use_gd": True, "hl": 30, "sig_cols": None, "rank_norm": True},
    {"name": "ols_sb_gd_hl45_rn", "use_gd": True, "hl": 45, "sig_cols": None, "rank_norm": True},
]


# ============================================================
# Gavish-Donoho optimal hard thresholding
# ============================================================
def _mp_eigenvalue_median(beta: float) -> float:
    beta = float(np.clip(beta, 1e-9, 1.0 - 1e-12))
    lp = (1.0 + np.sqrt(beta)) ** 2
    lm = (1.0 - np.sqrt(beta)) ** 2

    def pdf(x: float) -> float:
        v = (lp - x) * (x - lm)
        return np.sqrt(max(v, 0.0)) / (2.0 * np.pi * beta * max(x, 1e-15))

    total, _ = _sci_integrate.quad(pdf, lm, lp, limit=300, epsabs=1e-9)

    def cdf_minus_half(t: float) -> float:
        v, _ = _sci_integrate.quad(pdf, lm, t, limit=300, epsabs=1e-9)
        return v / total - 0.5

    return float(_sci_optimize.brentq(cdf_minus_half, lm + 1e-10, lp - 1e-10, xtol=1e-8))


def gavish_donoho_denoise(M: np.ndarray) -> tuple[np.ndarray, int, float]:
    M = np.asarray(M, dtype=float)
    m, n = M.shape
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return M.copy(), 0, 0.0
    if s.size == 0 or s.max() < 1e-14:
        return M.copy(), 0, 0.0
    y_med = float(np.median(s))
    if y_med < 1e-14:
        return np.zeros_like(M), 0, 0.0
    beta = min(m, n) / max(m, n)
    omega = np.sqrt(
        2.0 * (1.0 + beta)
        + 8.0 * beta / (1.0 + beta + np.sqrt(beta ** 2 + 14.0 * beta + 1.0))
    )
    mu_beta = _mp_eigenvalue_median(beta)
    lambda_star = (omega / np.sqrt(mu_beta)) * y_med
    s_thresh = np.where(s >= lambda_star, s, 0.0)
    M_denoised = (U * s_thresh) @ Vt
    return M_denoised, int((s_thresh > 0).sum()), float(lambda_star)


# ============================================================
# exp-decay OFI kernel
# ============================================================
def build_exp_decay_kernel(n_minutes: int, half_life: float) -> np.ndarray:
    lam = np.log(2) / half_life
    k = np.arange(n_minutes, 0, -1, dtype=float)
    weights = np.exp(-lam * k)
    weights /= weights.sum()
    return weights


# ============================================================
# helpers
# ============================================================
def get_minute_cols_from_columns(columns: list[str]) -> list[str]:
    return sorted(
        [c for c in columns if c.startswith("minute_")],
        key=lambda x: int(x.split("_")[1]),
    )


def zscore_with_train_stats(train_mat, test_mat):
    mean = train_mat.mean(axis=0, skipna=True)
    std  = train_mat.std(axis=0, skipna=True, ddof=0).replace(0, 1.0)
    return (train_mat - mean) / std, (test_mat - mean) / std, mean, std


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
    min_names_per_sector: int = 4,
) -> float:
    tmp = pd.concat([scores.rename("s"), returns.rename("r")], axis=1).dropna()
    if tmp.empty:
        return np.nan
    tmp = tmp.join(ticker_to_sector.rename("sec"), how="left").dropna(subset=["sec"])
    spreads = []
    for _, g in tmp.groupby("sec"):
        n = len(g)
        if n < min_names_per_sector:
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
# OLS fitting (replaces ridge)
# ============================================================
def fit_ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ordinary least squares: W = lstsq(X, Y)[0].

    Uses numpy's lstsq (SVD-based pseudo-inverse) which is numerically stable
    even when X has near-collinear columns.  Equivalent to ridge with λ → 0.
    """
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def fit_sector_block_ols(
    X_train_df: pd.DataFrame,
    Y_train_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
) -> pd.DataFrame:
    """Sector-block OLS: one lstsq fit per GICS sector."""
    y_tickers = Y_train_df.columns.tolist()
    x_cols    = X_train_df.columns.tolist()
    W_full    = pd.DataFrame(0.0, index=x_cols, columns=y_tickers, dtype=float)

    sector_to_tickers = (
        ticker_to_sector.rename("sector")
        .reset_index()
        .rename(columns={"index": "ticker"})
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    for sector, sec_tickers in sorted(sector_to_tickers.items()):
        sec_tickers = [t for t in sec_tickers if t in Y_train_df.columns]
        if len(sec_tickers) < MIN_SECTOR_ASSETS:
            continue
        sec_x_cols = [c for c in x_cols if c in sec_tickers]
        if not sec_x_cols:
            continue
        Xg = X_train_df[sec_x_cols].to_numpy(dtype=float)
        Yg = Y_train_df[sec_tickers].to_numpy(dtype=float)
        W_full.loc[sec_x_cols, sec_tickers] = fit_ols(Xg, Yg)

    return W_full


def predict_from_w(W: pd.DataFrame, X_eval_z_df: pd.DataFrame,
                   y_mean: pd.Series, y_std: pd.Series) -> pd.DataFrame:
    common_x = X_eval_z_df.columns.intersection(W.index)
    pred_z   = X_eval_z_df[common_x].to_numpy(float) @ W.loc[common_x].to_numpy(float)
    cols     = W.columns
    pred     = pred_z * y_std.loc[cols].to_numpy(float) + y_mean.loc[cols].to_numpy(float)
    return pd.DataFrame(pred, index=X_eval_z_df.index, columns=cols)


# ============================================================
# transaction cost functions  (identical to script 33)
# ============================================================
def build_gross_weights(pred_row: pd.Series, q: float) -> pd.Series:
    valid = pred_row.dropna()
    n = len(valid)
    weights = pd.Series(0.0, index=pred_row.index)
    if n < 2:
        return weights
    k = max(1, int(n * q))
    if 2 * k > n:
        return weights
    order = valid.sort_values()
    weights.loc[order.index[-k:]] =  1.0 / k
    weights.loc[order.index[:k]]  = -1.0 / k
    return weights


def build_sn_weights(
    pred_row: pd.Series,
    ticker_to_sector: pd.Series,
    q: float,
    min_per_sector: int,
) -> pd.Series:
    weights = pd.Series(0.0, index=pred_row.index)
    combined = pd.concat(
        [pred_row.rename("score"), ticker_to_sector.rename("sector")], axis=1
    ).dropna(subset=["score", "sector"])
    sector_dicts: dict = {}
    for sector, g in combined.groupby("sector"):
        n = len(g)
        if n < min_per_sector:
            continue
        k = max(1, int(n * q))
        if 2 * k > n:
            continue
        g_sorted = g.sort_values("score")
        sector_dicts[sector] = {
            "longs":  g_sorted.index[-k:].tolist(),
            "shorts": g_sorted.index[:k].tolist(),
            "k": k,
        }
    n_active = len(sector_dicts)
    if n_active == 0:
        return weights
    for d in sector_dicts.values():
        k = d["k"]
        for t in d["longs"]:
            if t in weights.index:
                weights.loc[t] += 1.0 / (k * n_active)
        for t in d["shorts"]:
            if t in weights.index:
                weights.loc[t] -= 1.0 / (k * n_active)
    return weights


def compute_turnover_series(
    pred_df: pd.DataFrame,
    q: float,
    ticker_to_sector: pd.Series | None = None,
    min_per_sector: int = 4,
    sn: bool = False,
) -> pd.Series:
    prev_w = pd.Series(0.0, index=pred_df.columns)
    turnovers: dict = {}
    for date, row in pred_df.iterrows():
        if sn and ticker_to_sector is not None:
            w = build_sn_weights(row, ticker_to_sector, q, min_per_sector)
        else:
            w = build_gross_weights(row, q)
        union = prev_w.index.union(w.index)
        delta = w.reindex(union).fillna(0.0) - prev_w.reindex(union).fillna(0.0)
        turnovers[date] = 0.5 * delta.abs().sum()
        prev_w = w.reindex(union).fillna(0.0)
    return pd.Series(turnovers)


def compute_net_spread(gross_spread: pd.Series, turnover: pd.Series, c_rt_bps: float) -> pd.Series:
    c = c_rt_bps / 10_000.0
    common = gross_spread.index.intersection(turnover.index)
    return (gross_spread.loc[common] - c * turnover.loc[common]).dropna()


def compute_breakeven_bps(gross_spread: pd.Series, turnover: pd.Series) -> float:
    common = gross_spread.index.intersection(turnover.index)
    mean_spread = gross_spread.loc[common].dropna().mean()
    mean_tau    = turnover.loc[common].dropna().mean()
    if mean_tau <= 0 or not np.isfinite(mean_tau):
        return np.nan
    return float(mean_spread / mean_tau * 10_000.0)


# ============================================================
# load and clean data
# ============================================================
print("Reading data...", flush=True)

needed_base_cols = ["date", "ticker", "residual_ret", "sector"]
header_df = pd.read_csv(INPUT_PATH, nrows=0)
minute_cols = get_minute_cols_from_columns(list(header_df.columns))
usecols = needed_base_cols + minute_cols

df = pd.read_csv(INPUT_PATH, usecols=usecols)
df["date"] = pd.to_datetime(df["date"])

if USE_LAST_N_YEARS is not None:
    cutoff = df["date"].max() - pd.DateOffset(years=USE_LAST_N_YEARS)
    df = df[df["date"] >= cutoff].copy()
    print(f"Using data from {cutoff.date()} onward", flush=True)

df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
df[minute_cols] = df[minute_cols].fillna(0.0)
df = df.dropna(subset=["ticker", "date", "residual_ret", "sector"]).copy()
# Drop GOOGL — identical LOB signals to GOOG, introduces near-collinear features
df = df[df["ticker"] != "GOOGL"].copy()

sector_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
df = df[df["ticker"].isin(sector_counts[sector_counts == 1].index)].copy()

ticker_to_sector_full = df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

# build exp-decay OFI signals
n_minutes = len(minute_cols)
minute_np = df[minute_cols].to_numpy(dtype=float)
print("Building exp-decay signals:", flush=True)
for _hl in HALF_LIFE_SWEEP:
    _kernel = build_exp_decay_kernel(n_minutes=n_minutes, half_life=_hl)
    df[f"ofi_hl{_hl}"] = minute_np @ _kernel
    print(f"  hl={_hl:3d}min  weight_ratio={_kernel[-1]/_kernel[0]:.1f}x", flush=True)
del minute_np

# rank-normalized OFI
print("Building rank-normalized signals...", flush=True)
for _hl in HALF_LIFE_SWEEP:
    df[f"ofi_hl{_hl}_rn"] = df.groupby("date")[f"ofi_hl{_hl}"].rank(pct=True)

all_dates = np.sort(df["date"].unique())

if len(all_dates) < INITIAL_TRAIN_DAYS + 2:
    raise ValueError(f"Not enough dates ({len(all_dates)}) for INITIAL_TRAIN_DAYS={INITIAL_TRAIN_DAYS}.")

# ============================================================
# global coverage filter
# ============================================================
sig_proxy = df.pivot(index="date", columns="ticker", values="ofi_hl45")
ret_proxy = df.pivot(index="date", columns="ticker", values="residual_ret")

cd = sig_proxy.index.intersection(ret_proxy.index)
ct = sig_proxy.columns.intersection(ret_proxy.columns)
sig_proxy, ret_proxy = sig_proxy.loc[cd, ct], ret_proxy.loc[cd, ct]

ticker_obs = (sig_proxy.notna() & ret_proxy.notna()).sum(axis=0)
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
# rolling walk-forward
# ============================================================
test_start_idx = INITIAL_TRAIN_DAYS
test_dates     = pd.Index(all_dates[test_start_idx:])
refit_points   = list(range(0, len(test_dates), REFIT_EVERY_DAYS))

all_model_summaries   = []
all_model_gd_histories = []

for model_spec in MODEL_SPECS:
    model_name = model_spec["name"]
    use_gd     = model_spec["use_gd"]
    rank_norm  = model_spec.get("rank_norm", False)
    hl         = model_spec["hl"]

    if model_spec["sig_cols"] is not None:
        sig_cols = model_spec["sig_cols"]
    else:
        base_col = f"ofi_hl{hl}"
        sig_cols = [f"{base_col}_rn" if rank_norm else base_col]

    print("\n" + "=" * 80, flush=True)
    print(f"RUNNING MODEL: {model_name}  (sig_cols={sig_cols}, use_gd={use_gd})", flush=True)
    print("=" * 80, flush=True)

    pred_chunks = []
    gd_records  = []

    for block_num, start_offset in enumerate(refit_points, start=1):
        block_test_dates = test_dates[start_offset:start_offset + REFIT_EVERY_DAYS]
        if len(block_test_dates) == 0:
            continue

        first_pred_date     = block_test_dates[0]
        eligible_train_dates = all_dates[all_dates < first_pred_date]
        if len(eligible_train_dates) < INITIAL_TRAIN_DAYS + 1:
            continue

        train_dates_window = eligible_train_dates[-INITIAL_TRAIN_DAYS:]

        df_train = df[df["date"].isin(train_dates_window)].copy()
        df_block = df[df["date"].isin(block_test_dates)].copy()

        if df_train.empty or df_block.empty:
            continue

        df_win = pd.concat([df_train, df_block]).sort_values(["date", "ticker"]).reset_index(drop=True)

        ret_mat   = df_win.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
        first_sig = df_win.pivot(index="date", columns="ticker", values=sig_cols[0]).sort_index()

        cd_b = first_sig.index.intersection(ret_mat.index)
        ct_b = first_sig.columns.intersection(ret_mat.columns)
        first_sig = first_sig.loc[cd_b, ct_b]
        ret_mat   = ret_mat.loc[cd_b, ct_b]

        block_ticker_obs = (first_sig.notna() & ret_mat.notna()).sum(axis=0)
        block_keep       = block_ticker_obs[block_ticker_obs >= 30].index
        ret_mat          = ret_mat[block_keep]
        ts_block         = ticker_to_sector.reindex(block_keep).dropna()
        block_keep       = ts_block.index.intersection(block_keep)
        ret_mat          = ret_mat[block_keep]

        if len(block_keep) == 0:
            continue

        signal_mat = df_win.pivot(
            index="date", columns="ticker", values=sig_cols[0]
        ).sort_index().loc[cd_b, block_keep]

        signal_train = signal_mat.loc[signal_mat.index.isin(train_dates_window)]
        ret_train    = ret_mat.loc[ret_mat.index.isin(train_dates_window)]
        signal_eval  = signal_mat.loc[signal_mat.index.isin(block_test_dates)]
        ret_eval     = ret_mat.loc[ret_mat.index.isin(block_test_dates)]

        # 1-day lag: X_t predicts Y_{t+1}
        X_train_df = signal_train.iloc[:-1].copy()
        Y_train_df = ret_train.iloc[1:].copy()
        X_train_df.index = Y_train_df.index

        X_eval_df = signal_eval.iloc[:-1].copy()
        Y_eval_df = ret_eval.iloc[1:].copy()
        X_eval_df.index = Y_eval_df.index

        if len(X_train_df) < 5 or len(X_eval_df) < 1:
            continue

        cc = (X_train_df.columns.intersection(Y_train_df.columns)
              .intersection(X_eval_df.columns).intersection(Y_eval_df.columns))
        X_train_df, Y_train_df = X_train_df[cc], Y_train_df[cc]
        X_eval_df,  Y_eval_df  = X_eval_df[cc],  Y_eval_df[cc]
        ts_block = ts_block.loc[cc]

        X_train_z_df, X_eval_z_df, _, _ = zscore_with_train_stats(X_train_df, X_eval_df)
        Y_train_z_df, _, y_mean, y_std   = zscore_with_train_stats(Y_train_df, Y_eval_df)

        X_train_z_df = X_train_z_df.fillna(0.0)
        X_eval_z_df  = X_eval_z_df.fillna(0.0)
        Y_train_z_df = Y_train_z_df.fillna(0.0)

        # variance filter
        y_var = Y_train_z_df.var(axis=0, ddof=0)
        x_var = X_train_z_df.var(axis=0, ddof=0)
        kept_cols = X_train_z_df.columns[(x_var > 0) & (y_var > 0)]
        if len(kept_cols) == 0:
            continue
        X_train_z_df = X_train_z_df[kept_cols]
        Y_train_z_df = Y_train_z_df[kept_cols]
        X_eval_z_df  = X_eval_z_df[kept_cols]
        Y_eval_df    = Y_eval_df[kept_cols]
        y_mean, y_std = y_mean[kept_cols], y_std[kept_cols]
        ts_block     = ts_block.loc[kept_cols]

        n_assets_block = len(kept_cols)
        print(
            f"[{model_name}] block {block_num}: "
            f"train {pd.Timestamp(train_dates_window[0]).date()} to "
            f"{pd.Timestamp(train_dates_window[-1]).date()} | "
            f"predict {pd.Timestamp(block_test_dates[0]).date()} to "
            f"{pd.Timestamp(block_test_dates[-1]).date()} | "
            f"assets={n_assets_block}",
            flush=True,
        )

        # ── OLS fit (sector-block) ────────────────────────────
        W = fit_sector_block_ols(X_train_z_df, Y_train_z_df, ts_block)

        # optional Gavish-Donoho SVD truncation
        gd_rank = None
        if use_gd:
            Wmat = W.to_numpy(dtype=float)
            W_gd, gd_rank, gd_thresh = gavish_donoho_denoise(Wmat)
            print(f"      GD: kept {gd_rank} singular values (threshold={gd_thresh:.4f})", flush=True)
            W = pd.DataFrame(W_gd, index=W.index, columns=W.columns)

        pred_eval_df = predict_from_w(W, X_eval_z_df, y_mean, y_std)
        pred_chunks.append(pred_eval_df)

        gd_records.append({"model": model_name, "block": block_num,
                            "gd_rank": gd_rank, "n_assets": n_assets_block,
                            "pred_start": pd.Timestamp(block_test_dates[0]).date()})

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

    # ── gross / SN spread ────────────────────────────────────
    daily_spreads = pred_all_df.apply(
        lambda row: daily_spread(row, ret_eval_df.loc[row.name], q=Q), axis=1
    ).dropna()

    daily_spreads_sn = pred_all_df.apply(
        lambda row: daily_sector_neutral_spread(
            row, ret_eval_df.loc[row.name],
            ticker_to_sector=ts_eval, q=Q,
            min_names_per_sector=MIN_NAMES_PER_SECTOR_NEUTRAL,
        ), axis=1
    ).dropna()

    daily_ic = pred_all_df.apply(
        lambda row: row.corr(ret_eval_df.loc[row.name], method="spearman"), axis=1
    ).dropna()

    # ── transaction costs ─────────────────────────────────────
    to_gross = compute_turnover_series(pred_all_df, Q)
    to_sn    = compute_turnover_series(pred_all_df, Q, ts_eval,
                                       MIN_NAMES_PER_SECTOR_NEUTRAL, sn=True)

    be_gross = compute_breakeven_bps(daily_spreads,    to_gross)
    be_sn    = compute_breakeven_bps(daily_spreads_sn, to_sn)

    net_sharpes: dict = {}
    for bps in TC_SCENARIOS:
        ns_gross = annualize_sharpe(sharpe_from_series(
            compute_net_spread(daily_spreads, to_gross, bps)))
        ns_sn    = annualize_sharpe(sharpe_from_series(
            compute_net_spread(daily_spreads_sn, to_sn, bps)))
        net_sharpes[f"net_sharpe_gross_{bps}bps"] = ns_gross
        net_sharpes[f"net_sharpe_sn_{bps}bps"]    = ns_sn

    mean_gd_rank = float(
        pd.DataFrame(gd_records)["gd_rank"].dropna().mean()
    ) if use_gd and gd_records else float("nan")

    summary_row = pd.DataFrame([{
        "model":           model_name,
        "use_gd":          use_gd,
        "sig_cols":        sig_cols[0],
        "half_life_min":   hl,
        "rank_norm":       rank_norm,
        "rolling_train_days": INITIAL_TRAIN_DAYS,
        "refit_every_days":   REFIT_EVERY_DAYS,
        "mean_daily_spread":  daily_spreads.mean(),
        "annualized_spread_sharpe": annualize_sharpe(sharpe_from_series(daily_spreads)),
        "mean_daily_spread_sn": daily_spreads_sn.mean(),
        "annualized_spread_sharpe_sn": annualize_sharpe(sharpe_from_series(daily_spreads_sn)),
        "mean_daily_ic": daily_ic.mean(),
        "mean_daily_turnover_gross": to_gross.mean(),
        "annualized_turnover_gross": to_gross.mean() * 252,
        "mean_daily_turnover_sn":   to_sn.mean(),
        "annualized_turnover_sn":   to_sn.mean() * 252,
        "breakeven_rt_bps_gross": be_gross,
        "breakeven_rt_bps_sn":    be_sn,
        "mean_gd_rank": mean_gd_rank,
        "n_eval_days":  len(pred_all_df),
        **net_sharpes,
    }])

    all_model_summaries.append(summary_row)
    all_model_gd_histories.append(pd.DataFrame(gd_records))

    summary_row.to_csv(OUT_DIR / f"rolling_summary_{model_name}.csv", index=False)
    pred_all_df.to_csv(OUT_DIR / f"predicted_returns_{model_name}_rolling.csv")
    to_gross.to_csv(OUT_DIR / f"turnover_gross_{model_name}.csv", header=["turnover"])
    to_sn.to_csv(OUT_DIR / f"turnover_sn_{model_name}.csv", header=["turnover"])

    print(f"\n[{model_name}] Rolling summary:", flush=True)
    print(summary_row.to_string(index=False), flush=True)

# ============================================================
# combined summary
# ============================================================
if len(all_model_summaries) == 0:
    raise ValueError("No models produced usable outputs.")

summary_all = pd.concat(all_model_summaries, axis=0, ignore_index=True)
summary_all = summary_all.sort_values(
    ["annualized_spread_sharpe_sn", "annualized_spread_sharpe", "mean_daily_ic"],
    ascending=False,
).reset_index(drop=True)

summary_all.to_csv(OUT_DIR / "rolling_summary_all_models.csv", index=False)

if all_model_gd_histories:
    pd.concat(all_model_gd_histories, ignore_index=True).to_csv(
        OUT_DIR / "gd_rank_history.csv", index=False
    )

print("\n" + "=" * 80, flush=True)
print("ALL MODEL SUMMARY", flush=True)
print("=" * 80, flush=True)
print(summary_all.to_string(index=False), flush=True)
print("\nSaved all outputs to:", OUT_DIR.resolve(), flush=True)
