#!/usr/bin/env python3
"""27_ridge_expdecayOFI.py

Rolling adjacency analysis — ridge only:
- exponential-decay daily OFI signal (replaces PCA)
- in-sample IC lambda selection
- comprehensive sweep: half-life, rank-norm, multi-scale (hl30+hl90), multi-lag

Run from repo root:
    python cont_cucuringu_zhang/27_ridge_expdecayOFI.py
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
OUT_DIR = Path("results/rolling_adjacency_ridge")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS = 6
MAX_ASSETS = 400
# All half-lives pre-computed upfront (minutes)
HALF_LIFE_SWEEP = [15, 20, 25, 30, 35, 40, 45, 60, 90]
Q = 0.10
MIN_TICKER_OBS = 252
MIN_SECTOR_ASSETS = 4
MIN_NAMES_PER_SECTOR_NEUTRAL = 4

INITIAL_TRAIN_DAYS = 750
REFIT_EVERY_DAYS = 21

# Restored intermediate values to avoid gaps between 10→100
RIDGE_GRID = [1.0, 10.0, 50.0, 250.0, 1000.0]

# sig_cols=None → infer from hl + rank_norm
# sig_cols=list → multi-signal (e.g. multi-scale hl30+hl90, or multi-lag hl30+lag1)
MODEL_SPECS = [
    # ── baselines ───────────────────────────────────────────────
    {"name": "ridge_sb",             "structure": "sector_block", "use_gd": False,
     "hl": 45, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd",          "structure": "sector_block", "use_gd": True,
     "hl": 45, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    # ── fine half-life sweep around hl=30 sweet spot ────────────
    {"name": "ridge_sb_gd_hl15",     "structure": "sector_block", "use_gd": True,
     "hl": 15, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd_hl20",     "structure": "sector_block", "use_gd": True,
     "hl": 20, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd_hl25",     "structure": "sector_block", "use_gd": True,
     "hl": 25, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd_hl30",     "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd_hl35",     "structure": "sector_block", "use_gd": True,
     "hl": 35, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd_hl40",     "structure": "sector_block", "use_gd": True,
     "hl": 40, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd_hl60",     "structure": "sector_block", "use_gd": True,
     "hl": 60, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_sb_gd_hl90",     "structure": "sector_block", "use_gd": True,
     "hl": 90, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    # ── rank-normalized OFI: cross-sectional percentile rank ────
    # Robust to earnings outliers; uniform marginal distribution
    {"name": "ridge_sb_gd_hl30_rn",  "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": None, "rank_norm": True,  "zero_diag": False},
    {"name": "ridge_sb_gd_hl90_rn",  "structure": "sector_block", "use_gd": True,
     "hl": 90, "sig_cols": None, "rank_norm": True,  "zero_diag": False},
    # ── multi-scale: hl=30 + hl=90 as 2N-feature model ─────────
    # W learns separate weights for fast vs slow order flow
    {"name": "ridge_sb_gd_ms3090",   "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": ["ofi_hl30", "ofi_hl90"], "rank_norm": False, "zero_diag": False},
    # ── multi-lag: today's + yesterday's hl=30 OFI ──────────────
    # Tests persistence of order flow imbalance
    {"name": "ridge_sb_gd_lag1",     "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": ["ofi_hl30", "ofi_hl30_lag1"], "rank_norm": False, "zero_diag": False},
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
    """Apply Gavish-Donoho (2014) optimal hard threshold to matrix M.

    Returns (M_denoised, n_singular_values_kept, threshold).
    """
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
    omega_star = omega / np.sqrt(mu_beta)
    lambda_star = omega_star * y_med

    s_thresh = np.where(s >= lambda_star, s, 0.0)
    M_denoised = (U * s_thresh) @ Vt
    return M_denoised, int((s_thresh > 0).sum()), float(lambda_star)


# ============================================================
# exponential decay kernel
# ============================================================
def build_exp_decay_kernel(n_minutes: int, half_life: float) -> np.ndarray:
    """
    Builds a normalized exponential decay kernel over intraday minutes.
    Weight of minute i (0=open, n-1=close) ~ exp(-lambda * (n-1-i))
    where lambda = log(2)/half_life.
    Minutes closer to close get exponentially more weight.
    """
    lam = np.log(2) / half_life
    k = np.arange(n_minutes, 0, -1, dtype=float)  # [n, n-1, ..., 1]
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


def _base_ticker(col: str) -> str:
    """Extract base ticker from possibly-suffixed column like 'AAPL__ofi_hl30'."""
    return col.split("__")[0]


def zscore_with_train_stats(
    train_mat: pd.DataFrame,
    test_mat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mean = train_mat.mean(axis=0, skipna=True)
    std = train_mat.std(axis=0, skipna=True, ddof=0).replace(0, 1.0)
    train_z = (train_mat - mean) / std
    test_z = (test_mat - mean) / std
    return train_z, test_z, mean, std


def daily_spread(scores: pd.Series, returns: pd.Series, q: float = 0.10) -> float:
    tmp = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    n = len(tmp)
    if n < 2:
        return np.nan
    k = max(1, int(n * q))
    if 2 * k > n:
        return np.nan
    tmp = tmp.sort_values("score")
    return float(tmp.iloc[-k:]["ret"].mean() - tmp.iloc[:k]["ret"].mean())


def daily_sector_neutral_spread(
    scores: pd.Series,
    returns: pd.Series,
    ticker_to_sector: pd.Series,
    q: float = 0.10,
    min_names_per_sector: int = 4,
) -> float:
    tmp = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    if tmp.empty:
        return np.nan
    tmp = tmp.join(ticker_to_sector.rename("sector"), how="left").dropna(subset=["sector"])
    if tmp.empty:
        return np.nan
    sector_spreads = []
    for _, g in tmp.groupby("sector"):
        n = len(g)
        if n < min_names_per_sector:
            continue
        k = max(1, int(n * q))
        if 2 * k > n:
            continue
        g = g.sort_values("score")
        sector_spreads.append(float(g.iloc[-k:]["ret"].mean() - g.iloc[:k]["ret"].mean()))
    return float(np.mean(sector_spreads)) if sector_spreads else np.nan


def sharpe_from_series(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    s = x.std(ddof=1)
    if s == 0 or not np.isfinite(s):
        return np.nan
    return float(x.mean() / s)


def annualize_sharpe(daily_sharpe: float) -> float:
    return float(daily_sharpe * np.sqrt(252)) if np.isfinite(daily_sharpe) else np.nan


# ============================================================
# model fitting — generalized for multi-signal X
# ============================================================
def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge: W = (X'X + λI)^{-1} X'Y"""
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def fit_sector_block_model(
    X_train_df: pd.DataFrame,
    Y_train_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    lam: float,
) -> pd.DataFrame:
    """Fit sector-block ridge.

    X columns may be base tickers (single-signal) OR 'TICKER__signal' (multi-signal).
    Y columns are always base tickers.
    W has shape (n_x_cols, n_y_tickers).
    """
    y_tickers = Y_train_df.columns.tolist()
    x_cols    = X_train_df.columns.tolist()
    W_full = pd.DataFrame(0.0, index=x_cols, columns=y_tickers, dtype=float)

    sector_to_tickers = (
        ticker_to_sector.rename("sector")
        .reset_index()
        .rename(columns={"index": "ticker"})
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    for sector, sec_y_tickers in sorted(sector_to_tickers.items()):
        sec_y_tickers = [t for t in sec_y_tickers if t in Y_train_df.columns]
        if len(sec_y_tickers) < MIN_SECTOR_ASSETS:
            continue
        sec_y_set = set(sec_y_tickers)
        # Collect all X cols whose base ticker belongs to this sector
        sec_x_cols = [c for c in x_cols if _base_ticker(c) in sec_y_set]
        if not sec_x_cols:
            continue
        Xg = X_train_df[sec_x_cols].to_numpy(dtype=float)
        Yg = Y_train_df[sec_y_tickers].to_numpy(dtype=float)
        W_full.loc[sec_x_cols, sec_y_tickers] = fit_ridge(Xg, Yg, lam)

    return W_full


def fit_model(
    structure: str,
    lam: float,
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
) -> np.ndarray | pd.DataFrame:
    X = X_train_z_df.to_numpy(dtype=float)
    Y = Y_train_z_df.to_numpy(dtype=float)
    if structure == "dense":
        return fit_ridge(X, Y, lam)
    if structure == "sector_block":
        return fit_sector_block_model(X_train_z_df, Y_train_z_df, ticker_to_sector, lam)
    raise ValueError(f"Unknown structure: {structure}")


def predict_from_model(W, X_eval_z_df, y_mean, y_std) -> pd.DataFrame:
    """Predict from W. Works for both single-signal (W square) and multi-signal (W rectangular)."""
    if isinstance(W, pd.DataFrame):
        common_x = X_eval_z_df.columns.intersection(W.index)
        pred_z = X_eval_z_df[common_x].to_numpy(dtype=float) @ W.loc[common_x].to_numpy(dtype=float)
        cols = W.columns
    else:
        pred_z = X_eval_z_df.to_numpy(dtype=float) @ W
        cols = X_eval_z_df.columns
    pred = pred_z * y_std.loc[cols].to_numpy(dtype=float) + y_mean.loc[cols].to_numpy(dtype=float)
    return pd.DataFrame(pred, index=X_eval_z_df.index, columns=cols)


def evaluate_predictions(pred_df, ret_df, ticker_to_sector, q, min_sn):
    ret_eval_df = ret_df[pred_df.columns].copy()
    sector_map = ticker_to_sector.loc[pred_df.columns]

    daily_spreads = pred_df.apply(
        lambda row: daily_spread(row, ret_eval_df.loc[row.name], q=q), axis=1
    ).dropna()

    daily_spreads_sn = pred_df.apply(
        lambda row: daily_sector_neutral_spread(
            row, ret_eval_df.loc[row.name],
            ticker_to_sector=sector_map, q=q, min_names_per_sector=min_sn,
        ), axis=1
    ).dropna()

    daily_ic = pred_df.apply(
        lambda row: row.corr(ret_eval_df.loc[row.name], method="spearman"), axis=1
    ).dropna()

    return daily_spreads, daily_spreads_sn, daily_ic


def choose_lambda_on_window(structure, lambda_grid, X_train_z_df, Y_train_z_df, ticker_to_sector):
    best_lam, best_score = lambda_grid[0], -np.inf

    for lam in lambda_grid:
        print(f"      trying lambda={lam}", flush=True)
        W = fit_model(structure, lam, X_train_z_df, Y_train_z_df, ticker_to_sector)
        # Handle multi-signal W (rectangular DataFrame)
        if isinstance(W, pd.DataFrame):
            common_x = X_train_z_df.columns.intersection(W.index)
            pred_np = X_train_z_df[common_x].to_numpy(float) @ W.loc[common_x].to_numpy(float)
            pred_cols = W.columns
        else:
            pred_np = X_train_z_df.to_numpy(float) @ W
            pred_cols = X_train_z_df.columns
        pred_df = pd.DataFrame(pred_np, index=X_train_z_df.index, columns=pred_cols)
        y_common = Y_train_z_df.columns.intersection(pred_cols)
        ic = pred_df[y_common].apply(
            lambda row: row.corr(Y_train_z_df.loc[row.name, y_common], method="spearman"), axis=1
        ).dropna()
        score = float(ic.mean()) if len(ic) else -np.inf
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lam = lam

    return float(best_lam)


# ============================================================
# load and clean
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
# Drop GOOGL — GOOG and GOOGL are both Alphabet share classes with nearly
# identical LOB signals; keeping both double-counts Alphabet in Communication
# Services and introduces near-collinear features. Keep GOOG (Class C, higher volume).
df = df[df["ticker"] != "GOOGL"].copy()

sector_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
df = df[df["ticker"].isin(sector_counts[sector_counts == 1].index)].copy()

ticker_to_sector_full = df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

# ── build all exp-decay OFI signals ─────────────────────────
n_minutes = len(minute_cols)
minute_np = df[minute_cols].to_numpy(dtype=float)
print("Building exp-decay signals:", flush=True)
for _hl in HALF_LIFE_SWEEP:
    _kernel = build_exp_decay_kernel(n_minutes=n_minutes, half_life=_hl)
    df[f"ofi_hl{_hl}"] = minute_np @ _kernel
    print(f"  hl={_hl:3d}min  weight_ratio={_kernel[-1]/_kernel[0]:.1f}x", flush=True)
del minute_np

# ── rank-normalized OFI: cross-sectional percentile per day ─
# Robust to heavy-tailed distributions (earnings, macro events)
print("Building rank-normalized signals...", flush=True)
for _hl in HALF_LIFE_SWEEP:
    df[f"ofi_hl{_hl}_rn"] = df.groupby("date")[f"ofi_hl{_hl}"].rank(pct=True)

# ── multi-lag: add yesterday's OFI for persistence tests ────
print("Building lag-1 signals...", flush=True)
df = df.sort_values(["ticker", "date"])
for _hl in [30, 90]:
    df[f"ofi_hl{_hl}_lag1"] = df.groupby("ticker")[f"ofi_hl{_hl}"].shift(1)
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

all_dates = np.sort(df["date"].unique())

if len(all_dates) < INITIAL_TRAIN_DAYS + 2:
    raise ValueError(f"Not enough dates ({len(all_dates)}) for INITIAL_TRAIN_DAYS={INITIAL_TRAIN_DAYS}.")

# ============================================================
# global coverage filter + asset cap (based on hl=45)
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

if len(keep_tickers) == 0:
    raise ValueError("No tickers remain after filtering.")

print(f"Using {len(keep_tickers)} assets after filtering/cap.", flush=True)

# ============================================================
# rolling walk-forward
# ============================================================
test_start_idx = INITIAL_TRAIN_DAYS
test_dates = pd.Index(all_dates[test_start_idx:])

if len(test_dates) == 0:
    raise ValueError("No test dates available.")

refit_points = list(range(0, len(test_dates), REFIT_EVERY_DAYS))

all_model_summaries = []
all_model_lambda_histories = []

for model_spec in MODEL_SPECS:
    model_name = model_spec["name"]
    structure  = model_spec["structure"]
    use_gd     = model_spec["use_gd"]
    zero_diag  = model_spec.get("zero_diag", False)
    rank_norm  = model_spec.get("rank_norm", False)
    hl         = model_spec["hl"]   # primary hl (metadata)

    # Resolve signal columns
    if model_spec["sig_cols"] is not None:
        sig_cols = model_spec["sig_cols"]
    else:
        base_col = f"ofi_hl{hl}"
        sig_cols = [f"{base_col}_rn" if rank_norm else base_col]

    print("\n" + "=" * 80, flush=True)
    print(f"RUNNING MODEL: {model_name}  (sig_cols={sig_cols}, use_gd={use_gd})", flush=True)
    print("=" * 80, flush=True)

    pred_chunks = []
    chosen_lambdas = []
    gd_ranks = []

    for block_num, start_offset in enumerate(refit_points, start=1):
        block_test_dates = test_dates[start_offset:start_offset + REFIT_EVERY_DAYS]
        if len(block_test_dates) == 0:
            continue

        first_pred_date = block_test_dates[0]
        eligible_train_dates = all_dates[all_dates < first_pred_date]
        if len(eligible_train_dates) < INITIAL_TRAIN_DAYS + 1:
            continue

        train_dates_window = eligible_train_dates[-INITIAL_TRAIN_DAYS:]

        df_train = df[df["date"].isin(train_dates_window)].copy()
        df_block = df[df["date"].isin(block_test_dates)].copy()

        if df_train.empty or df_block.empty:
            continue

        df_win = pd.concat([df_train, df_block]).sort_values(["date", "ticker"]).reset_index(drop=True)

        # ── returns matrix (always base tickers) ────────────────
        ret_mat = df_win.pivot(index="date", columns="ticker", values="residual_ret").sort_index()

        # ── coverage filter using first signal + returns ─────────
        first_sc = sig_cols[0]
        first_sig = df_win.pivot(index="date", columns="ticker", values=first_sc).sort_index()

        cd_b = first_sig.index.intersection(ret_mat.index)
        ct_b = first_sig.columns.intersection(ret_mat.columns)
        first_sig = first_sig.loc[cd_b, ct_b]
        ret_mat = ret_mat.loc[cd_b, ct_b]

        block_ticker_obs = (first_sig.notna() & ret_mat.notna()).sum(axis=0)
        block_keep = block_ticker_obs[block_ticker_obs >= 30].index
        ret_mat = ret_mat[block_keep]
        ts_block = ticker_to_sector.reindex(block_keep).dropna()
        block_keep = ts_block.index.intersection(block_keep)
        ret_mat = ret_mat[block_keep]

        if len(block_keep) == 0:
            continue

        # ── build full signal matrix (single or multi-signal) ───
        if len(sig_cols) == 1:
            signal_mat = df_win.pivot(
                index="date", columns="ticker", values=sig_cols[0]
            ).sort_index().loc[cd_b, block_keep]
        else:
            parts = []
            for sc in sig_cols:
                sm = df_win.pivot(index="date", columns="ticker", values=sc).sort_index()
                sm = sm.loc[cd_b, block_keep]
                sm.columns = [f"{t}__{sc}" for t in sm.columns]
                parts.append(sm)
            signal_mat = pd.concat(parts, axis=1)

        # ── split train / eval ───────────────────────────────────
        signal_train = signal_mat.loc[signal_mat.index.isin(train_dates_window)]
        ret_train    = ret_mat.loc[ret_mat.index.isin(train_dates_window)]
        signal_eval  = signal_mat.loc[signal_mat.index.isin(block_test_dates)]
        ret_eval     = ret_mat.loc[ret_mat.index.isin(block_test_dates)]

        X_train_df = signal_train.iloc[:-1].copy()
        Y_train_df = ret_train.iloc[1:].copy()
        X_train_df.index = Y_train_df.index

        X_eval_df = signal_eval.iloc[:-1].copy()
        Y_eval_df = ret_eval.iloc[1:].copy()
        X_eval_df.index = Y_eval_df.index

        if len(X_train_df) < 5 or len(X_eval_df) < 1:
            continue

        # ── align columns X↔Y ───────────────────────────────────
        if len(sig_cols) == 1:
            cc = (X_train_df.columns.intersection(Y_train_df.columns)
                  .intersection(X_eval_df.columns).intersection(Y_eval_df.columns))
            X_train_df, Y_train_df = X_train_df[cc], Y_train_df[cc]
            X_eval_df,  Y_eval_df  = X_eval_df[cc],  Y_eval_df[cc]
            ts_block = ts_block.loc[cc]
        else:
            # Multi-signal: X cols are 'TICKER__sc', Y cols are base tickers
            x_base = set(_base_ticker(c) for c in X_train_df.columns)
            cc_y = pd.Index([t for t in Y_train_df.columns
                              if t in x_base and t in Y_eval_df.columns])
            cc_y_set = set(cc_y)
            cc_x_tr = [c for c in X_train_df.columns if _base_ticker(c) in cc_y_set]
            cc_x_ev = [c for c in X_eval_df.columns  if _base_ticker(c) in cc_y_set]
            Y_train_df, Y_eval_df = Y_train_df[cc_y], Y_eval_df[cc_y]
            X_train_df, X_eval_df = X_train_df[cc_x_tr], X_eval_df[cc_x_ev]
            ts_block = ts_block.loc[cc_y]

        if len(Y_train_df.columns) == 0 or len(X_train_df.columns) == 0:
            continue

        # ── z-score (per signal block for multi-signal) ──────────
        if len(sig_cols) == 1:
            X_train_z_df, X_eval_z_df, _, _ = zscore_with_train_stats(X_train_df, X_eval_df)
        else:
            X_train_z_df = X_train_df.copy().astype(float)
            X_eval_z_df  = X_eval_df.copy().astype(float)
            for sc in sig_cols:
                cols_sc = [c for c in X_train_df.columns if c.endswith(f"__{sc}")]
                if not cols_sc:
                    continue
                tz, ez, _, _ = zscore_with_train_stats(X_train_df[cols_sc], X_eval_df[cols_sc])
                X_train_z_df[cols_sc] = tz.values
                X_eval_z_df[cols_sc]  = ez.values

        Y_train_z_df, _, y_mean, y_std = zscore_with_train_stats(Y_train_df, Y_eval_df)

        X_train_z_df = X_train_z_df.fillna(0.0)
        X_eval_z_df  = X_eval_z_df.fillna(0.0)
        Y_train_z_df = Y_train_z_df.fillna(0.0)

        # ── variance filter ──────────────────────────────────────
        y_var = Y_train_z_df.var(axis=0, ddof=0)
        x_var = X_train_z_df.var(axis=0, ddof=0)

        if len(sig_cols) == 1:
            kept_cols = X_train_z_df.columns[(x_var > 0) & (y_var > 0)]
            X_train_z_df = X_train_z_df[kept_cols]
            Y_train_z_df = Y_train_z_df[kept_cols]
            X_eval_z_df  = X_eval_z_df[kept_cols]
            Y_eval_df    = Y_eval_df[kept_cols]
            y_mean, y_std = y_mean[kept_cols], y_std[kept_cols]
            ts_block = ts_block.loc[kept_cols]
            if len(kept_cols) == 0:
                continue
        else:
            kept_y = Y_train_z_df.columns[y_var > 0]
            kept_x = X_train_z_df.columns[x_var > 0]
            kept_y_set = set(t for t in kept_y
                             if any(_base_ticker(xc) == t for xc in kept_x))
            kept_y = pd.Index([t for t in kept_y if t in kept_y_set])
            kept_x = pd.Index([c for c in kept_x if _base_ticker(c) in kept_y_set])
            if len(kept_y) == 0 or len(kept_x) == 0:
                continue
            Y_train_z_df = Y_train_z_df[kept_y]
            Y_eval_df    = Y_eval_df[kept_y]
            X_train_z_df = X_train_z_df[kept_x]
            X_eval_z_df  = X_eval_z_df[kept_x]
            y_mean, y_std = y_mean[kept_y], y_std[kept_y]
            ts_block = ts_block.loc[kept_y]

        n_assets_block = len(Y_train_z_df.columns)

        print(
            f"[{model_name}] block {block_num}: "
            f"train {pd.Timestamp(train_dates_window[0]).date()} to {pd.Timestamp(train_dates_window[-1]).date()} | "
            f"predict {pd.Timestamp(block_test_dates[0]).date()} to {pd.Timestamp(block_test_dates[-1]).date()} | "
            f"assets={n_assets_block}",
            flush=True,
        )

        lam_star = choose_lambda_on_window(structure, RIDGE_GRID, X_train_z_df, Y_train_z_df, ts_block)
        W = fit_model(structure, lam_star, X_train_z_df, Y_train_z_df, ts_block)

        # optional Gavish-Donoho SVD truncation of W
        gd_rank = None
        if use_gd:
            Wmat = W.to_numpy(dtype=float) if isinstance(W, pd.DataFrame) else W
            W_gd, gd_rank, gd_thresh = gavish_donoho_denoise(Wmat)
            print(f"      GD: kept {gd_rank} singular values (threshold={gd_thresh:.4f})", flush=True)
            if isinstance(W, pd.DataFrame):
                W = pd.DataFrame(W_gd, index=W.index, columns=W.columns)
            else:
                W = W_gd

        # optional diagonal zeroing
        if zero_diag:
            if isinstance(W, pd.DataFrame):
                W_arr = W.to_numpy(copy=True)
                np.fill_diagonal(W_arr, 0.0)
                W = pd.DataFrame(W_arr, index=W.index, columns=W.columns)
            else:
                W = W.copy()
                np.fill_diagonal(W, 0.0)

        pred_eval_df = predict_from_model(W, X_eval_z_df, y_mean, y_std)
        pred_chunks.append(pred_eval_df)

        chosen_lambdas.append(pd.DataFrame({
            "model": [model_name], "structure": [structure], "use_gd": [use_gd],
            "refit_block": [block_num],
            "pred_start": [pd.Timestamp(block_test_dates[0]).date()],
            "pred_end":   [pd.Timestamp(block_test_dates[-1]).date()],
            "lambda": [lam_star], "n_assets": [n_assets_block],
            "gd_rank": [gd_rank],
        }))

    if len(pred_chunks) == 0:
        print(f"[{model_name}] No rolling prediction blocks were produced.", flush=True)
        continue

    pred_all_df = pd.concat(pred_chunks, axis=0).sort_index()
    pred_all_df = pred_all_df[~pred_all_df.index.duplicated(keep="last")]

    ret_full = df.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
    ce_d = pred_all_df.index.intersection(ret_full.index)
    ce_c = pred_all_df.columns.intersection(ret_full.columns)
    pred_all_df = pred_all_df.loc[ce_d, ce_c]
    ret_eval_df = ret_full.loc[ce_d, ce_c]
    ts_eval = ticker_to_sector.loc[ce_c]

    daily_spreads, daily_spreads_sn, daily_ic = evaluate_predictions(
        pred_all_df, ret_eval_df, ts_eval, Q, MIN_NAMES_PER_SECTOR_NEUTRAL
    )

    lh_df = pd.concat(chosen_lambdas, axis=0, ignore_index=True)
    mean_gd_rank = float(lh_df["gd_rank"].dropna().mean()) if use_gd else float("nan")

    summary_row = pd.DataFrame([{
        "model": model_name, "structure": structure, "use_gd": use_gd,
        "sig_cols": "|".join(sig_cols), "half_life_minutes": hl,
        "rank_norm": rank_norm, "zero_diag": zero_diag,
        "rolling_train_days": INITIAL_TRAIN_DAYS,
        "refit_every_days": REFIT_EVERY_DAYS,
        "mean_daily_spread": daily_spreads.mean(),
        "annualized_spread_sharpe": annualize_sharpe(sharpe_from_series(daily_spreads)),
        "mean_daily_spread_sector_neutral": daily_spreads_sn.mean(),
        "annualized_spread_sharpe_sector_neutral": annualize_sharpe(sharpe_from_series(daily_spreads_sn)),
        "mean_daily_ic": daily_ic.mean(),
        "n_eval_days": len(pred_all_df),
        "n_refits": len(chosen_lambdas),
        "mean_gd_rank": mean_gd_rank,
    }])

    all_model_summaries.append(summary_row)
    all_model_lambda_histories.append(lh_df)

    summary_row.to_csv(OUT_DIR / f"rolling_summary_{model_name}.csv", index=False)
    lh_df.to_csv(OUT_DIR / f"rolling_lambda_history_{model_name}.csv", index=False)
    pred_all_df.to_csv(OUT_DIR / f"predicted_returns_{model_name}_rolling.csv")

    print(f"\n[{model_name}] Rolling summary:", flush=True)
    print(summary_row.to_string(index=False), flush=True)

# ============================================================
# combined summary
# ============================================================
if len(all_model_summaries) == 0:
    raise ValueError("No models produced usable outputs.")

summary_all = pd.concat(all_model_summaries, axis=0, ignore_index=True)
summary_all = summary_all.sort_values(
    ["annualized_spread_sharpe_sector_neutral", "annualized_spread_sharpe", "mean_daily_ic"],
    ascending=False
).reset_index(drop=True)

summary_all.to_csv(OUT_DIR / "rolling_summary_all_models.csv", index=False)

if all_model_lambda_histories:
    pd.concat(all_model_lambda_histories, axis=0, ignore_index=True).to_csv(
        OUT_DIR / "rolling_lambda_history_all_models.csv", index=False
    )

print("\n" + "=" * 80, flush=True)
print("ALL MODEL SUMMARY", flush=True)
print("=" * 80, flush=True)
print(summary_all.to_string(index=False), flush=True)
print("\nSaved all outputs to:", OUT_DIR.resolve(), flush=True)
