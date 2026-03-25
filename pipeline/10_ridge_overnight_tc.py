#!/usr/bin/env python3
"""10_ridge_overnight_tc.py

Transaction-cost-aware version of the overnight rolling ridge strategy (script 31).

Signal:  exp-decay weighted intraday OFI on day t  (emphasis toward close)
Target:  overnight_resid_t = (Open_{t+1}/Close_t - 1) - beta * SPY_overnight_t
         i.e. beta-residualized close-to-open return starting that same night.

No prediction lag needed: X_t and Y_t are aligned on the same date
(OFI through close of t predicts what happens after that close).

TRANSACTION COST MODEL  (Frazzini, Israel & Moskowitz 2015; Novy-Marx &
Velikov 2016; Grinold & Kahn 2000)
--------------------------------------------------------------------
One-way turnover on date t:
    τ_t = (1/2) Σ_i |w_{i,t} − w_{i,t−1}|
where w_{i,0} = 0 (portfolio starts from cash, so the first day is τ_1 = 1.0).

Net daily P&L after round-trip transaction costs:
    NetP_t = P_t − c_rt × τ_t
where c_rt is the round-trip cost in bps.

Breakeven cost:
    c* = E[P] / E[τ]  (in bps: c*_bps = c* × 10 000)

SECTOR-NEUTRAL VERSION
----------------------
Weights constructed within each GICS sector (rank within sector → long
top-q, short bottom-q per sector), then averaged equally across active sectors.

TIMING / EXECUTION ASSUMPTION
------------------------------
Signal X_t is observed at the close of day t (OFI through close).
Trade executed at the open of t+1 (market-on-open order).
Position liquidated at open of t+2 (or equivalently, the overnight gap
is monetized at the next open).  The overnight return accrues from
close_t to open_{t+1}.

References
----------
Frazzini, A., Israel, R., & Moskowitz, T. J. (2015). Trading costs of asset
    pricing anomalies. Fama-Miller Working Paper.
Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their
    trading costs. Review of Financial Studies, 29(1), 104-147.
Grinold, R. C., & Kahn, R. N. (2000). Active Portfolio Management (2nd ed.).
    McGraw-Hill.

Run from repo root:
    python pipeline/10_ridge_overnight_tc.py
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
INPUT_PATH       = Path("data/processed/feature_table_with_residuals_10level.csv")
OVERNIGHT_CACHE  = Path("data/processed/overnight_returns_residualized.csv")
OUT_DIR          = Path("results/rolling_adjacency_ridge_overnight_tc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS = 6
MAX_ASSETS       = 400
HALF_LIFE_SWEEP  = [15, 20, 25, 30, 35, 40, 45, 60, 90]
Q                = 0.10
MIN_TICKER_OBS   = 252
MIN_SECTOR_ASSETS        = 4
MIN_NAMES_PER_SECTOR_NEUTRAL = 4

INITIAL_TRAIN_DAYS = 750
REFIT_EVERY_DAYS   = 21
BETA_WINDOW        = 60   # days for rolling SPY beta on overnight returns

RIDGE_GRID = [1.0, 10.0, 50.0, 250.0, 1000.0]

# Round-trip transaction cost scenarios (bps).
TC_SCENARIOS_BPS = [5, 10, 20, 30]

MODEL_SPECS = [
    # ── baselines ───────────────────────────────────────────────
    {"name": "ridge_on_sb",             "structure": "sector_block", "use_gd": False,
     "hl": 45, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd",          "structure": "sector_block", "use_gd": True,
     "hl": 45, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    # ── half-life sweep ──────────────────────────────────────────
    {"name": "ridge_on_sb_gd_hl15",     "structure": "sector_block", "use_gd": True,
     "hl": 15, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl20",     "structure": "sector_block", "use_gd": True,
     "hl": 20, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl25",     "structure": "sector_block", "use_gd": True,
     "hl": 25, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl30",     "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl35",     "structure": "sector_block", "use_gd": True,
     "hl": 35, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl40",     "structure": "sector_block", "use_gd": True,
     "hl": 40, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl60",     "structure": "sector_block", "use_gd": True,
     "hl": 60, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl90",     "structure": "sector_block", "use_gd": True,
     "hl": 90, "sig_cols": None, "rank_norm": False, "zero_diag": False},
    # ── rank-normalized OFI ──────────────────────────────────────
    {"name": "ridge_on_sb_gd_hl30_rn",  "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": None, "rank_norm": True,  "zero_diag": False},
    {"name": "ridge_on_sb_gd_hl90_rn",  "structure": "sector_block", "use_gd": True,
     "hl": 90, "sig_cols": None, "rank_norm": True,  "zero_diag": False},
    # ── multi-scale ─────────────────────────────────────────────
    {"name": "ridge_on_sb_gd_ms3090",   "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": ["ofi_hl30", "ofi_hl90"], "rank_norm": False, "zero_diag": False},
    # ── multi-lag ────────────────────────────────────────────────
    {"name": "ridge_on_sb_gd_lag1",     "structure": "sector_block", "use_gd": True,
     "hl": 30, "sig_cols": ["ofi_hl30", "ofi_hl30_lag1"], "rank_norm": False, "zero_diag": False},
]


# ============================================================
# Gavish-Donoho optimal hard thresholding (unchanged from 31)
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
    omega_star = omega / np.sqrt(mu_beta)
    lambda_star = omega_star * y_med
    s_thresh = np.where(s >= lambda_star, s, 0.0)
    M_denoised = (U * s_thresh) @ Vt
    return M_denoised, int((s_thresh > 0).sum()), float(lambda_star)


# ============================================================
# exp-decay kernel (unchanged)
# ============================================================
def build_exp_decay_kernel(n_minutes: int, half_life: float) -> np.ndarray:
    lam = np.log(2) / half_life
    k = np.arange(n_minutes, 0, -1, dtype=float)
    weights = np.exp(-lam * k)
    weights /= weights.sum()
    return weights


# ============================================================
# helpers (unchanged from 31)
# ============================================================
def get_minute_cols_from_columns(columns: list[str]) -> list[str]:
    return sorted(
        [c for c in columns if c.startswith("minute_")],
        key=lambda x: int(x.split("_")[1]),
    )


def _base_ticker(col: str) -> str:
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


# ============================================================
# portfolio weight construction (from script 33)
# ============================================================
def build_gross_weights(pred_row: pd.Series, q: float) -> pd.Series:
    """Equal-weight long top-q, short bottom-q across all assets.

    w · r equals mean(top-k returns) - mean(bottom-k returns),
    i.e. exactly the gross spread from daily_spread().
    """
    valid = pred_row.dropna()
    n = len(valid)
    weights = pd.Series(0.0, index=pred_row.index)
    if n < 2:
        return weights
    k = max(1, int(n * q))
    if 2 * k > n:
        return weights
    order = valid.sort_values()
    weights.loc[order.index[-k:]] = 1.0 / k   # longs
    weights.loc[order.index[:k]]  = -1.0 / k  # shorts
    return weights


def build_sn_weights(
    pred_row: pd.Series,
    ticker_to_sector: pd.Series,
    q: float,
    min_per_sector: int,
) -> pd.Series:
    """Sector-neutral weights: rank within each sector, average equally across sectors.

    Consistent with daily_sector_neutral_spread() — the resulting w · r equals
    the sector-neutral spread.
    """
    weights = pd.Series(0.0, index=pred_row.index)
    combined = pd.concat(
        [pred_row.rename("score"), ticker_to_sector.rename("sector")], axis=1
    ).dropna(subset=["score", "sector"])

    sector_dicts: dict[str, dict] = {}
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


# ============================================================
# transaction cost computation (from script 33)
# ============================================================
def compute_turnover_series(
    pred_df: pd.DataFrame,
    q: float,
    ticker_to_sector: pd.Series | None = None,
    min_per_sector: int = 4,
    sn: bool = False,
) -> pd.Series:
    """One-way daily turnover τ_t = (1/2) Σ_i |w_{i,t} - w_{i,t-1}|.

    On the first date w_{t-1} = 0, so τ_1 = 1.0 (full portfolio setup cost).
    """
    all_tickers = pred_df.columns
    prev_w = pd.Series(0.0, index=all_tickers)
    turnovers: dict = {}

    for date, row in pred_df.iterrows():
        if sn and ticker_to_sector is not None:
            w = build_sn_weights(row, ticker_to_sector, q, min_per_sector)
        else:
            w = build_gross_weights(row, q)

        union = prev_w.index.union(w.index)
        delta = (
            w.reindex(union).fillna(0.0)
            - prev_w.reindex(union).fillna(0.0)
        )
        turnovers[date] = 0.5 * delta.abs().sum()
        prev_w = w.reindex(union).fillna(0.0)

    return pd.Series(turnovers)


def compute_net_spread(
    gross_spread: pd.Series,
    turnover: pd.Series,
    c_rt_bps: float,
) -> pd.Series:
    """Net spread = gross spread - c_rt * τ_t."""
    c = c_rt_bps / 10_000.0
    common = gross_spread.index.intersection(turnover.index)
    return (gross_spread.loc[common] - c * turnover.loc[common]).dropna()


def compute_breakeven_bps(
    gross_spread: pd.Series,
    turnover: pd.Series,
) -> float:
    """Round-trip breakeven cost in bps: c* = E[P_gross] / E[τ] × 10 000."""
    common = gross_spread.index.intersection(turnover.index)
    mean_spread = gross_spread.loc[common].dropna().mean()
    mean_tau    = turnover.loc[common].dropna().mean()
    if mean_tau <= 0 or not np.isfinite(mean_tau):
        return np.nan
    return float(mean_spread / mean_tau * 10_000.0)


# ============================================================
# model fitting (unchanged from 31)
# ============================================================
def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def fit_sector_block_model(
    X_train_df: pd.DataFrame,
    Y_train_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    lam: float,
) -> pd.DataFrame:
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
    if isinstance(W, pd.DataFrame):
        common_x = X_eval_z_df.columns.intersection(W.index)
        pred_z = X_eval_z_df[common_x].to_numpy(dtype=float) @ W.loc[common_x].to_numpy(dtype=float)
        cols = W.columns
    else:
        pred_z = X_eval_z_df.to_numpy(dtype=float) @ W
        cols = X_eval_z_df.columns
    pred = pred_z * y_std.loc[cols].to_numpy(dtype=float) + y_mean.loc[cols].to_numpy(dtype=float)
    return pd.DataFrame(pred, index=X_eval_z_df.index, columns=cols)


def choose_lambda_on_window(structure, lambda_grid, X_train_z_df, Y_train_z_df, ticker_to_sector):
    best_lam, best_score = lambda_grid[0], -np.inf
    for lam in lambda_grid:
        print(f"      trying lambda={lam}", flush=True)
        W = fit_model(structure, lam, X_train_z_df, Y_train_z_df, ticker_to_sector)
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
# overnight return computation (unchanged from 31)
# ============================================================
def build_overnight_returns(tickers: list[str], cache_path: Path) -> pd.DataFrame:
    """Download daily OHLC, compute overnight (close→open) returns residualized vs SPY.

    Returns wide DataFrame: index=date (the signal date, i.e. close date),
    columns=tickers, values=residualized overnight return starting that night.

    overnight_ret[t] = (Open_{t+1}/Close_t - 1) - beta_t * SPY_overnight[t]
    """
    if cache_path.exists():
        print(f"Loading cached overnight returns from {cache_path}", flush=True)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print("Downloading daily OHLC via yfinance...", flush=True)
    import yfinance as yf

    all_syms = tickers + ["SPY"]
    raw = yf.download(all_syms, start="2007-01-01", end="2022-06-01",
                      auto_adjust=True, progress=False)

    # raw columns: (price_type, ticker)
    open_df  = raw["Open"].sort_index()
    close_df = raw["Close"].sort_index()

    # overnight return for date t = open_{t+1} / close_t - 1
    overnight = open_df.shift(-1) / close_df - 1.0
    overnight = overnight.iloc[:-1]

    spy_on = overnight["SPY"].copy()
    overnight = overnight.drop(columns=["SPY"], errors="ignore")

    print(f"Residualizing overnight returns (rolling {BETA_WINDOW}-day beta)...", flush=True)
    overnight_resid = overnight.copy() * np.nan
    for ticker in overnight.columns:
        s = overnight[ticker].dropna()
        combined = pd.concat([s, spy_on.reindex(s.index)], axis=1).dropna()
        combined.columns = ["stk", "spy"]
        cov = combined["stk"].rolling(BETA_WINDOW).cov(combined["spy"])
        var = combined["spy"].rolling(BETA_WINDOW).var()
        beta = cov / var
        resid = combined["stk"] - beta * combined["spy"]
        overnight_resid.loc[resid.index, ticker] = resid.values

    overnight_resid.index.name = "date"
    overnight_resid.to_csv(cache_path)
    print(f"  Saved to {cache_path}", flush=True)
    return overnight_resid


# ============================================================
# load feature table
# ============================================================
print("Reading feature data...", flush=True)

needed_base_cols = ["date", "ticker", "sector"]
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
df = df.dropna(subset=["ticker", "date", "sector"]).copy()
# Drop GOOGL — GOOG and GOOGL are both Alphabet share classes with nearly
# identical LOB signals; keeping both double-counts Alphabet in Communication
# Services and introduces near-collinear features. Keep GOOG (Class C, higher volume).
df = df[df["ticker"] != "GOOGL"].copy()

sector_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
df = df[df["ticker"].isin(sector_counts[sector_counts == 1].index)].copy()

ticker_to_sector_full = df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

# ============================================================
# load / build overnight returns and merge into df
# ============================================================
tickers_in_feature = df["ticker"].unique().tolist()
overnight_wide = build_overnight_returns(tickers_in_feature, OVERNIGHT_CACHE)

overnight_long = (
    overnight_wide.reset_index()
    .melt(id_vars="date", var_name="ticker", value_name="overnight_resid")
    .dropna(subset=["overnight_resid"])
)
overnight_long["date"] = pd.to_datetime(overnight_long["date"])

df = pd.merge(df, overnight_long, on=["date", "ticker"], how="inner")
df = df.dropna(subset=["overnight_resid"]).copy()
print(f"  {len(df):,} rows after overnight merge", flush=True)

# ── build all exp-decay OFI signals ─────────────────────────
n_minutes = len(minute_cols)
minute_np = df[minute_cols].to_numpy(dtype=float)
print("Building exp-decay signals:", flush=True)
for _hl in HALF_LIFE_SWEEP:
    _kernel = build_exp_decay_kernel(n_minutes=n_minutes, half_life=_hl)
    df[f"ofi_hl{_hl}"] = minute_np @ _kernel
    print(f"  hl={_hl:3d}min  weight_ratio={_kernel[-1]/_kernel[0]:.1f}x", flush=True)
del minute_np

print("Building rank-normalized signals...", flush=True)
for _hl in HALF_LIFE_SWEEP:
    df[f"ofi_hl{_hl}_rn"] = df.groupby("date")[f"ofi_hl{_hl}"].rank(pct=True)

print("Building lag-1 signals...", flush=True)
df = df.sort_values(["ticker", "date"])
for _hl in [30, 90]:
    df[f"ofi_hl{_hl}_lag1"] = df.groupby("ticker")[f"ofi_hl{_hl}"].shift(1)
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

all_dates = np.sort(df["date"].unique())

if len(all_dates) < INITIAL_TRAIN_DAYS + 2:
    raise ValueError(f"Not enough dates ({len(all_dates)}) for INITIAL_TRAIN_DAYS={INITIAL_TRAIN_DAYS}.")

# ============================================================
# global coverage filter + asset cap
# ============================================================
sig_proxy = df.pivot(index="date", columns="ticker", values="ofi_hl45")
ret_proxy = df.pivot(index="date", columns="ticker", values="overnight_resid")

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

all_model_summaries        = []
all_model_lambda_histories = []

for model_spec in MODEL_SPECS:
    model_name = model_spec["name"]
    structure  = model_spec["structure"]
    use_gd     = model_spec["use_gd"]
    zero_diag  = model_spec.get("zero_diag", False)
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

    pred_chunks    = []
    chosen_lambdas = []
    gd_ranks       = []

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

        # ── overnight returns matrix ─────────────────────────────
        ret_mat = df_win.pivot(index="date", columns="ticker", values="overnight_resid").sort_index()

        # ── coverage filter ──────────────────────────────────────
        first_sc = sig_cols[0]
        first_sig = df_win.pivot(index="date", columns="ticker", values=first_sc).sort_index()

        cd_b = first_sig.index.intersection(ret_mat.index)
        ct_b = first_sig.columns.intersection(ret_mat.columns)
        first_sig = first_sig.loc[cd_b, ct_b]
        ret_mat   = ret_mat.loc[cd_b, ct_b]

        block_ticker_obs = (first_sig.notna() & ret_mat.notna()).sum(axis=0)
        block_keep = block_ticker_obs[block_ticker_obs >= 30].index
        ret_mat  = ret_mat[block_keep]
        ts_block = ticker_to_sector.reindex(block_keep).dropna()
        block_keep = ts_block.index.intersection(block_keep)
        ret_mat  = ret_mat[block_keep]

        if len(block_keep) == 0:
            continue

        # ── signal matrix ────────────────────────────────────────
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

        # ── OVERNIGHT ALIGNMENT: same date (no lag needed) ───────
        # X_t = today's OFI (through close of t)
        # Y_t = overnight_resid_t = close_t → open_{t+1}
        common_train = signal_train.index.intersection(ret_train.index)
        common_eval  = signal_eval.index.intersection(ret_eval.index)

        X_train_df = signal_train.loc[common_train].copy()
        Y_train_df = ret_train.loc[common_train].copy()
        X_eval_df  = signal_eval.loc[common_eval].copy()
        Y_eval_df  = ret_eval.loc[common_eval].copy()

        if len(X_train_df) < 5 or len(X_eval_df) < 1:
            continue

        # ── align tickers across X and Y ────────────────────────
        if len(sig_cols) == 1:
            cc = (X_train_df.columns.intersection(Y_train_df.columns)
                  .intersection(X_eval_df.columns).intersection(Y_eval_df.columns))
            X_train_df, Y_train_df = X_train_df[cc], Y_train_df[cc]
            X_eval_df,  Y_eval_df  = X_eval_df[cc],  Y_eval_df[cc]
            ts_block = ts_block.loc[cc]
        else:
            x_base = set(_base_ticker(c) for c in X_train_df.columns)
            cc_y = pd.Index([t for t in Y_train_df.columns
                              if t in x_base and t in Y_eval_df.columns])
            cc_y_set = set(cc_y)
            cc_x_tr = [c for c in X_train_df.columns if _base_ticker(c) in cc_y_set]
            cc_x_ev = [c for c in X_eval_df.columns  if _base_ticker(c) in cc_y_set]
            Y_train_df, Y_eval_df = Y_train_df[cc_y], Y_eval_df[cc_y]
            X_train_df, X_eval_df = X_train_df[cc_x_tr], X_eval_df[cc_x_ev]
            ts_block = ts_block.loc[cc_y]

        n_assets_block = len(Y_train_df.columns)
        if n_assets_block == 0:
            continue

        # ── z-score ──────────────────────────────────────────────
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

        # ── optional GD column alignment ─────────────────────────
        gd_rank = float("nan")
        if use_gd:
            if len(sig_cols) == 1:
                kept_cols = X_train_z_df.columns.intersection(Y_train_z_df.columns)
                if len(kept_cols) < 2:
                    continue
                X_train_z_df = X_train_z_df[kept_cols]
                Y_train_z_df = Y_train_z_df[kept_cols]
                X_eval_z_df  = X_eval_z_df[kept_cols]
                Y_eval_df    = Y_eval_df[kept_cols]
                y_mean, y_std = y_mean[kept_cols], y_std[kept_cols]
                ts_block = ts_block.loc[kept_cols]
            else:
                x_base_set = set(_base_ticker(c) for c in X_train_z_df.columns)
                kept_y = pd.Index([t for t in Y_train_z_df.columns if t in x_base_set])
                if len(kept_y) < 2:
                    continue
                cc_x_kept   = [c for c in X_train_z_df.columns if _base_ticker(c) in set(kept_y)]
                cc_x_ev_kept = [c for c in X_eval_z_df.columns if _base_ticker(c) in set(kept_y)]
                Y_train_z_df = Y_train_z_df[kept_y]
                Y_eval_df    = Y_eval_df[kept_y]
                X_train_z_df = X_train_z_df[cc_x_kept]
                X_eval_z_df  = X_eval_z_df[cc_x_ev_kept]
                y_mean, y_std = y_mean[kept_y], y_std[kept_y]
                ts_block = ts_block.loc[kept_y]

        print(
            f"[{model_name}] block {block_num}: "
            f"train {pd.Timestamp(train_dates_window[0]).date()} to "
            f"{pd.Timestamp(train_dates_window[-1]).date()} | "
            f"predict {pd.Timestamp(block_test_dates[0]).date()} to "
            f"{pd.Timestamp(block_test_dates[-1]).date()} | "
            f"assets={n_assets_block}",
            flush=True,
        )

        lam_star = choose_lambda_on_window(structure, RIDGE_GRID, X_train_z_df, Y_train_z_df, ts_block)
        W = fit_model(structure, lam_star, X_train_z_df, Y_train_z_df, ts_block)

        if use_gd:
            Wmat = W.to_numpy(dtype=float) if isinstance(W, pd.DataFrame) else W
            W_gd, gd_rank, gd_thresh = gavish_donoho_denoise(Wmat)
            print(f"      GD: kept {gd_rank} singular values (threshold={gd_thresh:.4f})", flush=True)
            if isinstance(W, pd.DataFrame):
                W = pd.DataFrame(W_gd, index=W.index, columns=W.columns)
            else:
                W = W_gd

        if zero_diag and isinstance(W, np.ndarray):
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

    if not pred_chunks:
        print(f"  No predictions for {model_name}, skipping.", flush=True)
        continue

    pred_all_df = pd.concat(pred_chunks, axis=0).sort_index()
    pred_all_df = pred_all_df[~pred_all_df.index.duplicated(keep="last")]

    ret_full = df.pivot(index="date", columns="ticker", values="overnight_resid").sort_index()
    ce_d = pred_all_df.index.intersection(ret_full.index)
    ce_c = pred_all_df.columns.intersection(ret_full.columns)
    pred_all_df = pred_all_df.loc[ce_d, ce_c]
    ret_eval_df = ret_full.loc[ce_d, ce_c]
    ts_eval = ticker_to_sector.loc[ce_c]

    # ── gross P&L series ────────────────────────────────────────
    gross_spread, gross_spread_sn, daily_ic = evaluate_predictions(
        pred_all_df, ret_eval_df, ts_eval, Q, MIN_NAMES_PER_SECTOR_NEUTRAL
    )

    # ── turnover series ─────────────────────────────────────────
    print(f"  Computing gross turnover...", flush=True)
    turnover_gross = compute_turnover_series(pred_all_df, Q, sn=False)

    print(f"  Computing sector-neutral turnover...", flush=True)
    turnover_sn = compute_turnover_series(
        pred_all_df, Q,
        ticker_to_sector=ts_eval,
        min_per_sector=MIN_NAMES_PER_SECTOR_NEUTRAL,
        sn=True,
    )

    mean_tau_gross = float(turnover_gross.mean())
    mean_tau_sn    = float(turnover_sn.mean())
    ann_tau_gross  = mean_tau_gross * 252
    ann_tau_sn     = mean_tau_sn    * 252

    breakeven_gross = compute_breakeven_bps(gross_spread, turnover_gross)
    breakeven_sn    = compute_breakeven_bps(gross_spread_sn, turnover_sn)

    print(f"  Mean daily gross turnover : {mean_tau_gross:.3f}  ({ann_tau_gross:.0f}x annualized)", flush=True)
    print(f"  Mean daily SN turnover    : {mean_tau_sn:.3f}  ({ann_tau_sn:.0f}x annualized)", flush=True)
    print(f"  Gross breakeven           : {breakeven_gross:.1f} bps round-trip", flush=True)
    print(f"  SN breakeven              : {breakeven_sn:.1f} bps round-trip", flush=True)

    # ── net Sharpe at each TC scenario ──────────────────────────
    net_sharpes_gross = {}
    net_sharpes_sn    = {}
    for c_bps in TC_SCENARIOS_BPS:
        net_g  = compute_net_spread(gross_spread,    turnover_gross, c_bps)
        net_sn = compute_net_spread(gross_spread_sn, turnover_sn,   c_bps)
        net_sharpes_gross[c_bps] = annualize_sharpe(sharpe_from_series(net_g))
        net_sharpes_sn[c_bps]    = annualize_sharpe(sharpe_from_series(net_sn))
        print(
            f"  TC={c_bps:2d}bps RT  |  "
            f"net gross Sharpe={net_sharpes_gross[c_bps]:.3f}  |  "
            f"net SN Sharpe={net_sharpes_sn[c_bps]:.3f}",
            flush=True,
        )

    lh_df = pd.concat(chosen_lambdas, axis=0, ignore_index=True)
    mean_gd_rank = float(lh_df["gd_rank"].dropna().mean()) if use_gd else float("nan")

    summary_row = {
        "model": model_name, "structure": structure, "use_gd": use_gd,
        "sig_cols": "|".join(sig_cols), "half_life_minutes": hl,
        "rank_norm": rank_norm, "zero_diag": zero_diag,
        "rolling_train_days": INITIAL_TRAIN_DAYS,
        "refit_every_days": REFIT_EVERY_DAYS,
        # ── gross stats ──────────────────────────────────────────
        "mean_daily_spread": gross_spread.mean(),
        "annualized_spread_sharpe": annualize_sharpe(sharpe_from_series(gross_spread)),
        "mean_daily_spread_sn": gross_spread_sn.mean(),
        "annualized_spread_sharpe_sn": annualize_sharpe(sharpe_from_series(gross_spread_sn)),
        "mean_daily_ic": daily_ic.mean(),
        # ── turnover ─────────────────────────────────────────────
        "mean_daily_turnover_gross": mean_tau_gross,
        "annualized_turnover_gross": ann_tau_gross,
        "mean_daily_turnover_sn":    mean_tau_sn,
        "annualized_turnover_sn":    ann_tau_sn,
        # ── breakeven ────────────────────────────────────────────
        "breakeven_rt_bps_gross": breakeven_gross,
        "breakeven_rt_bps_sn":    breakeven_sn,
        # ── net Sharpe at each scenario ──────────────────────────
        **{f"net_sharpe_gross_{c}bps": net_sharpes_gross[c] for c in TC_SCENARIOS_BPS},
        **{f"net_sharpe_sn_{c}bps":    net_sharpes_sn[c]    for c in TC_SCENARIOS_BPS},
        # ── misc ─────────────────────────────────────────────────
        "n_eval_days": len(pred_all_df),
        "n_refits": len(chosen_lambdas),
        "mean_gd_rank": mean_gd_rank,
    }
    summary_df = pd.DataFrame([summary_row])

    print(f"\n[{model_name}] Summary:", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    all_model_summaries.append(summary_df)
    all_model_lambda_histories.append(lh_df)

    summary_df.to_csv(OUT_DIR / f"rolling_summary_{model_name}.csv", index=False)
    lh_df.to_csv(OUT_DIR / f"rolling_lambda_history_{model_name}.csv", index=False)
    pred_all_df.to_csv(OUT_DIR / f"predicted_returns_{model_name}_rolling.csv")

    # ── save turnover series ─────────────────────────────────────
    pd.DataFrame({
        "date": turnover_gross.index,
        "turnover_gross": turnover_gross.values,
        "turnover_sn":    turnover_sn.reindex(turnover_gross.index).values,
    }).to_csv(OUT_DIR / f"turnover_{model_name}.csv", index=False)

    print(f"  Saved results for {model_name}", flush=True)

# ============================================================
# combined summary
# ============================================================
if not all_model_summaries:
    raise ValueError("No models produced outputs.")

summary_all = pd.concat(all_model_summaries, axis=0, ignore_index=True)
summary_all = summary_all.sort_values(
    ["annualized_spread_sharpe_sn", "annualized_spread_sharpe", "mean_daily_ic"],
    ascending=False,
).reset_index(drop=True)

summary_all.to_csv(OUT_DIR / "rolling_summary_all_models.csv", index=False)

if all_model_lambda_histories:
    pd.concat(all_model_lambda_histories, axis=0, ignore_index=True).to_csv(
        OUT_DIR / "rolling_lambda_history_all_models.csv", index=False
    )

print("\n\n" + "=" * 80)
print("FINAL SUMMARY — overnight strategy with transaction costs")
print("=" * 80)
print(summary_all.to_string(index=False))
print(f"\nDone. Results in {OUT_DIR.resolve()}", flush=True)
