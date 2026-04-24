"""pipeline_utils.py — shared utilities for the cross-impact OFI pipeline.

All functions that were previously duplicated across scripts 04–10 now live
here.  New scripts (13, 14, …) import directly from this module.  Existing
scripts are unchanged for backward compatibility but can migrate incrementally.

Usage (from repo root, with 'pip install -e .'):
    from pipeline_utils import (
        build_exp_decay_kernel,
        gavish_donoho_denoise,
        fit_ridge, fit_sector_block, fit_nuclear_norm_sector_block,
        fit_soft_sector_block,
        choose_lambda_holdout,
        smooth_scores,
        daily_spread, daily_sector_neutral_spread,
        sharpe_from_series, annualize_sharpe,
        zscore_with_train_stats,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate as _sci_integrate, optimize as _sci_optimize


# ══════════════════════════════════════════════════════════════════════════════
# Signal construction
# ══════════════════════════════════════════════════════════════════════════════

def build_exp_decay_kernel(n_minutes: int, half_life: float) -> np.ndarray:
    """Normalised exponential-decay kernel over intraday minutes.

    Minute 1 = market open, minute n_minutes = market close.
    Weighting emphasises the close: w_k ∝ exp(-λ * (n - k)) so that k = n
    gets weight exp(0) = 1 and earlier minutes get exponentially less weight.

    Parameters
    ----------
    n_minutes : int
        Total minutes in the trading day (typically 390).
    half_life : float
        Half-life in minutes.  λ = log(2) / half_life.

    Returns
    -------
    np.ndarray of shape (n_minutes,) summing to 1.
    """
    lam = np.log(2) / half_life
    k = np.arange(n_minutes, 0, -1, dtype=float)   # [n, n-1, …, 1]
    weights = np.exp(-lam * k)
    return weights / weights.sum()


# ══════════════════════════════════════════════════════════════════════════════
# Gavish–Donoho optimal hard SVD threshold (Gavish & Donoho, 2014)
# ══════════════════════════════════════════════════════════════════════════════

def _mp_eigenvalue_median(beta: float) -> float:
    """Median of the Marchenko–Pastur distribution with aspect ratio beta."""
    beta = float(np.clip(beta, 1e-9, 1.0 - 1e-12))
    lp = (1.0 + np.sqrt(beta)) ** 2
    lm = (1.0 - np.sqrt(beta)) ** 2

    def _pdf(x: float) -> float:
        v = (lp - x) * (x - lm)
        return np.sqrt(max(v, 0.0)) / (2.0 * np.pi * beta * max(x, 1e-15))

    total, _ = _sci_integrate.quad(_pdf, lm, lp, limit=300, epsabs=1e-9)

    def _cdf_minus_half(t: float) -> float:
        v, _ = _sci_integrate.quad(_pdf, lm, t, limit=300, epsabs=1e-9)
        return v / total - 0.5

    return float(_sci_optimize.brentq(_cdf_minus_half, lm + 1e-10, lp - 1e-10, xtol=1e-8))


def gavish_donoho_denoise(M: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Apply the Gavish–Donoho optimal hard threshold to matrix M.

    Returns
    -------
    M_denoised : np.ndarray
        Low-rank approximation retaining only singular values above the threshold.
    n_kept : int
        Number of singular values retained.
    threshold : float
        The optimal singular value threshold τ*.
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


# ══════════════════════════════════════════════════════════════════════════════
# Model fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression: W = (X'X + λI)^{-1} X'Y.

    Parameters
    ----------
    X : (T, p) array of features.
    Y : (T, q) array of targets.
    lam : float ridge penalty.

    Returns
    -------
    W : (p, q) coefficient matrix.
    """
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def _base_ticker(col: str) -> str:
    """Extract base ticker from possibly-suffixed column like 'AAPL__ofi_hl30'."""
    return col.split("__")[0]


def fit_sector_block(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    lam: float,
    min_sector_assets: int = 4,
) -> pd.DataFrame:
    """Ridge regression with sector-block structure.

    W[i,j] = 0 whenever assets i and j are in different GICS sectors.
    Within each sector block, standard ridge is applied.

    Parameters
    ----------
    X_df : (T, p) signal DataFrame.
    Y_df : (T, N) return DataFrame.
    ticker_to_sector : Series mapping ticker → sector label.
    lam : float ridge penalty applied within each block.
    min_sector_assets : int minimum sector size to fit (smaller sectors skipped).

    Returns
    -------
    W_full : DataFrame of shape (p, N), zero outside sector blocks.
    """
    y_tickers = Y_df.columns.tolist()
    x_cols = X_df.columns.tolist()
    W_full = pd.DataFrame(0.0, index=x_cols, columns=y_tickers, dtype=float)

    sector_to_tickers: dict[str, list[str]] = (
        ticker_to_sector.rename("sector")
        .reset_index()
        .rename(columns={"index": "ticker"})
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    for sector, sec_y in sorted(sector_to_tickers.items()):
        sec_y = [t for t in sec_y if t in Y_df.columns]
        if len(sec_y) < min_sector_assets:
            continue
        sec_y_set = set(sec_y)
        sec_x = [c for c in x_cols if _base_ticker(c) in sec_y_set]
        if not sec_x:
            continue
        Xg = X_df[sec_x].to_numpy(dtype=float)
        Yg = Y_df[sec_y].to_numpy(dtype=float)
        W_full.loc[sec_x, sec_y] = fit_ridge(Xg, Yg, lam)

    return W_full


def fit_soft_sector_block(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    lam_within: float,
    lam_cross: float,
) -> pd.DataFrame:
    """Non-uniform ridge with sector-differentiated penalties.

    Within-sector source–target pairs are penalised by lam_within.
    Cross-sector pairs are penalised by lam_cross (typically >> lam_within).
    Each target stock's weight vector is solved independently via:
        w_i = (X'X + diag(omega_i))^{-1} X'y_i
    where omega_i[j] = lam_within if j shares sector with i, else lam_cross.

    Unlike the hard sector-block, this allows weak cross-sector effects to
    survive if the data supports them.
    """
    tickers = Y_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)
    XtX = X.T @ X  # (p, p) — computed once
    W = pd.DataFrame(0.0, index=X_df.columns, columns=tickers, dtype=float)

    for target in tickers:
        target_sector = ticker_to_sector.get(target)
        y = Y_df[target].to_numpy(dtype=float)

        omega = np.array([
            lam_within if ticker_to_sector.get(_base_ticker(src)) == target_sector
            else lam_cross
            for src in X_df.columns
        ], dtype=float)

        w = np.linalg.solve(XtX + np.diag(omega), X.T @ y)
        W[target] = w

    return W


# ──────────────────────────────────────────────────────────────────────────────
# Nuclear norm (low-rank) regression via proximal gradient
# ──────────────────────────────────────────────────────────────────────────────

def _prox_nuclear(M: np.ndarray, tau: float) -> np.ndarray:
    """Soft-threshold singular values of M by tau (proximal operator of τ||·||_*)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U * np.maximum(s - tau, 0.0)) @ Vt


def fit_nuclear_norm(
    X: np.ndarray,
    Y: np.ndarray,
    lam_ridge: float,
    lam_nuc: float,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> np.ndarray:
    """Nuclear-norm + ridge penalised least squares via proximal gradient.

    Solves:
        min_W  (1/T) ||Y - X W||_F^2  +  lam_ridge ||W||_F^2  +  lam_nuc ||W||_*

    The nuclear norm is the convex surrogate for matrix rank and directly
    encourages a low-rank cross-impact matrix, rather than post-hoc SVD
    thresholding.  The smooth (ridge) part stabilises the solution.

    Parameters
    ----------
    X : (T, p) features.
    Y : (T, q) targets.
    lam_ridge : float  ℓ2 penalty on W entries.
    lam_nuc   : float  nuclear norm penalty.

    Returns
    -------
    W : (p, q) coefficient matrix.
    """
    T, p = X.shape
    q = Y.shape[1]
    W = np.zeros((p, q), dtype=float)

    XtX = X.T @ X / T
    XtY = X.T @ Y / T

    # Lipschitz constant of the smooth gradient
    L = 2.0 * (float(np.linalg.eigvalsh(XtX).max()) + lam_ridge)
    step = 1.0 / max(L, 1e-12)

    for _ in range(max_iter):
        W_old = W
        grad = 2.0 * (XtX @ W - XtY) + 2.0 * lam_ridge * W
        W = _prox_nuclear(W - step * grad, step * lam_nuc)
        delta = np.linalg.norm(W - W_old, "fro")
        scale = max(1.0, np.linalg.norm(W_old, "fro"))
        if delta < tol * scale:
            break

    return W


def fit_nuclear_norm_sector_block(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    lam_ridge: float,
    lam_nuc: float,
    min_sector_assets: int = 4,
) -> pd.DataFrame:
    """Nuclear-norm regression with sector-block structure.

    Nuclear norm is applied independently within each sector block.
    Cross-sector entries remain zero (same structural constraint as
    fit_sector_block, but within each block the estimator is nuclear-norm
    regularised rather than ridge).
    """
    y_tickers = Y_df.columns.tolist()
    x_cols = X_df.columns.tolist()
    W_full = pd.DataFrame(0.0, index=x_cols, columns=y_tickers, dtype=float)

    sector_to_tickers: dict[str, list[str]] = (
        ticker_to_sector.rename("sector")
        .reset_index()
        .rename(columns={"index": "ticker"})
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    for sector, sec_y in sorted(sector_to_tickers.items()):
        sec_y = [t for t in sec_y if t in Y_df.columns]
        if len(sec_y) < min_sector_assets:
            continue
        sec_y_set = set(sec_y)
        sec_x = [c for c in x_cols if _base_ticker(c) in sec_y_set]
        if not sec_x:
            continue
        Xg = X_df[sec_x].to_numpy(dtype=float)
        Yg = Y_df[sec_y].to_numpy(dtype=float)
        W_block = fit_nuclear_norm(Xg, Yg, lam_ridge=lam_ridge, lam_nuc=lam_nuc)
        W_full.loc[sec_x, sec_y] = W_block

    return W_full


# ──────────────────────────────────────────────────────────────────────────────
# Lambda selection
# ──────────────────────────────────────────────────────────────────────────────

def choose_lambda_holdout(
    structure: str,
    lambda_grid: list[float],
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    holdout_frac: float = 0.20,
    min_sector_assets: int = 4,
) -> float:
    """Select ridge λ using a holdout validation sub-window.

    The training window is split at (1 - holdout_frac): the first part is
    used to fit W candidates, the last part is the validation set on which
    IC is evaluated.  The best λ is re-fit on the full window before returning
    (the caller still does the final re-fit; this function only returns λ*).

    This avoids the subtle in-sample overfitting of choose_lambda_on_window,
    which scores IC on the same data used to fit W.
    """
    n = len(X_train_z_df)
    split = max(int(n * (1 - holdout_frac)), 5)

    X_fit = X_train_z_df.iloc[:split]
    Y_fit = Y_train_z_df.iloc[:split]
    X_val = X_train_z_df.iloc[split:]
    Y_val = Y_train_z_df.iloc[split:]

    if len(X_val) < 2:
        # Fallback: use in-sample IC (same as original behaviour)
        return choose_lambda_insample(
            structure, lambda_grid, X_train_z_df, Y_train_z_df,
            ticker_to_sector, min_sector_assets,
        )

    best_lam, best_score = lambda_grid[0], -np.inf

    for lam in lambda_grid:
        if structure == "sector_block":
            W = fit_sector_block(X_fit, Y_fit, ticker_to_sector, lam, min_sector_assets)
        else:
            W = pd.DataFrame(
                fit_ridge(X_fit.to_numpy(float), Y_fit.to_numpy(float), lam),
                index=X_fit.columns, columns=Y_fit.columns,
            )

        cx = X_val.columns.intersection(W.index)
        pred_np = X_val[cx].to_numpy(float) @ W.loc[cx].to_numpy(float)
        cols = W.columns
        pred_df = pd.DataFrame(pred_np, index=X_val.index, columns=cols)
        y_common = Y_val.columns.intersection(cols)

        # Vectorised Spearman IC across validation days
        P = pred_df[y_common].rank(axis=1)
        R = Y_val[y_common].rank(axis=1)
        ic_vals = P.corrwith(R, axis=1, method="pearson").dropna()
        score = float(ic_vals.mean()) if len(ic_vals) else -np.inf

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lam = lam

    return float(best_lam)


def choose_lambda_insample(
    structure: str,
    lambda_grid: list[float],
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    min_sector_assets: int = 4,
) -> float:
    """Original in-sample IC lambda selection (kept for backward compatibility)."""
    best_lam, best_score = lambda_grid[0], -np.inf

    for lam in lambda_grid:
        if structure == "sector_block":
            W = fit_sector_block(X_train_z_df, Y_train_z_df, ticker_to_sector, lam, min_sector_assets)
        else:
            W = pd.DataFrame(
                fit_ridge(X_train_z_df.to_numpy(float), Y_train_z_df.to_numpy(float), lam),
                index=X_train_z_df.columns, columns=Y_train_z_df.columns,
            )

        cx = X_train_z_df.columns.intersection(W.index)
        pred_np = X_train_z_df[cx].to_numpy(float) @ W.loc[cx].to_numpy(float)
        pred_df = pd.DataFrame(pred_np, index=X_train_z_df.index, columns=W.columns)
        y_common = Y_train_z_df.columns.intersection(W.columns)

        P = pred_df[y_common].rank(axis=1)
        R = Y_train_z_df[y_common].rank(axis=1)
        ic_vals = P.corrwith(R, axis=1, method="pearson").dropna()
        score = float(ic_vals.mean()) if len(ic_vals) else -np.inf

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lam = lam

    return float(best_lam)


# ══════════════════════════════════════════════════════════════════════════════
# Score smoothing (turnover reduction)
# ══════════════════════════════════════════════════════════════════════════════

def smooth_scores(pred_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply EMA smoothing to predicted return scores.

    score_smooth_t = alpha * score_raw_t + (1 - alpha) * score_smooth_{t-1}

    alpha = 1.0  → no smoothing (pass-through, identical to input).
    alpha = 0.5  → half-weight on history.
    alpha = 0.2  → heavy smoothing, large turnover reduction.

    Lower alpha → slower adaptation → lower turnover → less alpha capture.
    The tradeoff is evaluated empirically in script 09.

    Parameters
    ----------
    pred_df : DataFrame (T × N) of raw predicted scores, sorted by date.
    alpha   : float in (0, 1].

    Returns
    -------
    DataFrame (T × N) of smoothed scores.
    """
    if alpha >= 1.0:
        return pred_df.copy()

    dates = pred_df.index.sort_values()
    smoothed = pd.DataFrame(np.nan, index=dates, columns=pred_df.columns, dtype=float)
    prev: pd.Series | None = None

    for date in dates:
        raw = pred_df.loc[date]
        if prev is None:
            cur = raw.copy()
        else:
            # Align on the current universe; for new assets use raw; for dropped assets ignore
            cur = alpha * raw + (1.0 - alpha) * prev.reindex(raw.index).fillna(raw)
        smoothed.loc[date] = cur
        prev = cur

    return smoothed


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def daily_spread_vec(
    pred_mat: np.ndarray,
    ret_mat: np.ndarray,
    q: float = 0.10,
) -> np.ndarray:
    """Vectorised daily long-short spread across all dates.

    Parameters
    ----------
    pred_mat : (T, N) array of predicted scores.
    ret_mat  : (T, N) array of realised returns.
    q        : float top/bottom quantile.

    Returns
    -------
    spreads : (T,) array of daily spread returns (nan where not computable).
    """
    T, N = pred_mat.shape
    spreads = np.full(T, np.nan)
    k = max(1, int(N * q))
    if 2 * k > N:
        return spreads

    for t in range(T):
        p = pred_mat[t]
        r = ret_mat[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < 2 * k:
            continue
        idx = np.argsort(p[mask])
        r_sub = r[mask][idx]
        spreads[t] = r_sub[-k:].mean() - r_sub[:k].mean()

    return spreads


def daily_spread(scores: pd.Series, returns: pd.Series, q: float = 0.10) -> float:
    """Scalar daily long-short spread (single date)."""
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
    """Sector-neutral spread: rank within each sector, average across sectors."""
    tmp = pd.concat([scores.rename("s"), returns.rename("r")], axis=1).dropna()
    if tmp.empty:
        return np.nan
    tmp = tmp.join(ticker_to_sector.rename("sec"), how="left").dropna(subset=["sec"])
    if tmp.empty:
        return np.nan
    sec_spreads = []
    for _, g in tmp.groupby("sec"):
        n = len(g)
        if n < min_names_per_sector:
            continue
        k = max(1, int(n * q))
        if 2 * k > n:
            continue
        g = g.sort_values("s")
        sec_spreads.append(float(g.iloc[-k:]["r"].mean() - g.iloc[:k]["r"].mean()))
    return float(np.mean(sec_spreads)) if sec_spreads else np.nan


def evaluate_predictions(
    pred_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    q: float = 0.10,
    min_sn: int = 4,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (daily_gross_spreads, daily_sn_spreads, daily_ic) Series."""
    ret_aligned = ret_df[pred_df.columns].copy()
    sec_map = ticker_to_sector.loc[pred_df.columns]

    gross = pred_df.apply(
        lambda row: daily_spread(row, ret_aligned.loc[row.name], q=q), axis=1
    ).dropna()

    sn = pred_df.apply(
        lambda row: daily_sector_neutral_spread(
            row, ret_aligned.loc[row.name], sec_map, q=q, min_names_per_sector=min_sn
        ), axis=1
    ).dropna()

    # Vectorised Spearman IC using rank correlation
    ic = pred_df.apply(
        lambda row: row.corr(ret_aligned.loc[row.name], method="spearman"), axis=1
    ).dropna()

    return gross, sn, ic


def sharpe_from_series(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    s = x.std(ddof=1)
    return float(x.mean() / s) if (s > 0 and np.isfinite(s)) else np.nan


def annualize_sharpe(daily_sharpe: float) -> float:
    return float(daily_sharpe * np.sqrt(252)) if np.isfinite(daily_sharpe) else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def zscore_with_train_stats(
    train_mat: pd.DataFrame,
    test_mat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Z-score train and test using training mean/std (no data leakage)."""
    mean = train_mat.mean(axis=0, skipna=True)
    std = train_mat.std(axis=0, skipna=True, ddof=0).replace(0, 1.0)
    return (train_mat - mean) / std, (test_mat - mean) / std, mean, std


# ══════════════════════════════════════════════════════════════════════════════
# Transaction-cost helpers (from script 09)
# ══════════════════════════════════════════════════════════════════════════════

def build_gross_weights(pred_row: pd.Series, q: float) -> pd.Series:
    """Equal-weight long top-q, short bottom-q.  Dollar-neutral."""
    valid = pred_row.dropna()
    n = len(valid)
    weights = pd.Series(0.0, index=pred_row.index)
    if n < 2:
        return weights
    k = max(1, int(n * q))
    if 2 * k > n:
        return weights
    order = valid.sort_values()
    weights.loc[order.index[-k:]] = 1.0 / k
    weights.loc[order.index[:k]] = -1.0 / k
    return weights


def build_sn_weights(
    pred_row: pd.Series,
    ticker_to_sector: pd.Series,
    q: float,
    min_per_sector: int,
) -> pd.Series:
    """Sector-neutral weights: rank within sector, average equally across sectors."""
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
            "longs": g_sorted.index[-k:].tolist(),
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
    """One-way daily turnover τ_t = (1/2) Σ_i |w_{i,t} - w_{i,t-1}|."""
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
            w.reindex(union).fillna(0.0) - prev_w.reindex(union).fillna(0.0)
        )
        turnovers[date] = 0.5 * delta.abs().sum()
        prev_w = w.reindex(union).fillna(0.0)

    return pd.Series(turnovers)


def compute_net_spread(
    gross_spread: pd.Series,
    turnover: pd.Series,
    c_rt_bps: float,
) -> pd.Series:
    """Net spread = gross spread - (c_rt_bps / 10000) * turnover."""
    c = c_rt_bps / 10_000.0
    common = gross_spread.index.intersection(turnover.index)
    return (gross_spread.loc[common] - c * turnover.loc[common]).dropna()


def compute_breakeven_bps(
    gross_spread: pd.Series,
    turnover: pd.Series,
) -> float:
    """Breakeven round-trip cost c* = E[P_gross] / E[τ] × 10000 (in bps)."""
    common = gross_spread.index.intersection(turnover.index)
    mean_spread = gross_spread.loc[common].dropna().mean()
    mean_tau = turnover.loc[common].dropna().mean()
    if mean_tau <= 0 or not np.isfinite(mean_tau):
        return np.nan
    return float(mean_spread / mean_tau * 10_000.0)
