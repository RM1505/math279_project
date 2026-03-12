"""
Parameter sweep focused on:
  - LAMBDA_GRID           : extended upward since holdout selection pegs to 250
  - DIAGONAL_BLEND_ALPHA  : how much auto-signal to blend back into sector_offdiag W
  - VOL_SCALE_DAYS        : rolling vol window for position vol-scaling (0 = off)
  - HOLDOUT_DAYS          : 21 vs 63 (1 month vs 1 quarter holdout for lambda selection)

Lambda grids all start at 50+ since the previous run showed lambda always hitting 250 ceiling.
Results saved to data/processed/sweep_results/sweep_summary.csv ranked by vol-scaled Sharpe.
"""

import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# fixed config
# ============================================================
INPUT_PATH = Path("data/processed/feature_table_with_residuals.csv")
OUT_DIR = Path("data/processed/sweep_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sector_block"
Q = 0.10
MIN_TICKER_OBS = 252
MIN_SECTOR_ASSETS = 2
MIN_NAMES_PER_SECTOR_NEUTRAL = 4
INITIAL_TRAIN_DAYS = 750
REFIT_EVERY_DAYS = 21
N_PCA_COMPONENTS = 1

# ============================================================
# sweep grid  (4 x 4 x 2 x 2 = 64 combos)
# ============================================================
SWEEP_DIAGONAL_BLEND_ALPHA = [0.0, 0.15, 0.3, 0.5]
SWEEP_VOL_SCALE_DAYS        = [0, 20, 30, 60]
SWEEP_LAMBDA_GRID_NAME      = ["high", "very_high"]
SWEEP_HOLDOUT_DAYS          = [21, 63]

LAMBDA_GRIDS = {
    # previous ceiling was 250 — start grid there and push up
    "high":      [50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0],
    "very_high": [250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0],
}


# ============================================================
# helpers
# ============================================================
def get_minute_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("minute_")],
        key=lambda x: int(x.split("_")[1])
    )


def zscore_with_train_stats(train_mat, test_mat):
    mean = train_mat.mean(axis=0, skipna=True)
    std = train_mat.std(axis=0, skipna=True, ddof=0).replace(0, 1.0)
    return (train_mat - mean) / std, (test_mat - mean) / std, mean, std


def daily_spread(scores: pd.Series, returns: pd.Series, q: float = 0.10) -> float:
    tmp = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    n = len(tmp)
    if n < 2:
        return np.nan
    k = max(1, int(n * q))
    if 2 * k > n:
        return np.nan
    tmp = tmp.sort_values("score")
    return tmp.iloc[-k:]["ret"].mean() - tmp.iloc[:k]["ret"].mean()


def daily_spread_volscaled(scores: pd.Series, returns: pd.Series, vol: pd.Series, q: float = 0.10) -> float:
    tmp = pd.concat(
        [scores.rename("score"), returns.rename("ret"), vol.rename("vol")], axis=1
    ).dropna()
    n = len(tmp)
    if n < 2:
        return np.nan
    k = max(1, int(n * q))
    if 2 * k > n:
        return np.nan
    tmp["score_adj"] = tmp["score"] / tmp["vol"].clip(lower=1e-8)
    tmp = tmp.sort_values("score_adj")
    inv_vol = 1.0 / tmp["vol"].clip(lower=1e-8)
    short_inv = inv_vol.iloc[:k]
    long_inv  = inv_vol.iloc[-k:]
    short_ret = (tmp.iloc[:k]["ret"] * short_inv).sum() / short_inv.sum()
    long_ret  = (tmp.iloc[-k:]["ret"] * long_inv).sum() / long_inv.sum()
    return long_ret - short_ret


def daily_sector_neutral_spread(scores, returns, ticker_to_sector, q=0.10, min_names=4):
    tmp = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    if tmp.empty:
        return np.nan
    tmp = tmp.join(ticker_to_sector.rename("sector"), how="left").dropna(subset=["sector"])
    if tmp.empty:
        return np.nan
    sector_spreads = []
    for _, g in tmp.groupby("sector"):
        n = len(g)
        if n < min_names:
            continue
        k = max(1, int(n * q))
        if 2 * k > n:
            continue
        g = g.sort_values("score")
        sector_spreads.append(g.iloc[-k:]["ret"].mean() - g.iloc[:k]["ret"].mean())
    return float(np.mean(sector_spreads)) if sector_spreads else np.nan


def sharpe_from_series(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    s = x.std(ddof=1)
    return np.nan if (s == 0 or not np.isfinite(s)) else x.mean() / s


def annualize_sharpe(ds: float) -> float:
    return ds * np.sqrt(252) if np.isfinite(ds) else np.nan


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def fit_sector_block_model(X_df, Y_df, ticker_to_sector, lam, min_sector_assets=2):
    tickers = X_df.columns.tolist()
    W_full = pd.DataFrame(0.0, index=tickers, columns=tickers)
    sector_to_tickers = (
        ticker_to_sector.reset_index()
        .groupby("sector")["ticker"].apply(list).to_dict()
    )
    for _, st in sorted(sector_to_tickers.items()):
        st = [t for t in st if t in X_df.columns]
        if len(st) < min_sector_assets:
            continue
        Wg = fit_ridge(
            X_df[st].to_numpy(dtype=float),
            Y_df[st].to_numpy(dtype=float),
            lam,
        )
        W_full.loc[st, st] = Wg
    return W_full


def fit_model(X_df, Y_df, ticker_to_sector, lam, diagonal_blend_alpha, min_sector_assets=2):
    W_sector = fit_sector_block_model(X_df, Y_df, ticker_to_sector, lam, min_sector_assets)
    diag_vals = np.diag(W_sector.to_numpy())
    W_vals = W_sector.to_numpy().copy()
    np.fill_diagonal(W_vals, diagonal_blend_alpha * diag_vals)
    return pd.DataFrame(W_vals, index=W_sector.index, columns=W_sector.columns)


def choose_lambda(lambda_grid, X_df, Y_df, ticker_to_sector,
                  min_sector_assets, holdout_days, diagonal_blend_alpha):
    n = len(X_df)
    if n <= holdout_days + 30:
        return float(lambda_grid[len(lambda_grid) // 2])

    X_fit = X_df.iloc[:-holdout_days]
    Y_fit = Y_df.iloc[:-holdout_days]
    X_val = X_df.iloc[-holdout_days:]
    Y_val = Y_df.iloc[-holdout_days:]

    best_lam, best_score = None, -np.inf

    for lam in lambda_grid:
        W = fit_model(X_fit, Y_fit, ticker_to_sector, lam, diagonal_blend_alpha, min_sector_assets)
        pred = X_val.to_numpy(dtype=float) @ W.to_numpy(dtype=float)
        pred_df = pd.DataFrame(pred, index=X_val.index, columns=X_val.columns)
        ic = pred_df.apply(
            lambda row: row.corr(Y_val.loc[row.name], method="spearman"), axis=1
        ).dropna().mean()
        if np.isfinite(ic) and ic > best_score:
            best_score, best_lam = ic, lam

    return float(best_lam) if best_lam is not None else float(lambda_grid[len(lambda_grid) // 2])


def predict_from_model(W, X_eval_z_df, y_mean, y_std):
    W_aligned = W.loc[X_eval_z_df.columns, X_eval_z_df.columns]
    pred_z = X_eval_z_df.to_numpy(dtype=float) @ W_aligned.to_numpy(dtype=float)
    pred = pred_z * y_std.loc[W_aligned.columns].to_numpy() + y_mean.loc[W_aligned.columns].to_numpy()
    return pd.DataFrame(pred, index=X_eval_z_df.index, columns=W_aligned.columns)


# ============================================================
# load and clean once
# ============================================================
print("Loading data...")
df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

minute_cols = get_minute_cols(df)
df[minute_cols] = df[minute_cols].fillna(0.0)

required_cols = ["ticker", "date", "residual_ret", "sector"]
df = df.dropna(subset=required_cols).copy()

sector_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
df = df[df["ticker"].isin(sector_counts[sector_counts == 1].index)].copy()

ticker_to_sector_full = df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]
all_dates = np.sort(df["date"].unique())

# global coverage filter
tmp_signal_proxy = (
    df.groupby(["date", "ticker"])[minute_cols].sum().reset_index()
    .pivot(index="date", columns="ticker", values=minute_cols[0])
)
ret_proxy = df.pivot(index="date", columns="ticker", values="residual_ret")
cd = tmp_signal_proxy.index.intersection(ret_proxy.index)
ct = tmp_signal_proxy.columns.intersection(ret_proxy.columns)
ticker_obs = (tmp_signal_proxy.loc[cd, ct].notna() & ret_proxy.loc[cd, ct].notna()).sum(axis=0)
keep_tickers = ticker_obs[ticker_obs >= MIN_TICKER_OBS].index.tolist()
df = df[df["ticker"].isin(keep_tickers)].copy()
ticker_to_sector = ticker_to_sector_full.loc[keep_tickers].copy()

print(f"Loaded: {len(all_dates)} dates, {len(keep_tickers)} tickers")

test_start_idx = INITIAL_TRAIN_DAYS
test_dates = pd.Index(all_dates[test_start_idx:])
refit_points = list(range(0, len(test_dates), REFIT_EVERY_DAYS))

# ============================================================
# precompute PCA signals once per block — reused across all sweep combos
# ============================================================
print("Precomputing PCA signals for all blocks...")
block_data = []

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
    df_block  = df[df["date"].isin(block_test_dates)].copy()

    if df_train.empty or df_block.empty:
        continue

    minute_scaler = StandardScaler()
    X_train_sc = minute_scaler.fit_transform(df_train[minute_cols].to_numpy(dtype=float))
    X_block_sc = minute_scaler.transform(df_block[minute_cols].to_numpy(dtype=float))

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    train_scores = pca.fit_transform(X_train_sc).ravel()
    block_scores = pca.transform(X_block_sc).ravel()

    raw_sum = df_train[minute_cols].sum(axis=1).to_numpy()
    if np.isfinite(np.corrcoef(train_scores, raw_sum)[0, 1]) and np.corrcoef(train_scores, raw_sum)[0, 1] < 0:
        train_scores = -train_scores
        block_scores = -block_scores

    df_train["ofi_pca_signal"] = train_scores
    df_block["ofi_pca_signal"]  = block_scores

    df_window = (
        pd.concat([df_train, df_block])
        .sort_values(["date", "ticker"]).reset_index(drop=True)
    )

    signal_mat = df_window.pivot(index="date", columns="ticker", values="ofi_pca_signal").sort_index().sort_index(axis=1)
    ret_mat    = df_window.pivot(index="date", columns="ticker", values="residual_ret").sort_index().sort_index(axis=1)

    cd2 = signal_mat.index.intersection(ret_mat.index)
    ct2 = signal_mat.columns.intersection(ret_mat.columns)
    signal_mat = signal_mat.loc[cd2, ct2]
    ret_mat    = ret_mat.loc[cd2, ct2]

    block_ticker_obs = (signal_mat.notna() & ret_mat.notna()).sum(axis=0)
    keep = block_ticker_obs[block_ticker_obs >= MIN_TICKER_OBS].index
    signal_mat = signal_mat[keep]
    ret_mat    = ret_mat[keep]

    if signal_mat.shape[1] == 0:
        continue

    signal_train = signal_mat.loc[signal_mat.index.isin(train_dates_window)]
    ret_train    = ret_mat.loc[ret_mat.index.isin(train_dates_window)]
    signal_eval  = signal_mat.loc[signal_mat.index.isin(block_test_dates)]
    ret_eval     = ret_mat.loc[ret_mat.index.isin(block_test_dates)]

    X_train_df = signal_train.iloc[:-1].copy(); X_train_df.index = ret_train.iloc[1:].index
    Y_train_df = ret_train.iloc[1:].copy()
    X_eval_df  = signal_eval.iloc[:-1].copy();  X_eval_df.index  = ret_eval.iloc[1:].index
    Y_eval_df  = ret_eval.iloc[1:].copy()

    if len(X_train_df) < 30 or len(X_eval_df) < 1:
        continue

    common_cols = (
        X_train_df.columns.intersection(Y_train_df.columns)
        .intersection(X_eval_df.columns).intersection(Y_eval_df.columns)
    )
    X_train_df = X_train_df[common_cols]; Y_train_df = Y_train_df[common_cols]
    X_eval_df  = X_eval_df[common_cols];  Y_eval_df  = Y_eval_df[common_cols]

    X_train_z_df, X_eval_z_df, _, _          = zscore_with_train_stats(X_train_df, X_eval_df)
    Y_train_z_df, Y_eval_z_df, y_mean, y_std = zscore_with_train_stats(Y_train_df, Y_eval_df)

    for mat in [X_train_z_df, X_eval_z_df, Y_train_z_df, Y_eval_z_df]:
        mat.fillna(0.0, inplace=True)

    finite_mask = (
        np.isfinite(X_train_z_df.to_numpy()).any(axis=0) &
        np.isfinite(Y_train_z_df.to_numpy()).any(axis=0) &
        np.isfinite(X_eval_z_df.to_numpy()).any(axis=0) &
        np.isfinite(Y_eval_z_df.to_numpy()).any(axis=0)
    )
    kept = X_train_z_df.columns[finite_mask]
    if len(kept) == 0:
        continue

    block_data.append({
        "block_num":        block_num,
        "X_train_z_df":     X_train_z_df[kept],
        "Y_train_z_df":     Y_train_z_df[kept],
        "X_eval_z_df":      X_eval_z_df[kept],
        "y_mean":           y_mean[kept],
        "y_std":            y_std[kept],
        "ticker_to_sector": ticker_to_sector.loc[kept],
    })

print(f"Precomputed {len(block_data)} valid blocks.")

ret_full = df.pivot(index="date", columns="ticker", values="residual_ret").sort_index().sort_index(axis=1)

# ============================================================
# sweep
# ============================================================
all_results = []
combos = list(itertools.product(
    SWEEP_DIAGONAL_BLEND_ALPHA,
    SWEEP_VOL_SCALE_DAYS,
    SWEEP_LAMBDA_GRID_NAME,
    SWEEP_HOLDOUT_DAYS,
))

print(f"\nRunning {len(combos)} combinations x {len(block_data)} blocks...\n")

for combo_idx, (diag_alpha, vol_days, lam_grid_name, holdout_days) in enumerate(combos, start=1):
    lambda_grid = LAMBDA_GRIDS[lam_grid_name]
    label = f"diag={diag_alpha} | vol={vol_days} | lam={lam_grid_name} | holdout={holdout_days}"
    print(f"[{combo_idx:>3}/{len(combos)}] {label}")

    pred_chunks    = []
    lambda_history = []

    for bd in block_data:
        lam_star = choose_lambda(
            lambda_grid,
            bd["X_train_z_df"], bd["Y_train_z_df"],
            bd["ticker_to_sector"], MIN_SECTOR_ASSETS,
            holdout_days, diag_alpha,
        )
        W = fit_model(
            bd["X_train_z_df"], bd["Y_train_z_df"],
            bd["ticker_to_sector"], lam_star, diag_alpha, MIN_SECTOR_ASSETS,
        )
        pred_eval_df = predict_from_model(W, bd["X_eval_z_df"], bd["y_mean"], bd["y_std"])
        pred_chunks.append(pred_eval_df)
        lambda_history.append(lam_star)

    if not pred_chunks:
        continue

    pred_all_df = pd.concat(pred_chunks).sort_index()
    pred_all_df = pred_all_df[~pred_all_df.index.duplicated(keep="last")]

    ced = pred_all_df.index.intersection(ret_full.index)
    cec = pred_all_df.columns.intersection(ret_full.columns)
    pred_all_df = pred_all_df.loc[ced, cec]
    ret_eval_df = ret_full.loc[ced, cec]
    ts_eval     = ticker_to_sector.loc[cec]

    daily_spreads = pred_all_df.apply(
        lambda row: daily_spread(row, ret_eval_df.loc[row.name], q=Q), axis=1
    ).dropna()

    daily_spreads_sn = pred_all_df.apply(
        lambda row: daily_sector_neutral_spread(
            row, ret_eval_df.loc[row.name], ts_eval, q=Q, min_names=MIN_NAMES_PER_SECTOR_NEUTRAL
        ), axis=1
    ).dropna()

    daily_ic = pred_all_df.apply(
        lambda row: row.corr(ret_eval_df.loc[row.name], method="spearman"), axis=1
    ).dropna()

    if vol_days > 0:
        rolling_vol = ret_eval_df.rolling(vol_days, min_periods=5).std()
        daily_spreads_vs = pred_all_df.apply(
            lambda row: daily_spread_volscaled(
                row, ret_eval_df.loc[row.name],
                vol=rolling_vol.loc[row.name].dropna(), q=Q,
            ), axis=1
        ).dropna()
    else:
        daily_spreads_vs = daily_spreads.copy()

    lam_arr = np.array(lambda_history)

    all_results.append({
        "diagonal_blend_alpha":        diag_alpha,
        "vol_scale_days":              vol_days,
        "lambda_grid":                 lam_grid_name,
        "holdout_days":                holdout_days,
        "lambda_mean":                 float(np.mean(lam_arr)),
        "lambda_min":                  float(np.min(lam_arr)),
        "lambda_max":                  float(np.max(lam_arr)),
        "lambda_pct_at_ceiling":       float(np.mean(lam_arr == max(lambda_grid))),
        "annualized_sharpe":           annualize_sharpe(sharpe_from_series(daily_spreads)),
        "annualized_sharpe_sn":        annualize_sharpe(sharpe_from_series(daily_spreads_sn)),
        "annualized_sharpe_volscaled": annualize_sharpe(sharpe_from_series(daily_spreads_vs)),
        "mean_daily_spread":           float(daily_spreads.mean()),
        "std_daily_spread":            float(daily_spreads.std(ddof=1)),
        "mean_daily_ic":               float(daily_ic.mean()),
        "n_eval_days":                 len(pred_all_df),
    })

# ============================================================
# save + print
# ============================================================
results_df = (
    pd.DataFrame(all_results)
    .sort_values("annualized_sharpe_volscaled", ascending=False)
    .reset_index(drop=True)
)

results_df.to_csv(OUT_DIR / "sweep_summary.csv", index=False)

print("\n" + "=" * 90)
print("TOP 15 COMBOS — ranked by annualized vol-scaled Sharpe")
print("=" * 90)
display_cols = [
    "diagonal_blend_alpha", "vol_scale_days", "lambda_grid", "holdout_days",
    "lambda_mean", "lambda_pct_at_ceiling",
    "annualized_sharpe", "annualized_sharpe_sn", "annualized_sharpe_volscaled",
    "mean_daily_ic",
]
print(results_df[display_cols].head(15).to_string(index=True))
print(f"\nFull results saved to: {(OUT_DIR / 'sweep_summary.csv').resolve()}")