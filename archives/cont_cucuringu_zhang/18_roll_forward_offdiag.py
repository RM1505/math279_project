import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# ============================================================
# config
# ============================================================
INPUT_PATH = Path("data/processed/feature_table_with_residuals.csv")
OUT_DIR = Path("data/processed/rolling_adjacency_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sector_offdiag"   # recommended best model from your current results
LAMBDA_GRID = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]

Q = 0.10
MIN_TICKER_OBS = 252
MIN_SECTOR_ASSETS = 2
MIN_NAMES_PER_SECTOR_NEUTRAL = 4

# rolling settings
INITIAL_TRAIN_DAYS = 250     # about 3 years
REFIT_EVERY_DAYS = 5        # monthly refit

# PCA settings
N_PCA_COMPONENTS = 1


# ============================================================
# helpers
# ============================================================
def get_minute_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("minute_")],
        key=lambda x: int(x.split("_")[1])
    )


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
    short_ret = tmp.iloc[:k]["ret"].mean()
    long_ret = tmp.iloc[-k:]["ret"].mean()
    return long_ret - short_ret


def daily_sector_neutral_spread(
    scores: pd.Series,
    returns: pd.Series,
    ticker_to_sector: pd.Series,
    q: float = 0.10,
    min_names_per_sector: int = 4,
) -> float:
    tmp = pd.concat(
        [scores.rename("score"), returns.rename("ret")],
        axis=1
    ).dropna()

    if tmp.empty:
        return np.nan

    tmp = tmp.join(ticker_to_sector.rename("sector"), how="left")
    tmp = tmp.dropna(subset=["sector"])

    if tmp.empty:
        return np.nan

    sector_spreads = []

    for sector, g in tmp.groupby("sector"):
        n = len(g)
        if n < min_names_per_sector:
            continue

        k = max(1, int(n * q))
        if 2 * k > n:
            continue

        g = g.sort_values("score")
        short_ret = g.iloc[:k]["ret"].mean()
        long_ret = g.iloc[-k:]["ret"].mean()
        sector_spreads.append(long_ret - short_ret)

    if len(sector_spreads) == 0:
        return np.nan

    return float(np.mean(sector_spreads))


def sharpe_from_series(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    s = x.std(ddof=1)
    if s == 0 or not np.isfinite(s):
        return np.nan
    return x.mean() / s


def annualize_sharpe(daily_sharpe: float) -> float:
    return daily_sharpe * np.sqrt(252) if np.isfinite(daily_sharpe) else np.nan


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X
    XtY = X.T @ Y
    return np.linalg.solve(XtX + lam * np.eye(X.shape[1]), XtY)

def fit_lasso(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fit Y ≈ XW column-by-column using sklearn Lasso.

    X: shape (n_samples, n_features)
    Y: shape (n_samples, n_targets)

    Returns:
        W: shape (n_features, n_targets)
    """
    n_features = X.shape[1]
    n_targets = Y.shape[1]
    W = np.zeros((n_features, n_targets), dtype=float)

    for j in range(n_targets):
        y = Y[:, j]

        model = Lasso(
            alpha=alpha,
            fit_intercept=False,   # data already z-scored
            max_iter=20000,
            tol=1e-4,
            selection="cyclic",
            random_state=42,
        )
        model.fit(X, y)
        W[:, j] = model.coef_

    return W


def fit_dense_model(X_train: np.ndarray, Y_train: np.ndarray, lam: float) -> np.ndarray:
    return fit_lasso(X_train, Y_train, lam)


def soft_threshold(z: float, gamma: float) -> float:
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


def fit_diagonal_model(X_train: np.ndarray, Y_train: np.ndarray, lam: float) -> np.ndarray:
    """
    1D lasso for each asset separately:
        min_beta (1/(2n)) * ||y - x beta||^2 + lam * |beta|

    Closed-form via soft-thresholding.
    """
    n_assets = X_train.shape[1]
    n = X_train.shape[0]
    W = np.zeros((n_assets, n_assets), dtype=float)

    for j in range(n_assets):
        x = X_train[:, j]
        y = Y_train[:, j]

        a = np.dot(x, x) / n
        c = np.dot(x, y) / n

        if a <= 0 or not np.isfinite(a):
            beta = 0.0
        else:
            beta = soft_threshold(c, lam) / a

        W[j, j] = beta

    return W

def fit_sector_block_model(
    X_train_df: pd.DataFrame,
    Y_train_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    lam: float,
    min_sector_assets: int = 2,
) -> pd.DataFrame:
    tickers = X_train_df.columns.tolist()
    W_full = pd.DataFrame(0.0, index=tickers, columns=tickers, dtype=float)

    sector_to_tickers = (
        ticker_to_sector.reset_index()
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    for sector, sector_tickers in sorted(sector_to_tickers.items()):
        sector_tickers = [t for t in sector_tickers if t in X_train_df.columns]

        if len(sector_tickers) < min_sector_assets:
            continue

        Xg = X_train_df[sector_tickers].to_numpy(dtype=float)
        Yg = Y_train_df[sector_tickers].to_numpy(dtype=float)

        Wg = fit_lasso(Xg, Yg, alpha=lam)
        W_full.loc[sector_tickers, sector_tickers] = Wg

    return W_full

def build_sector_offdiag(W_sector: pd.DataFrame) -> pd.DataFrame:
    vals = W_sector.to_numpy(copy=True)
    np.fill_diagonal(vals, 0.0)
    return pd.DataFrame(vals, index=W_sector.index, columns=W_sector.columns)


def fit_model(
    model_name: str,
    lam: float,
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    min_sector_assets: int,
):
    X_train_z = X_train_z_df.to_numpy(dtype=float)
    Y_train_z = Y_train_z_df.to_numpy(dtype=float)

    if model_name == "dense":
        return fit_dense_model(X_train_z, Y_train_z, lam)

    if model_name == "diagonal_only":
        return fit_diagonal_model(X_train_z, Y_train_z, lam)

    if model_name == "sector_block":
        return fit_sector_block_model(
            X_train_z_df,
            Y_train_z_df,
            ticker_to_sector=ticker_to_sector,
            lam=lam,
            min_sector_assets=min_sector_assets,
        )

    if model_name == "sector_offdiag":
        W_sector = fit_sector_block_model(
            X_train_z_df,
            Y_train_z_df,
            ticker_to_sector=ticker_to_sector,
            lam=lam,
            min_sector_assets=min_sector_assets,
        )
        return build_sector_offdiag(W_sector)

    raise ValueError(f"Unknown model_name: {model_name}")


def predict_from_model(
    W,
    X_eval_z_df: pd.DataFrame,
    y_mean: pd.Series,
    y_std: pd.Series,
) -> pd.DataFrame:
    if isinstance(W, pd.DataFrame):
        W = W.loc[X_eval_z_df.columns, X_eval_z_df.columns]
        pred_eval_z = X_eval_z_df.to_numpy(dtype=float) @ W.to_numpy(dtype=float)
        pred_cols = W.columns
    else:
        pred_eval_z = X_eval_z_df.to_numpy(dtype=float) @ W
        pred_cols = X_eval_z_df.columns

    pred_eval = (
        pred_eval_z * y_std.loc[pred_cols].to_numpy(dtype=float)
        + y_mean.loc[pred_cols].to_numpy(dtype=float)
    )

    return pd.DataFrame(
        pred_eval,
        index=X_eval_z_df.index,
        columns=pred_cols,
    )


def evaluate_predictions(
    pred_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    q: float,
    min_names_per_sector_neutral: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ret_eval_df = ret_df[pred_df.columns].copy()
    sector_map_eval = ticker_to_sector.loc[pred_df.columns]

    daily_spreads = pred_df.apply(
        lambda row: daily_spread(row, ret_eval_df.loc[row.name], q=q),
        axis=1
    ).dropna()

    daily_spreads_sector_neutral = pred_df.apply(
        lambda row: daily_sector_neutral_spread(
            row,
            ret_eval_df.loc[row.name],
            ticker_to_sector=sector_map_eval,
            q=q,
            min_names_per_sector=min_names_per_sector_neutral,
        ),
        axis=1
    ).dropna()

    daily_ic = pred_df.apply(
        lambda row: row.corr(ret_eval_df.loc[row.name], method="spearman"),
        axis=1
    ).dropna()

    return daily_spreads, daily_spreads_sector_neutral, daily_ic


def choose_lambda_on_window(
    model_name: str,
    lambda_grid: list[float],
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    min_sector_assets: int,
) -> float:
    """
    Simple in-sample proxy for lambda choice.
    Since you're mainly trying to get adaptive OOS performance quickly,
    this picks lambda by in-sample fit quality with a ridge penalty already baked in.
    If you want, later we can replace this with a nested rolling validation window.
    """
    X = X_train_z_df.to_numpy(dtype=float)
    Y = Y_train_z_df.to_numpy(dtype=float)

    best_lam = None
    best_score = -np.inf

    for lam in lambda_grid:
        W = fit_model(
            model_name=model_name,
            lam=lam,
            X_train_z_df=X_train_z_df,
            Y_train_z_df=Y_train_z_df,
            ticker_to_sector=ticker_to_sector,
            min_sector_assets=min_sector_assets,
        )

        if isinstance(W, pd.DataFrame):
            pred = X @ W.to_numpy(dtype=float)
        else:
            pred = X @ W

        # use mean cross-sectional Spearman on train as quick lambda selector
        pred_df = pd.DataFrame(pred, index=X_train_z_df.index, columns=X_train_z_df.columns)
        y_df = Y_train_z_df.copy()

        ic = pred_df.apply(
            lambda row: row.corr(y_df.loc[row.name], method="spearman"),
            axis=1
        ).dropna()

        score = ic.mean()

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lam = lam

    if best_lam is None:
        raise ValueError("Failed to choose lambda on rolling window.")

    return float(best_lam)


# ============================================================
# load and clean
# ============================================================
df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

minute_cols = get_minute_cols(df)
df[minute_cols] = df[minute_cols].fillna(0.0)

required_cols = ["ticker", "date", "residual_ret", "sector"]
df = df.dropna(subset=required_cols).copy()

sector_counts = df[["ticker", "sector"]].drop_duplicates().groupby("ticker").size()
valid_sector_tickers = sector_counts[sector_counts == 1].index
df = df[df["ticker"].isin(valid_sector_tickers)].copy()

ticker_to_sector_full = (
    df[["ticker", "sector"]]
    .drop_duplicates()
    .set_index("ticker")["sector"]
)

all_dates = np.sort(df["date"].unique())

if len(all_dates) < INITIAL_TRAIN_DAYS + 2:
    raise ValueError(
        f"Not enough dates ({len(all_dates)}) for INITIAL_TRAIN_DAYS={INITIAL_TRAIN_DAYS}."
    )

# earliest possible test start date: first date after a full training window
test_start_idx = INITIAL_TRAIN_DAYS
test_start_date = all_dates[test_start_idx]

print("Total dates:", len(all_dates))
print("Earliest possible walk-forward test starts at:", pd.Timestamp(test_start_date).date())
print("Dataset runs from:", pd.Timestamp(all_dates[0]).date(), "to", pd.Timestamp(all_dates[-1]).date())

# ============================================================
# global coverage filter first
# ============================================================
tmp_signal_proxy = (
    df.groupby(["date", "ticker"])[minute_cols]
    .sum()
    .reset_index()
    .pivot(index="date", columns="ticker", values=minute_cols[0])
)

ret_proxy = (
    df.pivot(index="date", columns="ticker", values="residual_ret")
)

common_dates = tmp_signal_proxy.index.intersection(ret_proxy.index)
common_tickers = tmp_signal_proxy.columns.intersection(ret_proxy.columns)

tmp_signal_proxy = tmp_signal_proxy.loc[common_dates, common_tickers]
ret_proxy = ret_proxy.loc[common_dates, common_tickers]

ticker_obs = (tmp_signal_proxy.notna() & ret_proxy.notna()).sum(axis=0)
keep_tickers = ticker_obs[ticker_obs >= MIN_TICKER_OBS].index.tolist()

df = df[df["ticker"].isin(keep_tickers)].copy()
ticker_to_sector = ticker_to_sector_full.loc[keep_tickers].copy()

if len(keep_tickers) == 0:
    raise ValueError("No tickers remain after global coverage filtering.")

# ============================================================
# rolling walk-forward
# ============================================================
test_dates = pd.Index(all_dates[test_start_idx:])
if len(test_dates) == 0:
    raise ValueError("No test dates available for walk-forward.")

pred_chunks = []
chosen_lambdas = []
refit_metadata = []
last_kernel = None
last_W = None

refit_points = list(range(0, len(test_dates), REFIT_EVERY_DAYS))

for block_num, start_offset in enumerate(refit_points, start=1):
    block_test_dates = test_dates[start_offset:start_offset + REFIT_EVERY_DAYS]
    if len(block_test_dates) == 0:
        continue

    first_pred_date = block_test_dates[0]

    # trailing train window ends right before first prediction date
    eligible_train_dates = all_dates[all_dates < first_pred_date]
    if len(eligible_train_dates) < INITIAL_TRAIN_DAYS + 1:
        print(f"Skipping block {block_num}: not enough history before {pd.Timestamp(first_pred_date).date()}")
        continue

    train_dates_window = eligible_train_dates[-INITIAL_TRAIN_DAYS:]

    df_train = df[df["date"].isin(train_dates_window)].copy()
    df_block = df[df["date"].isin(block_test_dates)].copy()

    if df_train.empty or df_block.empty:
        continue

    # --------------------------------------------------------
    # fit rolling PCA on training window only
    # --------------------------------------------------------
    X_train_minutes = df_train[minute_cols].to_numpy(dtype=float)
    X_block_minutes = df_block[minute_cols].to_numpy(dtype=float)

    minute_scaler = StandardScaler()
    X_train_minutes_scaled = minute_scaler.fit_transform(X_train_minutes)
    X_block_minutes_scaled = minute_scaler.transform(X_block_minutes)

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    train_scores = pca.fit_transform(X_train_minutes_scaled).ravel()
    block_scores = pca.transform(X_block_minutes_scaled).ravel()

    ofi_kernel = pca.components_[0].copy()

    raw_sum_train = df_train[minute_cols].sum(axis=1).to_numpy()
    corr = np.corrcoef(train_scores, raw_sum_train)[0, 1]
    if np.isfinite(corr) and corr < 0:
        train_scores = -train_scores
        block_scores = -block_scores
        ofi_kernel = -ofi_kernel

    df_train["ofi_pca_signal"] = train_scores
    df_block["ofi_pca_signal"] = block_scores

    # combine just enough dates to form X_t -> Y_{t+1} in this block
    # we need the last train date plus block dates for alignment
    df_window_signal = (
        pd.concat([df_train, df_block], axis=0)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    signal_mat = (
        df_window_signal.pivot(index="date", columns="ticker", values="ofi_pca_signal")
        .sort_index()
        .sort_index(axis=1)
    )

    ret_mat = (
        df_window_signal.pivot(index="date", columns="ticker", values="residual_ret")
        .sort_index()
        .sort_index(axis=1)
    )

    common_dates = signal_mat.index.intersection(ret_mat.index)
    common_tickers = signal_mat.columns.intersection(ret_mat.columns)

    signal_mat = signal_mat.loc[common_dates, common_tickers]
    ret_mat = ret_mat.loc[common_dates, common_tickers]

    block_ticker_obs = (signal_mat.notna() & ret_mat.notna()).sum(axis=0)
    block_keep_tickers = block_ticker_obs[block_ticker_obs >= MIN_TICKER_OBS].index

    signal_mat = signal_mat[block_keep_tickers]
    ret_mat = ret_mat[block_keep_tickers]
    ticker_to_sector_block = ticker_to_sector.loc[block_keep_tickers]

    if signal_mat.shape[1] == 0:
        print(f"Skipping block {block_num}: no assets remain after block filtering.")
        continue

    # split into train part and eval part
    signal_train = signal_mat.loc[signal_mat.index.isin(train_dates_window)].copy()
    ret_train = ret_mat.loc[ret_mat.index.isin(train_dates_window)].copy()

    signal_eval = signal_mat.loc[signal_mat.index.isin(block_test_dates)].copy()
    ret_eval = ret_mat.loc[ret_mat.index.isin(block_test_dates)].copy()

    # build X_t -> Y_{t+1}
    X_train_df = signal_train.iloc[:-1].copy()
    Y_train_df = ret_train.iloc[1:].copy()
    X_train_df.index = Y_train_df.index

    X_eval_df = signal_eval.iloc[:-1].copy()
    Y_eval_df = ret_eval.iloc[1:].copy()
    X_eval_df.index = Y_eval_df.index

    # IMPORTANT:
    # for the first day of the block, prediction uses same-day signal to predict next-day return.
    # so metrics live on dates 2..end of block.
    if len(X_train_df) < 30 or len(X_eval_df) < 1:
        print(f"Skipping block {block_num}: insufficient aligned rows.")
        continue

    # align common columns
    common_cols = (
        X_train_df.columns
        .intersection(Y_train_df.columns)
        .intersection(X_eval_df.columns)
        .intersection(Y_eval_df.columns)
    )

    X_train_df = X_train_df[common_cols]
    Y_train_df = Y_train_df[common_cols]
    X_eval_df = X_eval_df[common_cols]
    Y_eval_df = Y_eval_df[common_cols]
    ticker_to_sector_block = ticker_to_sector_block.loc[common_cols]

    # z-score using train stats only
    X_train_z_df, X_eval_z_df, x_mean, x_std = zscore_with_train_stats(X_train_df, X_eval_df)
    Y_train_z_df, Y_eval_z_df, y_mean, y_std = zscore_with_train_stats(Y_train_df, Y_eval_df)

    X_train_z_df = X_train_z_df.fillna(0.0)
    X_eval_z_df = X_eval_z_df.fillna(0.0)
    Y_train_z_df = Y_train_z_df.fillna(0.0)
    Y_eval_z_df = Y_eval_z_df.fillna(0.0)

    finite_cols = (
        np.isfinite(X_train_z_df.to_numpy()).any(axis=0) &
        np.isfinite(Y_train_z_df.to_numpy()).any(axis=0) &
        np.isfinite(X_eval_z_df.to_numpy()).any(axis=0) &
        np.isfinite(Y_eval_z_df.to_numpy()).any(axis=0)
    )

    kept_cols = X_train_z_df.columns[finite_cols]

    X_train_z_df = X_train_z_df[kept_cols]
    Y_train_z_df = Y_train_z_df[kept_cols]
    X_eval_z_df = X_eval_z_df[kept_cols]
    Y_eval_z_df = Y_eval_z_df[kept_cols]
    y_mean = y_mean[kept_cols]
    y_std = y_std[kept_cols]
    ticker_to_sector_block = ticker_to_sector_block.loc[kept_cols]

    if len(kept_cols) == 0:
        print(f"Skipping block {block_num}: no usable columns after finite filtering.")
        continue

    # choose lambda on this training window
    lam_star = choose_lambda_on_window(
        model_name=MODEL_NAME,
        lambda_grid=LAMBDA_GRID,
        X_train_z_df=X_train_z_df,
        Y_train_z_df=Y_train_z_df,
        ticker_to_sector=ticker_to_sector_block,
        min_sector_assets=MIN_SECTOR_ASSETS,
    )

    W = fit_model(
        model_name=MODEL_NAME,
        lam=lam_star,
        X_train_z_df=X_train_z_df,
        Y_train_z_df=Y_train_z_df,
        ticker_to_sector=ticker_to_sector_block,
        min_sector_assets=MIN_SECTOR_ASSETS,
    )

    pred_eval_df = predict_from_model(
        W=W,
        X_eval_z_df=X_eval_z_df,
        y_mean=y_mean,
        y_std=y_std,
    )

    pred_chunks.append(pred_eval_df)
    chosen_lambdas.append(
        pd.DataFrame({
            "refit_block": [block_num],
            "train_start": [pd.Timestamp(train_dates_window[0]).date()],
            "train_end": [pd.Timestamp(train_dates_window[-1]).date()],
            "pred_start": [pd.Timestamp(block_test_dates[0]).date()],
            "pred_end": [pd.Timestamp(block_test_dates[-1]).date()],
            "lambda": [lam_star],
            "n_assets": [len(kept_cols)],
            #"pca_explained_variance_ratio": [float(pca.explained_variance_ratio_[0])],
        })
    )

    refit_metadata.append({
        "block_num": block_num,
        "lambda": lam_star,
        "n_assets": len(kept_cols),
        "train_start": pd.Timestamp(train_dates_window[0]).date(),
        "train_end": pd.Timestamp(train_dates_window[-1]).date(),
        "pred_start": pd.Timestamp(block_test_dates[0]).date(),
        "pred_end": pd.Timestamp(block_test_dates[-1]).date(),
    })

    last_kernel = ofi_kernel
    last_W = W

    print(
        f"Block {block_num:>3}: "
        f"train {pd.Timestamp(train_dates_window[0]).date()} -> {pd.Timestamp(train_dates_window[-1]).date()} | "
        f"predict {pd.Timestamp(block_test_dates[0]).date()} -> {pd.Timestamp(block_test_dates[-1]).date()} | "
        f"assets={len(kept_cols)} | lambda={lam_star}"
    )

if len(pred_chunks) == 0:
    raise ValueError("No rolling prediction blocks were produced.")

# ============================================================
# stitch predictions and evaluate
# ============================================================
pred_all_df = pd.concat(pred_chunks, axis=0).sort_index()
pred_all_df = pred_all_df[~pred_all_df.index.duplicated(keep="last")]

# realized returns matrix on same dates/columns
ret_full = (
    df.pivot(index="date", columns="ticker", values="residual_ret")
    .sort_index()
    .sort_index(axis=1)
)

common_eval_dates = pred_all_df.index.intersection(ret_full.index)
common_eval_cols = pred_all_df.columns.intersection(ret_full.columns)

pred_all_df = pred_all_df.loc[common_eval_dates, common_eval_cols]
ret_eval_df = ret_full.loc[common_eval_dates, common_eval_cols]
ticker_to_sector_eval = ticker_to_sector.loc[common_eval_cols]

daily_spreads, daily_spreads_sector_neutral, daily_ic = evaluate_predictions(
    pred_df=pred_all_df,
    ret_df=ret_eval_df,
    ticker_to_sector=ticker_to_sector_eval,
    q=Q,
    min_names_per_sector_neutral=MIN_NAMES_PER_SECTOR_NEUTRAL,
)

summary = pd.DataFrame([{
    "model": MODEL_NAME,
    "rolling_train_days": INITIAL_TRAIN_DAYS,
    "refit_every_days": REFIT_EVERY_DAYS,
    "mean_daily_spread": daily_spreads.mean(),
    "std_daily_spread": daily_spreads.std(ddof=1),
    "daily_spread_sharpe": sharpe_from_series(daily_spreads),
    "annualized_spread_sharpe": annualize_sharpe(sharpe_from_series(daily_spreads)),
    "mean_daily_spread_sector_neutral": daily_spreads_sector_neutral.mean(),
    "std_daily_spread_sector_neutral": daily_spreads_sector_neutral.std(ddof=1),
    "daily_spread_sharpe_sector_neutral": sharpe_from_series(daily_spreads_sector_neutral),
    "annualized_spread_sharpe_sector_neutral": annualize_sharpe(sharpe_from_series(daily_spreads_sector_neutral)),
    "mean_daily_ic": daily_ic.mean(),
    "std_daily_ic": daily_ic.std(ddof=1),
    "n_eval_days": len(pred_all_df),
    "n_refits": len(refit_metadata),
}])

print("\nRolling summary:")
print(summary.to_string(index=False))

# ============================================================
# save outputs
# ============================================================
pred_all_df.to_csv(OUT_DIR / f"predicted_returns_{MODEL_NAME}_rolling.csv")
daily_spreads.to_csv(OUT_DIR / f"daily_spreads_{MODEL_NAME}_rolling.csv", header=["spread"])
daily_spreads_sector_neutral.to_csv(
    OUT_DIR / f"daily_spreads_sector_neutral_{MODEL_NAME}_rolling.csv",
    header=["spread_sector_neutral"]
)
daily_ic.to_csv(OUT_DIR / f"daily_ic_{MODEL_NAME}_rolling.csv", header=["ic"])
summary.to_csv(OUT_DIR / f"rolling_summary_{MODEL_NAME}.csv", index=False)

pd.concat(chosen_lambdas, axis=0, ignore_index=True).to_csv(
    OUT_DIR / f"rolling_lambda_history_{MODEL_NAME}.csv",
    index=False
)

if last_kernel is not None:
    np.save(OUT_DIR / f"last_ofi_market_pca_kernel_{MODEL_NAME}_rolling.npy", last_kernel)

if last_W is not None:
    if isinstance(last_W, pd.DataFrame):
        last_W.to_csv(OUT_DIR / f"last_adjacency_matrix_{MODEL_NAME}_rolling.csv")
    else:
        pd.DataFrame(last_W, index=pred_all_df.columns, columns=pred_all_df.columns).to_csv(
            OUT_DIR / f"last_adjacency_matrix_{MODEL_NAME}_rolling.csv"
        )

print("\nSaved rolling outputs to:", OUT_DIR.resolve())