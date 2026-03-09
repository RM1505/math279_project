import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("data/processed/feature_table_with_residuals.csv")
OUT_DIR = Path("data/processed")

LAMBDA_GRID = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]
Q = 0.10
MIN_TICKER_OBS = 252
TRAIN_FRAC = 0.80
MIN_SECTOR_ASSETS = 2
MIN_NAMES_PER_SECTOR_NEUTRAL = 4


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
    """
    For one date:
      - rank within each sector
      - long top q fraction, short bottom q fraction
      - average sector spreads equally across sectors
    """
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


def fit_dense_model(X_train: np.ndarray, Y_train: np.ndarray, lam: float) -> np.ndarray:
    return fit_ridge(X_train, Y_train, lam)


def fit_diagonal_model(X_train: np.ndarray, Y_train: np.ndarray, lam: float) -> np.ndarray:
    n_assets = X_train.shape[1]
    W = np.zeros((n_assets, n_assets), dtype=float)

    for j in range(n_assets):
        x = X_train[:, j]
        y = Y_train[:, j]
        denom = np.dot(x, x) + lam
        beta = 0.0 if denom == 0 else np.dot(x, y) / denom
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

        Wg = fit_ridge(Xg, Yg, lam)
        W_full.loc[sector_tickers, sector_tickers] = Wg

    return W_full


def build_sector_offdiag(W_sector: pd.DataFrame) -> pd.DataFrame:
    vals = W_sector.to_numpy(copy=True)
    np.fill_diagonal(vals, 0.0)
    return pd.DataFrame(vals, index=W_sector.index, columns=W_sector.columns)


def evaluate_model(
    name: str,
    W,
    X_test_z_df: pd.DataFrame,
    Y_test_df: pd.DataFrame,
    y_mean: pd.Series,
    y_std: pd.Series,
    ticker_to_sector: pd.Series,
    q: float,
    min_names_per_sector_neutral: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    if isinstance(W, pd.DataFrame):
        W = W.loc[X_test_z_df.columns, X_test_z_df.columns]
        pred_test_z = X_test_z_df.to_numpy(dtype=float) @ W.to_numpy(dtype=float)
        pred_cols = W.columns
    else:
        pred_test_z = X_test_z_df.to_numpy(dtype=float) @ W
        pred_cols = X_test_z_df.columns

    pred_test = (
        pred_test_z * y_std.loc[pred_cols].to_numpy(dtype=float)
        + y_mean.loc[pred_cols].to_numpy(dtype=float)
    )

    pred_test_df = pd.DataFrame(
        pred_test,
        index=X_test_z_df.index,
        columns=pred_cols,
    )

    ret_test_eval_df = Y_test_df[pred_cols].copy()
    sector_map_eval = ticker_to_sector.loc[pred_cols]

    daily_spreads = pred_test_df.apply(
        lambda row: daily_spread(row, ret_test_eval_df.loc[row.name], q=q),
        axis=1
    ).dropna()

    daily_spreads_sector_neutral = pred_test_df.apply(
        lambda row: daily_sector_neutral_spread(
            row,
            ret_test_eval_df.loc[row.name],
            ticker_to_sector=sector_map_eval,
            q=q,
            min_names_per_sector=min_names_per_sector_neutral,
        ),
        axis=1
    ).dropna()

    daily_ic = pred_test_df.apply(
        lambda row: row.corr(ret_test_eval_df.loc[row.name], method="spearman"),
        axis=1
    ).dropna()

    daily_sharpe = sharpe_from_series(daily_spreads)
    annual_sharpe = annualize_sharpe(daily_sharpe)

    daily_sharpe_sector_neutral = sharpe_from_series(daily_spreads_sector_neutral)
    annual_sharpe_sector_neutral = annualize_sharpe(daily_sharpe_sector_neutral)

    metrics = {
        "model": name,
        "mean_daily_spread": daily_spreads.mean(),
        "std_daily_spread": daily_spreads.std(ddof=1),
        "daily_spread_sharpe": daily_sharpe,
        "annualized_spread_sharpe": annual_sharpe,
        "mean_daily_spread_sector_neutral": daily_spreads_sector_neutral.mean(),
        "std_daily_spread_sector_neutral": daily_spreads_sector_neutral.std(ddof=1),
        "daily_spread_sharpe_sector_neutral": daily_sharpe_sector_neutral,
        "annualized_spread_sharpe_sector_neutral": annual_sharpe_sector_neutral,
        "mean_daily_ic": daily_ic.mean(),
        "std_daily_ic": daily_ic.std(ddof=1),
        "n_pred_assets": pred_test_df.shape[1],
        "n_test_days": pred_test_df.shape[0],
    }

    return pred_test_df, daily_spreads, daily_spreads_sector_neutral, daily_ic, metrics


def sector_sharpe_breakdown(
    pred_test_df: pd.DataFrame,
    ret_test_df: pd.DataFrame,
    ticker_to_sector: pd.Series,
    q: float,
) -> pd.DataFrame:
    rows = []

    sector_to_tickers = (
        ticker_to_sector.loc[pred_test_df.columns]
        .reset_index()
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    for sector, tickers in sorted(sector_to_tickers.items()):
        tickers = [t for t in tickers if t in pred_test_df.columns and t in ret_test_df.columns]

        if len(tickers) < 2:
            continue

        pred_sector = pred_test_df[tickers]
        ret_sector = ret_test_df[tickers]

        daily_spreads_sector = pred_sector.apply(
            lambda row: daily_spread(row, ret_sector.loc[row.name], q=q),
            axis=1
        ).dropna()

        daily_ic_sector = pred_sector.apply(
            lambda row: row.corr(ret_sector.loc[row.name], method="spearman"),
            axis=1
        ).dropna()

        rows.append({
            "sector": sector,
            "n_assets": len(tickers),
            "mean_daily_spread": daily_spreads_sector.mean(),
            "std_daily_spread": daily_spreads_sector.std(ddof=1),
            "daily_spread_sharpe": sharpe_from_series(daily_spreads_sector),
            "annualized_spread_sharpe": annualize_sharpe(sharpe_from_series(daily_spreads_sector)),
            "mean_daily_ic": daily_ic_sector.mean(),
            "std_daily_ic": daily_ic_sector.std(ddof=1),
            "n_days": len(daily_spreads_sector),
        })

    return pd.DataFrame(rows).sort_values(
        ["annualized_spread_sharpe", "sector"],
        ascending=[False, True]
    )


def fit_and_eval_single_model(
    model_name: str,
    lam: float,
    X_train_z: np.ndarray,
    Y_train_z: np.ndarray,
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    X_test_z_df: pd.DataFrame,
    Y_test_df: pd.DataFrame,
    y_mean: pd.Series,
    y_std: pd.Series,
    ticker_to_sector: pd.Series,
    q: float,
    min_sector_assets: int,
    min_names_per_sector_neutral: int,
):
    if model_name == "dense":
        W = fit_dense_model(X_train_z, Y_train_z, lam)

    elif model_name == "diagonal_only":
        W = fit_diagonal_model(X_train_z, Y_train_z, lam)

    elif model_name == "sector_block":
        W = fit_sector_block_model(
            X_train_z_df,
            Y_train_z_df,
            ticker_to_sector=ticker_to_sector,
            lam=lam,
            min_sector_assets=min_sector_assets,
        )

    elif model_name == "sector_offdiag":
        W_sector = fit_sector_block_model(
            X_train_z_df,
            Y_train_z_df,
            ticker_to_sector=ticker_to_sector,
            lam=lam,
            min_sector_assets=min_sector_assets,
        )
        W = build_sector_offdiag(W_sector)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pred_test_df, daily_spreads, daily_spreads_sector_neutral, daily_ic, metrics = evaluate_model(
        model_name,
        W,
        X_test_z_df,
        Y_test_df,
        y_mean,
        y_std,
        ticker_to_sector,
        q,
        min_names_per_sector_neutral,
    )

    sector_breakdown_df = sector_sharpe_breakdown(
        pred_test_df,
        Y_test_df[pred_test_df.columns],
        ticker_to_sector,
        q=q,
    )

    metrics["lambda"] = lam

    return W, pred_test_df, daily_spreads, daily_spreads_sector_neutral, daily_ic, metrics, sector_breakdown_df


def lambda_sweep_for_model(
    model_name: str,
    lambda_grid: list[float],
    X_train_z: np.ndarray,
    Y_train_z: np.ndarray,
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    X_test_z_df: pd.DataFrame,
    Y_test_df: pd.DataFrame,
    y_mean: pd.Series,
    y_std: pd.Series,
    ticker_to_sector: pd.Series,
    q: float,
    min_sector_assets: int,
    min_names_per_sector_neutral: int,
):
    sweep_rows = []
    results_by_lambda = {}

    for lam in lambda_grid:
        W, pred_test_df, daily_spreads, daily_spreads_sector_neutral, daily_ic, metrics, sector_breakdown_df = fit_and_eval_single_model(
            model_name=model_name,
            lam=lam,
            X_train_z=X_train_z,
            Y_train_z=Y_train_z,
            X_train_z_df=X_train_z_df,
            Y_train_z_df=Y_train_z_df,
            X_test_z_df=X_test_z_df,
            Y_test_df=Y_test_df,
            y_mean=y_mean,
            y_std=y_std,
            ticker_to_sector=ticker_to_sector,
            q=q,
            min_sector_assets=min_sector_assets,
            min_names_per_sector_neutral=min_names_per_sector_neutral,
        )

        sweep_rows.append(metrics)
        results_by_lambda[lam] = {
            "W": W,
            "pred_test_df": pred_test_df,
            "daily_spreads": daily_spreads,
            "daily_spreads_sector_neutral": daily_spreads_sector_neutral,
            "daily_ic": daily_ic,
            "sector_breakdown_df": sector_breakdown_df,
            "metrics": metrics,
        }

        print(
            f"{model_name:>15} | lambda={lam:>6} | "
            f"ann_sharpe={metrics['annualized_spread_sharpe']:.6f} | "
            f"ann_sharpe_sector_neutral={metrics['annualized_spread_sharpe_sector_neutral']:.6f} | "
            f"mean_ic={metrics['mean_daily_ic']:.6f}"
        )

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ["annualized_spread_sharpe_sector_neutral", "annualized_spread_sharpe", "mean_daily_ic", "lambda"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    best_lambda = sweep_df.iloc[0]["lambda"]
    best_result = results_by_lambda[best_lambda]

    return sweep_df, best_lambda, best_result


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

ticker_to_sector = (
    df[["ticker", "sector"]]
    .drop_duplicates()
    .set_index("ticker")["sector"]
)


all_dates = np.sort(df["date"].unique())
split_idx = int(len(all_dates) * TRAIN_FRAC)

train_dates = all_dates[:split_idx]
test_dates = all_dates[split_idx:]

if len(train_dates) < 2 or len(test_dates) < 2:
    raise ValueError("Not enough dates for a train/test split.")

df_train = df[df["date"].isin(train_dates)].copy()
df_test = df[df["date"].isin(test_dates)].copy()

X_train_minutes = df_train[minute_cols].to_numpy(dtype=float)
X_test_minutes = df_test[minute_cols].to_numpy(dtype=float)

minute_scaler = StandardScaler()
X_train_minutes_scaled = minute_scaler.fit_transform(X_train_minutes)
X_test_minutes_scaled = minute_scaler.transform(X_test_minutes)

pca = PCA(n_components=1, random_state=42)
train_scores = pca.fit_transform(X_train_minutes_scaled).ravel()
test_scores = pca.transform(X_test_minutes_scaled).ravel()

ofi_kernel = pca.components_[0].copy()

raw_sum_train = df_train[minute_cols].sum(axis=1).to_numpy()
corr = np.corrcoef(train_scores, raw_sum_train)[0, 1]
if np.isfinite(corr) and corr < 0:
    train_scores = -train_scores
    test_scores = -test_scores
    ofi_kernel = -ofi_kernel

df_train["ofi_pca_signal"] = train_scores
df_test["ofi_pca_signal"] = test_scores

df_with_signal = (
    pd.concat([df_train, df_test], axis=0)
    .sort_values(["date", "ticker"])
    .reset_index(drop=True)
)

signal_mat = (
    df_with_signal.pivot(index="date", columns="ticker", values="ofi_pca_signal")
    .sort_index()
    .sort_index(axis=1)
)

ret_mat = (
    df_with_signal.pivot(index="date", columns="ticker", values="residual_ret")
    .sort_index()
    .sort_index(axis=1)
)

common_dates = signal_mat.index.intersection(ret_mat.index)
common_tickers = signal_mat.columns.intersection(ret_mat.columns)

signal_mat = signal_mat.loc[common_dates, common_tickers]
ret_mat = ret_mat.loc[common_dates, common_tickers]

ticker_obs = (signal_mat.notna() & ret_mat.notna()).sum(axis=0)
keep_tickers = ticker_obs[ticker_obs >= MIN_TICKER_OBS].index

signal_mat = signal_mat[keep_tickers]
ret_mat = ret_mat[keep_tickers]
ticker_to_sector = ticker_to_sector.loc[keep_tickers]

if signal_mat.shape[1] == 0:
    raise ValueError("No tickers remain after coverage filtering.")

if signal_mat.shape[0] < 3:
    raise ValueError("Not enough dates after alignment.")


train_dates_idx = signal_mat.index.intersection(pd.Index(train_dates))
test_dates_idx = signal_mat.index.intersection(pd.Index(test_dates))

signal_train = signal_mat.loc[train_dates_idx].copy()
ret_train = ret_mat.loc[train_dates_idx].copy()

signal_test = signal_mat.loc[test_dates_idx].copy()
ret_test = ret_mat.loc[test_dates_idx].copy()

if len(signal_train) < 2 or len(signal_test) < 2:
    raise ValueError("Not enough train/test dates after pivot alignment.")


X_train_df = signal_train.iloc[:-1].copy()
Y_train_df = ret_train.iloc[1:].copy()
X_train_df.index = Y_train_df.index

X_test_df = signal_test.iloc[:-1].copy()
Y_test_df = ret_test.iloc[1:].copy()
X_test_df.index = Y_test_df.index

if X_train_df.shape != Y_train_df.shape:
    raise ValueError(f"Train shape mismatch: {X_train_df.shape} vs {Y_train_df.shape}")

if X_test_df.shape != Y_test_df.shape:
    raise ValueError(f"Test shape mismatch: {X_test_df.shape} vs {Y_test_df.shape}")


X_train_z_df, X_test_z_df, x_mean, x_std = zscore_with_train_stats(X_train_df, X_test_df)
Y_train_z_df, Y_test_z_df, y_mean, y_std = zscore_with_train_stats(Y_train_df, Y_test_df)

X_train_z_df = X_train_z_df.fillna(0.0)
X_test_z_df = X_test_z_df.fillna(0.0)
Y_train_z_df = Y_train_z_df.fillna(0.0)
Y_test_z_df = Y_test_z_df.fillna(0.0)

valid_cols = (
    np.isfinite(X_train_z_df.to_numpy()).any(axis=0) &
    np.isfinite(Y_train_z_df.to_numpy()).any(axis=0) &
    np.isfinite(X_test_z_df.to_numpy()).any(axis=0) &
    np.isfinite(Y_test_z_df.to_numpy()).any(axis=0)
)

kept_cols = X_train_df.columns[valid_cols]

X_train_df = X_train_df[kept_cols]
Y_train_df = Y_train_df[kept_cols]
X_test_df = X_test_df[kept_cols]
Y_test_df = Y_test_df[kept_cols]

X_train_z_df = X_train_z_df[kept_cols]
Y_train_z_df = Y_train_z_df[kept_cols]
X_test_z_df = X_test_z_df[kept_cols]
Y_test_z_df = Y_test_z_df[kept_cols]

y_mean = y_mean[kept_cols]
y_std = y_std[kept_cols]
ticker_to_sector = ticker_to_sector.loc[kept_cols]

X_train_z = X_train_z_df.to_numpy(dtype=float)
Y_train_z = Y_train_z_df.to_numpy(dtype=float)

n_train_dates, n_assets = X_train_z.shape
n_test_dates = X_test_z_df.shape[0]

if n_assets == 0:
    raise ValueError("No usable assets remain after filtering.")

print(f"Using {n_train_dates} train dates, {n_test_dates} test dates, and {n_assets} assets.")
print("Train dates:", pd.Timestamp(train_dates[0]).date(), "to", pd.Timestamp(train_dates[-1]).date())
print("Test dates:", pd.Timestamp(test_dates[0]).date(), "to", pd.Timestamp(test_dates[-1]).date())
print("PCA explained variance ratio:", float(pca.explained_variance_ratio_[0]))
print("Lambda grid:", LAMBDA_GRID)

model_names = ["dense", "sector_block", "diagonal_only", "sector_offdiag"]

all_sweeps = []
best_summary_rows = []
best_results = {}

for model_name in model_names:
    print(f"\n=== Sweeping {model_name} ===")

    sweep_df, best_lambda, best_result = lambda_sweep_for_model(
        model_name=model_name,
        lambda_grid=LAMBDA_GRID,
        X_train_z=X_train_z,
        Y_train_z=Y_train_z,
        X_train_z_df=X_train_z_df,
        Y_train_z_df=Y_train_z_df,
        X_test_z_df=X_test_z_df,
        Y_test_df=Y_test_df,
        y_mean=y_mean,
        y_std=y_std,
        ticker_to_sector=ticker_to_sector,
        q=Q,
        min_sector_assets=MIN_SECTOR_ASSETS,
        min_names_per_sector_neutral=MIN_NAMES_PER_SECTOR_NEUTRAL,
    )

    sweep_df["model"] = model_name
    all_sweeps.append(sweep_df)

    best_metrics = best_result["metrics"].copy()
    best_metrics["best_lambda"] = best_lambda
    best_summary_rows.append(best_metrics)

    best_results[model_name] = {
        "best_lambda": best_lambda,
        **best_result,
    }

lambda_sweep_df = pd.concat(all_sweeps, axis=0, ignore_index=True)
best_models_df = pd.DataFrame(best_summary_rows).sort_values(
    ["annualized_spread_sharpe_sector_neutral", "annualized_spread_sharpe", "mean_daily_ic"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("\nBest model summary:")
print(best_models_df.to_string(index=False))

OUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUT_DIR / "ofi_market_pca_kernel_train_only.npy", ofi_kernel)
df_with_signal.to_csv(OUT_DIR / "feature_table_with_market_pca_signal_train_test.csv", index=False)
signal_mat.to_csv(OUT_DIR / "signal_matrix_unbalanced.csv")
ret_mat.to_csv(OUT_DIR / "return_matrix_unbalanced.csv")

lambda_sweep_df.to_csv(OUT_DIR / "lambda_sweep_results.csv", index=False)
best_models_df.to_csv(OUT_DIR / "best_model_summary.csv", index=False)
ticker_to_sector.rename("sector").to_csv(OUT_DIR / "ticker_sector_used.csv", header=True)

for model_name, result in best_results.items():
    best_lambda = result["best_lambda"]
    pred_test_df = result["pred_test_df"]
    daily_spreads = result["daily_spreads"]
    daily_spreads_sector_neutral = result["daily_spreads_sector_neutral"]
    daily_ic = result["daily_ic"]
    sector_breakdown_df = result["sector_breakdown_df"]
    W = result["W"]

    safe_model = model_name.replace(" ", "_")

    pred_test_df.to_csv(OUT_DIR / f"predicted_returns_{safe_model}_best_lambda.csv")
    daily_spreads.to_csv(OUT_DIR / f"daily_spreads_{safe_model}_best_lambda.csv", header=["spread"])
    daily_spreads_sector_neutral.to_csv(
        OUT_DIR / f"daily_spreads_sector_neutral_{safe_model}_best_lambda.csv",
        header=["spread_sector_neutral"]
    )
    daily_ic.to_csv(OUT_DIR / f"daily_ic_{safe_model}_best_lambda.csv", header=["ic"])
    sector_breakdown_df.to_csv(OUT_DIR / f"sector_breakdown_{safe_model}_best_lambda.csv", index=False)

    if isinstance(W, pd.DataFrame):
        W.to_csv(OUT_DIR / f"adjacency_matrix_{safe_model}_best_lambda.csv")
    else:
        W_df = pd.DataFrame(W, index=kept_cols, columns=kept_cols)
        W_df.to_csv(OUT_DIR / f"adjacency_matrix_{safe_model}_best_lambda.csv")

print("\nSaved outputs to:", OUT_DIR.resolve())