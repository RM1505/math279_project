import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# config
# ============================================================
INPUT_PATH = Path("data/processed/feature_table_with_residuals.csv")
OUT_DIR = Path("data/processed/full_day_sum_sector_models")

TRAIN_FRAC = 0.80
MIN_TICKER_OBS = 252
Q = 0.10
MIN_SECTOR_ASSETS = 2
MIN_NAMES_PER_SECTOR_NEUTRAL = 4

LAMBDA_GRID = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]


# ============================================================
# helpers
# ============================================================
def get_minute_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("minute_")],
        key=lambda x: int(x.split("_")[1])
    )


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

    for _, g in tmp.groupby("sector"):
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


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X
    XtY = X.T @ Y
    return np.linalg.solve(XtX + lam * np.eye(X.shape[1]), XtY)


# ============================================================
# feature construction: full-day sum only
# ============================================================
def make_full_day_sum_feature_df(df: pd.DataFrame, minute_cols: list[str]) -> pd.DataFrame:
    out = df[["ticker", "date", "residual_ret", "sector"]].copy()
    out["signal"] = df[minute_cols].sum(axis=1)
    return out


def build_signal_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_df.pivot(index="date", columns="ticker", values="signal")
        .sort_index()
        .sort_index(axis=1)
    )


def build_return_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_df.pivot(index="date", columns="ticker", values="residual_ret")
        .sort_index()
        .sort_index(axis=1)
    )


# ============================================================
# model fitting
# ============================================================
def fit_dense_model(X_train_df: pd.DataFrame, Y_train_df: pd.DataFrame, lam: float) -> pd.DataFrame:
    X = X_train_df.to_numpy(dtype=float)
    Y = Y_train_df.to_numpy(dtype=float)
    W = fit_ridge(X, Y, lam)
    return pd.DataFrame(W, index=X_train_df.columns, columns=Y_train_df.columns)


def fit_diagonal_model(X_train_df: pd.DataFrame, Y_train_df: pd.DataFrame, lam: float) -> pd.DataFrame:
    tickers = X_train_df.columns.tolist()
    W = pd.DataFrame(0.0, index=tickers, columns=tickers, dtype=float)

    for t in tickers:
        x = X_train_df[t].to_numpy(dtype=float)
        y = Y_train_df[t].to_numpy(dtype=float)

        denom = np.dot(x, x) + lam
        beta = 0.0 if denom == 0 else np.dot(x, y) / denom
        W.loc[t, t] = beta

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
        ticker_to_sector.loc[tickers]
        .reset_index()
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    for sector, sector_tickers in sorted(sector_to_tickers.items()):
        sector_tickers = [t for t in sector_tickers if t in X_train_df.columns and t in Y_train_df.columns]

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


# ============================================================
# evaluation
# ============================================================
def evaluate_model(
    model_name: str,
    W: pd.DataFrame,
    X_test_z_df: pd.DataFrame,
    Y_test_df: pd.DataFrame,
    y_mean: pd.Series,
    y_std: pd.Series,
    ticker_to_sector: pd.Series,
    q: float,
    min_names_per_sector_neutral: int,
):
    W = W.loc[X_test_z_df.columns, Y_test_df.columns]

    pred_test_z = X_test_z_df.to_numpy(dtype=float) @ W.to_numpy(dtype=float)
    pred_test = (
        pred_test_z * y_std.loc[W.columns].to_numpy(dtype=float)
        + y_mean.loc[W.columns].to_numpy(dtype=float)
    )

    pred_test_df = pd.DataFrame(
        pred_test,
        index=X_test_z_df.index,
        columns=W.columns,
    )

    ret_test_eval_df = Y_test_df[W.columns].copy()
    sector_map_eval = ticker_to_sector.loc[W.columns]

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

    metrics = {
        "model": model_name,
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

        daily_spreads_sector_neutral = pred_sector.apply(
            lambda row: daily_sector_neutral_spread(
                row,
                ret_sector.loc[row.name],
                ticker_to_sector=ticker_to_sector.loc[tickers],
                q=q,
                min_names_per_sector=MIN_NAMES_PER_SECTOR_NEUTRAL,
            ),
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
            "mean_daily_spread_sector_neutral": daily_spreads_sector_neutral.mean(),
            "std_daily_spread_sector_neutral": daily_spreads_sector_neutral.std(ddof=1),
            "daily_spread_sharpe_sector_neutral": sharpe_from_series(daily_spreads_sector_neutral),
            "annualized_spread_sharpe_sector_neutral": annualize_sharpe(sharpe_from_series(daily_spreads_sector_neutral)),
            "mean_daily_ic": daily_ic_sector.mean(),
            "std_daily_ic": daily_ic_sector.std(ddof=1),
            "n_days": len(daily_spreads_sector),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["annualized_spread_sharpe_sector_neutral", "annualized_spread_sharpe", "sector"],
        ascending=[False, False, True]
    )


# ============================================================
# prepare data once
# ============================================================
def prepare_data():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    minute_cols = get_minute_cols(df)
    df[minute_cols] = df[minute_cols].fillna(0.0)

    df = df.dropna(subset=["ticker", "date", "residual_ret", "sector"]).copy()

    # enforce one sector per ticker
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
        raise ValueError("Not enough dates for train/test split.")

    df_train = df[df["date"].isin(train_dates)].copy()
    df_test = df[df["date"].isin(test_dates)].copy()

    feat_train = make_full_day_sum_feature_df(df_train, minute_cols)
    feat_test = make_full_day_sum_feature_df(df_test, minute_cols)

    feat_all = (
        pd.concat([feat_train, feat_test], axis=0)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    signal_mat = build_signal_matrix(feat_all)
    ret_mat = build_return_matrix(feat_all)

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

    X_train_z_df, X_test_z_df, _, _ = zscore_with_train_stats(X_train_df, X_test_df)
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

    return {
        "X_train_df": X_train_df,
        "Y_train_df": Y_train_df,
        "X_test_df": X_test_df,
        "Y_test_df": Y_test_df,
        "X_train_z_df": X_train_z_df,
        "Y_train_z_df": Y_train_z_df,
        "X_test_z_df": X_test_z_df,
        "Y_test_z_df": Y_test_z_df,
        "y_mean": y_mean,
        "y_std": y_std,
        "ticker_to_sector": ticker_to_sector,
        "n_train_days": X_train_z_df.shape[0],
        "n_test_days": X_test_z_df.shape[0],
        "n_assets": X_train_z_df.shape[1],
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
    }


# ============================================================
# run one model/lambda
# ============================================================
def fit_model(model_name: str, lam: float, data: dict) -> pd.DataFrame:
    X_train_z_df = data["X_train_z_df"]
    Y_train_z_df = data["Y_train_z_df"]
    ticker_to_sector = data["ticker_to_sector"]

    if model_name == "sector_block":
        return fit_sector_block_model(
            X_train_z_df, Y_train_z_df, ticker_to_sector,
            lam=lam,
            min_sector_assets=MIN_SECTOR_ASSETS,
        )

    if model_name == "sector_offdiag":
        W_sector = fit_sector_block_model(
            X_train_z_df, Y_train_z_df, ticker_to_sector,
            lam=lam,
            min_sector_assets=MIN_SECTOR_ASSETS,
        )
        return build_sector_offdiag(W_sector)

    if model_name == "diagonal_only":
        return fit_diagonal_model(X_train_z_df, Y_train_z_df, lam=lam)

    if model_name == "dense":
        return fit_dense_model(X_train_z_df, Y_train_z_df, lam=lam)

    raise ValueError(f"Unknown model_name: {model_name}")


def sweep_model(model_name: str, data: dict):
    rows = []
    results = {}

    for lam in LAMBDA_GRID:
        W = fit_model(model_name, lam, data)

        pred_test_df, daily_spreads, daily_spreads_sector_neutral, daily_ic, metrics = evaluate_model(
            model_name=model_name,
            W=W,
            X_test_z_df=data["X_test_z_df"],
            Y_test_df=data["Y_test_df"],
            y_mean=data["y_mean"],
            y_std=data["y_std"],
            ticker_to_sector=data["ticker_to_sector"],
            q=Q,
            min_names_per_sector_neutral=MIN_NAMES_PER_SECTOR_NEUTRAL,
        )

        metrics["lambda"] = lam
        rows.append(metrics)

        sector_breakdown_df = sector_sharpe_breakdown(
            pred_test_df,
            data["Y_test_df"][pred_test_df.columns],
            data["ticker_to_sector"],
            q=Q,
        )

        results[lam] = {
            "W": W,
            "pred_test_df": pred_test_df,
            "daily_spreads": daily_spreads,
            "daily_spreads_sector_neutral": daily_spreads_sector_neutral,
            "daily_ic": daily_ic,
            "metrics": metrics,
            "sector_breakdown_df": sector_breakdown_df,
        }

        print(
            f"{model_name:>15} | lambda={lam:>6} | "
            f"ann_sharpe={metrics['annualized_spread_sharpe']:.6f} | "
            f"ann_sharpe_sector_neutral={metrics['annualized_spread_sharpe_sector_neutral']:.6f} | "
            f"mean_ic={metrics['mean_daily_ic']:.6f}"
        )

    sweep_df = pd.DataFrame(rows).sort_values(
        ["annualized_spread_sharpe_sector_neutral", "annualized_spread_sharpe", "mean_daily_ic", "lambda"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    best_lambda = sweep_df.iloc[0]["lambda"]
    best_result = results[best_lambda]

    return sweep_df, best_lambda, best_result


# ============================================================
# main
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_data()

    print(
        f"Using {data['n_train_days']} train dates, {data['n_test_days']} test dates, "
        f"and {data['n_assets']} assets."
    )
    print("Train dates:", pd.Timestamp(data["train_start"]).date(), "to", pd.Timestamp(data["train_end"]).date())
    print("Test dates:", pd.Timestamp(data["test_start"]).date(), "to", pd.Timestamp(data["test_end"]).date())
    print("Lambda grid:", LAMBDA_GRID)

    model_names = ["sector_block", "sector_offdiag", "diagonal_only", "dense"]

    all_sweeps = []
    best_summary_rows = []
    best_results = {}

    for model_name in model_names:
        print(f"\n=== Sweeping {model_name} ===")
        sweep_df, best_lambda, best_result = sweep_model(model_name, data)

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

    print("\n=== Best model summary ===")
    print(best_models_df.to_string(index=False))

    lambda_sweep_df.to_csv(OUT_DIR / "lambda_sweep_results.csv", index=False)
    best_models_df.to_csv(OUT_DIR / "best_model_summary.csv", index=False)
    data["ticker_to_sector"].rename("sector").to_csv(OUT_DIR / "ticker_sector_used.csv", header=True)

    for model_name, result in best_results.items():
        safe_model = model_name.replace(" ", "_")
        W = result["W"]
        pred_test_df = result["pred_test_df"]
        daily_spreads = result["daily_spreads"]
        daily_spreads_sector_neutral = result["daily_spreads_sector_neutral"]
        daily_ic = result["daily_ic"]
        sector_breakdown_df = result["sector_breakdown_df"]

        pred_test_df.to_csv(OUT_DIR / f"predicted_returns_{safe_model}_best_lambda.csv")
        daily_spreads.to_csv(OUT_DIR / f"daily_spreads_{safe_model}_best_lambda.csv", header=["spread"])
        daily_spreads_sector_neutral.to_csv(
            OUT_DIR / f"daily_spreads_sector_neutral_{safe_model}_best_lambda.csv",
            header=["spread_sector_neutral"]
        )
        daily_ic.to_csv(OUT_DIR / f"daily_ic_{safe_model}_best_lambda.csv", header=["ic"])
        sector_breakdown_df.to_csv(OUT_DIR / f"sector_breakdown_{safe_model}_best_lambda.csv", index=False)
        W.to_csv(OUT_DIR / f"adjacency_matrix_{safe_model}_best_lambda.csv")

    print(f"\nSaved outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()