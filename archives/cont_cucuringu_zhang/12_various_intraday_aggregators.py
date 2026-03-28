import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# config
# ============================================================
INPUT_PATH = Path("data/processed/feature_table_with_residuals.csv")
OUT_DIR = Path("data/processed/intraday_aggregation_tests")

TRAIN_FRAC = 0.80
MIN_TICKER_OBS = 252
Q = 0.10
MIN_SECTOR_ASSETS = 2
MIN_NAMES_PER_SECTOR_NEUTRAL = 4

# use your current best sector-block lambda as the default starting point
RIDGE_LAMBDA = 250.0

AGGREGATIONS = [
    "full_day_sum",
    "open_mid_close_3",
    "five_equal_buckets",
    "ten_equal_buckets",
    "pca_k3",
    "pca_k5",
]


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


def _make_feature_df_from_array(
    base_df: pd.DataFrame,
    feature_array: np.ndarray,
    feature_prefix: str = "feat",
) -> pd.DataFrame:
    out = base_df[["ticker", "date", "residual_ret", "sector"]].copy()

    n_features = feature_array.shape[1]
    feat_cols = [f"{feature_prefix}_{j}" for j in range(n_features)]
    feat_df = pd.DataFrame(feature_array, columns=feat_cols, index=base_df.index)

    out = pd.concat([out, feat_df], axis=1)
    return out


# ============================================================
# intraday aggregations
# ============================================================
def aggregate_full_day_sum(df: pd.DataFrame, minute_cols: list[str]) -> pd.DataFrame:
    X = df[minute_cols].to_numpy(dtype=float)
    feats = X.sum(axis=1, keepdims=True)
    return _make_feature_df_from_array(df, feats, "feat")


def aggregate_bucket_sums(
    df: pd.DataFrame,
    minute_cols: list[str],
    bucket_edges: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    bucket_edges are 1-based inclusive minute ranges
    e.g. [(1, 60), (61, 240), (241, 390)]
    """
    X = df[minute_cols].to_numpy(dtype=float)

    feats = []
    for start_min, end_min in bucket_edges:
        s = start_min - 1
        e = end_min
        feats.append(X[:, s:e].sum(axis=1))

    feats = np.column_stack(feats)
    return _make_feature_df_from_array(df, feats, "feat")


def make_equal_buckets(n_minutes: int, n_buckets: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_minutes, n_buckets + 1, dtype=int)
    out = []
    for i in range(n_buckets):
        start0 = edges[i]
        end0 = edges[i + 1]
        out.append((start0 + 1, end0))
    return out


def aggregate_pca_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    minute_cols: list[str],
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, PCA, StandardScaler]:
    X_train = df_train[minute_cols].to_numpy(dtype=float)
    X_test = df_test[minute_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(X_train_scaled)
    Z_test = pca.transform(X_test_scaled)

    comps = pca.components_.copy()
    raw_sum_train = X_train.sum(axis=1)

    for j in range(n_components):
        corr = np.corrcoef(Z_train[:, j], raw_sum_train)[0, 1]
        if np.isfinite(corr) and corr < 0:
            Z_train[:, j] *= -1
            Z_test[:, j] *= -1
            comps[j, :] *= -1

    df_train_out = _make_feature_df_from_array(df_train, Z_train, "feat")
    df_test_out = _make_feature_df_from_array(df_test, Z_test, "feat")

    return df_train_out, df_test_out, comps, pca, scaler


def build_intraday_features_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    minute_cols: list[str],
    aggregation_name: str,
):
    n_minutes = len(minute_cols)

    if aggregation_name == "full_day_sum":
        train_out = aggregate_full_day_sum(df_train, minute_cols)
        test_out = aggregate_full_day_sum(df_test, minute_cols)
        meta = {"aggregation": aggregation_name}

    elif aggregation_name == "open_mid_close_3":
        bucket_edges = [(1, 60), (61, 240), (241, n_minutes)]
        train_out = aggregate_bucket_sums(df_train, minute_cols, bucket_edges)
        test_out = aggregate_bucket_sums(df_test, minute_cols, bucket_edges)
        meta = {"aggregation": aggregation_name, "bucket_edges": bucket_edges}

    elif aggregation_name == "five_equal_buckets":
        bucket_edges = make_equal_buckets(n_minutes, 5)
        train_out = aggregate_bucket_sums(df_train, minute_cols, bucket_edges)
        test_out = aggregate_bucket_sums(df_test, minute_cols, bucket_edges)
        meta = {"aggregation": aggregation_name, "bucket_edges": bucket_edges}

    elif aggregation_name == "ten_equal_buckets":
        bucket_edges = make_equal_buckets(n_minutes, 10)
        train_out = aggregate_bucket_sums(df_train, minute_cols, bucket_edges)
        test_out = aggregate_bucket_sums(df_test, minute_cols, bucket_edges)
        meta = {"aggregation": aggregation_name, "bucket_edges": bucket_edges}

    elif aggregation_name == "pca_k3":
        train_out, test_out, comps, pca, scaler = aggregate_pca_train_test(
            df_train, df_test, minute_cols, n_components=3
        )
        meta = {
            "aggregation": aggregation_name,
            "pca_components": comps,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

    elif aggregation_name == "pca_k5":
        train_out, test_out, comps, pca, scaler = aggregate_pca_train_test(
            df_train, df_test, minute_cols, n_components=5
        )
        meta = {
            "aggregation": aggregation_name,
            "pca_components": comps,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

    else:
        raise ValueError(f"Unknown aggregation_name: {aggregation_name}")

    return train_out, test_out, meta


# ============================================================
# wide matrices
# ============================================================
def get_feature_cols(feature_df: pd.DataFrame) -> list[str]:
    return [c for c in feature_df.columns if c.startswith("feat_")]


def pivot_features_wide(feature_df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = get_feature_cols(feature_df)

    pieces = []
    for feat_col in feat_cols:
        tmp = feature_df.pivot(index="date", columns="ticker", values=feat_col)
        tmp.columns = [f"{ticker}__{feat_col}" for ticker in tmp.columns]
        pieces.append(tmp)

    out = pd.concat(pieces, axis=1).sort_index().sort_index(axis=1)
    return out


def build_return_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_df.pivot(index="date", columns="ticker", values="residual_ret")
        .sort_index()
        .sort_index(axis=1)
    )


def build_featurecol_to_sector(feature_cols: list[str], ticker_to_sector: pd.Series) -> pd.Series:
    mapping = {}
    for col in feature_cols:
        ticker = col.split("__")[0]
        mapping[col] = ticker_to_sector.get(ticker, np.nan)
    return pd.Series(mapping, name="sector")


# ============================================================
# model: sector-block multifeature
# ============================================================
def fit_sector_block_multifeature_model(
    X_train_z_df: pd.DataFrame,
    Y_train_z_df: pd.DataFrame,
    featurecol_to_sector: pd.Series,
    ticker_to_sector: pd.Series,
    lam: float,
    min_sector_assets: int = 2,
) -> pd.DataFrame:
    """
    W has shape:
      (# feature columns) x (# target tickers)

    target ticker in sector g is predicted only from feature cols in sector g
    """
    W_full = pd.DataFrame(
        0.0,
        index=X_train_z_df.columns,
        columns=Y_train_z_df.columns,
        dtype=float,
    )

    sector_to_targets = (
        ticker_to_sector.loc[Y_train_z_df.columns]
        .reset_index()
        .groupby("sector")["ticker"]
        .apply(list)
        .to_dict()
    )

    featurecol_sector_df = featurecol_to_sector.loc[X_train_z_df.columns].rename("sector").reset_index()
    sector_to_features = (
        featurecol_sector_df
        .groupby("sector")["index"]
        .apply(list)
        .to_dict()
    )

    for sector, target_tickers in sorted(sector_to_targets.items()):
        feature_cols = sector_to_features.get(sector, [])

        target_tickers = [t for t in target_tickers if t in Y_train_z_df.columns]
        feature_cols = [c for c in feature_cols if c in X_train_z_df.columns]

        if len(target_tickers) < min_sector_assets or len(feature_cols) == 0:
            continue

        Xg = X_train_z_df[feature_cols].to_numpy(dtype=float)
        Yg = Y_train_z_df[target_tickers].to_numpy(dtype=float)

        Wg = fit_ridge(Xg, Yg, lam)
        W_full.loc[feature_cols, target_tickers] = Wg

    return W_full


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

    if not rows:
        return pd.DataFrame(columns=[
            "sector", "n_assets", "mean_daily_spread", "std_daily_spread",
            "daily_spread_sharpe", "annualized_spread_sharpe",
            "mean_daily_ic", "std_daily_ic", "n_days"
        ])

    return pd.DataFrame(rows).sort_values(
        ["annualized_spread_sharpe", "sector"],
        ascending=[False, True]
    )


# ============================================================
# one full run for one aggregation
# ============================================================
def run_one_aggregation_sector_block(
    df: pd.DataFrame,
    aggregation_name: str,
    lam: float = 250.0,
    train_frac: float = 0.80,
    min_ticker_obs: int = 252,
    q: float = 0.10,
    min_sector_assets: int = 2,
    min_names_per_sector_neutral: int = 4,
):
    minute_cols = get_minute_cols(df)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
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
    split_idx = int(len(all_dates) * train_frac)

    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]

    if len(train_dates) < 2 or len(test_dates) < 2:
        raise ValueError(f"{aggregation_name}: not enough dates for split")

    df_train = df[df["date"].isin(train_dates)].copy()
    df_test = df[df["date"].isin(test_dates)].copy()

    # build aggregated features
    feat_train, feat_test, meta = build_intraday_features_train_test(
        df_train, df_test, minute_cols, aggregation_name
    )

    feat_all = (
        pd.concat([feat_train, feat_test], axis=0)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    X_mat = pivot_features_wide(feat_all)
    Y_mat = build_return_matrix(feat_all)

    common_dates = X_mat.index.intersection(Y_mat.index)
    X_mat = X_mat.loc[common_dates]
    Y_mat = Y_mat.loc[common_dates]

    # coverage filter at ticker level
    ticker_obs = Y_mat.notna().sum(axis=0)
    keep_tickers = ticker_obs[ticker_obs >= min_ticker_obs].index

    Y_mat = Y_mat[keep_tickers]
    ticker_to_sector = ticker_to_sector.loc[keep_tickers]

    keep_feature_cols = [c for c in X_mat.columns if c.split("__")[0] in keep_tickers]
    X_mat = X_mat[keep_feature_cols]

    if len(keep_tickers) == 0 or X_mat.shape[1] == 0:
        raise ValueError(f"{aggregation_name}: no tickers/features remain after filtering")

    featurecol_to_sector = build_featurecol_to_sector(X_mat.columns.tolist(), ticker_to_sector)

    # split after pivot
    train_dates_idx = X_mat.index.intersection(pd.Index(train_dates))
    test_dates_idx = X_mat.index.intersection(pd.Index(test_dates))

    X_train = X_mat.loc[train_dates_idx].copy()
    Y_train = Y_mat.loc[train_dates_idx].copy()
    X_test = X_mat.loc[test_dates_idx].copy()
    Y_test = Y_mat.loc[test_dates_idx].copy()

    if len(X_train) < 2 or len(X_test) < 2:
        raise ValueError(f"{aggregation_name}: not enough train/test dates after alignment")

    # lag
    X_train_df = X_train.iloc[:-1].copy()
    Y_train_df = Y_train.iloc[1:].copy()
    X_train_df.index = Y_train_df.index

    X_test_df = X_test.iloc[:-1].copy()
    Y_test_df = Y_test.iloc[1:].copy()
    X_test_df.index = Y_test_df.index

    # z-score
    X_train_z_df, X_test_z_df, _, _ = zscore_with_train_stats(X_train_df, X_test_df)
    Y_train_z_df, Y_test_z_df, y_mean, y_std = zscore_with_train_stats(Y_train_df, Y_test_df)

    X_train_z_df = X_train_z_df.fillna(0.0)
    X_test_z_df = X_test_z_df.fillna(0.0)
    Y_train_z_df = Y_train_z_df.fillna(0.0)
    Y_test_z_df = Y_test_z_df.fillna(0.0)

    # keep target columns that are usable in train/test
    valid_y_cols = (
        np.isfinite(Y_train_z_df.to_numpy()).any(axis=0) &
        np.isfinite(Y_test_z_df.to_numpy()).any(axis=0)
    )
    kept_y_cols = Y_train_df.columns[valid_y_cols]

    Y_train_df = Y_train_df[kept_y_cols]
    Y_test_df = Y_test_df[kept_y_cols]
    Y_train_z_df = Y_train_z_df[kept_y_cols]
    Y_test_z_df = Y_test_z_df[kept_y_cols]
    y_mean = y_mean[kept_y_cols]
    y_std = y_std[kept_y_cols]
    ticker_to_sector = ticker_to_sector.loc[kept_y_cols]

    # keep feature columns whose ticker survived
    keep_x_cols = [c for c in X_train_z_df.columns if c.split("__")[0] in kept_y_cols]
    X_train_z_df = X_train_z_df[keep_x_cols]
    X_test_z_df = X_test_z_df[keep_x_cols]
    featurecol_to_sector = featurecol_to_sector.loc[keep_x_cols]

    if X_train_z_df.shape[1] == 0 or Y_train_z_df.shape[1] == 0:
        raise ValueError(f"{aggregation_name}: no usable X/Y columns remain after filtering")

    # fit sector-block multifeature model
    W = fit_sector_block_multifeature_model(
        X_train_z_df=X_train_z_df,
        Y_train_z_df=Y_train_z_df,
        featurecol_to_sector=featurecol_to_sector,
        ticker_to_sector=ticker_to_sector,
        lam=lam,
        min_sector_assets=min_sector_assets,
    )

    # predict
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

    # metrics
    daily_spreads = pred_test_df.apply(
        lambda row: daily_spread(row, Y_test_df.loc[row.name], q=q),
        axis=1
    ).dropna()

    daily_spreads_sector_neutral = pred_test_df.apply(
        lambda row: daily_sector_neutral_spread(
            row,
            Y_test_df.loc[row.name],
            ticker_to_sector=ticker_to_sector.loc[pred_test_df.columns],
            q=q,
            min_names_per_sector=min_names_per_sector_neutral,
        ),
        axis=1
    ).dropna()

    daily_ic = pred_test_df.apply(
        lambda row: row.corr(Y_test_df.loc[row.name], method="spearman"),
        axis=1
    ).dropna()

    summary = {
        "aggregation": aggregation_name,
        "lambda": lam,
        "n_x_features": X_train_z_df.shape[1],
        "n_y_assets": Y_train_z_df.shape[1],
        "n_train_days": X_train_z_df.shape[0],
        "n_test_days": X_test_z_df.shape[0],
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
    }

    sector_breakdown_df = sector_sharpe_breakdown(
        pred_test_df=pred_test_df,
        ret_test_df=Y_test_df[pred_test_df.columns],
        ticker_to_sector=ticker_to_sector,
        q=q,
    )

    return summary, pred_test_df, W, meta, sector_breakdown_df


# ============================================================
# main
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    all_results = []
    all_meta_rows = []

    for agg in AGGREGATIONS:
        print(f"\n=== Running aggregation: {agg} ===")

        try:
            summary, pred_df, W, meta, sector_breakdown_df = run_one_aggregation_sector_block(
                df=df,
                aggregation_name=agg,
                lam=RIDGE_LAMBDA,
                train_frac=TRAIN_FRAC,
                min_ticker_obs=MIN_TICKER_OBS,
                q=Q,
                min_sector_assets=MIN_SECTOR_ASSETS,
                min_names_per_sector_neutral=MIN_NAMES_PER_SECTOR_NEUTRAL,
            )

            all_results.append(summary)

            meta_row = {"aggregation": agg}
            if "explained_variance_ratio" in meta:
                evr = meta["explained_variance_ratio"]
                for i, v in enumerate(evr):
                    meta_row[f"explained_variance_ratio_{i}"] = float(v)
            all_meta_rows.append(meta_row)

            safe_agg = agg.replace("/", "_").replace("\\", "_").replace(" ", "_")

            pred_df.to_csv(OUT_DIR / f"predicted_returns_{safe_agg}.csv")
            W.to_csv(OUT_DIR / f"adjacency_matrix_{safe_agg}.csv")
            sector_breakdown_df.to_csv(OUT_DIR / f"sector_breakdown_{safe_agg}.csv", index=False)

            print(
                f"{agg:>20} | "
                f"ann_sharpe={summary['annualized_spread_sharpe']:.6f} | "
                f"ann_sharpe_sector_neutral={summary['annualized_spread_sharpe_sector_neutral']:.6f} | "
                f"mean_ic={summary['mean_daily_ic']:.6f}"
            )

        except Exception as e:
            print(f"{agg} FAILED: {e}")
            all_results.append({
                "aggregation": agg,
                "lambda": RIDGE_LAMBDA,
                "n_x_features": np.nan,
                "n_y_assets": np.nan,
                "n_train_days": np.nan,
                "n_test_days": np.nan,
                "mean_daily_spread": np.nan,
                "std_daily_spread": np.nan,
                "daily_spread_sharpe": np.nan,
                "annualized_spread_sharpe": np.nan,
                "mean_daily_spread_sector_neutral": np.nan,
                "std_daily_spread_sector_neutral": np.nan,
                "daily_spread_sharpe_sector_neutral": np.nan,
                "annualized_spread_sharpe_sector_neutral": np.nan,
                "mean_daily_ic": np.nan,
                "std_daily_ic": np.nan,
                "error": str(e),
            })

    results_df = pd.DataFrame(all_results).sort_values(
        ["annualized_spread_sharpe_sector_neutral", "annualized_spread_sharpe", "mean_daily_ic"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    meta_df = pd.DataFrame(all_meta_rows)

    results_df.to_csv(OUT_DIR / "aggregation_comparison_results.csv", index=False)
    meta_df.to_csv(OUT_DIR / "aggregation_metadata.csv", index=False)

    print("\n=== Final aggregation comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()