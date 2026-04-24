#!/usr/bin/env python3
"""15_nonlinear_nextday.py

Non-linear models for next-day residual return prediction.

Feature representation (per ticker-day observation):
  own_ofi_hl{k}        : EMA-aggregated OFI for this ticker at half-life k
  sector_mean_hl{k}    : cross-sectional mean of own_ofi within GICS sector
  = 18 features total (9 own + 9 sector-mean), or 9 for "own-only" models

Because features are fixed-width regardless of sector size, a single model is
trained across all tickers (panel regression), capturing cross-asset OFI effects
without the N×N adjacency matrix of scripts 06–10.

Models
------
  ridge_own      : Ridge, own OFI only (linear, 9-feature baseline)
  ridge_ctx      : Ridge, own + sector-mean OFI (linear, 18 features)
  ridge_ctx_rn   : Ridge, rank-normalised features
  rf_ctx_rn      : Random Forest, rank-normalised ctx features
  hgb_ctx_rn     : HistGradientBoosting, rank-normalised ctx features

Walk-forward: 750-day train window, refit every 21 days.
Output: results/nonlinear/

Run from repo root:
    python pipeline/15_nonlinear_nextday.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    build_exp_decay_kernel,
    evaluate_predictions,
    sharpe_from_series,
    annualize_sharpe,
)

# ── config ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("data/processed/feature_table_with_residuals_10level.csv")
OUT_DIR     = Path("results/nonlinear")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS            = 6
HALF_LIFE_SWEEP             = [15, 20, 25, 30, 35, 40, 45, 60, 90]
Q                           = 0.10
MIN_TICKER_OBS              = 252
MIN_SECTOR_ASSETS           = 4
MIN_NAMES_PER_SECTOR_NEUTRAL = 4
INITIAL_TRAIN_DAYS          = 750
REFIT_EVERY_DAYS            = 21
MAX_ASSETS                  = 400
RIDGE_ALPHAS                = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]

MODEL_SPECS = [
    # ── linear baselines ───────────────────────────────────────────────────────
    {"name": "ridge_own",    "features": "own", "model_cls": "ridge", "rank_norm": False},
    {"name": "ridge_ctx",    "features": "ctx", "model_cls": "ridge", "rank_norm": False},
    {"name": "ridge_ctx_rn", "features": "ctx", "model_cls": "ridge", "rank_norm": True},
    # ── tree models ────────────────────────────────────────────────────────────
    {"name": "rf_ctx_rn",    "features": "ctx", "model_cls": "rf",    "rank_norm": True},
    {"name": "hgb_ctx_rn",   "features": "ctx", "model_cls": "hgb",   "rank_norm": True},
]


def make_model(model_cls: str):
    if model_cls == "ridge":
        return RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)
    if model_cls == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features=0.5, n_jobs=-1, random_state=42,
        )
    if model_cls == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=50,
            learning_rate=0.05, random_state=42,
        )
    raise ValueError(f"Unknown model_cls: {model_cls!r}")


# ── load and preprocess ────────────────────────────────────────────────────────
print("Reading data...", flush=True)
header_df   = pd.read_csv(INPUT_PATH, nrows=0)
minute_cols = [c for c in header_df.columns if c.startswith("minute_")]
usecols     = ["date", "ticker", "sector", "residual_ret"] + minute_cols

df = pd.read_csv(INPUT_PATH, usecols=usecols)
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
ticker_to_sector_full = (
    df[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]
)

# ── build EMA OFI signals ──────────────────────────────────────────────────────
n_minutes = len(minute_cols)
minute_np = df[minute_cols].to_numpy(dtype=float)
print("Building exp-decay OFI signals:", flush=True)
ofi_dict: dict[str, np.ndarray] = {}
for hl in HALF_LIFE_SWEEP:
    kernel = build_exp_decay_kernel(n_minutes=n_minutes, half_life=hl)
    ofi_dict[f"ofi_hl{hl}"] = minute_np @ kernel
    print(f"  hl={hl:3d}min  weight_ratio={kernel[-1]/kernel[0]:.1f}x", flush=True)
del minute_np
df = pd.concat([df, pd.DataFrame(ofi_dict, index=df.index)], axis=1)

# ── sector-mean OFI ────────────────────────────────────────────────────────────
print("Building sector-mean OFI signals...", flush=True)
smean_dict: dict[str, pd.Series] = {}
for hl in HALF_LIFE_SWEEP:
    smean_dict[f"sector_mean_hl{hl}"] = (
        df.groupby(["date", "sector"])[f"ofi_hl{hl}"].transform("mean")
    )
df = pd.concat([df, pd.DataFrame(smean_dict, index=df.index)], axis=1)

# ── rank-normalised versions (cross-sectional percentile per day) ──────────────
print("Building rank-normalised signals...", flush=True)
rn_dict: dict[str, pd.Series] = {}
for hl in HALF_LIFE_SWEEP:
    rn_dict[f"ofi_hl{hl}_rn"]         = df.groupby("date")[f"ofi_hl{hl}"].rank(pct=True)
    rn_dict[f"sector_mean_hl{hl}_rn"] = df.groupby("date")[f"sector_mean_hl{hl}"].rank(pct=True)
df = pd.concat([df, pd.DataFrame(rn_dict, index=df.index)], axis=1)

# ── global ticker filter (coverage + cap) ─────────────────────────────────────
print("Filtering tickers...", flush=True)
sig_proxy  = df.pivot(index="date", columns="ticker", values="ofi_hl45")
ret_proxy  = df.pivot(index="date", columns="ticker", values="residual_ret")
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

all_dates = np.sort(df["date"].unique())
if len(all_dates) < INITIAL_TRAIN_DAYS + 2:
    raise ValueError(f"Not enough dates ({len(all_dates)}) for walk-forward.")

# ── walk-forward ───────────────────────────────────────────────────────────────
test_dates   = pd.Index(all_dates[INITIAL_TRAIN_DAYS:])
refit_points = list(range(0, len(test_dates), REFIT_EVERY_DAYS))

all_summaries: list[pd.DataFrame] = []


def build_panel_Xy(
    df_window: pd.DataFrame,
    dates_for_features: np.ndarray,
    dates_for_targets:  np.ndarray,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack (ticker, date) observations into X, y arrays.

    Features at dates_for_features[i], targets at dates_for_targets[i].
    """
    date_to_next = dict(zip(dates_for_features, dates_for_targets))
    df_feat = (
        df_window[df_window["date"].isin(dates_for_features)][["date", "ticker"] + feat_cols]
        .copy()
    )
    df_feat["ret_date"] = df_feat["date"].map(date_to_next)

    df_ret = (
        df_window[df_window["date"].isin(dates_for_targets)][["date", "ticker", "residual_ret"]]
        .rename(columns={"date": "ret_date"})
    )

    merged = df_feat.merge(df_ret, on=["ret_date", "ticker"], how="inner").dropna(
        subset=feat_cols + ["residual_ret"]
    )

    X = merged[feat_cols].to_numpy(dtype=float)
    y = merged["residual_ret"].to_numpy(dtype=float)
    return X, y


for spec in MODEL_SPECS:
    model_name = spec["name"]
    model_cls  = spec["model_cls"]
    features   = spec["features"]
    rank_norm  = spec["rank_norm"]

    suffix   = "_rn" if rank_norm else ""
    own_cols = [f"ofi_hl{hl}{suffix}" for hl in HALF_LIFE_SWEEP]
    ctx_cols = own_cols + [f"sector_mean_hl{hl}{suffix}" for hl in HALF_LIFE_SWEEP]
    feat_cols = own_cols if features == "own" else ctx_cols

    print(f"\n{'='*70}", flush=True)
    print(
        f"RUNNING: {model_name}  "
        f"({len(feat_cols)} features, rank_norm={rank_norm}, cls={model_cls})",
        flush=True,
    )

    pred_chunks: list[pd.DataFrame] = []
    t0 = time.time()

    for block_num, start_offset in enumerate(refit_points, start=1):
        block_test_dates = test_dates[start_offset : start_offset + REFIT_EVERY_DAYS]
        if len(block_test_dates) == 0:
            continue

        first_pred         = block_test_dates[0]
        train_dates_window = all_dates[all_dates < first_pred][-INITIAL_TRAIN_DAYS:]
        if len(train_dates_window) < INITIAL_TRAIN_DAYS:
            continue

        # ── window data ──────────────────────────────────────────────────────
        window_dates = np.concatenate([train_dates_window, block_test_dates])
        df_win = df[df["date"].isin(window_dates)].copy()

        # Drop sectors that are too small in the training window
        sector_sizes = (
            df_win[df_win["date"].isin(train_dates_window)]
            .groupby(["date", "sector"])["ticker"]
            .nunique()
        )
        ok_sectors = sector_sizes.groupby("sector").min()
        ok_sectors = ok_sectors[ok_sectors >= MIN_SECTOR_ASSETS].index.tolist()
        df_win = df_win[df_win["sector"].isin(ok_sectors)].copy()

        if df_win.empty:
            continue

        # ── build training set ────────────────────────────────────────────────
        # features at d, targets at next trading day d+1 (within window)
        tr_sorted = np.sort(np.intersect1d(train_dates_window, df_win["date"].unique()))
        if len(tr_sorted) < 2:
            continue
        feat_tr_dates = tr_sorted[:-1]
        ret_tr_dates  = tr_sorted[1:]

        df_tr = df_win[df_win["date"].isin(tr_sorted)]
        X_train, y_train = build_panel_Xy(df_tr, feat_tr_dates, ret_tr_dates, feat_cols)

        if len(X_train) < 100:
            continue

        # ── fit ───────────────────────────────────────────────────────────────
        model = make_model(model_cls)
        if model_cls == "ridge":
            scaler    = StandardScaler()
            X_fit     = scaler.fit_transform(X_train)
        else:
            scaler    = None
            X_fit     = X_train

        model.fit(X_fit, y_train)

        # ── predict each test date ────────────────────────────────────────────
        pred_rows: dict = {}
        df_test = df_win[df_win["date"].isin(block_test_dates)]

        for test_date, grp in df_test.groupby("date"):
            grp_clean = grp[["ticker"] + feat_cols].dropna(subset=feat_cols)
            if grp_clean.empty:
                continue
            X_test = grp_clean[feat_cols].to_numpy(dtype=float)
            if scaler is not None:
                X_test = scaler.transform(X_test)
            preds = model.predict(X_test)
            pred_rows[test_date] = pd.Series(preds, index=grp_clean["ticker"].values)

        if not pred_rows:
            continue

        pred_block = pd.DataFrame(pred_rows).T
        pred_block.index = pd.DatetimeIndex(pred_block.index)
        pred_chunks.append(pred_block)

        alpha_tag = f"  alpha={model.alpha_:.3g}" if model_cls == "ridge" else ""
        print(
            f"  block {block_num:3d}/{len(refit_points)}"
            f"  train={len(X_train):,}"
            f"  test_days={len(block_test_dates)}"
            f"{alpha_tag}",
            flush=True,
        )

    if not pred_chunks:
        print(f"[{model_name}] No predictions produced.", flush=True)
        continue

    pred_all = pd.concat(pred_chunks, axis=0).sort_index()
    pred_all = pred_all[~pred_all.index.duplicated(keep="last")]

    # ── evaluate ──────────────────────────────────────────────────────────────
    ret_full = df.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
    ce_d = pred_all.index.intersection(ret_full.index)
    ce_c = pred_all.columns.intersection(ret_full.columns)
    pred_aligned = pred_all.loc[ce_d, ce_c]
    ret_aligned  = ret_full.loc[ce_d, ce_c]
    ts_eval = ticker_to_sector.reindex(ce_c).dropna()
    pred_aligned = pred_aligned[ts_eval.index]
    ret_aligned  = ret_aligned[ts_eval.index]

    gross, sn, ic = evaluate_predictions(
        pred_aligned, ret_aligned, ts_eval, Q, MIN_NAMES_PER_SECTOR_NEUTRAL
    )

    summary = pd.DataFrame([{
        "model":                        model_name,
        "model_cls":                    model_cls,
        "features":                     features,
        "rank_norm":                    rank_norm,
        "n_features":                   len(feat_cols),
        "rolling_train_days":           INITIAL_TRAIN_DAYS,
        "refit_every_days":             REFIT_EVERY_DAYS,
        "mean_daily_spread":            gross.mean(),
        "annualized_spread_sharpe":     annualize_sharpe(sharpe_from_series(gross)),
        "mean_daily_spread_sn":         sn.mean(),
        "annualized_spread_sharpe_sn":  annualize_sharpe(sharpe_from_series(sn)),
        "mean_daily_ic":                ic.mean(),
        "n_eval_days":                  len(pred_aligned),
        "n_refits":                     len(pred_chunks),
        "elapsed_sec":                  round(time.time() - t0, 1),
    }])

    all_summaries.append(summary)
    summary.to_csv(OUT_DIR / f"rolling_summary_{model_name}.csv", index=False)
    pred_aligned.to_csv(OUT_DIR / f"predicted_returns_{model_name}_rolling.csv")

    print(f"\n[{model_name}] Rolling summary:", flush=True)
    show_cols = ["model", "annualized_spread_sharpe_sn", "mean_daily_ic", "elapsed_sec"]
    print(summary[show_cols].to_string(index=False), flush=True)

# ── combined output ────────────────────────────────────────────────────────────
if all_summaries:
    combined = (
        pd.concat(all_summaries, ignore_index=True)
        .sort_values("annualized_spread_sharpe_sn", ascending=False)
    )
    combined.to_csv(OUT_DIR / "rolling_summary_all_models.csv", index=False)
    print("\n" + "=" * 70, flush=True)
    print("NON-LINEAR MODELS — COMBINED SUMMARY", flush=True)
    print("=" * 70, flush=True)
    show = ["model", "annualized_spread_sharpe_sn", "annualized_spread_sharpe",
            "mean_daily_ic", "elapsed_sec"]
    print(combined[[c for c in show if c in combined.columns]].to_string(index=False), flush=True)

print(f"\nOutputs saved to: {OUT_DIR.resolve()}", flush=True)
