#!/usr/bin/env python3
"""16_nonlinear_sector_block.py

True non-linear analog of script 06's sector-block ridge regression.

For each target ticker i in sector s on date t, one training observation is:
  features = [ofi_hl{k}[j] for k in HLs for j in sorted(sector_tickers)]
             + [rank of ticker i in sorted sector list]   (integer, 0-indexed)
  target   = next-day residual return of ticker i

One model is trained per sector per refit block, pooling all (ticker, date)
pairs within the sector.  This exactly mirrors script 06's sector-block W but
replaces the linear map with RF or HGB.

The ticker-rank feature lets the model learn ticker-specific intercepts and
cross-asset effects, analogous to each row i of W having different weights.

Models
------
  rf_sector_hl30_rn    : RF,  single HL=30 rank-normed  (matches ridge_sb_gd_hl30_rn)
  hgb_sector_hl30_rn   : HGB, single HL=30 rank-normed
  rf_sector_allhl_rn   : RF,  all 9 HLs rank-normed     (richest feature set)
  hgb_sector_allhl_rn  : HGB, all 9 HLs rank-normed

Walk-forward: 750-day train, refit every 21 days.
Output: results/nonlinear_sector_block/

Run from repo root:
    python pipeline/16_nonlinear_sector_block.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from pipeline_utils import (
    build_exp_decay_kernel,
    evaluate_predictions,
    sharpe_from_series,
    annualize_sharpe,
)

# ── config ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("data/processed/feature_table_with_residuals_10level.csv")
OUT_DIR     = Path("results/nonlinear_sector_block")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LAST_N_YEARS             = 6
HALF_LIFE_SWEEP              = [15, 20, 25, 30, 35, 40, 45, 60, 90]
Q                            = 0.10
MIN_TICKER_OBS               = 252
MIN_SECTOR_ASSETS            = 4
MIN_NAMES_PER_SECTOR_NEUTRAL = 4
INITIAL_TRAIN_DAYS           = 750
REFIT_EVERY_DAYS             = 21
MAX_ASSETS                   = 400

MODEL_SPECS = [
    {"name": "rf_sector_hl30_rn",   "hl_list": [30],            "model_cls": "rf"},
    {"name": "hgb_sector_hl30_rn",  "hl_list": [30],            "model_cls": "hgb"},
    {"name": "hgb_sector_allhl_rn", "hl_list": HALF_LIFE_SWEEP, "model_cls": "hgb"},
]


def make_model(model_cls: str):
    if model_cls == "rf":
        return RandomForestRegressor(
            n_estimators=50, max_depth=4, min_samples_leaf=20,
            max_features=0.5, n_jobs=-1, random_state=42,
        )
    if model_cls == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
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

# ── rank-normalised per date (cross-sectional percentile) ──────────────────────
print("Building rank-normalised signals...", flush=True)
rn_dict: dict[str, pd.Series] = {}
for hl in HALF_LIFE_SWEEP:
    rn_dict[f"ofi_hl{hl}_rn"] = df.groupby("date")[f"ofi_hl{hl}"].rank(pct=True)
df = pd.concat([df, pd.DataFrame(rn_dict, index=df.index)], axis=1)

# ── global ticker filter ───────────────────────────────────────────────────────
print("Filtering tickers...", flush=True)
sig_proxy = df.pivot(index="date", columns="ticker", values="ofi_hl45")
ret_proxy = df.pivot(index="date", columns="ticker", values="residual_ret")
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


def build_sector_Xy(
    sig_mats: dict[int, pd.DataFrame],   # hl → (date × ticker) signal matrix
    ret_mat:  pd.DataFrame,               # date × ticker returns
    feat_dates: np.ndarray,               # feature dates (t)
    ret_dates:  np.ndarray,               # return dates  (t+1)
    sector_tickers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build X, y for one sector's training set.

    X shape: (N_obs, N_sector * N_hl + 1)
      Columns: flattened OFI [hl0_tick0, hl0_tick1, ..., hlK_tickN] + ticker_rank
    y shape: (N_obs,)
    """
    n_hl   = len(sig_mats)
    n_s    = len(sector_tickers)
    n_feat = n_hl * n_s + 1   # +1 for ticker rank

    feat_date_to_ret = dict(zip(feat_dates, ret_dates))

    X_rows: list[np.ndarray] = []
    y_rows: list[float]      = []

    for d_t, d_t1 in feat_date_to_ret.items():
        # OFI row for this date: (N_hl × N_sector) flattened
        ofi_parts = []
        valid = True
        for sig_mat in sig_mats.values():
            if d_t not in sig_mat.index:
                valid = False
                break
            row = sig_mat.loc[d_t, sector_tickers].fillna(0.0).to_numpy(dtype=float)
            ofi_parts.append(row)
        if not valid:
            continue

        ofi_flat = np.concatenate(ofi_parts)   # shape (N_hl * N_sector,)

        if d_t1 not in ret_mat.index:
            continue

        ret_row = ret_mat.loc[d_t1, sector_tickers]

        for rank_i, ticker_i in enumerate(sector_tickers):
            r = ret_row.get(ticker_i, np.nan)
            if np.isnan(r):
                continue
            feat = np.append(ofi_flat, rank_i)   # ticker rank as last feature
            X_rows.append(feat)
            y_rows.append(float(r))

    if not X_rows:
        return np.empty((0, n_feat)), np.empty(0)

    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float)


for spec in MODEL_SPECS:
    model_name = spec["name"]
    model_cls  = spec["model_cls"]
    hl_list    = spec["hl_list"]
    sig_cols   = {hl: f"ofi_hl{hl}_rn" for hl in hl_list}

    print(f"\n{'='*70}", flush=True)
    print(
        f"RUNNING: {model_name}  "
        f"(hl={hl_list}, cls={model_cls}, sector-block)",
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

        window_dates = np.concatenate([train_dates_window, block_test_dates])
        df_win = df[df["date"].isin(window_dates)].copy()

        tr_sorted = np.sort(
            np.intersect1d(train_dates_window, df_win["date"].unique())
        )
        if len(tr_sorted) < 2:
            continue

        feat_tr_dates = tr_sorted[:-1]
        ret_tr_dates  = tr_sorted[1:]

        # ── pre-pivot signal and return matrices for this window ──────────────
        sig_mats_win: dict[int, pd.DataFrame] = {}
        for hl, col in sig_cols.items():
            sig_mats_win[hl] = (
                df_win.pivot(index="date", columns="ticker", values=col)
                .sort_index()
            )
        ret_mat_win = (
            df_win.pivot(index="date", columns="ticker", values="residual_ret")
            .sort_index()
        )

        # ── identify valid sectors for this window ───────────────────────────
        train_tickers = df_win[df_win["date"].isin(tr_sorted)]["ticker"].unique()
        sectors_in_window = ticker_to_sector.reindex(train_tickers).dropna()

        sector_groups = sectors_in_window.groupby(sectors_in_window).apply(
            lambda g: sorted(g.index.tolist())
        )

        pred_block_rows: dict = {}   # date → {ticker: pred}

        n_sectors_fit = 0
        for sector, sector_tickers in sector_groups.items():
            if len(sector_tickers) < MIN_SECTOR_ASSETS:
                continue

            # ── training set for this sector ──────────────────────────────────
            sm_sector = {
                hl: mat[sector_tickers] for hl, mat in sig_mats_win.items()
                if all(t in mat.columns for t in sector_tickers)
            }
            if len(sm_sector) < len(hl_list):
                continue

            rm_sector = ret_mat_win.reindex(columns=sector_tickers)

            X_tr, y_tr = build_sector_Xy(
                sm_sector, rm_sector,
                feat_tr_dates, ret_tr_dates,
                sector_tickers,
            )
            if len(X_tr) < 20:
                continue

            model = make_model(model_cls)
            model.fit(X_tr, y_tr)
            n_sectors_fit += 1

            # ── predict each test date ────────────────────────────────────────
            for d_test in block_test_dates:
                if d_test not in sig_mats_win[hl_list[0]].index:
                    continue

                ofi_parts = []
                for hl in hl_list:
                    row = (
                        sig_mats_win[hl]
                        .loc[d_test, sector_tickers]
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    ofi_parts.append(row)
                ofi_flat = np.concatenate(ofi_parts)

                for rank_i, ticker_i in enumerate(sector_tickers):
                    feat = np.append(ofi_flat, rank_i).reshape(1, -1)
                    pred = float(model.predict(feat)[0])
                    pred_block_rows.setdefault(d_test, {})[ticker_i] = pred

        if not pred_block_rows:
            continue

        pred_block = pd.DataFrame(pred_block_rows).T
        pred_block.index = pd.DatetimeIndex(pred_block.index)
        pred_chunks.append(pred_block)

        elapsed = time.time() - t0
        print(
            f"  block {block_num:3d}/{len(refit_points)}"
            f"  sectors_fit={n_sectors_fit}"
            f"  test_days={len(block_test_dates)}"
            f"  elapsed={elapsed:.0f}s",
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
        "hl_list":                      str(hl_list),
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
    show = ["model", "annualized_spread_sharpe_sn", "mean_daily_ic", "elapsed_sec"]
    print(summary[show].to_string(index=False), flush=True)

# ── combined output ────────────────────────────────────────────────────────────
if all_summaries:
    combined = (
        pd.concat(all_summaries, ignore_index=True)
        .sort_values("annualized_spread_sharpe_sn", ascending=False)
    )
    combined.to_csv(OUT_DIR / "rolling_summary_all_models.csv", index=False)
    print("\n" + "=" * 70, flush=True)
    print("NON-LINEAR SECTOR-BLOCK — COMBINED SUMMARY", flush=True)
    print("=" * 70, flush=True)
    show = ["model", "annualized_spread_sharpe_sn", "annualized_spread_sharpe",
            "mean_daily_ic", "elapsed_sec"]
    print(combined[[c for c in show if c in combined.columns]].to_string(index=False), flush=True)

print(f"\nOutputs saved to: {OUT_DIR.resolve()}", flush=True)
