#!/usr/bin/env python3
"""13_ensemble.py

Combine the overnight (script 07) and next-day (script 06) ridge predictions
into an ensemble strategy and evaluate its out-of-sample performance.

Methodology
-----------
The overnight and next-day strategies use the SAME intraday OFI signal but
predict DIFFERENT return targets and hold positions at different times of day:

  - Next-day (script 06): OFI on day t-1 predicts intraday return (open_t → close_t)
    Prediction CSV indexed by return date t.

  - Overnight (script 07): OFI on day t predicts overnight return (close_t → open_{t+1})
    Prediction CSV indexed by OFI date t (same date as the overnight return).

Because they trade at different times, their daily P&L series are additive.
The correct ensemble combines the DAILY SPREAD SERIES, not the raw score matrices
(which have different return targets and cannot be cross-evaluated).

Ensemble approaches
-------------------
1. Equal-weight PnL: spread = (spread_nd + spread_on) / 2
2. Rolling IC-weighted PnL: weight each strategy by its recent rolling Sharpe
3. EMA-smoothed scores (within each strategy) then equal-weight PnL combination

Outputs -> results/ensemble/

Run from repo root (requires scripts 06 and 07 to have completed):
    python pipeline/13_ensemble.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from pipeline_utils import (
    daily_spread,
    daily_sector_neutral_spread,
    smooth_scores,
    sharpe_from_series,
    annualize_sharpe,
    compute_turnover_series,
    compute_net_spread,
    compute_breakeven_bps,
)

# ── config ────────────────────────────────────────────────────────────────────
NEXTDAY_DIR    = Path("results/rolling_adjacency_ridge")
OVERNIGHT_DIR  = Path("results/rolling_adjacency_ridge_overnight")
FEATURE_TABLE  = Path("data/processed/feature_table_with_residuals_10level.csv")
OVERNIGHT_RET  = Path("data/processed/overnight_returns_residualized.csv")
OUT_DIR        = Path("results/ensemble")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best model from each strategy (by SN Sharpe)
NEXTDAY_MODEL   = "ridge_sb_gd_hl30_rn"
OVERNIGHT_MODEL = "ridge_on_sb_gd_hl30_rn"

Q               = 0.10
MIN_SN_NAMES    = 4
TC_SCENARIOS    = [5, 10, 20, 30]          # round-trip bps

# Rolling Sharpe window for IC-weighted ensemble (trading days)
SR_WEIGHT_WINDOW = 60

# EMA alpha for the smoothed-scores variant
EMA_ALPHA = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Load predicted score DataFrames
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(directory: Path, model_name: str) -> pd.DataFrame:
    path = directory / f"predicted_returns_{model_name}_rolling.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Predicted returns not found: {path}\n"
            f"Run the corresponding script first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


print("Loading next-day predictions...", flush=True)
pred_nd = load_predictions(NEXTDAY_DIR, NEXTDAY_MODEL)
print(f"  {NEXTDAY_MODEL}: {pred_nd.shape[0]} dates x {pred_nd.shape[1]} assets", flush=True)

print("Loading overnight predictions...", flush=True)
pred_on = load_predictions(OVERNIGHT_DIR, OVERNIGHT_MODEL)
print(f"  {OVERNIGHT_MODEL}: {pred_on.shape[0]} dates x {pred_on.shape[1]} assets", flush=True)

# ── Load next-day returns (intraday residualized) ─────────────────────────────
print("Loading next-day returns (intraday)...", flush=True)
meta_cols = ["date", "ticker", "sector", "residual_ret"]
feat = pd.read_csv(FEATURE_TABLE, usecols=meta_cols)
feat["date"] = pd.to_datetime(feat["date"])
ticker_to_sector = (
    feat[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]
)
ret_nd_full = feat.pivot(index="date", columns="ticker", values="residual_ret").sort_index()
del feat

# ── Load overnight returns ────────────────────────────────────────────────────
print("Loading overnight returns...", flush=True)
if not OVERNIGHT_RET.exists():
    raise FileNotFoundError(
        f"Overnight return cache not found: {OVERNIGHT_RET}\n"
        "Run pipeline/07_ridge_overnight.py first."
    )
ret_on_raw = pd.read_csv(OVERNIGHT_RET, index_col=0, parse_dates=True)
ret_on_raw.index = pd.to_datetime(ret_on_raw.index)
ret_on_full = ret_on_raw.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# Compute daily spread series for each strategy against its own return target
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_spreads(
    pred_df: pd.DataFrame,
    ret_full: pd.DataFrame,
    ts: pd.Series,
    q: float = 0.10,
    min_sn: int = 4,
    ema_alpha: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Compute gross and sector-neutral daily spread series.

    Returns (gross_spreads, sn_spreads) as pd.Series indexed by date.
    """
    if ema_alpha < 1.0:
        pred_df = smooth_scores(pred_df, alpha=ema_alpha)

    cd = pred_df.index.intersection(ret_full.index)
    cc = pred_df.columns.intersection(ret_full.columns)
    # Further restrict to tickers that have a sector label
    ts_local = ts.reindex(cc).dropna()
    cc = ts_local.index

    P = pred_df.loc[cd, cc]
    R = ret_full.loc[cd, cc]

    gross = P.apply(lambda row: daily_spread(row, R.loc[row.name], q=q), axis=1).dropna()
    sn = P.apply(
        lambda row: daily_sector_neutral_spread(
            row, R.loc[row.name], ts_local, q=q, min_names_per_sector=min_sn
        ),
        axis=1,
    ).dropna()
    return gross, sn


print("\nComputing next-day daily spreads...", flush=True)
gross_nd, sn_nd = compute_daily_spreads(pred_nd, ret_nd_full, ticker_to_sector, Q, MIN_SN_NAMES)
print(f"  Next-day: {len(sn_nd)} dates, SN Sharpe = {annualize_sharpe(sharpe_from_series(sn_nd)):.3f}", flush=True)

print("Computing overnight daily spreads...", flush=True)
gross_on, sn_on = compute_daily_spreads(pred_on, ret_on_full, ticker_to_sector, Q, MIN_SN_NAMES)
print(f"  Overnight: {len(sn_on)} dates, SN Sharpe = {annualize_sharpe(sharpe_from_series(sn_on)):.3f}", flush=True)

# EMA-smoothed versions
print("Computing EMA-smoothed next-day spreads...", flush=True)
gross_nd_ema, sn_nd_ema = compute_daily_spreads(
    pred_nd, ret_nd_full, ticker_to_sector, Q, MIN_SN_NAMES, ema_alpha=EMA_ALPHA
)
print("Computing EMA-smoothed overnight spreads...", flush=True)
gross_on_ema, sn_on_ema = compute_daily_spreads(
    pred_on, ret_on_full, ticker_to_sector, Q, MIN_SN_NAMES, ema_alpha=EMA_ALPHA
)


# ══════════════════════════════════════════════════════════════════════════════
# Build ensemble PnL series
# ══════════════════════════════════════════════════════════════════════════════
# The overnight spread for date t covers close_t -> open_{t+1}.
# The next-day spread for date t covers open_t -> close_t (predicted by OFI_{t-1}).
# Both strategies trade at different times of day so their P&L streams add.
# We align on common calendar dates.

common_gross = gross_nd.index.intersection(gross_on.index)
common_sn    = sn_nd.index.intersection(sn_on.index)

print(f"\nCommon gross dates : {len(common_gross)}", flush=True)
print(f"Common SN dates    : {len(common_sn)}", flush=True)

# ── Ensemble 1: equal-weight PnL ─────────────────────────────────────────────
eq_gross = (gross_nd.loc[common_gross] + gross_on.loc[common_gross]) / 2.0
eq_sn    = (sn_nd.loc[common_sn]      + sn_on.loc[common_sn])      / 2.0

# ── Ensemble 2: rolling Sharpe-weighted PnL ───────────────────────────────────
# Weight each strategy proportional to its rolling Sharpe over past SR_WEIGHT_WINDOW days.

def rolling_sharpe_weights(
    s1: pd.Series, s2: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """Return (w1, w2) where weights are proportional to rolling Sharpe (clipped at 0)."""
    common = s1.index.intersection(s2.index)
    sr1 = s1.loc[common].rolling(window).mean() / (s1.loc[common].rolling(window).std() + 1e-9)
    sr2 = s2.loc[common].rolling(window).mean() / (s2.loc[common].rolling(window).std() + 1e-9)
    sr1 = sr1.clip(lower=0.0)
    sr2 = sr2.clip(lower=0.0)
    total = (sr1 + sr2).replace(0.0, np.nan)
    w1 = (sr1 / total).fillna(0.5)
    w2 = (sr2 / total).fillna(0.5)
    return w1, w2


w_nd_gross, w_on_gross = rolling_sharpe_weights(
    gross_nd.loc[common_gross], gross_on.loc[common_gross], SR_WEIGHT_WINDOW
)
w_nd_sn, w_on_sn = rolling_sharpe_weights(
    sn_nd.loc[common_sn], sn_on.loc[common_sn], SR_WEIGHT_WINDOW
)

srw_gross = w_nd_gross * gross_nd.loc[common_gross] + w_on_gross * gross_on.loc[common_gross]
srw_sn    = w_nd_sn   * sn_nd.loc[common_sn]        + w_on_sn   * sn_on.loc[common_sn]

# ── Ensemble 3: equal-weight PnL after EMA-smoothing each strategy ────────────
common_gross_ema = gross_nd_ema.index.intersection(gross_on_ema.index)
common_sn_ema    = sn_nd_ema.index.intersection(sn_on_ema.index)

eq_ema_gross = (gross_nd_ema.loc[common_gross_ema] + gross_on_ema.loc[common_gross_ema]) / 2.0
eq_ema_sn    = (sn_nd_ema.loc[common_sn_ema]       + sn_on_ema.loc[common_sn_ema])       / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate all strategies
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_spread_pair(
    gross: pd.Series,
    sn: pd.Series,
    name: str,
    tc_scenarios: list[int],
) -> dict:
    """Summarise a (gross, SN) spread pair."""
    # Turnover proxy: use constant fraction since we don't have position-level data here.
    # Instead report TC-adjusted Sharpe at each scenario directly from the spread series.
    # We can't compute exact turnover from PnL-level data so we report gross metrics only.
    row: dict = {
        "model":                       name,
        "n_eval_days_gross":           len(gross),
        "n_eval_days_sn":              len(sn),
        "mean_daily_spread":           gross.mean(),
        "annualized_spread_sharpe":    annualize_sharpe(sharpe_from_series(gross)),
        "mean_daily_spread_sn":        sn.mean(),
        "annualized_spread_sharpe_sn": annualize_sharpe(sharpe_from_series(sn)),
    }
    return row


# Build pred-level turnover for the two base strategies (to get TC-adjusted Sharpe)
def tc_adjusted_from_pred(
    pred_df: pd.DataFrame,
    ret_full: pd.DataFrame,
    ts: pd.Series,
    q: float,
    min_sn: int,
    tc_scenarios: list[int],
    ema_alpha: float = 1.0,
) -> dict[str, float]:
    """Compute turnover-based TC-adjusted net Sharpe for a prediction matrix."""
    if ema_alpha < 1.0:
        pred_df = smooth_scores(pred_df, alpha=ema_alpha)
    cd = pred_df.index.intersection(ret_full.index)
    cc = pred_df.columns.intersection(ret_full.columns)
    ts_local = ts.reindex(cc).dropna()
    cc = ts_local.index
    P = pred_df.loc[cd, cc]
    R = ret_full.loc[cd, cc]

    gross = P.apply(lambda row: daily_spread(row, R.loc[row.name], q=q), axis=1).dropna()
    sn = P.apply(
        lambda row: daily_sector_neutral_spread(row, R.loc[row.name], ts_local, q=q,
                                                 min_names_per_sector=min_sn),
        axis=1,
    ).dropna()
    tau_gross = compute_turnover_series(P, q)
    tau_sn = compute_turnover_series(P, q, ticker_to_sector=ts_local, min_per_sector=min_sn, sn=True)
    out: dict[str, float] = {
        "breakeven_rt_bps_gross": compute_breakeven_bps(gross, tau_gross),
        "breakeven_rt_bps_sn":    compute_breakeven_bps(sn, tau_sn),
        "mean_daily_turnover_gross": tau_gross.mean(),
        "mean_daily_turnover_sn":    tau_sn.mean(),
    }
    for tc in tc_scenarios:
        out[f"net_sharpe_gross_{tc}bps"] = annualize_sharpe(sharpe_from_series(
            compute_net_spread(gross, tau_gross, tc)
        ))
        out[f"net_sharpe_sn_{tc}bps"] = annualize_sharpe(sharpe_from_series(
            compute_net_spread(sn, tau_sn, tc)
        ))
    return out


print("\nComputing TC-adjusted metrics for individual strategies...", flush=True)
tc_nd = tc_adjusted_from_pred(pred_nd, ret_nd_full, ticker_to_sector, Q, MIN_SN_NAMES, TC_SCENARIOS)
tc_on = tc_adjusted_from_pred(pred_on, ret_on_full, ticker_to_sector, Q, MIN_SN_NAMES, TC_SCENARIOS)
tc_nd_ema = tc_adjusted_from_pred(pred_nd, ret_nd_full, ticker_to_sector, Q, MIN_SN_NAMES,
                                   TC_SCENARIOS, ema_alpha=EMA_ALPHA)
tc_on_ema = tc_adjusted_from_pred(pred_on, ret_on_full, ticker_to_sector, Q, MIN_SN_NAMES,
                                   TC_SCENARIOS, ema_alpha=EMA_ALPHA)

print("Building summary table...", flush=True)

base_records = [
    {"name": "nextday_only",    "gross": gross_nd, "sn": sn_nd,    "tc": tc_nd},
    {"name": "overnight_only",  "gross": gross_on, "sn": sn_on,    "tc": tc_on},
    {"name": "nd_ema_only",     "gross": gross_nd_ema, "sn": sn_nd_ema, "tc": tc_nd_ema},
    {"name": "on_ema_only",     "gross": gross_on_ema, "sn": sn_on_ema, "tc": tc_on_ema},
]

ensemble_records = [
    {"name": "ensemble_eq",      "gross": eq_gross,     "sn": eq_sn,     "tc": {}},
    {"name": "ensemble_srw",     "gross": srw_gross,    "sn": srw_sn,    "tc": {}},
    {"name": "ensemble_eq_ema",  "gross": eq_ema_gross, "sn": eq_ema_sn, "tc": {}},
]

rows = []
for rec in base_records + ensemble_records:
    r = evaluate_spread_pair(rec["gross"], rec["sn"], rec["name"], TC_SCENARIOS)
    r.update(rec["tc"])
    rows.append(r)

summary = pd.DataFrame(rows).sort_values("annualized_spread_sharpe_sn", ascending=False)
summary.to_csv(OUT_DIR / "ensemble_summary.csv", index=False)

# Save daily PnL series for the two base strategies and the best ensemble
for name, gross_ser, sn_ser in [
    ("nextday_only", gross_nd, sn_nd),
    ("overnight_only", gross_on, sn_on),
    ("ensemble_eq", eq_gross, eq_sn),
    ("ensemble_srw", srw_gross, srw_sn),
]:
    pd.DataFrame({"gross_spread": gross_ser, "sn_spread": sn_ser}).to_csv(
        OUT_DIR / f"daily_spreads_{name}.csv"
    )

print("\n" + "=" * 70, flush=True)
print("ENSEMBLE SUMMARY (PnL-level combination)", flush=True)
print("=" * 70, flush=True)
cols_show = [
    "model", "annualized_spread_sharpe_sn", "annualized_spread_sharpe",
    "breakeven_rt_bps_sn", "net_sharpe_sn_5bps", "net_sharpe_sn_10bps",
]
avail = [c for c in cols_show if c in summary.columns]
print(summary[avail].to_string(index=False), flush=True)
print(f"\nOutputs saved to: {OUT_DIR.resolve()}", flush=True)
