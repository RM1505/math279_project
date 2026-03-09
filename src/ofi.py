from pdb import main

import numpy as np
import pandas as pd
from pathlib import Path

def compute_ofi(df: pd.DataFrame, level = "1") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame({"time": pd.Series(dtype=df["time"].dtype if df is not None and "time" in df else "datetime64[ns]"),
                             "ofi":  pd.Series(dtype=float)})
    
    b  = df[f"bid_{level}"].to_numpy(dtype=float)
    a  = df[f"ask_{level}"].to_numpy(dtype=float)
    qb = df[f" bid_size_{level}"].to_numpy(dtype=float)
    qa = df[f"ask_size_{level}"].to_numpy(dtype=float)

    b1  = np.roll(b, 1).astype(float, copy=False)
    a1  = np.roll(a, 1).astype(float, copy=False)
    qb1 = np.roll(qb, 1).astype(float, copy=False)
    qa1 = np.roll(qa, 1).astype(float, copy=False)

    b1[0]  = np.nan
    a1[0]  = np.nan
    qb1[0] = np.nan
    qa1[0] = np.nan

    dB = np.select(
        [b == b1, b > b1, b < b1],
        [qb - qb1, qb,     -qb1],
        default=np.nan
    )

    dA = np.select(
        [a == a1, a > a1, a < a1],
        [qa1 - qa, -qa1,  qa],
        default=np.nan
    )

    D = (qb + qa) / 2.0

    num = dB + dA
    ofi = np.full_like(num, np.nan, dtype=float)
    np.divide(num, D, out=ofi, where=(D != 0) & np.isfinite(D))

    return pd.DataFrame(
        {
            "time": df["time"].to_numpy(),
            "ofi": ofi,
        }
    )

def make_weights(lambda_, n_minutes=390, emphasize_close=True):
    if emphasize_close:
        return np.exp(-lambda_ * (n_minutes - np.arange(1, n_minutes + 1, dtype=float)))
    return np.exp(-lambda_ * np.arange(1, n_minutes + 1, dtype=float))

def minute_index_fast(df, time_col="time", n_minutes=390):
    h = df[time_col].str.slice(0, 2).astype(np.int32)
    m = df[time_col].str.slice(3, 5).astype(np.int32)
    minute = 1 + (h - 9) * 60 + (m - 30)
    return np.clip(minute.to_numpy(), 1, n_minutes)

def compress_day_df_fast(day_df, weights, time_col="time", ofi_col="ofi"):
    minute = minute_index_fast(day_df, time_col=time_col)
    ofi = day_df[ofi_col].to_numpy(dtype=float)
    return float(np.nansum(ofi * weights[minute - 1]))

def compress_ticker_folder_fast(ticker_dir: Path, out_csv: Path, weights,
                               time_col="time", ofi_col="ofi"):
    rows = []
    bad = 0

    for f in sorted(ticker_dir.glob("*.csv")):
        if f.stat().st_size == 0:
            bad += 1
            continue
        try:
            day_df = pd.read_csv(
                f,
                usecols=[time_col, ofi_col],
                dtype={time_col: "string", ofi_col: "float64"},
                engine="c",
            )
        except (pd.errors.EmptyDataError, ValueError):
            bad += 1
            continue

        if day_df.empty:
            bad += 1
            continue

        ofi_daily = compress_day_df_fast(day_df, weights, time_col=time_col, ofi_col=ofi_col)
        rows.append({"date": f.stem, "ofi_daily": ofi_daily})

    if not rows:
        return (ticker_dir.name, False, bad)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("date").to_csv(out_csv, index=False)

    return (ticker_dir.name, True, bad)

def worker(job):
    ticker_dir, out_csv, lambda_, time_col, ofi_col, emphasize_close = job
    weights = make_weights(lambda_, emphasize_close=emphasize_close)
    return compress_ticker_folder_fast(
        Path(ticker_dir),
        Path(out_csv),
        weights,
        time_col=time_col,
        ofi_col=ofi_col
    )
