from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


def build_opcl_panel_and_save_per_ticker(
    universe_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    crsp_zip_dir: str | Path = "data/raw/US_CRSP",
    out_dir: str | Path = "data/processed/opcl_by_ticker",
    ticker_col: str = "ticker",
    opcl_col: str = "OPCL",
    fill_value: float | None = None,  
    dtype=np.float32,                 
) -> pd.DataFrame:
    """
    Reads daily YYYYMMDD.csv.gz files inside yearly ZIPs, extracts OPCL for tickers in universe,
    builds a date x ticker matrix for all calendar days in [start_date, end_date], and saves
    one CSV per ticker (date, OPCL).

    Returns the full panel DataFrame (index=date, columns=ticker).
    """

    crsp_zip_dir = Path(crsp_zip_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = (
        universe_df[ticker_col]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    n = len(tickers)
    tick_set = set(tickers)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_dates = pd.date_range(start=start, end=end, freq="D")
    T = len(all_dates)

    data = np.full((T, n), np.nan, dtype=dtype)

    date_to_row = {d.normalize(): i for i, d in enumerate(all_dates)}

    for year in range(start.year, end.year + 1):
        zip_path = crsp_zip_dir / f"{year}.zip"
        if not zip_path.exists():
            continue

        with zipfile.ZipFile(zip_path, "r") as z:
            daily_files = [name for name in z.namelist() if name.endswith(".csv.gz")]
            daily_files.sort()

            for name in daily_files:
                fname = Path(name).name
                ymd = fname.split(".")[0]
                if len(ymd) != 8 or not ymd.isdigit():
                    continue

                d = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
                if pd.isna(d):
                    continue
                if d < start or d > end:
                    continue

                row = date_to_row.get(d.normalize())
                if row is None:
                    continue

                with z.open(name) as f:
                    df = pd.read_csv(
                        f,
                        compression="gzip",
                        usecols=[ticker_col, opcl_col],
                    )

                df[ticker_col] = df[ticker_col].astype(str)

                df = df[df[ticker_col].isin(tick_set)]
                if df.empty:
                    continue

                s = (
                    pd.to_numeric(df[opcl_col], errors="coerce")
                    .astype(dtype, copy=False)
                )
                by_ticker = pd.Series(s.to_numpy(), index=df[ticker_col].to_numpy())
                aligned = by_ticker.reindex(tickers)

                data[row, :] = aligned.to_numpy(dtype=dtype)

    panel = pd.DataFrame(data, index=all_dates, columns=tickers)

    if fill_value is not None:
        panel = panel.fillna(fill_value)

    for t in tickers:
        out_path = out_dir / f"{t}_opcl.csv"
        panel[[t]].rename(columns={t: opcl_col}).to_csv(out_path, index_label="date")

    return panel