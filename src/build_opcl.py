import pandas as pd
import numpy as np
import zipfile
from pathlib import Path

def build_opcl_panel_and_save_per_ticker(
    universe_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    crsp_zip_dir: str | Path = "data/raw/US_CRSP",
    out_dir: str | Path = "data/processed/opcl_by_ticker",
    ticker_col: str = "ticker",
    opcl_col: str = "OPCL",
    sic_col: str = "HSICMG",
    fill_value: float | None = None,  
    dtype=np.float32,                 
) -> pd.DataFrame:
    """
    Reads daily CRSP files, extracts OPCL and HSICMG, and saves 
    per-ticker CSVs containing both columns.
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

    # Initialize two matrices: one for Returns, one for SIC codes
    data_opcl = np.full((T, n), np.nan, dtype=dtype)
    data_sic = np.full((T, n), np.nan, dtype=dtype)

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
                if pd.isna(d) or d < start or d > end:
                    continue

                row = date_to_row.get(d.normalize())
                if row is None:
                    continue

                with z.open(name) as f:
                    # Load both columns
                    df = pd.read_csv(
                        f,
                        compression="gzip",
                        usecols=[ticker_col, opcl_col, sic_col],
                    )

                df[ticker_col] = df[ticker_col].astype(str)
                df = df[df[ticker_col].isin(tick_set)]
                
                if df.empty:
                    continue

                # Align OPCL
                opcl_vals = pd.to_numeric(df[opcl_col], errors="coerce").astype(dtype)
                s_opcl = pd.Series(opcl_vals.to_numpy(), index=df[ticker_col])
                data_opcl[row, :] = s_opcl.reindex(tickers).to_numpy(dtype=dtype)

                # Align HSICMG
                sic_vals = pd.to_numeric(df[sic_col], errors="coerce").astype(dtype)
                s_sic = pd.Series(sic_vals.to_numpy(), index=df[ticker_col])
                data_sic[row, :] = s_sic.reindex(tickers).to_numpy(dtype=dtype)

    # Convert matrices to DataFrames
    panel_opcl = pd.DataFrame(data_opcl, index=all_dates, columns=tickers)
    panel_sic = pd.DataFrame(data_sic, index=all_dates, columns=tickers)

    if fill_value is not None:
        panel_opcl = panel_opcl.fillna(fill_value)
        # Note: Usually you don't fillna industry codes with 0.0, 
        # but you can if you want them to match.

    # Save 2-column CSV per ticker
    for t in tickers:
        out_path = out_dir / f"{t}_opcl.csv"
        ticker_df = pd.DataFrame({
            opcl_col: panel_opcl[t],
            sic_col: panel_sic[t]
        })
        ticker_df.to_csv(out_path, index_label="date")

    return panel_opcl