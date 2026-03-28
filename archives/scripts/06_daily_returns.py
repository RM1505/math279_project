import pandas as pd
from pathlib import Path
import numpy as np
from build_opcl import build_opcl_panel_and_save_per_ticker


df = pd.read_csv("data/index.csv", index_col=0)
start = pd.to_datetime(df["start"], format="%Y-%m-%d").min()
end = pd.to_datetime(df["end"], format="%Y-%m-%d").max()

panel = build_opcl_panel_and_save_per_ticker(
    universe_df=df,
    start_date=start,
    end_date=end,
    crsp_zip_dir="data/raw/US_CRSP",
    out_dir="data/processed/opcl_by_ticker",
    ticker_col="ticker",
    opcl_col="OPCL",
    fill_value=None,   # keep NaN for weekends/missing; use 0.0 if you want zeros
)