import pandas as pd
from pathlib import Path
import numpy as np

df = pd.read_csv("data/index.csv", index_col=0)
tickers = df["ticker"]
n = len(df) #516

start = pd.to_datetime(df["start"], format="%Y-%m-%d").min()
end = pd.to_datetime(df["end"], format="%Y-%m-%d").max()
dates = pd.date_range(start=start, end=end, freq="D")
T = len(dates)

base = Path("data/processed/daily")

P = np.full((T, n), np.nan, dtype=np.float32)
dates_idx = pd.DatetimeIndex(dates)

for j, ticker in enumerate(tickers):
    fp = base / f"{ticker}_ofi_daily.csv"
    s = pd.read_csv(fp, usecols=["date", "ofi_z60"])

    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index("date")["ofi_z60"]

    s = s.reindex(dates_idx)

    P[:, j] = s.to_numpy(dtype=np.float32)

np.save("data/processed/P.npy", P) #To load: P = np.load("data/processed/P.npy")
