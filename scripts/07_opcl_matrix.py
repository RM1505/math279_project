import pandas as pd
from pathlib import Path
import numpy as np

df = pd.read_csv("data/index.csv", index_col=0)
tickers = df["ticker"]
n = len(df) #514

start = pd.to_datetime(df["start"], format="%Y-%m-%d").min()
end = pd.to_datetime(df["end"], format="%Y-%m-%d").max()
dates = pd.date_range(start=start, end=end, freq="D")
T = len(dates)

base = Path("data/processed/opcl_by_ticker")

R = np.full((T, n), np.nan, dtype=np.float32)
dates_idx = pd.DatetimeIndex(dates)

for j, ticker in enumerate(tickers):
    fp = base / f"{ticker}_opcl.csv"
    s = pd.read_csv(fp, usecols=["date", "OPCL"])

    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index("date")["OPCL"]

    s = s.reindex(dates_idx)

    R[:, j] = s.to_numpy(dtype=np.float32)

np.save("data/processed/R.npy", R) #To load: R = np.load("data/processed/R.npy")
