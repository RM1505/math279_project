import pandas as pd
from pathlib import Path

df = pd.read_csv("data/index.csv", index_col=0)
df["daily"] = df["ticker"] + "_ofi_daily.csv"
df.to_csv("data/index.csv", index=True)