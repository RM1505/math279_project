import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/feature_table.csv")
spy_df = df[df["ticker"]=="SPY"]
spy_ret = spy_df.set_index("date")["OPCL"]

asset_df = df[df["ticker"]!="SPY"]
asset_df["spy_ret"] = asset_df["date"].map(spy_ret)

window = 60

asset_df["beta"] = (
    asset_df.groupby("ticker")
    .apply(lambda x: x["OPCL"].rolling(window).cov(x["spy_ret"]) /
                     x["spy_ret"].rolling(window).var())
    .reset_index(level=0, drop=True)
)

asset_df["residual_ret"] = asset_df["OPCL"] - asset_df["beta"] * asset_df["spy_ret"]
asset_df = asset_df.dropna(subset=["beta"])

file_path = Path("data/processed/feature_table_with_residuals.csv")
asset_df.to_csv(file_path, index=False)
