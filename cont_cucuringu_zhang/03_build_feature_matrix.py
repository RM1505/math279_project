import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
folder = Path(f"data/processed/integrated_ofi")

dfs = []

for file in tqdm(folder.glob("*.csv"), desc="Processing files", total=len(list(folder.glob("*.csv")))):
    df = pd.read_csv(file) 

    df = df.sort_values(["source", "time"]).reset_index(drop=True)

    df["minute"] = df.groupby("source").cumcount() + 1

    df_wide = df.pivot(index="source", columns="minute", values="IOFI")

    df_wide.columns = [f"minute_{int(col)}" for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    df_wide["ticker"] = file.stem
    dfs.append(df_wide)

combined = pd.concat(dfs, ignore_index=True)

def load_opcl_returns(folder):
    dfs = []

    for file in tqdm(Path(folder).glob("*.csv"), desc="Loading OPCL returns", total=len(list(Path(folder).glob("*.csv")))):
        ticker = file.stem.replace("_opcl", "")  # or parse if filename has extra text
        df = pd.read_csv(file)

        # adjust these column names to match your files
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"])

        df = df[["ticker", "date", "OPCL"]]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

combined["date"] = pd.to_datetime(combined["source"])
combined.drop(columns=["source"], inplace=True)

ret_df = load_opcl_returns("data/processed/opcl_by_ticker")
combined = combined.merge(ret_df, on=["ticker", "date"], how="left")

indices = pd.read_csv("data/index.csv")

combined["ticker"] = combined["ticker"].astype(str)
indices["ticker"] = indices["ticker"].astype(str)

combined = combined.merge(indices[["ticker", "sector"]], on="ticker", how="left")

combined.to_csv("data/processed/feature_table.csv", index=False)