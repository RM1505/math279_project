from pathlib import Path
from z_score import add_rolling_zscore
import pandas as pd
from tqdm import tqdm

folder = Path("data/processed/daily")

for file in tqdm(folder.glob("*.csv"), total=len(list(folder.glob("*.csv"))), desc="Files"):
    df = pd.read_csv(file)
    df = add_rolling_zscore(df, "ofi_daily", 60, out_col="ofi_z60")
    df.to_csv(file, index=False)