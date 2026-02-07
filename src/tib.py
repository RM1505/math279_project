import numpy as np
import pandas as pd

def compute_tib(df: pd.DataFrame) -> pd.DataFrame:
    
    bid = df[" bid_size_1"].to_numpy(dtype=float) # For some reason the column name has a leading space in the raw data :sob:
    ask = df["ask_size_1"].to_numpy(dtype=float)
    
    denom = bid + ask

    tib = np.divide(
        bid - ask,
        denom,
        out=np.zeros_like(bid),
        where=denom != 0,
    )

    return pd.DataFrame(
        {
            "time": df["time"].values,
            "tib": tib,
        }
    )
