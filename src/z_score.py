from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_zscore(
    df: pd.DataFrame,
    col: str,
    window: int,
    *,
    out_col: str | None = None,
    min_periods: int | None = None,
    ddof: int = 0,
    center: bool = False,
    avoid_lookahead: bool = True,
    fillna: float | None = np.nan,
) -> pd.DataFrame:
    """
    Add a rolling z-score column to df and return df.

    z_t = (x_t - mean_t) / std_t

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Column to z-score.
    window : int
        Rolling window length.
    out_col : str, optional
        Name of output z-score column. Default: f"{col}_z{window}".
    min_periods : int, optional
        Minimum periods for rolling stats. Default: window (i.e., require full window).
    ddof : int
        Degrees of freedom for std. ddof=0 is faster/stabler for large scale usage.
    center : bool
        Centered rolling window (usually False for time series).
    avoid_lookahead : bool
        If True, compute stats on past data only by shifting mean/std by 1.
        This is typically what you want for backtests.
    fillna : float or None
        Value to fill NaNs in output. Use None to leave as-is.

    Returns
    -------
    pd.DataFrame
        Same df object with added column.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in df")

    if window <= 0:
        raise ValueError("window must be a positive integer")

    if out_col is None:
        out_col = f"{col}_z{window}"

    if min_periods is None:
        min_periods = window

    x = df[col].to_numpy(dtype=np.float64, copy=False)

    s = pd.Series(x, index=df.index, copy=False)

    r = s.rolling(window=window, min_periods=min_periods, center=center)

    mean = r.mean()
    std = r.std(ddof=ddof)

    if avoid_lookahead:
        mean = mean.shift(1)
        std = std.shift(1)

    z = (s - mean) / std
    z = z.where(std != 0.0, np.nan)

    if fillna is not None:
        z = z.fillna(fillna)

    df[out_col] = z.to_numpy(copy=False)
    return df