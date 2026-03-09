from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import time


_NAME_RE = re.compile(
    r"""
    ^(.+__)?                          
    (?P<ticker>[A-Z][A-Z0-9.\-]*)     
    _(?P<start>\d{4}-\d{2}-\d{2})     
    _(?P<end>\d{4}-\d{2}-\d{2})       
    _(?P<rest>.+)                     
    \.7z$
    """,
    re.VERBOSE,
)


def _fetch_sector_map(tickers: list[str], cache_path: Path) -> pd.DataFrame:
    """
    Fetch ticker -> sector mapping from yfinance and cache to disk.
    """
    if cache_path.exists():
        return pd.read_csv(cache_path)

    rows = []

    for t in tickers:
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector")
        except Exception:
            sector = None

        rows.append({"ticker": t, "sector": sector})

        time.sleep(0.2)  # avoid rate limits

    sector_df = pd.DataFrame(rows)
    sector_df.to_csv(cache_path, index=False)

    return sector_df


def index_7z_folder(folder: str | Path, sector_cache: str | Path = "ticker_sectors.csv") -> pd.DataFrame:
    """
    Return a DataFrame of all archives with parsed ticker/date-range
    and Yahoo Finance sector classification.
    """
    folder = Path(folder)
    rows = []

    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() != ".7z":
            continue

        m = _NAME_RE.match(p.name)

        if not m:
            rows.append(
                dict(
                    ticker=None,
                    start=None,
                    end=None,
                    archive=p.name,
                )
            )
            continue

        ticker = m.group("ticker")

        if "." in ticker:
            continue

        rows.append(
            dict(
                ticker=ticker,
                start=m.group("start"),
                end=m.group("end"),
                archive=p.name,
            )
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(["ticker", "start", "end"])
    df.reset_index(drop=True, inplace=True)

    # -------------------------
    # Add sector info
    # -------------------------

    unique_tickers = sorted(df["ticker"].dropna().unique().tolist())

    sector_df = _fetch_sector_map(
        unique_tickers,
        Path(sector_cache),
    )

    df = df.merge(sector_df, on="ticker", how="left")

    return df