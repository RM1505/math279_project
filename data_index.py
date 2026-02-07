from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


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


def index_7z_folder(folder: str | Path) -> pd.DataFrame:
    """
    Return a DataFrame of all with parsed ticker/date-range.
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

        rows.append(
            dict(
                ticker=m.group("ticker"),
                start=m.group("start"),
                end=m.group("end"),
                archive=p.name,
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "ticker" in df.columns:
        df = df.sort_values(["ticker", "start", "end"], ascending=[True, True, True])

    return df

if __name__ == "__main__":
    df = index_7z_folder("data")
    print(df) 