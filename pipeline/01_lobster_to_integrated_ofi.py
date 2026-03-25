#!/usr/bin/env python3
"""Pipeline: LOBSTER archives → integrated OFI per ticker.

Merges old scripts 01 and 02 to avoid storing 40+ GB of intermediate
per-day per-level CSVs.  For each archive (one ticker):
  1. Extract to a temp dir (auto-deleted afterward)
  2. Compute per-level OFI for every day
  3. Stack all days, PCA across OFI levels → IOFI (first principal component)
  4. Write data/processed/integrated_ofi/{ticker}.csv
     columns: source (date), time, IOFI

For 1-level tickers IOFI degenerates to OFI_1 (only one level to integrate).
For 10-level tickers IOFI is the first PC across all ten OFI signals.

Run from repo root:
    python pipeline/01_lobster_to_integrated_ofi.py
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
from pandas.errors import EmptyDataError
from tqdm import tqdm

from ofi import compute_ofi


# ── config ────────────────────────────────────────────────────────────
DATA_RAW   = Path("data/raw")
DATA_INDEX = Path("data/index.csv")
OUT_DIR    = Path("data/processed/integrated_ofi")

_NAME_RE = re.compile(r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<date>\d{4}-\d{2}-\d{2})_")


# ── per-level OFI helpers ─────────────────────────────────────────────
def group_by_lob_level(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return one sub-DataFrame per LOB level, each with 'time' + level cols."""
    groups: dict[str, list[str]] = {}
    for col in df.columns:
        if col == "time":
            continue
        m = re.search(r"\d+$", col)
        if m is None:
            continue
        groups.setdefault(m.group(), ["time"]).append(col)
    return {k: df[v].copy() for k, v in groups.items()}


def ofi_for_day(csv_path: Path) -> pd.DataFrame | None:
    """Read one daily snapshot CSV and return a DataFrame of per-level OFI.

    Returns columns: time, ofi_1 [, ofi_2, ..., ofi_10], source (date stem).
    First row is dropped (NaN OFI — no previous tick to diff against).
    """
    try:
        df = pd.read_csv(csv_path)
    except (EmptyDataError, Exception):
        return None

    if df.empty:
        return None

    m = _NAME_RE.match(csv_path.name)
    if m is None:
        return None
    date = m.group("date")

    level_dfs = group_by_lob_level(df)
    ofi_frames: list[pd.DataFrame] = []

    for lvl in sorted(level_dfs, key=int):
        sub = level_dfs[lvl]
        try:
            ofi_df = compute_ofi(sub, level=lvl)
        except Exception:
            return None
        ofi_frames.append(
            ofi_df[["time", "ofi"]].rename(columns={"ofi": f"ofi_{lvl}"})
        )

    merged = ofi_frames[0]
    for frame in ofi_frames[1:]:
        merged = merged.merge(frame, on="time", how="inner")

    # drop row 0: OFI is NaN there (no prior snapshot to diff)
    merged = merged.iloc[1:].reset_index(drop=True)
    merged["source"] = date
    return merged


# ── PCA integration ───────────────────────────────────────────────────
def integrate_via_pca(combined: pd.DataFrame) -> pd.DataFrame:
    """First PC of the ofi_* columns across all ticker-days.

    For 1-level tickers this is just ofi_1 (trivial PCA).
    """
    X = combined.filter(like="ofi_").to_numpy(dtype=float, copy=True)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X -= X.mean(axis=0)

    if X.shape[1] == 1:
        # single level — no PCA needed
        iofi = X[:, 0]
    else:
        cov = (X.T @ X) / max(X.shape[0] - 1, 1)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            iofi = X[:, 0]
        else:
            first_pc = eigenvectors[:, np.argmax(eigenvalues)]
            iofi = X @ first_pc

    out = combined[["source", "time"]].copy()
    out["IOFI"] = iofi
    return out


# ── per-archive processing ────────────────────────────────────────────
def process_archive(archive_name: str) -> str:
    """Extract archive, compute integrated OFI, write one CSV. Returns status."""
    archive_path = DATA_RAW / archive_name
    if not archive_path.exists():
        return "missing"

    # derive ticker from the archive filename
    m = re.search(r"__(?P<ticker>[A-Z0-9.\-]+)_\d{4}", archive_name)
    if m is None:
        return "bad_name"
    ticker = m.group("ticker")

    out_path = OUT_DIR / f"{ticker}.csv"
    if out_path.exists():
        return "skipped"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as z:
                csv_names = [n for n in z.getnames() if n.lower().endswith(".csv")]
                z.extract(path=tmpdir, targets=csv_names)
        except Exception as e:
            return f"extract_error: {e}"

        csv_files = sorted(tmpdir_path.rglob("*.csv"))
        day_frames: list[pd.DataFrame] = []

        for f in csv_files:
            result = ofi_for_day(f)
            if result is not None and not result.empty:
                day_frames.append(result)

    if not day_frames:
        return "no_data"

    combined = pd.concat(day_frames, ignore_index=True)

    try:
        iofi_df = integrate_via_pca(combined)
    except Exception as e:
        return f"pca_error: {e}"

    iofi_df.to_csv(out_path, index=False)
    return "ok"


# ── main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx           = pd.read_csv(DATA_INDEX)
    archive_names = idx["archive"].dropna().astype(str).tolist()

    status_counts: dict[str, int] = {}

    for archive_name in tqdm(archive_names, desc="Archives"):
        status = process_archive(archive_name)
        status_counts[status] = status_counts.get(status, 0) + 1
        if not status.startswith(("ok", "skipped")):
            tqdm.write(f"  [{status}] {archive_name}")

    print("\nDone.")
    for s, c in sorted(status_counts.items()):
        print(f"  {s}: {c}")
