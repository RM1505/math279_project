#!/usr/bin/env python3
"""
19_dhillon_bipartite_graphs.py

Quick plotting utilities for results produced by the
`18_dhillon_bipartite_source_target.py` backtest.

Expected directory structure:

    results/18_dhillon_bipartite/
        daily_backtest.csv       <- record of every day
        rebalance_history.csv    <- which block was chosen each rebalance
        cumulative_pnl.csv       <- convenience cumulative pnl series
        summary.json             <- summary stats from backtest

The script will read these files and produce a handful of simple
diagnostic charts in a `graphs/` subdirectory.

Usage:

    python3 scripts/19_dhillon_bipartite_graphs.py \
        --results results/18_dhillon_bipartite

You can also call the main() function from elsewhere if you prefer.

"""

from __future__ import annotations

from pathlib import Path
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# global style tweaks (mimic 17_ script)
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def plot_cumulative_pnl(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["cum_net_pnl"], label="net", linewidth=1)
    ax.plot(df.index, df["cum_gross_pnl"], label="gross", linewidth=1)
    ax.set_title("Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_daily_histogram(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = df["net_pnl"].dropna().to_numpy(dtype=float)
    ax.hist(vals, bins=50, edgecolor="black")
    ax.set_title("Histogram of daily net PnL")
    ax.set_xlabel("net_pnl")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[xcol], df[ycol], alpha=0.6, s=20)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{ycol} vs {xcol}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)



def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with open(summary_path, "r") as f:
        return json.load(f)



def main(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    backtest_csv = results_dir / "daily_backtest.csv"
    cum_csv = results_dir / "cumulative_pnl.csv"
    rebalance_csv = results_dir / "rebalance_history.csv"
    summary_json = results_dir / "summary.json"

    if not backtest_csv.exists():
        raise FileNotFoundError(f"Missing backtest file: {backtest_csv}")

    df = pd.read_csv(backtest_csv, parse_dates=["date"]).set_index("date")
    cum = pd.read_csv(cum_csv, parse_dates=["date"]).set_index("date") if cum_csv.exists() else None
    reb = pd.read_csv(rebalance_csv) if rebalance_csv.exists() else None
    summary = load_summary(summary_json)

    graph_dir = results_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # cumulative pnl (use provided file if available, else compute)
    if cum is not None and "cum_net_pnl" in cum.columns:
        plot_cumulative_pnl(cum, graph_dir / "cumulative_pnl.png")
    else:
        run = df.copy()
        run["cum_net_pnl"] = run["net_pnl"].cumsum()
        run["cum_gross_pnl"] = run["gross_pnl"].cumsum()
        plot_cumulative_pnl(run, graph_dir / "cumulative_pnl.png")

    plot_daily_histogram(df, graph_dir / "hist_net_pnl.png")

    # relationships
    plot_scatter(df, "gross_exposure", "net_pnl", graph_dir / "netpnl_vs_exposure.png")
    plot_scatter(df, "turnover", "net_pnl", graph_dir / "netpnl_vs_turnover.png")

    # if rebalance info present, maybe plot cluster changes over time
    if reb is not None and "rebalance_date" in reb.columns:
        # create a simple bar chart of active days per block_score quantile
        reb["rebalance_date"] = pd.to_datetime(reb["rebalance_date"])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(reb["rebalance_date"], reb["block_score"], marker="o", linestyle="-", markersize=3)
        ax.set_title("Block score over time")
        ax.set_xlabel("rebalance date")
        ax.set_ylabel("block_score")
        fig.tight_layout()
        fig.savefig(graph_dir / "block_score_time.png", dpi=200)
        plt.close(fig)

    # write summary back out if present
    if summary:
        with open(graph_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved graphs to {graph_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for 18_dhillon_bipartite results")
    parser.add_argument("--results", default="results/18_dhillon_bipartite", help="Directory containing 18_ results")
    args = parser.parse_args()
    main(Path(args.results))
