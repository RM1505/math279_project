#!/usr/bin/env python3
"""
19_dhillon_bipartite_graphs.py
"""

from __future__ import annotations

from pathlib import Path
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    old_rebalance_csv = results_dir / "rebalance_history.csv"
    new_block_csv = results_dir / "daily_block_history.csv"
    summary_json = results_dir / "summary.json"

    if not backtest_csv.exists():
        raise FileNotFoundError(f"Missing backtest file: {backtest_csv}")

    df = pd.read_csv(backtest_csv, parse_dates=["date"]).set_index("date")
    cum = pd.read_csv(cum_csv, parse_dates=["date"]).set_index("date") if cum_csv.exists() else None

    reb = None
    date_col = None
    if new_block_csv.exists():
        reb = pd.read_csv(new_block_csv)
        date_col = "date"
    elif old_rebalance_csv.exists():
        reb = pd.read_csv(old_rebalance_csv)
        date_col = "rebalance_date"

    summary = load_summary(summary_json)

    graph_dir = results_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    if cum is not None and {"cum_net_pnl", "cum_gross_pnl"}.issubset(cum.columns):
        plot_cumulative_pnl(cum, graph_dir / "cumulative_pnl.png")
    else:
        run = df.copy()
        run["cum_net_pnl"] = run["net_pnl"].cumsum()
        run["cum_gross_pnl"] = run["gross_pnl"].cumsum()
        plot_cumulative_pnl(run, graph_dir / "cumulative_pnl.png")

    plot_daily_histogram(df, graph_dir / "hist_net_pnl.png")
    plot_scatter(df, "gross_exposure", "net_pnl", graph_dir / "netpnl_vs_exposure.png")
    plot_scatter(df, "turnover", "net_pnl", graph_dir / "netpnl_vs_turnover.png")

    if reb is not None and date_col in reb.columns and "block_score" in reb.columns:
        reb[date_col] = pd.to_datetime(reb[date_col])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(reb[date_col], reb["block_score"], marker="o", linestyle="-", markersize=3)
        ax.set_title("Block score over time")
        ax.set_xlabel(date_col)
        ax.set_ylabel("block_score")
        fig.tight_layout()
        fig.savefig(graph_dir / "block_score_time.png", dpi=200)
        plt.close(fig)

    if summary:
        with open(graph_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved graphs to {graph_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for Dhillon bipartite results")
    parser.add_argument("--results", default="data/processed/dhillon_bipartite_daily")
    args = parser.parse_args()
    main(Path(args.results))