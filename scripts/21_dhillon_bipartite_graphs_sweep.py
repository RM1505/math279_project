#!/usr/bin/env python3
"""
21_dhillon_bipartite_graphs_sweep.py

Analyze outputs from `scripts/20_dhillon_bipartite_sweep.py`.
Creates heatmaps and summary plots across the parameter grid.

Usage:
    python3 scripts/21_dhillon_bipartite_graphs_sweep.py --results results/20_dhillon_bipartite
"""
from __future__ import annotations

import argparse
from pathlib import Path
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


def make_heatmap(pivot: pd.DataFrame, outpath: Path, title: str):
    vals = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vals, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("q")
    ax.set_ylabel("min_count")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(x) for x in pivot.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i,j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main(results_dir: Path):
    results_dir = Path(results_dir)
    summary_csv = results_dir / "summary.csv"
    graph_dir = results_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_csv}")

    df = pd.read_csv(summary_csv)

    # Pivot net_sharpe (if present) or net_pnl_sharpe-like keys
    score_col = None
    for c in ["net_pnl_sharpe", "net_sharpe", "gross_pnl_sharpe", "net_pnl_mean"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        # maybe summary.json fields are nested; attempt to coerce
        possible = [c for c in df.columns if "sharpe" in c]
        score_col = possible[0] if possible else None

    # Build heatmaps grouped by n_clusters
    for k, sub in df.groupby("n_clusters"):
        # pivot index=min_count, columns=fdr q, values=score_col (mean over runs)
        if score_col is None:
            print("No sharpe column found; skipping heatmap")
            break
        pivot = sub.pivot(index="min_count", columns="fdr_q", values=score_col)
        if pivot.empty:
            continue
        pivot = pivot.sort_index().sort_index(axis=1)
        make_heatmap(pivot, graph_dir / f"heatmap_k_{k}.png", title=f"{score_col} (k={k})")

    # Scatter: net_sharpe vs active_fraction if present
    if "active_fraction" in df.columns and score_col in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df["active_fraction"], df[score_col], alpha=0.7)
        ax.set_xlabel("active_fraction")
        ax.set_ylabel(score_col)
        ax.set_title(f"{score_col} vs active_fraction")
        fig.tight_layout()
        fig.savefig(graph_dir / "scatter_active_vs_sharpe.png", dpi=200)
        plt.close(fig)

    # For top 3 runs by score, copy and plot their cumulative_pnl if available
    if score_col in df.columns:
        best = df.sort_values(score_col, ascending=False).head(3)
        for i, row in best.iterrows():
            outdir = Path(row.get("outdir", ""))
            cum_file = outdir + "/cumulative_pnl.csv" if isinstance(outdir, str) else Path(outdir) / "cumulative_pnl.csv"
            if Path(cum_file).exists():
                cum = pd.read_csv(cum_file, parse_dates=["date"]).set_index("date")
                fig, ax = plt.subplots(figsize=(8, 4))
                if "cum_net_pnl" in cum.columns:
                    ax.plot(cum.index, cum["cum_net_pnl"], label="net")
                if "cum_gross_pnl" in cum.columns:
                    ax.plot(cum.index, cum["cum_gross_pnl"], label="gross")
                ax.set_title(f"Top run {row.get('run_name', i)}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(graph_dir / f"top_run_{i}_cum_pnl.png", dpi=200)
                plt.close(fig)

    print(f"Saved sweep graphs to: {graph_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/20_dhillon_bipartite")
    args = parser.parse_args()
    main(Path(args.results))
