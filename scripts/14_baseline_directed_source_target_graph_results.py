from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results/13_directed_source_target_generalrun"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
GRAPH_DIR = RESULTS_DIR / "graphs"


print("RUNNING GRAPH SCRIPT")

def make_heatmap(pivot_df: pd.DataFrame, title: str, outpath: Path, fmt: str = ".3f"):
    vals = pivot_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vals, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("q")
    ax.set_ylabel("min_count")

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels([str(x) for x in pivot_df.columns])

    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels([str(x) for x in pivot_df.index])

    # annotate cells
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            text = format(vals[i, j], fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Could not find summary file: {SUMMARY_CSV}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)

    # Sort cleanly
    df = df.sort_values(["min_count", "q"]).reset_index(drop=True)

    # Heatmaps
    metrics_to_plot = [
        ("net_sharpe", "Net Sharpe"),
        ("gross_sharpe", "Gross Sharpe"),
        ("avg_kept_edges", "Average Kept Edges"),
        ("avg_strategy_edges", "Average Strategy Edges"),
        ("gross_nonzero_fraction", "Gross Nonzero Fraction"),
        ("avg_direction_gap", "Average Direction Gap"),
        ("turnover_mean", "Mean Turnover"),
    ]

    for metric, title in metrics_to_plot:
        pivot = df.pivot(index="min_count", columns="q", values=metric)
        outpath = GRAPH_DIR / f"{metric}_heatmap.png"
        make_heatmap(pivot, title, outpath)

    # Scatter: net Sharpe vs average kept edges
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["avg_kept_edges"], df["net_sharpe"])
    for _, row in df.iterrows():
        ax.annotate(
            f"q={row['q']}, m={int(row['min_count'])}",
            (row["avg_kept_edges"], row["net_sharpe"]),
            fontsize=7,
            alpha=0.8,
        )
    ax.set_xlabel("Average kept edges")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Net Sharpe vs Average Kept Edges")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "net_sharpe_vs_avg_kept_edges.png", dpi=200)
    plt.close(fig)

    # Best runs table
    best = df.sort_values("net_sharpe", ascending=False).head(10)
    best.to_csv(GRAPH_DIR / "top_10_runs_by_net_sharpe.csv", index=False)

    print(f"Saved graphs to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()