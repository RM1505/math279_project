from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global font tuning (keeps titles from being gigantic)
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results/16_bicluster_block_factor_generalrun"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
GRAPH_DIR = RESULTS_DIR / "graphs"

print("RUNNING GRAPH SCRIPT")


# -----------------------------
# Plot helpers
# -----------------------------

def two_line_title(title_prefix: str, title_suffix: str) -> str:
    """
    Force a two-line title:
      line 1 = title_prefix
      line 2 = title_suffix (parameter stuff)
    """
    title_suffix = title_suffix.strip()
    if not title_suffix:
        return title_prefix
    return f"{title_prefix}\n{title_suffix}"


def make_heatmap(
    pivot_df: pd.DataFrame,
    title_prefix: str,
    title_suffix: str,
    outpath: Path,
    fmt: str = ".3f",
):
    vals = pivot_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vals, aspect="auto")

    ax.set_title(two_line_title(title_prefix, title_suffix), pad=10)
    ax.set_xlabel("q")
    ax.set_ylabel("min_count")

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels([str(x) for x in pivot_df.columns])

    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels([str(x) for x in pivot_df.index])

    # annotate cells
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, format(vals[i, j], fmt), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def grouped_best_heatmap(
    df: pd.DataFrame,
    *,
    value_col: str,
    by_cols: list[str],
    outname_prefix: str,
    title_prefix: str,
    agg: str = "max",
    fmt: str = ".3f",
):
    """
    For each (selection_mode, factor_mode, rebalance_every), build a heatmap of value_col
    across (min_count x q).

    The title is forced to be two lines.
    """
    grouped = df.groupby(by_cols + ["min_count", "q"], as_index=False)[value_col]

    if agg == "max":
        grouped = grouped.max()
    elif agg == "mean":
        grouped = grouped.mean()
    else:
        raise ValueError("agg must be 'max' or 'mean'")

    for keys, sub in grouped.groupby(by_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        label = "__".join(f"{col}_{val}" for col, val in zip(by_cols, keys))
        title_suffix = ", ".join(f"{col}={val}" for col, val in zip(by_cols, keys))

        pivot = sub.pivot(index="min_count", columns="q", values=value_col)
        pivot = pivot.sort_index().sort_index(axis=1)

        outpath = GRAPH_DIR / f"{outname_prefix}__{label}.png"
        make_heatmap(
            pivot,
            title_prefix=title_prefix,
            title_suffix=title_suffix,
            outpath=outpath,
            fmt=fmt,
        )


def make_scatter(df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Let matplotlib choose colors (don’t over-style)
    for selection_mode, sub in df.groupby("selection_mode"):
        ax.scatter(
            sub["active_fraction"],
            sub["net_sharpe"],
            label=selection_mode,
            alpha=0.75,
            s=40,
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Active fraction")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Net Sharpe vs Active Fraction", pad=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------
# Tables
# -----------------------------

def make_top_tables(df: pd.DataFrame):
    (df.sort_values("net_sharpe", ascending=False).head(20)
       .to_csv(GRAPH_DIR / "top_20_by_net_sharpe.csv", index=False))

    (df.sort_values("active_fraction", ascending=False).head(20)
       .to_csv(GRAPH_DIR / "top_20_by_active_fraction.csv", index=False))

    (df.sort_values("net_active_only_sharpe", ascending=False).head(20)
       .to_csv(GRAPH_DIR / "top_20_by_net_active_only_sharpe.csv", index=False))


def make_mode_summary_table(df: pd.DataFrame):
    summary = (
        df.groupby(["selection_mode", "factor_mode", "rebalance_every"], as_index=False)
          .agg(
              mean_net_sharpe=("net_sharpe", "mean"),
              best_net_sharpe=("net_sharpe", "max"),
              mean_active_fraction=("active_fraction", "mean"),
              best_active_fraction=("active_fraction", "max"),
              mean_net_active_only_sharpe=("net_active_only_sharpe", "mean"),
              best_net_active_only_sharpe=("net_active_only_sharpe", "max"),
          )
          .sort_values(["best_net_sharpe", "best_active_fraction"], ascending=False)
    )
    summary.to_csv(GRAPH_DIR / "mode_summary.csv", index=False)


# -----------------------------
# Main
# -----------------------------

def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Could not find summary file: {SUMMARY_CSV}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)
    df = df.sort_values(
        ["selection_mode", "factor_mode", "rebalance_every", "min_count", "q"]
    ).reset_index(drop=True)

    # Heatmaps: your 3 core diagnostics
    grouped_best_heatmap(
        df,
        value_col="net_sharpe",
        by_cols=["selection_mode", "factor_mode", "rebalance_every"],
        outname_prefix="best_net_sharpe_heatmap",
        title_prefix="Net Sharpe",
        agg="max",
        fmt=".3f",
    )

    grouped_best_heatmap(
        df,
        value_col="active_fraction",
        by_cols=["selection_mode", "factor_mode", "rebalance_every"],
        outname_prefix="best_active_fraction_heatmap",
        title_prefix="Active Fraction",
        agg="max",
        fmt=".3f",
    )

    grouped_best_heatmap(
        df,
        value_col="net_active_only_sharpe",
        by_cols=["selection_mode", "factor_mode", "rebalance_every"],
        outname_prefix="best_net_active_only_sharpe_heatmap",
        title_prefix="Active-Only Net Sharpe",
        agg="max",
        fmt=".3f",
    )

    # One scatter: tradeoff plot
    make_scatter(df, GRAPH_DIR / "net_sharpe_vs_active_fraction.png")

    # Tables: quick “best configs” references
    make_top_tables(df)
    make_mode_summary_table(df)

    print(f"Saved graphs and tables to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()