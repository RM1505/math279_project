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

    vabs = np.nanmax(np.abs(vals))
    if np.isnan(vabs) or vabs == 0:
        vabs = 1.0

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vabs, vmax=vabs)

    ax.set_title(title)
    ax.set_xlabel("tstat_min")
    ax.set_ylabel("min_count")

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels([str(x) for x in pivot_df.columns])

    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels([str(x) for x in pivot_df.index])

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            text = "nan" if np.isnan(v) else format(v, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="black" if abs(v) < 0.6 * vabs else "white")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Could not find summary file: {SUMMARY_CSV}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)

    # ── Check which sweep dimensions are present ──
    if "tstat_min" not in df.columns:
        raise ValueError("summary.csv does not have 'tstat_min' column. "
                         "Re-run script 13 with the updated version.")

    df = df.sort_values(["min_count", "tstat_min"]).reset_index(drop=True)

    # ── Heatmaps (rows=min_count, cols=tstat_min) ─────────────
    metrics_to_plot = [
        ("net_sharpe",             "Net Sharpe",                      ".3f"),
        ("gross_sharpe",           "Gross Sharpe",                    ".3f"),
        ("avg_kept_edges",         "Average Kept Edges",              ".1f"),
        ("frac_nonzero_windows",   "Fraction of Windows with ≥1 Edge",".2f"),
        ("gross_nonzero_fraction", "Gross Nonzero PnL Fraction",      ".3f"),
        ("avg_direction_gap",      "Average Direction Gap",           ".2f"),
        ("turnover_mean",          "Mean Turnover",                   ".3f"),
    ]

    print("Generating heatmaps...")
    for metric, title, fmt in metrics_to_plot:
        if metric not in df.columns:
            print(f"  skipping {metric} (not in CSV)")
            continue
        pivot = df.pivot(index="min_count", columns="tstat_min", values=metric)
        make_heatmap(pivot, title, GRAPH_DIR / f"{metric}_heatmap.png", fmt=fmt)

    # ── Scatter: net Sharpe vs avg_kept_edges ──────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        df["avg_kept_edges"], df["net_sharpe"],
        c=df["tstat_min"], cmap="viridis", s=80, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="tstat_min")
    for _, row in df.iterrows():
        ax.annotate(
            f"t={row['tstat_min']},m={int(row['min_count'])}",
            (row["avg_kept_edges"], row["net_sharpe"]),
            fontsize=7, alpha=0.8,
        )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Average kept edges per window")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Net Sharpe vs Average Kept Edges (coloured by tstat_min)")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "net_sharpe_vs_avg_kept_edges.png", dpi=200)
    plt.close(fig)
    print("  Saved net_sharpe_vs_avg_kept_edges.png")

    # ── Cumulative PnL for top-5 runs ─────────────────────────
    top5 = df.sort_values("net_sharpe", ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(12, 5))
    for _, row in top5.iterrows():
        pnl_path = RESULTS_DIR / row["run_name"] / "pnl_gross.npy"
        if not pnl_path.exists():
            continue
        pnl = np.load(pnl_path)
        cum = np.nancumsum(np.where(np.isfinite(pnl), pnl, 0.0))
        ax.plot(cum, label=f"{row['run_name']} (SR={row['net_sharpe']:.3f})", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Cumulative Gross PnL — Top 5 Runs")
    ax.set_xlabel("Time (index)")
    ax.set_ylabel("Cumulative PnL")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "top5_cumulative_pnl.png", dpi=200)
    plt.close(fig)
    print("  Saved top5_cumulative_pnl.png")

    # ── Best runs table ────────────────────────────────────────
    best = df.sort_values("net_sharpe", ascending=False).head(10)
    best.to_csv(GRAPH_DIR / "top_10_runs_by_net_sharpe.csv", index=False)
    print("  Saved top_10_runs_by_net_sharpe.csv")

    # ── Print summary ──────────────────────────────────────────
    print(f"\nSaved all graphs to: {GRAPH_DIR}")
    print("\nTop 5 by net Sharpe:")
    for _, row in best.head(5).iterrows():
        print(f"  {row['run_name']}: net_sharpe={row['net_sharpe']:.4f}, "
              f"avg_kept_edges={row['avg_kept_edges']:.1f}, "
              f"frac_nonzero_windows={row.get('frac_nonzero_windows', float('nan')):.2f}")


if __name__ == "__main__":
    main()
