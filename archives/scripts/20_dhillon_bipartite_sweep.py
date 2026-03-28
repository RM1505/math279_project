#!/usr/bin/env python3
"""
20_dhillon_bipartite_sweep.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results/20_dhillon_bipartite"

MODULE_FILE = PROJECT_ROOT / "scripts/18_dhillon_bipartite_daily.py"
SIGNALS_PATH = PROJECT_ROOT / "data/processed/signal_matrix_unbalanced.csv"
RETURNS_PATH = PROJECT_ROOT / "data/processed/return_matrix_unbalanced.csv"


def load_module():
    if not MODULE_FILE.exists():
        raise FileNotFoundError(f"Could not find module: {MODULE_FILE}")
    spec = importlib.util.spec_from_file_location("mod18_daily", MODULE_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_cumulative(bt: pd.DataFrame, outdir: Path) -> None:
    cum = pd.DataFrame(index=bt.index)
    cum["cum_gross_pnl"] = bt["gross_pnl"].cumsum()
    cum["cum_net_pnl"] = bt["net_pnl"].cumsum()
    cum.to_csv(outdir / "cumulative_pnl.csv")


def run_one(module, signals, returns, params: dict[str, Any], outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        module.TRAIN_DAYS = int(params["train_days"])
        module.N_CLUSTERS = int(params["n_clusters"])
        module.FDR_Q = float(params["fdr_q"])
        module.MIN_COUNT = int(params["min_count"])
        module.CLUSTER_WEIGHT_STAT = params["cluster_weight_stat"]
        module.BLOCK_SCORE_STAT = params["block_score_stat"]
        module.COST_BPS = float(params.get("cost_bps", 5.0))
        module.ALLOW_SAME_CLUSTER_PAIR = bool(params.get("allow_same_cluster_pair", True))
        module.RANDOM_STATE = int(params.get("random_state", 0))

        bt, bh, summary = module.run_daily_walk_forward(signals, returns)

        bt.to_csv(outdir / "daily_backtest.csv")
        bh.to_csv(outdir / "daily_block_history.csv", index=False)
        save_cumulative(bt, outdir)

        with open(outdir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        canonical = {
            "gross_pnl_mean": summary.get("gross_pnl_mean"),
            "gross_pnl_sharpe": summary.get("gross_pnl_sharpe"),
            "net_pnl_mean": summary.get("net_pnl_mean"),
            "net_pnl_sharpe": summary.get("net_pnl_sharpe"),
            "active_fraction": summary.get("active_fraction"),
            "avg_turnover": summary.get("avg_turnover"),
            "avg_gross_exposure": summary.get("avg_gross_exposure"),
            "active_days": summary.get("active_days"),
            "total_days": summary.get("total_days"),
        }

        out = {**canonical}
        out.update({k: params[k] for k in ["fdr_q", "min_count", "n_clusters", "train_days"]})
        out["outdir"] = str(outdir)
        out["error"] = ""
        return out

    except Exception as e:
        return {"error": str(e), **params, "outdir": str(outdir)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(DEFAULT_RESULTS))
    p.add_argument("--q", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--min-count", nargs="*", type=int, default=[40, 100, 120])
    p.add_argument("--n-clusters", nargs="*", type=int, default=[2, 4, 6])
    p.add_argument("--train-days", nargs="*", type=int, default=[252])
    p.add_argument("--cluster-weight-stat", default="abs_t")
    p.add_argument("--block-score-stat", default="sr")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--random-state", type=int, default=0)
    args = p.parse_args()

    results_dir = Path(args.outdir)
    results_dir.mkdir(parents=True, exist_ok=True)

    module = load_module()

    signals = module.read_matrix(SIGNALS_PATH)
    returns = module.read_matrix(RETURNS_PATH)
    signals, returns = module.align_signal_and_returns(signals, returns)

    records: list[dict[str, Any]] = []

    for q in args.q:
        for min_count in args.min_count:
            for n_clusters in args.n_clusters:
                for train_days in args.train_days:
                    run_name = f"q_{q:.2f}_mincount_{min_count}_k_{n_clusters}_train_{train_days}"
                    outdir = results_dir / run_name
                    params = {
                        "fdr_q": float(q),
                        "min_count": int(min_count),
                        "n_clusters": int(n_clusters),
                        "train_days": int(train_days),
                        "cluster_weight_stat": args.cluster_weight_stat,
                        "block_score_stat": args.block_score_stat,
                        "cost_bps": args.cost_bps,
                        "random_state": args.random_state,
                    }
                    print("Running:", run_name)
                    rec = run_one(module, signals, returns, params, outdir)
                    rec["run_name"] = run_name
                    records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(results_dir / "summary.csv", index=False)
    print("Sweep complete. Summary saved to:", results_dir / "summary.csv")


if __name__ == "__main__":
    main()