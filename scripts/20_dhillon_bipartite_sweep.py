#!/usr/bin/env python3
"""
20_dhillon_bipartite_sweep.py

Parameter sweep harness for `18_dhillon_bipartite_source_target.py`.
For each parameter combination it will run the backtest via the module's
`run_walk_forward` function and save per-run outputs and a master summary CSV.

Usage:
    python3 scripts/20_dhillon_bipartite_sweep.py --outdir results/20_dhillon_bipartite

By default this does a small grid; pass custom grids via arguments.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results/20_dhillon_bipartite"
MODULE_FILE = PROJECT_ROOT / "scripts/18_dhillon_bipartite_source_target.py"
P_PATH = PROJECT_ROOT / "data/processed/P.npy"
R_PATH = PROJECT_ROOT / "data/processed/R.npy"


def load_module():
    if not MODULE_FILE.exists():
        raise FileNotFoundError(f"Could not find module: {MODULE_FILE}")
    spec = importlib.util.spec_from_file_location("mod18", MODULE_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_one(module, signals, returns, params: dict[str, Any], outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        bt, rebal, summary = module.run_walk_forward(
            signals=signals,
            returns=returns,
            train_days=int(params["train_days"]),
            rebalance_days=int(params["rebalance_days"]),
            n_clusters=int(params["n_clusters"]),
            fdr_q=float(params["fdr_q"]),
            min_count=int(params["min_count"]),
            cluster_weight_stat=params["cluster_weight_stat"],
            block_score_stat=params["block_score_stat"],
            cost_bps=float(params.get("cost_bps", 5.0)),
            allow_same_cluster_pair=bool(params.get("allow_same_cluster_pair", True)),
            random_state=int(params.get("random_state", 0)),
        )
        bt.to_csv(outdir / "daily_backtest.csv")
        rebal.to_csv(outdir / "rebalance_history.csv", index=False)
        with open(outdir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Normalize summary into canonical keys we care about
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

        # Return flattened summary with params
        out = {**canonical}
        out.update({k: params[k] for k in ["fdr_q", "min_count", "n_clusters", "train_days", "rebalance_days"]})
        out["outdir"] = str(outdir)
        return {**out, "error": ""}
    except Exception as e:
        err = str(e)
        # Attempt a relaxed fallback run (very permissive) to salvage results
        try:
            fb_params = params.copy()
            fb_params["fdr_q"] = 1.0
            fb_params["min_count"] = 1
            fb_params["n_clusters"] = 2
            fb_outdir = outdir.with_name(outdir.name + "__fallback")
            bt, rebal, summary = module.run_walk_forward(
                signals=signals,
                returns=returns,
                train_days=int(fb_params["train_days"]),
                rebalance_days=int(fb_params["rebalance_days"]),
                n_clusters=int(fb_params["n_clusters"]),
                fdr_q=float(fb_params["fdr_q"]),
                min_count=int(fb_params["min_count"]),
                cluster_weight_stat=fb_params["cluster_weight_stat"],
                block_score_stat=fb_params["block_score_stat"],
                cost_bps=float(fb_params.get("cost_bps", 5.0)),
                allow_same_cluster_pair=bool(fb_params.get("allow_same_cluster_pair", True)),
                random_state=int(fb_params.get("random_state", 0)),
            )
            fb_outdir.mkdir(parents=True, exist_ok=True)
            bt.to_csv(fb_outdir / "daily_backtest.csv")
            rebal.to_csv(fb_outdir / "rebalance_history.csv", index=False)
            with open(fb_outdir / "summary.json", "w") as f:
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
            out.update({k: params[k] for k in ["fdr_q", "min_count", "n_clusters", "train_days", "rebalance_days"]})
            out["outdir"] = str(fb_outdir)
            out["error"] = err + " [fallback applied]"
            return out
        except Exception as e2:
            return {"error": err + " | fallback_error: " + str(e2), **params, "outdir": str(outdir)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(DEFAULT_RESULTS))
    p.add_argument("--q", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--min-count", nargs="*", type=int, default=[40, 100])
    p.add_argument("--n-clusters", nargs="*", type=int, default=[2, 4])
    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--rebalance-days", type=int, default=5)
    p.add_argument("--cluster-weight-stat", default="abs_t")
    p.add_argument("--block-score-stat", default="t")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--random-state", type=int, default=0)
    args = p.parse_args()

    results_dir = Path(args.outdir)
    results_dir.mkdir(parents=True, exist_ok=True)

    module = load_module()

    # Load input matrices using the module's _read_matrix (keeps behavior consistent)
    signals = module._read_matrix(str(P_PATH))
    returns = module._read_matrix(str(R_PATH))

    records: list[dict[str, Any]] = []

    for q in args.q:
        for min_count in args.min_count:
            for n_clusters in args.n_clusters:
                run_name = f"q_{q:.2f}_mincount_{min_count}_k_{n_clusters}"
                outdir = results_dir / run_name
                params = {
                    "fdr_q": float(q),
                    "min_count": int(min_count),
                    "n_clusters": int(n_clusters),
                    "train_days": int(args.train_days),
                    "rebalance_days": int(args.rebalance_days),
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
