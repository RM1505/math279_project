from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


# =========================================================
# Default settings for the 25-run sweep
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

R_PATH = PROJECT_ROOT / "data/processed/R.npy"
P_PATH = PROJECT_ROOT / "data/processed/P.npy"
BASELINE_FILE = Path(__file__).resolve().with_name("12_baseline_directed_source_target.py")

OUTDIR = PROJECT_ROOT / "results/13_directed_source_target_generalrun"

# 5 x 5 = 25 runs
Q_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
MIN_COUNT_GRID = [40, 60, 80, 100, 150]

# Keep these fixed for now
LAG = 1
KS = 25
KT = 25
WEIGHT_MODE = "abs_tstat"
TRAIN_LEN = 1260
TEST_LEN = 63
STEP = 21            # more dynamic than 63
TC_BPS = 1.0
PERIODS_PER_YEAR = 252
DOLLAR_NEUTRAL = True
REBALANCE_EVERY = 1
DTYPE = np.float64


# =========================================================
# Load functions from 12_baseline_directed_source_target.py
# =========================================================

def load_baseline_module():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"Could not find baseline file: {BASELINE_FILE}")

    spec = importlib.util.spec_from_file_location("baseline12", BASELINE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {BASELINE_FILE}")

    module = importlib.util.module_from_spec(spec)

    # Important for dataclass / annotations on Python 3.13
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module


# =========================================================
# Small helpers
# =========================================================

def nonzero_fraction(x: np.ndarray, eps: float = 1e-12) -> float:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(vals) > eps))


def run_name(q: float, min_count: int) -> str:
    q_str = f"{q:.2f}".replace(".", "p")
    return f"q_{q_str}_mincount_{min_count}"


# =========================================================
# Main sweep
# =========================================================

def main():
    base = load_baseline_module()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading data once...")
    R = base.load_matrix(R_PATH)
    P = base.load_matrix(P_PATH)

    if R.shape != P.shape:
        raise ValueError(f"R and P shape mismatch: R={R.shape}, P={P.shape}")

    T, N = R.shape
    print(f"R,P shape = {(T, N)}")

    windows = base.make_walkforward_windows(
        T,
        train_len=TRAIN_LEN,
        test_len=TEST_LEN,
        step=STEP,
        lag=LAG,
        start=0,
        end=None,
    )
    if not windows:
        raise ValueError("No walk-forward windows created.")

    print(f"Created {len(windows)} windows")
    print(f"Running {len(Q_GRID) * len(MIN_COUNT_GRID)} total configurations...")

    summary_rows = []
    summary_json = []

    total_runs = len(Q_GRID) * len(MIN_COUNT_GRID)
    run_idx = 0

    for q in Q_GRID:
        for min_count in MIN_COUNT_GRID:
            run_idx += 1
            name = run_name(q, min_count)
            run_outdir = OUTDIR / name
            run_outdir.mkdir(parents=True, exist_ok=True)

            print(f"[{run_idx}/{total_runs}] Running {name} ...")

            result, window_reports = base.run_walkforward(
                R,
                P,
                windows,
                lag=LAG,
                ks=KS,
                kt=KT,
                q=q,
                min_count=min_count,
                weight_mode=WEIGHT_MODE,
                tc_bps=TC_BPS,
                periods_per_year=PERIODS_PER_YEAR,
                dollar_neutral=DOLLAR_NEUTRAL,
                rebalance_every=REBALANCE_EVERY,
                dtype=DTYPE,
            )

            # Save per-run arrays
            base.save_npy(run_outdir / "pnl_gross.npy", result.pnl_gross)
            base.save_npy(run_outdir / "pnl_net.npy", result.pnl_net)
            base.save_npy(run_outdir / "turnover.npy", result.turnover)
            base.save_npy(run_outdir / "active.npy", result.active.astype(np.uint8))

            # Save per-run reports
            (run_outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
            (run_outdir / "window_reports.json").write_text(json.dumps(window_reports, indent=2))

            # Aggregate diagnostics
            avg_kept_edges = float(np.mean([w["n_kept_edges"] for w in window_reports])) if window_reports else 0.0
            avg_strategy_edges = float(np.mean([w["n_strategy_edges"] for w in window_reports])) if window_reports else 0.0
            avg_direction_gap = float(np.mean([w["direction_gap"] for w in window_reports])) if window_reports else 0.0
            avg_forward_mass = float(np.mean([w["forward_mass"] for w in window_reports])) if window_reports else 0.0
            avg_reverse_mass = float(np.mean([w["reverse_mass"] for w in window_reports])) if window_reports else 0.0

            gross_nonzero_frac = nonzero_fraction(result.pnl_gross)
            net_nonzero_frac = nonzero_fraction(result.pnl_net)

            row = {
                "run_name": name,
                "q": q,
                "min_count": min_count,
                "lag": LAG,
                "ks": KS,
                "kt": KT,
                "weight_mode": WEIGHT_MODE,
                "train_len": TRAIN_LEN,
                "test_len": TEST_LEN,
                "step": STEP,
                "tc_bps": TC_BPS,
                "rebalance_every": REBALANCE_EVERY,
                "dollar_neutral": int(DOLLAR_NEUTRAL),

                "active_obs": result.metrics["active_obs"],

                "gross_sharpe": result.metrics["gross"]["sharpe"],
                "gross_ann_mean": result.metrics["gross"]["ann_mean"],
                "gross_ann_vol": result.metrics["gross"]["ann_vol"],
                "gross_cum_pnl": result.metrics["gross"]["cum_pnl"],
                "gross_hit_rate": result.metrics["gross"]["hit_rate"],
                "gross_max_drawdown": result.metrics["gross"]["max_drawdown"],
                "gross_nonzero_fraction": gross_nonzero_frac,

                "net_sharpe": result.metrics["net"]["sharpe"],
                "net_ann_mean": result.metrics["net"]["ann_mean"],
                "net_ann_vol": result.metrics["net"]["ann_vol"],
                "net_cum_pnl": result.metrics["net"]["cum_pnl"],
                "net_hit_rate": result.metrics["net"]["hit_rate"],
                "net_max_drawdown": result.metrics["net"]["max_drawdown"],
                "net_nonzero_fraction": net_nonzero_frac,

                "turnover_mean": result.metrics["turnover"]["mean"],
                "turnover_median": result.metrics["turnover"]["median"],

                "avg_kept_edges": avg_kept_edges,
                "avg_strategy_edges": avg_strategy_edges,
                "avg_forward_mass": avg_forward_mass,
                "avg_reverse_mass": avg_reverse_mass,
                "avg_direction_gap": avg_direction_gap,
            }

            summary_rows.append(row)
            summary_json.append(row)

            print(
                f"    net_sharpe={row['net_sharpe']:.4f}, "
                f"gross_sharpe={row['gross_sharpe']:.4f}, "
                f"avg_kept_edges={row['avg_kept_edges']:.1f}, "
                f"gross_nonzero_fraction={row['gross_nonzero_fraction']:.4f}"
            )

    # Save master CSV
    csv_path = OUTDIR / "summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["run_name"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save master JSON
    (OUTDIR / "summary.json").write_text(json.dumps(summary_json, indent=2))

    # Save best runs by net Sharpe
    best_by_net = sorted(summary_rows, key=lambda x: x["net_sharpe"], reverse=True)
    with (OUTDIR / "best_runs_by_net_sharpe.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_by_net[:10])

    print(f"\nDone. Saved all results to: {OUTDIR}")
    print(f"Master summary: {csv_path}")


if __name__ == "__main__":
    main()