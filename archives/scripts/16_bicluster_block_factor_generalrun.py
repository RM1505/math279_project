from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


# =========================================================
# Default settings for the block-factor sweep
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

R_PATH = PROJECT_ROOT / "data/processed/R.npy"
P_PATH = PROJECT_ROOT / "data/processed/P.npy"
MODEL_FILE = Path(__file__).resolve().with_name("15_bicluster_block_factor_source_target.py")

OUTDIR = PROJECT_ROOT / "results/16_bicluster_block_factor_generalrun"

# Main grids for a compact first-pass sweep.
# Total runs = 4 q values x 4 min_count values x 2 selection modes x 2 factor modes x 2 rebalance settings = 64.
Q_GRID = [0.05, 0.10, 0.15, 0.20]
MIN_COUNT_GRID = [40, 60, 100, 150]
SELECTION_MODE_GRID = ["abs_tstat", "abs_A"]
FACTOR_MODE_GRID = ["signed_svd", "signed_rowcol_sums"]
REBALANCE_EVERY_GRID = [1, 5]

# Keep these fixed initially
LAG = 1
KS = 25
KT = 25
WINSOR_Q = 0.99
SCALE_ROWS_COLS = True
INIT = "degree"
MAX_ITER = 100
TOL = 1e-8
TRAIN_LEN = 1260
TEST_LEN = 63
STEP = 21
TC_BPS = 1.0
PERIODS_PER_YEAR = 252
DOLLAR_NEUTRAL = True
DTYPE = np.float64


# =========================================================
# Load model module
# =========================================================

def load_model_module():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Could not find model file: {MODEL_FILE}")

    spec = importlib.util.spec_from_file_location("blockfactor15", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {MODEL_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =========================================================
# Helpers
# =========================================================

def nonzero_fraction(x: np.ndarray, eps: float = 1e-12) -> float:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(vals) > eps))


def run_name(selection_mode: str, factor_mode: str, q: float, min_count: int, rebalance_every: int) -> str:
    q_str = f"{q:.2f}".replace(".", "p")
    return (
        f"sel_{selection_mode}__fac_{factor_mode}__"
        f"q_{q_str}__mincount_{min_count}__reb_{rebalance_every}"
    )


def jaccard(a: list[int], b: list[int]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa and sb:
        return 0.0
    if sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def average_window_stability(window_reports: list[dict], key: str) -> float:
    if len(window_reports) <= 1:
        return 0.0
    vals = []
    for prev, cur in zip(window_reports[:-1], window_reports[1:]):
        vals.append(jaccard(prev.get(key, []), cur.get(key, [])))
    return float(np.mean(vals)) if vals else 0.0


# =========================================================
# Main sweep
# =========================================================

def main():
    model = load_model_module()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading data once...")
    R = model.load_matrix(R_PATH)
    P = model.load_matrix(P_PATH)

    if R.shape != P.shape:
        raise ValueError(f"R and P shape mismatch: R={R.shape}, P={P.shape}")

    T, N = R.shape
    print(f"R,P shape = {(T, N)}")

    windows = model.make_walkforward_windows(
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

    total_runs = (
        len(Q_GRID)
        * len(MIN_COUNT_GRID)
        * len(SELECTION_MODE_GRID)
        * len(FACTOR_MODE_GRID)
        * len(REBALANCE_EVERY_GRID)
    )
    print(f"Running {total_runs} total configurations...")

    summary_rows = []
    run_idx = 0

    for selection_mode in SELECTION_MODE_GRID:
        for factor_mode in FACTOR_MODE_GRID:
            for rebalance_every in REBALANCE_EVERY_GRID:
                for q in Q_GRID:
                    for min_count in MIN_COUNT_GRID:
                        run_idx += 1
                        name = run_name(selection_mode, factor_mode, q, min_count, rebalance_every)
                        run_outdir = OUTDIR / name
                        run_outdir.mkdir(parents=True, exist_ok=True)

                        print(f"[{run_idx}/{total_runs}] Running {name} ...")

                        result, window_reports = model.run_walkforward(
                            R,
                            P,
                            windows,
                            lag=LAG,
                            ks=KS,
                            kt=KT,
                            q=q,
                            min_count=min_count,
                            selection_mode=selection_mode,
                            winsor_q=WINSOR_Q,
                            scale_rows_cols=SCALE_ROWS_COLS,
                            init=INIT,
                            factor_mode=factor_mode,
                            tc_bps=TC_BPS,
                            periods_per_year=PERIODS_PER_YEAR,
                            dollar_neutral=DOLLAR_NEUTRAL,
                            rebalance_every=rebalance_every,
                            max_iter=MAX_ITER,
                            tol=TOL,
                            dtype=DTYPE,
                        )

                        model.save_npy(run_outdir / "pnl_gross.npy", result.pnl_gross)
                        model.save_npy(run_outdir / "pnl_net.npy", result.pnl_net)
                        model.save_npy(run_outdir / "turnover.npy", result.turnover)
                        model.save_npy(run_outdir / "active.npy", result.active.astype(np.uint8))

                        (run_outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
                        (run_outdir / "window_reports.json").write_text(json.dumps(window_reports, indent=2))

                        avg_kept_edges = float(np.mean([w["n_kept_edges"] for w in window_reports])) if window_reports else 0.0
                        avg_block_edges = float(np.mean([w["n_block_edges"] for w in window_reports])) if window_reports else 0.0
                        avg_direction_gap = float(np.mean([w["direction_gap"] for w in window_reports])) if window_reports else 0.0
                        avg_forward_mass = float(np.mean([w["forward_mass"] for w in window_reports])) if window_reports else 0.0
                        avg_reverse_mass = float(np.mean([w["reverse_mass"] for w in window_reports])) if window_reports else 0.0
                        avg_singular_value = float(np.mean([w["singular_value"] for w in window_reports])) if window_reports else 0.0
                        avg_converged = float(np.mean([w["converged"] for w in window_reports])) if window_reports else 0.0
                        avg_source_stability = average_window_stability(window_reports, "sources")
                        avg_target_stability = average_window_stability(window_reports, "targets")

                        gross_nonzero_frac = nonzero_fraction(result.pnl_gross)
                        net_nonzero_frac = nonzero_fraction(result.pnl_net)

                        row = {
                            "run_name": name,
                            "selection_mode": selection_mode,
                            "factor_mode": factor_mode,
                            "q": q,
                            "min_count": min_count,
                            "rebalance_every": rebalance_every,
                            "lag": LAG,
                            "ks": KS,
                            "kt": KT,
                            "winsor_q": WINSOR_Q,
                            "scale_rows_cols": int(SCALE_ROWS_COLS),
                            "init": INIT,
                            "train_len": TRAIN_LEN,
                            "test_len": TEST_LEN,
                            "step": STEP,
                            "tc_bps": TC_BPS,
                            "dollar_neutral": int(DOLLAR_NEUTRAL),

                            "active_obs": result.metrics["active_obs"],
                            "active_fraction": result.metrics["active_fraction"],

                            "gross_sharpe": result.metrics["gross"]["sharpe"],
                            "gross_ann_mean": result.metrics["gross"]["ann_mean"],
                            "gross_ann_vol": result.metrics["gross"]["ann_vol"],
                            "gross_cum_pnl": result.metrics["gross"]["cum_pnl"],
                            "gross_hit_rate": result.metrics["gross"]["hit_rate"],
                            "gross_max_drawdown": result.metrics["gross"]["max_drawdown"],
                            "gross_active_only_sharpe": result.metrics["gross_active_only"]["sharpe"],
                            "gross_nonzero_fraction": gross_nonzero_frac,

                            "net_sharpe": result.metrics["net"]["sharpe"],
                            "net_ann_mean": result.metrics["net"]["ann_mean"],
                            "net_ann_vol": result.metrics["net"]["ann_vol"],
                            "net_cum_pnl": result.metrics["net"]["cum_pnl"],
                            "net_hit_rate": result.metrics["net"]["hit_rate"],
                            "net_max_drawdown": result.metrics["net"]["max_drawdown"],
                            "net_active_only_sharpe": result.metrics["net_active_only"]["sharpe"],
                            "net_nonzero_fraction": net_nonzero_frac,

                            "turnover_mean": result.metrics["turnover"]["mean"],
                            "turnover_median": result.metrics["turnover"]["median"],

                            "avg_kept_edges": avg_kept_edges,
                            "avg_block_edges": avg_block_edges,
                            "avg_forward_mass": avg_forward_mass,
                            "avg_reverse_mass": avg_reverse_mass,
                            "avg_direction_gap": avg_direction_gap,
                            "avg_singular_value": avg_singular_value,
                            "avg_converged": avg_converged,
                            "avg_source_stability": avg_source_stability,
                            "avg_target_stability": avg_target_stability,
                        }

                        summary_rows.append(row)

                        print(
                            f"    net_sharpe={row['net_sharpe']:.4f}, "
                            f"net_active_only_sharpe={row['net_active_only_sharpe']:.4f}, "
                            f"active_fraction={row['active_fraction']:.4f}, "
                            f"avg_block_edges={row['avg_block_edges']:.1f}"
                        )

    csv_path = OUTDIR / "summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["run_name"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    (OUTDIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))

    best_by_net = sorted(summary_rows, key=lambda x: x["net_sharpe"], reverse=True)
    with (OUTDIR / "best_runs_by_net_sharpe.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_by_net[:20])

    best_by_active = sorted(summary_rows, key=lambda x: x["active_fraction"], reverse=True)
    with (OUTDIR / "best_runs_by_active_fraction.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_by_active[:20])

    print(f"\nDone. Saved all results to: {OUTDIR}")
    print(f"Master summary: {csv_path}")


if __name__ == "__main__":
    main()
