from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

R_PATH = PROJECT_ROOT / "data/processed/R.npy"
P_PATH = PROJECT_ROOT / "data/processed/P.npy"
BASELINE_FILE = Path(__file__).resolve().with_name("12_baseline_directed_source_target.py")

OUTDIR = PROJECT_ROOT / "results/13_directed_source_target_generalrun"

# ── universe ──────────────────────────────────────────────
# Subset to the N_TOP assets with the most complete data to reduce the
# multiple-testing burden. With N=514 there are 263K edge tests; at N=150
# that drops to ~22K, making BH 12x less conservative and direct-threshold
# runs much cleaner.
N_TOP = 150

# ── walk-forward ──────────────────────────────────────────
LAG = 1
KS = 25
KT = 25
WEIGHT_MODE = "abs_tstat"
TRAIN_LEN = 1260        # ~5 years of training
TEST_LEN  = 21          # ~1 month test per block (was 63 — keeps model fresh with step=10)
STEP      = 10          # refit every 10 trading days
TC_BPS = 0
PERIODS_PER_YEAR = 252
DOLLAR_NEUTRAL = True
REBALANCE_EVERY = 5
DTYPE = np.float64

# ── sweep parameters ──────────────────────────────────────
# We sweep |t|-threshold × min_count with BH disabled (use_bh=False).
#
# Rationale: BH with 150*149 ≈ 22K tests still requires |t|≈4 for any edge
# to survive at q=0.10. Sweeping tstat_min directly gives transparent
# control over statistical stringency.  Expected false-positives under H0:
#   FP ≈ 22350 × erfc(tstat_min / sqrt(2))
#   tstat_min=3.0 → FP≈ 22350×0.0027 ≈  60
#   tstat_min=3.5 → FP≈ 22350×0.0005 ≈  10
#   tstat_min=4.0 → FP≈ 22350×6e-5  ≈   1
#   tstat_min=4.5 → FP≈ 22350×7e-6  ≈  0.2
#   tstat_min=5.0 → FP≈ 22350×6e-7  ≈  0.01
USE_BH = False
TSTAT_MIN_GRID = [3.0, 3.5, 4.0, 4.5, 5.0]   # binding statistical bar
MIN_COUNT_GRID = [2, 5, 10, 20, 50]           # minimum joint observations per edge
Q = 0.10   # only used if USE_BH = True


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


def run_name(tstat_min: float, min_count: int) -> str:
    t_str = f"{tstat_min:.1f}".replace(".", "p")
    return f"tmin_{t_str}_mincount_{min_count}"


def select_top_assets(R: np.ndarray, P: np.ndarray, n_top: int) -> np.ndarray:
    """
    Return indices of the n_top assets with the most jointly finite (R, P) observations.
    Uses the full dataset for selection — acceptable as a pre-filtering step.
    """
    joint_finite = np.isfinite(R) & np.isfinite(P)   # (T, N)
    completeness = joint_finite.sum(axis=0)           # (N,)
    return np.argsort(-completeness)[:n_top]


# =========================================================
# Main sweep
# =========================================================

def main():
    base = load_baseline_module()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading data once...")
    R_full = base.load_matrix(R_PATH)
    P_full = base.load_matrix(P_PATH)

    if R_full.shape != P_full.shape:
        raise ValueError(f"R and P shape mismatch: R={R_full.shape}, P={P_full.shape}")

    T, N_full = R_full.shape
    print(f"Full R,P shape = {(T, N_full)}")

    # Subset universe
    top_idx = select_top_assets(R_full, P_full, N_TOP)
    R = R_full[:, top_idx]
    P = P_full[:, top_idx]
    print(f"Subsetted to top {N_TOP} assets by data completeness → R,P shape = {R.shape}")
    N = R.shape[1]

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

    print(f"Created {len(windows)} windows (step={STEP}, test_len={TEST_LEN})")
    print(f"Running {len(TSTAT_MIN_GRID) * len(MIN_COUNT_GRID)} total configurations...")
    n_tests = N * (N - 1)
    print(f"Edge tests per window: {n_tests:,} (N={N})")
    for tm in TSTAT_MIN_GRID:
        import math
        fp = n_tests * math.erfc(tm / math.sqrt(2.0))
        print(f"  tstat_min={tm:.1f}: expected false positives under H0 ≈ {fp:.1f}")
    print()

    summary_rows = []
    summary_json = []

    total_runs = len(TSTAT_MIN_GRID) * len(MIN_COUNT_GRID)
    run_idx = 0

    for tstat_min in TSTAT_MIN_GRID:
        for min_count in MIN_COUNT_GRID:
            run_idx += 1
            name = run_name(tstat_min, min_count)
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
                q=Q,
                min_count=min_count,
                weight_mode=WEIGHT_MODE,
                tc_bps=TC_BPS,
                periods_per_year=PERIODS_PER_YEAR,
                dollar_neutral=DOLLAR_NEUTRAL,
                rebalance_every=REBALANCE_EVERY,
                use_bh=USE_BH,
                tstat_min=tstat_min,
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
            avg_kept_edges   = float(np.mean([w["n_kept_edges"]   for w in window_reports])) if window_reports else 0.0
            avg_strat_edges  = float(np.mean([w["n_strategy_edges"] for w in window_reports])) if window_reports else 0.0
            avg_dir_gap      = float(np.mean([w["direction_gap"]  for w in window_reports])) if window_reports else 0.0
            avg_fwd_mass     = float(np.mean([w["forward_mass"]   for w in window_reports])) if window_reports else 0.0
            avg_rev_mass     = float(np.mean([w["reverse_mass"]   for w in window_reports])) if window_reports else 0.0
            frac_nonzero_windows = float(np.mean([w["n_kept_edges"] > 0 for w in window_reports])) if window_reports else 0.0

            gross_nonzero_frac = nonzero_fraction(result.pnl_gross)
            net_nonzero_frac   = nonzero_fraction(result.pnl_net)

            row = {
                "run_name": name,
                "tstat_min": tstat_min,
                "min_count": min_count,
                "use_bh": int(USE_BH),
                "lag": LAG,
                "ks": KS,
                "kt": KT,
                "q": Q,
                "weight_mode": WEIGHT_MODE,
                "train_len": TRAIN_LEN,
                "test_len": TEST_LEN,
                "step": STEP,
                "tc_bps": TC_BPS,
                "rebalance_every": REBALANCE_EVERY,
                "n_assets": N,
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
                "avg_strategy_edges": avg_strat_edges,
                "avg_forward_mass": avg_fwd_mass,
                "avg_reverse_mass": avg_rev_mass,
                "avg_direction_gap": avg_dir_gap,
                "frac_nonzero_windows": frac_nonzero_windows,
            }

            summary_rows.append(row)
            summary_json.append(row)

            print(
                f"    net_sharpe={row['net_sharpe']:.4f}, "
                f"avg_kept_edges={row['avg_kept_edges']:.1f}, "
                f"frac_nonzero_windows={row['frac_nonzero_windows']:.2f}, "
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
    print("\nTop 5 by net Sharpe:")
    for row in best_by_net[:5]:
        print(f"  {row['run_name']}: net_sharpe={row['net_sharpe']:.4f}, "
              f"avg_kept_edges={row['avg_kept_edges']:.1f}, "
              f"frac_nonzero_windows={row['frac_nonzero_windows']:.2f}")


if __name__ == "__main__":
    main()
