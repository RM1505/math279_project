# Cross-Asset OFI Prediction Pipeline

Code for the Math 279 project: predicting residualized equity returns from
cross-sectional intraday order flow imbalance (OFI) using ridge regression,
directed source-target (DST) biclustering, and Dhillon bipartite co-clustering.

Run all scripts from the **repository root** (not from inside `pipeline/`):

```bash
python pipeline/06_ridge_nextday.py
```

---

## Data Requirements

### Option A — Full pipeline from raw LOBSTER data

You need the raw LOBSTER limit order book archives in `data/raw/`, one `.7z`
file per ticker. Scripts 01–03 process these into the feature table.

### Option B — Start from the processed feature table (recommended)

If you have `data/processed/feature_table_with_residuals_10level.csv`, you can
skip scripts 01–03 and run the analysis directly from script 04 onward.

The feature table is a CSV with columns:
- `date`, `ticker`, `sector`, `residual_ret` — date, ticker ID, GICS sector, beta-residualized intraday return
- `minute_1` … `minute_390` — per-minute integrated OFI (IOFI, first PC across LOB levels) for each trading minute

Scripts 07, 08, 10 (overnight and close-to-close) additionally download return
data automatically via `yfinance` on first run and cache to `data/processed/`.

### Dependencies

Install the project package (needed for the `ofi` module used in script 01):

```bash
pip install -e .
```

Core dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`,
`tqdm`, `yfinance`, `py7zr` (for LOBSTER extraction, script 01 only).

---

## Script Reference

### Data Pipeline (scripts 01–03)

| Script | Input | Output |
|--------|-------|--------|
| `01_lobster_to_integrated_ofi.py` | `data/raw/*.7z` | `data/processed/integrated_ofi/{ticker}.csv` |
| `02_build_feature_matrix.py` | `data/processed/integrated_ofi/` | `data/processed/feature_table.csv` |
| `03_residualize_returns.py` | `data/processed/feature_table.csv` | `data/processed/feature_table_with_residuals_10level.csv` |

**01** — Extracts each LOBSTER archive, computes per-level OFI at each tick,
runs PCA across the 10 LOB levels to produce a scalar Integrated OFI (IOFI)
per minute, and writes one CSV per ticker.

**02** — Pivots the per-ticker minute-level IOFI into a wide feature table:
one row per (ticker, date), columns `minute_1`…`minute_T`.

**03** — Adds beta-residualized open-to-close returns by regressing on SPY
in a 60-day rolling window. Produces the main input for all analysis scripts.

---

### Analysis Scripts (scripts 04–12)

All analysis scripts use a **walk-forward** framework: 750-day training window,
refit every 21 days. The exp-decay OFI signal uses half-lives
`[15, 20, 25, 30, 35, 40, 45, 60, 90]` minutes, emphasising late-day order flow.

| Script | Method | Target return | Output directory |
|--------|--------|---------------|-----------------|
| `04_directed_source_target.py` | DST bicluster | next-day residual | `results/rolling_adjacency_dst/` |
| `05_dhillon_bipartite.py` | Dhillon co-clustering | next-day residual | `results/dhillon_adapted/` |
| `06_ridge_nextday.py` | Ridge + sector-block + GD | next-day residual | `results/rolling_adjacency_ridge/` |
| `07_ridge_overnight.py` | Ridge + sector-block + GD | overnight (close→open) | `results/rolling_adjacency_ridge_overnight/` |
| `08_ridge_close_to_close.py` | Ridge + sector-block + GD | close-to-close | `results/rolling_adjacency_ridge_ctc/` |
| `09_ridge_nextday_tc.py` | Ridge + sector-block + GD + TC | next-day residual | `results/rolling_adjacency_ridge_tc/` |
| `10_ridge_overnight_tc.py` | Ridge + sector-block + GD + TC | overnight | `results/rolling_adjacency_ridge_overnight_tc/` |
| `11_plot_ridge_results.py` | — | — | `results/figures/` |
| `12_plot_dst_dhillon_results.py` | — | — | `results/figures_dst_dhillon/` |

**Key modeling choices (scripts 06–10):**

- **Sector-block W**: cross-impact estimated separately per GICS sector;
  cross-sector entries are zero by construction.
- **Gavish–Donoho denoising**: optimal hard SVD threshold applied to W
  after fitting; retains ~9 singular values on average.
- **`MIN_SECTOR_ASSETS = 4`**: sectors with fewer than 4 assets in a window
  are skipped.
- **GOOGL/GOOG filter**: only GOOG is kept (GOOGL duplicate removed).

**DST (script 04):**
Estimates a pairwise edge-stat `A[i,j] = mean(R_i * P_j) / std(R_i * P_j)` for
each (source j, target i) pair over the training window. Edges with `|t| < 2`
are dropped. Alternating top-k selection (KS=KT=15) finds the densest
signal-bearing bicluster. Prediction: `score = M @ ofi_lag`.

**Dhillon (script 05):**
Normalizes the edge-stat matrix as `D_r^{-1/2} A D_c^{-1/2}`, runs SVD, and
clusters rows and columns jointly via KMeans on the stacked embedding. The best
block is selected by mean weighted Sharpe. Within-sector masking (same GICS
sector only) is used to keep the multiple-testing burden manageable.

---

## Suggested Run Order

```bash
# ── data pipeline (skip if you have feature_table_with_residuals_10level.csv) ──
python pipeline/01_lobster_to_integrated_ofi.py
python pipeline/02_build_feature_matrix.py
python pipeline/03_residualize_returns.py

# ── analysis (can run independently once feature table exists) ──────────────
python pipeline/04_directed_source_target.py     # ~10 min
python pipeline/05_dhillon_bipartite.py          # ~20 min
python pipeline/06_ridge_nextday.py              # ~2 hr
python pipeline/07_ridge_overnight.py            # ~2 hr
python pipeline/08_ridge_close_to_close.py       # ~2 hr
python pipeline/09_ridge_nextday_tc.py           # ~3 hr
python pipeline/10_ridge_overnight_tc.py         # ~3 hr

# ── figures ──────────────────────────────────────────────────────────────────
python pipeline/11_plot_ridge_results.py         # requires 06,07,09,10 outputs
python pipeline/12_plot_dst_dhillon_results.py   # requires 04,05,06 outputs
```
