import numpy as np
import matplotlib.pyplot as plt

mean_pnl = np.load("data/processed/mean_pnl.npy")
sr_ann   = np.load("data/processed/sr_annualized.npy")

N = mean_pnl.shape[0]

# Create diagonal mask
diag_mask = np.eye(N, dtype=bool)

# Remove diagonal first
mean_no_diag = mean_pnl[~diag_mask]
sr_no_diag   = sr_ann[~diag_mask]

# Now remove NaNs
mean_vals = mean_no_diag[np.isfinite(mean_no_diag)]
sr_vals   = sr_no_diag[np.isfinite(sr_no_diag)]

# ----------------------------
# Histogram 1 — Mean Daily PnL
# ----------------------------
plt.figure()
plt.hist(mean_vals, bins=150)
plt.title("Distribution of Mean Daily PnL (Off-Diagonal)")
plt.xlabel("Mean Daily PnL")
plt.ylabel("Frequency")
plt.show()

# ----------------------------
# Histogram 2 — Annualized Sharpe
# ----------------------------
plt.figure()
plt.hist(sr_vals, bins=150)
plt.title("Distribution of Annualized Sharpe (Off-Diagonal)")
plt.xlabel("Annualized Sharpe")
plt.ylabel("Frequency")
plt.show()