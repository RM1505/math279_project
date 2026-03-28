import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# example inputs
# -----------------------------

# replace with your actual results

model_sharpes = {
    "Diagonal": 0.74,
    "Sector Block": 1.55,
    "Sector Off-Diag": 1.83,
    "Dense": 0.44 
}

daily_ic = pd.read_csv("data/processed/daily_ic_sector_offdiag_best_lambda.csv")["ic"]
daily_spread = pd.read_csv("data/processed/daily_spreads_sector_offdiag_best_lambda.csv")["spread"]

cum_returns = (1 + daily_spread).cumprod()

# -----------------------------
# figure
# -----------------------------

fig, axes = plt.subplots(1,4, figsize=(20,5))

# -----------------------------
# Panel 1: Model Sharpe
# -----------------------------

models = list(model_sharpes.keys())
sharpes = list(model_sharpes.values())

axes[0].bar(models, sharpes)

axes[0].set_title("Model Performance")
axes[0].set_ylabel("Annualized Sharpe")

axes[0].set_xticklabels(models, rotation=30)

for i, v in enumerate(sharpes):
    axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center")

# -----------------------------
# Panel 2: IC Distribution
# -----------------------------

axes[1].hist(daily_ic, bins=50)

axes[1].set_title("Daily Information Coefficient")
axes[1].set_xlabel("IC")

axes[1].axvline(daily_ic.mean(), linestyle="--")

# -----------------------------
# Panel 3: Cumulative Returns
# -----------------------------

axes[2].plot(cum_returns)

axes[2].set_title("Long-Short Strategy Performance")
axes[2].set_ylabel("Cumulative Return")

axes[2].axhline(1, linestyle="--")


rolling_sr = (
    daily_spread.rolling(252).mean() /
    daily_spread.rolling(252).std()
) * np.sqrt(252)

axes[3].plot(rolling_sr)
axes[3].axhline(0, linestyle="--")
axes[3].axhline(daily_spread.mean()/daily_spread.std()*np.sqrt(252), linestyle="--", label = "Overall Sharpe")

# -----------------------------
# formatting
# -----------------------------

plt.suptitle("Order Flow Imbalance Cross-Asset Signal Summary", fontsize=16)

plt.tight_layout()

plt.show()