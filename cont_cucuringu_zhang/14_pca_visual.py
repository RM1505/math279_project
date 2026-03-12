import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_PATH = Path("data/processed/feature_table_with_residuals.csv")

df = pd.read_csv(INPUT_PATH)

minute_cols = sorted(
    [c for c in df.columns if c.startswith("minute_")],
    key=lambda x: int(x.split("_")[1])
)

X = df[minute_cols].fillna(0).to_numpy()

# -------------------------
# PCA
# -------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=1)
pca.fit(X_scaled)

kernel = pca.components_[0]
scores = X_scaled @ kernel

# align sign with raw OFI
raw_sum = X.sum(axis=1)
if np.corrcoef(scores, raw_sum)[0,1] < 0:
    kernel = -kernel

explained = pca.explained_variance_ratio_[0]

# -------------------------
# smoothing
# -------------------------

def smooth(x, w=11):
    return np.convolve(x, np.ones(w)/w, mode='same')

kernel_smooth = smooth(kernel)

minutes = np.arange(len(kernel))

# -------------------------
# time ticks
# -------------------------

tick_positions = [0,60,120,180,240,300,360]
tick_labels = ["9:30","10:30","11:30","12:30","13:30","14:30","15:30"]

# -------------------------
# peak detection
# -------------------------

imax = np.argmax(kernel_smooth)

# -------------------------
# plotting
# -------------------------

plt.figure(figsize=(13,6))

# raw PCA
plt.plot(
    minutes,
    kernel,
    color="steelblue",
    alpha=0.35,
    linewidth=1.2,
    label="Raw PCA weight"
)

# smoothed PCA
plt.plot(
    minutes,
    kernel_smooth,
    color="darkorange",
    linewidth=3.5,
    label="Smoothed PCA weight"
)

# open / close shading
plt.axvspan(0,60,color="grey",alpha=0.18,label="Opening hour")
plt.axvspan(330,390,color="grey",alpha=0.18,label="Closing hour")

# vertical reference lines
plt.axvline(0,color="black",linestyle="--",alpha=0.6)
plt.axvline(150,color="black",linestyle=":",alpha=0.5)
plt.axvline(390,color="black",linestyle="--",alpha=0.6)

# peak markers
plt.scatter(imax,kernel_smooth[imax],s=70,zorder=5,color="black")

plt.annotate(
    "Highest signal weight",
    xy=(imax,kernel_smooth[imax]),
    xytext=(imax+25,kernel_smooth[imax]+0.01),
    arrowprops=dict(arrowstyle="->"),
    fontsize=11
)

plt.title(
    f"Market-Wide Intraday OFI PCA Kernel",
    fontsize=16
)

plt.xlabel("Time of day",fontsize=13)
plt.ylabel("Kernel weight",fontsize=13)

plt.xticks(tick_positions,tick_labels)

plt.grid(alpha=0.25)

plt.legend(frameon=True)

plt.tight_layout()

plt.show()