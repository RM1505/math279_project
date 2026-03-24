
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



daily_spread = pd.read_csv("data/processed/rolling_adjacency_analysis/daily_spreads_sector_offdiag_rolling.csv")
daily_spread["date"] = pd.to_datetime(daily_spread["date"])

cum_returns = (1 + daily_spread["spread"]).cumprod()

fig, axes = plt.subplots(1,2, figsize=(12,5))

rolling_sr = (
    daily_spread["spread"].rolling(504).mean() /
    daily_spread["spread"].rolling(504).std()
) * np.sqrt(252)

axes[0].plot(daily_spread["date"],rolling_sr)
axes[0].axhline(0, linestyle="--")
axes[0].axhline(daily_spread["spread"].mean()/daily_spread["spread"].std()*np.sqrt(252), linestyle="--", label = "Overall Sharpe")
axes[0].set_title("Rolling 2-Year Sharpe Ratio")
axes[0].set_ylabel("Sharpe Ratio")

axes[1].plot(daily_spread["date"],cum_returns)
axes[1].set_ylabel("Cumulative Returns")
axes[1].set_title("Cumulative Returns of Long-Short Strategy")

fig.autofmt_xdate()

plt.show()
