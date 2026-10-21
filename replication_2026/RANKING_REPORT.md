# Ranking follow-up: the minute forecasts

Rigel's top-20 idea was tested, alongside top-5/top-50 sensitivity checks, spread-filtered rankings, and a spread-scaled queue-state feature. The tested minute-level rules do not recover profitability. This says nothing definitive about next-day prediction.

The rules and model family were fixed before reviewing these outcomes. Selection on 2018 chose **cash**: every candidate lost after spreads. All 27 candidates also lost in the 2019 matched sample, including turnover-aware accounting that carries positions rather than charging a new round trip every minute. These are exploratory results on previously examined dates.

For the cross-book model's top 20 by absolute forecast, the average predicted move was **2.13 bps**, but the selected stocks' entry spreads averaged **9.26 bps**. Independent round trips lost **7.71 bps per selected stock-minute** on average. Ranking magnitudes disproportionately selects expensive stocks; it does not calibrate the predictions or guarantee a tradable edge.

Requiring the forecast to exceed one current spread reduced activity to **0.79 positions per minute** on average (out of 20 available slots), but the selected round trips still lost **1.98 bps** on average. A two-spread requirement reduced activity further without producing positive results. Unused allocation remained cash. The simple spread-scaled book feature did not fix this.

Turnover-aware results use constant target notional and charge changes in share holdings at the observed bid/ask, with daily liquidation. This accounting passed an independent cash/share ledger test. It still assumes quote availability, excludes fees/borrow/impact/capacity, and is not an executable strategy claim. Daily results are profits per constant gross capital allocation, not compounded returns. The reported core/close split attributes a continuous portfolio's P&L; it is not a separately optimized closing-only strategy.

The 2019 comparison uses 244 common complete dates, with two additional unscorable dates removed from the prior 246-day minute sample. Selected missing outcomes were not replaced by zero or by another stock. Six targeted tests pass; reused forecasts were also checked against the original saved scores.

Detailed results: `full_study/ranking/summary.csv`, `daily.csv`, `selected_rule.json`, `excluded.csv`, and `common_day_exclusions.json`. Specification: [RANKING_PROTOCOL.md](RANKING_PROTOCOL.md). Code: `ranking_study.py` and `test_ranking.py`.
