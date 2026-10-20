# Research context and local evidence

## What the existing folder contains

The actual project is `/Users/brendanconnelly/Documents/Coding/math279_project`. The configured path containing `Coding:Github` is stale. A complete file-size inventory is saved in `full_study/project_file_inventory.csv`. Raw data is linked from `data/raw` to `/Users/brendanconnelly/Documents/raw`.

Approximate disk usage before this study: 49 GiB in the project, of which 46 GiB is processed data. Major components are 24 GiB of per-stock integrated-OFI CSVs, a 12.61 GB feature table, a 7.45 GB feature table with residuals, and 3.5 GiB of V2 caches. Results occupy about 2 GiB, Git objects 0.7 GiB, and the environment 0.3 GiB. Raw archives outside the project total 9.006 decimal GB. GB and GiB are different units, so these approximate figures should not be added as exact totals.

All 516 archive directories were inspected. They contain 1,693,301 dated members across 2007-06-27 through 2021-12-31. All archive names encode sixty-second sampling. There are 119 ten-level and 397 one-level archives. For 2017–2019 the archive inventory contains 377,826 stock-day members, approximately 11.886 GB uncompressed. The study reads those files, not just the first few names.

The 24 GiB integrated-OFI directory contains three-column transformed files (`source`, `time`, `IOFI`), not extra raw order-book events. The giant feature tables hold wide derived minute features. Neither recovers the missing within-minute event path.

## Local code versus the older remote audit

- `src/ofi.py` already contains the ask-side sign fix. The old remote-code finding must not be described as a current local bug.
- `data/processed/v2/ofi_levels` and `ofi_levels_buggy_backup` both exist and differ. Both contain daily compressed features; mixing their provenance would invalidate comparisons.
- `pipeline_v2/25_final_results.py` currently assigns the evaluation index from `sig_ev.iloc[1:]`, whereas the earlier remote audit described same-date evaluation. This local revision differs. That does not by itself validate overnight target semantics.
- `pipeline_v2/common_v2.py` still assigns each refit's projections to both `train_mask` and `eval_mask`, allowing later fits to replace earlier projections in the final table.
- `ofi_scrutiny/integrated_ofi_test.py` computes PCA from the entire supplied matrix, not a chronological forecasting fit. Its contemporaneous diagnostic is distinct from its forward prediction interpretation.
- `ofi_scrutiny/outputs/characterize_run.log` includes divide-by-zero/invalid-value warnings and NaN market-timing outputs. Some other diagnostics are finite; the point is that this run cannot support an unqualified successful test of every channel.
- `OFI_Brendan_Rigel_Paper/newmain_final.tex` already reports the sign error and trailing-mean attribution. Its daily Sharpe values are historical reported results, not independently reproduced by this new study. Broad claims such as a clean event-OFI replication or no predictability at any horizon go beyond what these sampled-book tests establish.

The new code leaves all these originals unchanged and rebuilds from raw snapshots.

## Relevant primary research

[Cont, Cucuringu and Zhang, Cross-impact of order flow imbalance in equity markets](https://arxiv.org/html/2112.13213): motivates the distinction between contemporaneous impact and forward prediction, integration across book levels, and own/cross-stock comparisons. Our sampled flows, equal-weight depth summaries, common-factor cross-stock model, annual chronological split and ridge regressions are explicit adaptations, not their exact replication.

[Gould and Bonart, Queue Imbalance as a One-Tick-Ahead Price Predictor](https://arxiv.org/abs/1512.03492): studies queue imbalance and the direction of the next midpoint price move using logistic models for ten Nasdaq names. It provides a reason to test book *state* directly. Its event-based next-tick target differs from our fixed one-minute return, so its positive findings do not guarantee ours.

[Xu, Gould and Howison, Multi-Level Order-Flow Imbalance in a Limit Order Book](https://arxiv.org/abs/1907.06230): studies contemporaneous changes in midprice for six Nasdaq stocks and reports improving fit with additional levels. This supports a matched-depth construction diagnostic, not an assumption that deeper snapshots predict future returns.

[Kolm, Turiel and Westray, Deep order flow imbalance](https://onlinelibrary.wiley.com/doi/abs/10.1111/mafi.12413): studies granular book data for 115 Nasdaq stocks and reports an effective stock-specific prediction horizon of roughly two average price changes. It reinforces the importance of temporal resolution; we are not reproducing its neural-network models or event horizons.

[Su et al., The Price Impact of Generalized Order Flow Imbalance](https://arxiv.org/abs/2112.02947): explicitly studies high-frequency snapshots and constructions accommodating non-minimal price changes. Its results concern explanatory fit in Chinese equities. It shows that snapshot-based research is legitimate while also cautioning that the flow definition matters. We do not claim to implement GOFI here.

[NYSE historical calendar announcement](https://ir.theice.com/press/news-details/2017/NYSE-Group-Announces-2018-2019-and-2020-Holiday-and-Early-Closings-Calendar/default.aspx) and [2017 calendar](https://ir.theice.com/press/news-details/2016/NYSE-Group-Announces-2017-Holiday-and-Early-Closings-Calendar/default.aspx): support excluding shortened sessions from a regular closing-half-hour analysis.

Novelty remains unestablished. A positive ordinary queue-imbalance result would primarily reproduce an established mechanism at a different sampling horizon. A robust incremental closing-session/depth interaction could motivate further work, but neither the size of the dataset nor a significant p-value alone makes a paper.

## Additional closing-session benchmarks

[Baltussen, Da and Soebhag, End-of-Day Reversal](https://academicweb.nd.edu/~zda/EOD.pdf), April 2025 author manuscript, documents cross-sectional reversals during the final half-hour associated with earlier same-day performance. This motivates the separate open-to-current return control. A book variable correlated with earlier intraday price pressure must not be credited with a distinct closing effect without such controls.

[Akyildirim et al., Big data analytics, order imbalance and the predictability of stock returns](https://www.sciencedirect.com/science/article/pii/S1042444X21000402), 2021, studies Borsa Istanbul analytics and one-minute excess returns with momentum controls. Its different market and disseminated analytics distinguish it from our inputs, but it is prior art against any generic claim that predicting one-minute returns with book imbalance is novel.

[NYSE closing-auction research](https://beta.nyse.com/data-insights/closing-auction-immediate-market-impact-price-drift-and-transaction-cost-of-trading) analyzes auction imbalance and surrounding price impact. Continuous-book queue imbalance is not the auction imbalance feed; the current study excludes outcomes reaching the 16:00 boundary and does not claim to explain the auction itself.
