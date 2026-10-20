# OFI reconstruction: bounded first pass

Status on 2026-10-18: **data audit and snapshot pilot completed; event-level replication blocked on appropriate data.**

The existing project is actually at `/Users/brendanconnelly/Documents/Coding/math279_project`, not the stale path configured in the chat. Existing scripts, caches, raw archives, and remote repositories were left untouched. All new work is in this folder. No data was uploaded.

## What was found

`/Users/brendanconnelly/Documents/raw` contains 516 archives totaling 9.006 GB (decimal): 119 labeled ten-level and 397 labeled one-level. Every archive filename ends in a 60-second sampling label. This is a filename inventory, not a full content audit.

We inspected 100 actual files: AAPL, MSFT, JPM, JNJ, XOM, all twenty trading days from 2017-01-03 through 2017-01-31. Every inspected file contains 391 rows, exactly 60 seconds apart, from 09:30 to 16:00. These are endpoint book snapshots. Intervening additions, cancellations, and executions cannot be recovered uniquely from them. A unit test demonstrates distinct event flow hidden by identical endpoint states.

The prior local `ofi_scrutiny` scripts and outputs do exist. Their minute tests also difference these sampled books. Their results should not be interpreted as an event-level OFI replication, regardless of whether their ask signs were corrected.

## Completed pilot

Run with the existing project environment; no packages installed. Single process and single-thread numerical libraries. Only selected days extracted, in temporary directories that are cleaned on exit. Caps: ten names, forty days, 100 MB extracted per name. The completed five-name/twenty-day run took about 13.4 seconds.

The experiment uses a ten-level endpoint-difference proxy, scaled by mean endpoint depth across levels. Per-stock PCA is fitted on each rolling training window, with L1-normalized loadings. It is not fitted on future rows. Features are standardized in that window. Ridge uses a fixed penalty of 30 and an unpenalized intercept; this choice was not tuned to performance. Return history uses the latest one-minute return of all five stocks.

Models: zero, trailing training-label mean, return history, own integrated proxy, all-stock integrated proxies, returns plus own proxy, and returns plus all proxies. All models are scored on the same rows. We test one-minute and five-minute forward log midquote returns and a one-minute forecast with one-minute delayed entry. Thirty fully matured training labels are used; the latest ends no later than the forecast origin. Sessions never cross overnight. Core targets end by 15:30; close-session origins start at 15:30 and targets end strictly before 16:00, excluding the auction boundary.

One-minute forecast results, measured as percentage-point R² improvement over the rolling mean forecast:

| Model | Core | Last half-hour |
| --- | ---: | ---: |
| Zero | +3.956 | +1.314 |
| Return history | -4.379 | -3.211 |
| Own proxy | -0.828 | -0.772 |
| All-stock proxies | -4.224 | -2.037 |
| Returns + own proxy | -4.881 | -3.814 |
| Returns + all proxies | -7.515 | -4.931 |

Negative means worse squared prediction error than the rolling-mean reference. Adding all proxies to return history increases MSE by approximately 3.00% in core hours and 1.67% near the close. These are exploratory results for this particular proxy/model/sample, not evidence that true OFI cannot forecast returns. No profitability, executable-price, or Sharpe claim is made. No trading-cost model has been implemented for this pilot.

`outputs/paired_loss_intervals.csv` contains exploratory paired differences with circular five-day block resampling of daily cross-section-averaged losses (1,000 draws, seed 279). Days, not stocks or overlapping minute targets, are the resampling unit. Twenty days is a very small sample; intervals are not multiplicity-adjusted. Existing research may already have used these dates; they are NOT described as untouched holdout data.

## Reproduce

From this directory:

```sh
../.venv/bin/python -m unittest -v
../.venv/bin/python pilot.py --raw /Users/brendanconnelly/Documents/raw
```

Output inventory: `archive_inventory.csv`, `sample_audit.csv`, `input_manifest.csv` (member hashes), `predictions.csv.gz` (explicit decision/target/training times), `metrics.csv`, `paired_loss_intervals.csv`, `run.json`. Outputs stay local and are ignored by Git. Rerunning replaces this pilot's outputs.

Seven tests check price-direction signs, previous queue sizes, information loss under sampling, invariance to future perturbation, matured target timestamps, event aggregation across chunk boundaries, and missing opening states. Event ingestion has only synthetic validation so far; real vendor-file validation is pending.

## Replication target and remaining work

Target: Cont, Cucuringu, Zhang (2023), *Cross-impact of order flow imbalance in equity markets*, [paper](https://arxiv.org/html/2112.13213), [published version](https://doi.org/10.1080/14697688.2023.2236159). Their study uses event-derived OFI, ten book levels, historical PCA, and rolling short-horizon models. Its predictive comparison uses OLS for own-stock models and LASSO for cross-stock models, with multiple lookback horizons. The published predictive table reports negative R² even for its better models: improvement between models is a different claim from positive absolute R². Their closing-session discussion motivates an extension; it does not establish current novelty.

This pilot differs in data frequency, universe, period, lag set, estimator, normalization, and scoring aggregation. Do not compare its numbers directly with a paper table.

Next steps, in order:

1. Obtain paired unsampled LOBSTER files using `DATA_REQUEST.md`; validate one day first.
2. Verify the paper's OFI convention against original typeset equations/authors' code. The arXiv HTML shows current-size notation in price-decrease branches, whereas the canonical previous-queue convention used here differs for unequal queues. Do not silently declare these equivalent. Record an explicit resolution or report both conventions as a prespecified sensitivity.
3. Validate event ingestion against real files: initial book state, message alignment, halts, auction events, sentinels, timestamps and session calendar. The bounded loader rejects several unsupported cases instead of silently repairing them.
4. Implement and validate the paper's OLS/LASSO models, full lag set, training-only regularization selection, and its precise reported scoring metric. Produce a table mapping each reproduced result to the source specification and deviations.
5. Replicate contemporaneous best-level/integrated comparisons first, then genuinely forward returns with matured-label embargoes. Contemporaneous fit is a construction diagnostic, not a trading signal.
6. Freeze the analysis before opening new evaluation dates. Use zero, mean, return-only, own-OFI, cross-OFI, and combined baselines; test incremental OFI beyond return history on identical samples.
7. Only then compare late-day incremental predictive value against core hours. Keep continuous trading and auctions separate. Add quote-age/latency controls and executable bid/ask prices before any trading claim. Compare actual portfolio turnover, including opening/closing costs, if a portfolio study is justified.

The extension remains deliberately narrow: **does OFI add information beyond return history, and does its incremental contribution change in the final half-hour?** A secondary data-quality experiment can compare event OFI with the same data downsampled to sixty seconds; this would quantify the loss caused by our current inputs. Neither idea is presented as established novelty. Broader literature review remains pending.

`event_ingest.py` is a bounded streaming adapter for standard paired CSVs. It sums event contributions into right-closed minute bins, scales by event-averaged mean depth, preserves state across chunks, and does not bridge sessions. It requires a valid initial book at/before the open, rejects halts/invalid books, and excludes auction-cross events from OFI. It is not yet wired to the pilot or a full paper replication.

Git operations through the system executable are currently blocked by an unaccepted Xcode license. No license was accepted, no remote visibility changed, and no commits were pushed. A private standalone research repository can be created later if useful; it is not necessary for local analysis.
