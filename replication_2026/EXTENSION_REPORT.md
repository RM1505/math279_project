# What the extension adds — 24 October 2026

We wanted to know whether the measurement difference changes a research decision, whether a realistic correction helps, and whether this is distinct enough to justify a paper. The evidence is stronger on the first two questions than on novelty. The paper is now a more useful measurement study, but it is not a discovery that snapshots lose information and it does not establish a profitable strategy.

## The new results

| Question | Result | What it supports |
|---|---|---|
| Do the largest measurements select the same stocks? | Raw top-20 overlap is **51.71%** (95% interval 50.24–53.20%); the common-training-scale version is **49.59%** (48.09–51.21%). | Measurement choice changes roughly half the selected names. These are OFI-feature ranks, not forecast ranks or trades. |
| Can endpoint regressions reconstruct the minute sum? | Calibration increases raw MSE by **6.38%**; a fixed nine-feature endpoint ridge model increases it by **4.97%**. Material sign errors are 17.04% and 18.93%, versus 17.03% for raw endpoint OFI. | These particular fitted repairs do not help. This is not a lower bound for all decoders. |
| Does generalized OFI help? | On identical all-four-material windows, the price-aligned version has **7.02%** disagreement with its own sum versus **21.71%** for standard OFI. | Greater aggregation stability for another operator on the selected sample. |
| What does deeper-book correction require? | Both price-aligned constructions are available for **57.60%** of 29,820 valid candidate windows; the literal contiguous-tick interpretation has **32.03%** coverage. | Depth coverage must accompany the improvement; ten recorded levels are not always sufficient. |
| Can we prevent the additional loss? | Tested chunk summaries retain observed additive contributions, boundary states, spacing and normalization statistics. | Exact preservation in mathematical arithmetic over unions of chunks, with floating-point roundoff in implementation. No hidden-event recovery. |
| Did this cause our old strategy to fail? | The current original helper already sums weighted minute contributions. | No evidence for that causal explanation. Historical versions and execution issues remain separate. |

## Selection details

The main extension uses the same 483-stock universe and five-minute grid in 2019. At each timestamp we use the same valid stocks for both features, require at least 100 names, and compare the sets with the largest absolute values. Overlap means intersection divided by k, not Jaccard overlap and not the fraction of all stocks selected. We average timestamps within date and then dates. Intervals use the existing 2,000-draw, 20-date circular bootstrap.

The raw top-20 result means an average **10.34 shared stocks** out of 20. The remaining 9.66 names in each list differ. Among the shared selections, signs disagree in 9.34% of comparisons (interval 9.05–9.63%). Absolute rank correlation is 0.4215 (0.4153–0.4279). Thus the measures are related; the result should not be described as random or unrelated rankings.

For the scale check, each stock's 2017 ninetieth percentile of absolute minute-sum OFI divides BOTH its minute-sum and endpoint values. This is a common, training-only scale, not separate transformations chosen to make the two ranks agree. Scaled top-20 directional disagreement is 8.76% among shared selections, and rank correlation is 0.4373. The results survive changing k: raw top-5/top-50 overlap is 54.66%/47.42%, and scaled overlap is 54.28%/45.92%. All specified variants are retained in the results.

This is a different experiment from Rigel's earlier proposal to trade the strongest predicted returns. That earlier proposal was evaluated in the shared research branch and did not overcome spreads under the tested rules. The present ranking calculation asks whether changing OFI measurement alters an OFI-based selection, without asserting which selection is profitable.

## Reconstruction details

The target is the observed minute sum F, not true event OFI and not future returns. Each stock is fitted on 2017, its ridge penalty is selected on 2018, and the model is refitted through 2018 for 2019 evaluation. The richer inputs are endpoint C, frozen-size Z, starting/ending queue imbalance, log-depth change, bid/ask changes divided by starting spread, tanh(C), and abs(C). No intermediate states are used as predictors. The target and training scales do use observed paths, which is necessary for supervised fitting and available in this archive.

Both model MSE comparisons use all eligible windows and the same stock/date-balanced loss definition. Calibration's MSE-increase interval is 2.66–11.41%; the richer model's is 1.37–10.08%. The sign-error diagnostic uses exactly the headline both-material F/C sample, including a predicted zero as an error against material F. The raw endpoint counts reproduce all 8,436,055 valid, 8,091,184 material and 1,377,269 opposite-sign 2019 comparisons.

This negative result does not show that a sufficiently flexible conditional model could never improve average error. The specified ridge family, penalties and training-only feature clipping are limited. It does show that a simple fitted correction is not demonstrated here. Exact path recovery is a separate impossibility statement; average statistical approximation can be useful under assumptions.

## Generalized OFI: improvement and selection

We implemented the 2021 Su–Sun–Li–Yuan equations and an explicitly labeled price-aligned adaptation for gaps in occupied US book levels. A literal one-cent tick interpretation is applied only when all required levels are contiguous and covered. The adaptation sums occupied levels through the old/new best-price boundary. They agree on covered contiguous ladders. Missing deeper queues are not filled with zero, and neither implementation is log-GOFI.

The predefined bounded sample includes all 105 ten-level stocks on January 2, April 1, July 1 and October 1, 2019: 420 stock-days and 29,820 valid five-minute windows. Each source grid was checked against 391 one-minute observations, 09:30–16:00; source-file hashes remain local. Only these four members per archive were extracted into temporary storage. There was no full raw-data expansion.

The initial output used each pair's own material sample. Price-aligned generalized disagreement was 7.53% versus standard 21.99%; contiguous-grid generalized disagreement was 3.70% versus standard 21.39%. To avoid suggesting an exactly paired comparison from those numbers, we then added a disclosed post-result check requiring all four measures to exceed 0.1. Those are the manuscript's numbers:

| Convention | Covered windows | Common material windows | Nonempty stock-days | Standard opposite-sign rate | Generalized opposite-sign rate |
|---|---:|---:|---:|---:|---:|
| Price-aligned adaptation | 17,177 | 16,264 | 414 | 21.71% | 7.02% |
| Literal contiguous ticks | 9,550 | 9,222 | 234 | 21.22% | 3.47% |

Rates give equal weight to nonempty stock-days, not individual windows. Counts therefore do not directly divide to the displayed rates. Neither row estimates a full-year temporal effect, and no confidence interval is supplied for four dates. The two conventions use different retained windows and cannot be ranked against each other as a clean estimator contest.

Standard OFI's full four-date rate before any depth filter is 18.50%. Coverage filtering changes that population. Separately, generalized endpoint versus standard minute-sum disagreement is 22.03% for the price-aligned convention under that pair's material filter. This is not evidence that generalized OFI is wrong: it defines another observed quantity. Stability with its own sum and recovery of standard OFI are different objectives. Neither compares against actual events.

## What the practical remedy preserves

Compute contributions on the finest available observations, then aggregate the contributions. The supplied summary can preserve the unnormalized bid and reversed-ask sums and support either starting-depth or mean-destination-depth normalization. It rejects inconsistent boundaries, gaps, duplicate observations and mixed spacings. The caller must keep instruments and units consistent.

A single total does not preserve arbitrary later choices of within-chunk boundaries, nonlinear transformations, or exponential weighting. Store the fine increments if those choices remain open; otherwise store the sufficient statistics for the declared calculation. Our original daily project uses per-minute normalization and time weights, so its statistic should not be silently replaced by the paper's common-start-depth sum.

## How much is new?

The general principle, standard temporal summation, snapshot limitations and generalized constructions predate this paper. The targeted search did not identify the exact empirical package of same-path material disagreement, top-k instability, matched generalized-coverage comparison and price-fit diagnostics. That is a candidate distinction, not a claim of first discovery. The search is not systematic or exhaustive. The [literature and usage audit](EXTENSION_LITERATURE.md) records evidence and access limits.

The strongest plausible contribution is: **a reproducible audit showing that aggregation choice can change research selections, with an explicit comparison of feasible remedies and their data requirements.** The weakest framing would be: “OFI is broken, we discovered compression bias, and past papers are invalid.” The latter is not supported.

## Where we stop

The specified experiment family is complete. More feature sweeps on the same explored years would not resolve the main uncertainty. The highest-value next input is expert feedback on distinctness and usefulness, especially whether Mihai knows a directly matching comparison. A truly independent event-level or new-market sample could answer a stronger question, but it is not available in this task and is not required to establish the narrower within-archive facts.

My assessment is that there is now enough for a serious coauthor/adviser discussion and a carefully framed measurement note. There is still no basis for promising a journal acceptance or describing the result as a major unknown flaw in market data. We should discuss the paper as a method and measurement contribution, not a rescued trading strategy.
