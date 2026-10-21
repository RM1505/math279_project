# When snapshot OFI changes meaning: a measurement study

**21 October 2026 — exploratory research-note candidate.** This is the strongest distinct direction found so far. It establishes an aggregation problem in the available data and supplies a practical diagnostic. It does **not** establish a publishable novelty claim, profitable strategy, or a failure of correctly computed event OFI.

The concrete result: replacing a five-minute observed quote path with its endpoints reverses the sign of measured pressure in **17.03%** of materially nonzero comparisons (95% day-block interval 16.82%–17.24%). At thirty minutes the rate is **21.61%**. A simple fixed-queue null shows that two measures can each achieve 63.66% contemporaneous explanatory R² while disagreeing on signs in about 20.6% of long windows. High explanatory fit therefore does not certify measurement fidelity. In the empirical five-minute test, freezing within-block queue-size changes raises explanatory R² from **24.25% to 54.32%**. However, bounding ordinary endpoint OFI raises its fit to **49.88%**, so much of the raw gap reflects functional form. The frozen control retains price changes by construction; it is not a tradable forecast.

## What was implemented

For each block, compare (1) the sum of its observed one-minute quote-transition increments, (2) the standard operator applied only to the two endpoint snapshots, and (3) the endpoint operator with displayed queue sizes frozen at their initial values. All three use the same positive starting mean depth. The first is a finer *snapshot reference*, not true event flow. The third retains observed price changes, so it is a price-input control rather than evidence of no market activity.

The study covers **483 stocks**, 2017–2019, and **25,475,162 eligible five-minute windows** across 358,807 stock-days. It excludes 135 five-minute windows for invalid observations or the stated extreme-return filter. Other block sizes are 1, 2, 10 and 30 minutes; the five-minute grid is also shifted by two minutes. Blocks lie between 10:00 and 15:55; shortened/exceptional sessions were removed upstream. Only best-level queues are used even for ten-level source books.

## The measured disagreement survives materiality and phase checks

“Material” means the absolute value of **each** normalized measure exceeds 0.1 starting-depth units. Rates average stocks equally within dates, then dates equally. Intervals resample whole dates in circular 20-day blocks, with 2,000 draws, preserving contemporaneous stock dependence. They describe uncertainty within this archive, not representativeness of all markets.

| Block length, minutes | Opposite signs in 2019; 95% interval |
|---|---|
| 1 | 0.00% [0.00%, 0.00%] |
| 2 | 11.79% [11.51%, 12.09%] |
| 5 | 17.03% [16.82%, 17.24%] |
| 10 | 19.49% [19.28%, 19.68%] |
| 30 | 21.61% [21.36%, 21.87%] |

| Year | Five-minute opposite signs; 95% interval |
|---|---|
| 2017 | 15.98% [15.75%, 16.25%] |
| 2018 | 17.08% [16.77%, 17.38%] |
| 2019 | 17.03% [16.82%, 17.24%] |

The two-minute phase shift gives **17.27% [17.09%, 17.45%]** in 2019. Across stocks, the median five-minute disagreement rate is 17.64%; every stock with a reported rate has some disagreements. Both one-level and ten-level source groups exhibit the effect. There are 1,377,269 opposite-sign instances among 8,091,184 material five-minute comparisons in 2019; this pooled count ratio differs slightly from the equal-stock/day estimand above. Material comparisons retain 95.91% of valid windows.

![Resolution audit](full_study/resolution/resolution.png)

The left panel uses the materiality threshold; the right panel shows same-block explanatory fit. Neither panel establishes tradable alpha.

## Explanatory fit and genuine forward forecasts

Models train on 2017, choose ridge strength on 2018, refit through 2018, and evaluate 2019. All clipping and scaling use training data. These are chronologically separated evaluations, but the archive has already been explored extensively, so 2019 is not a fresh untouched research holdout.

The forward baseline includes current and previous block returns, current spread/depth/queue imbalance and trailing 5/30-minute returns. The target begins at the current endpoint and ends one block later. No future outcomes enter its features. Explanatory models intentionally use information from the return interval itself.

| Five-minute specification | 2019 R² versus training mean | 95% interval |
|---|---|---|
| Summed minute increments: same-block explanation | 23.38% | [21.23%, 25.22%] |
| Endpoint increment: same-block explanation | 24.25% | [22.19%, 25.97%] |
| Frozen queue sizes: same-block explanation | 54.32% | [53.28%, 55.49%] |
| Previous-block endpoint pressure: separate control sample | -0.0031% | [-0.0073%, 0.0010%] |
| Next-block forecast: baseline | 0.0778% | [0.0355%, 0.12%] |
| Next-block forecast: baseline + minute-sum | 0.0732% | [0.0300%, 0.12%] |
| Next-block forecast: baseline + endpoint | 0.0742% | [0.0317%, 0.12%] |

R² is 1 minus the ratio of equal-stock/day average model loss to training-mean benchmark loss. It is not the mean of stock-level R² values. The shifted-pressure negative control has fewer windows and is not a paired comparison with the contemporaneous models.

| Paired contrast (positive favors first model) | Relative MSE improvement | 95% interval | Holm p, four primary contrasts |
|---|---|---|---|
| same_coarse vs same_fine | 1.14% | [0.64%, 1.55%] | 0.0020 |
| same_frozen vs same_coarse | 39.70% | [38.37%, 41.03%] | 0.0020 |
| next_fine vs next_base | -0.0046% | [-0.0066%, -0.0029%] | 0.0020 |
| next_coarse vs next_base | -0.0036% | [-0.0053%, -0.0021%] | 0.0020 |

A frozen-size model retaining substantial fit shows that price inputs alone can generate strong association. It does not estimate a causal percentage of true flow information. In the primary five-minute forward test, adding either pressure measure slightly worsens the baseline loss (roughly 0.004% relative MSE). The statistically detectable deterioration is economically tiny. These forecasts do not rescue the alpha hypothesis. All scales and model outcomes are retained in the accompanying tables.

### Bounded-transform check: an important qualification

After seeing the primary fit comparison, I tested fixed tanh and sign transforms of both pressure measures, plus the sign of the current return itself, using the same chronological fitting procedure across all 483 stocks. Tanh uses the original starting-depth units with scale one; its scale was not tuned. This is a **post-result descriptive sensitivity**, not an additional pre-specified primary test.

| Five-minute single-feature model | 2019 explanatory R² | 95% interval |
|---|---|---|
| sign_coarse | 48.72% | [47.69%, 49.88%] |
| sign_fine | 32.31% | [31.61%, 33.12%] |
| sign_return | 52.49% | [51.46%, 53.67%] |
| tanh_coarse | 49.88% | [48.84%, 51.05%] |
| tanh_fine | 37.45% | [36.66%, 38.38%] |

Bounding endpoint OFI raises its fit from 24.25% to 49.88%, near the frozen-queue control's 54.32%. Therefore the raw fit gap must **not** be interpreted entirely as an economic effect of removing queue changes. Even the sign of the return alone gives 52.49% fit; this feature intentionally contains the outcome and serves only as an explanatory benchmark. The robust point is that large same-block fit can coexist with severe path-measurement disagreement, not that all OFI information is mechanically explained or that frozen queues constitute a superior trading model.

## Mathematical and practical contribution

[The derivation](RESOLUTION_THEORY.md) gives an exact difference between the sum of observed increments and the endpoint calculation. It also constructs two paths with identical endpoints and opposite summed flow; without path restrictions, no endpoint-only estimator can recover both. A Gaussian fixed-queue calculation gives identical 2/π explanatory R² for the fine and endpoint measures while their sign-disagreement probability approaches arccos(√(2/π))/π. The supplied simulation uses 100,000 paths at each scale, including 1,001 increments, and obtains 20.52% disagreement at that long scale.

The useful remedy is simple: preserve summed observed bid/ask increments and depth/count summaries before discarding intermediate observations. The supplied utility merges adjacent chunks, checks their shared boundary and rejects gaps. It reproduces the direct sum exactly in the unit test. This is ordinary additive aggregation, not a novel compression algorithm, and it cannot recover sub-minute activity never recorded.

## Verification and numerical precision

All 17 relevant measurement, original-study and statistical tests pass. Seven targeted tests cover the identity, flat-price telescoping, endpoint ambiguity, merging, the Gaussian control, future-state isolation and gap rejection. The identity was also checked on every eligible empirical block; maximum normalized discrepancy was 4.85e-11. Saved parameters were independently replayed for **210 models and 7,560 stock-day losses**, with maximum loss difference 3.39e-21.

An original-file audit covers 18 stock-days from six stocks and all three years. Their cached bid/ask prices match exactly. Reconstructing queue sizes from float32 log-depth and imbalance creates small errors: maximum sampled normalized five-minute flow error 0.00012239. Four exact raw zeros became tiny nonzero cached values. Accordingly the original near-zero threshold results remain available but are not the headline; sensitivities at 0.001, 0.01 and 0.1 are all reported. This sample check is not a universal numerical error bound.

## Is this worth taking to Mihai?

**Yes, as a concrete methods-paper proposal, with the novelty question still open.** The contribution would be the combination of a transparent identification argument, a large within-path aggregation audit, price-input controls and a reproducible preservation tool. This is more specific than another weak/untradeable OFI signal.

The closest prior work is [Su et al. (2021)](https://arxiv.org/pdf/2112.02947), which already recognizes snapshot limitations and proposes generalized OFI using deeper levels. [Cont–Kukanov–Stoikov](https://arxiv.org/html/1011.6402v3) already defines OFI as a sum of event contributions. We cannot claim that sampling loss, summing increments, or contemporaneous-versus-forward distinctions are new. [The literature memo](RESOLUTION_LITERATURE.md) states the overlap explicitly. We have not implemented or outperformed generalized OFI here.

A sensible pitch is: *We found that endpoint substitution changes the sign of snapshot pressure surprisingly often, and that strong price-fit statistics do not detect this measurement failure. We have an exact decomposition, a 483-stock audit and an aggregation-preserving implementation. Is this specific combination new enough for a short methods note?*

The unresolved publication hurdle is whether this is useful new evidence rather than a familiar implementation pitfall. An event-level benchmark would strengthen the scope but is not needed to establish the narrower observed-path facts. The available minute data cannot measure the amount of information lost below one minute, reconstruct cancellations/trades, identify causality, or validate a general correction to event OFI. Archive composition may also be subject to survivorship selection.

## Reproduction

From this directory, with the existing 2017–2019 caches and archive inventories:

```bash
../.venv/bin/python -m unittest test_resolution
../.venv/bin/python resolution_study.py
../.venv/bin/python verify_resolution.py
../.venv/bin/python resolution_sensitivity.py
../.venv/bin/python resolution_heterogeneity.py
../.venv/bin/python replay_resolution.py
../.venv/bin/python resolution_null.py
../.venv/bin/python resolution_transform.py
../.venv/bin/python summarize_resolution.py
../.venv/bin/python report_resolution.py
```

Outputs are under `full_study/resolution/`. Per-stock files are resumable; use a fresh output directory if changing the specification. Raw data, caches, per-stock losses and saved model parameters remain local; source, tests, protocol, theory, compact aggregate results and figures are shared.
