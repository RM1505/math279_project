# How this paper compares with the earlier OFI project

The new paper is a narrower measurement study. It does not rescue the earlier trading results, invalidate established event-level OFI research, or constitute an exact replication of the original cross-impact paper.

## The original project

The earlier manuscript was titled *Structured Cross-Asset Order Flow Imbalance for Daily Return Prediction*. It asked whether intraday order-flow information helped predict overnight and next-day returns, including cross-stock effects. It combined book levels, exponentially weighted minute signals, learned cross-impact matrices, sector-block restrictions and singular-value denoising. Historical reports highlighted sector-neutral Sharpe ratios around 1.4 at selected decay half-lives, with materially weaker performance after costs.

Those were historical reported results, not independently validated findings of the new paper. The local/remote history contains differing versions of sign handling, projection fitting and target alignment. Later fixes do not automatically validate every earlier result file. The original manuscript and outputs remain preserved as research history rather than being silently rewritten as if their conclusions had been confirmed.

## The subsequent reconstruction

The reconstruction started from the available minute snapshots, used chronological fitting, matched baseline and book models, and distinguished queue state from sampled quote-transition flow. It found modest short-lived minute information. Delaying execution and accounting for observed spreads weakened the trading interpretation.

Rigel's proposal to rank the largest predictions was tested directly. In the minute ranking study, validation selected cash, and all 27 candidate rules lost after spreads in the matched 2019 sample. For one representative top-20 rule, average forecast magnitude was 2.13 bps versus 9.26 bps average entry spread; independent round trips lost 7.71 bps per selected stock-minute. Ranking does not itself fix calibration, liquidity or execution costs.

The broader daily study investigated an apparent +19.51 bps/cycle result. Common-market/sector attribution, concentration and multiple-testing checks weakened its interpretation as incremental stock-specific alpha. The separate 2015–2016 reconstruction did not establish a corresponding significant book increment. Relative-return and risk extensions likewise did not establish a repeatable positive alternative. These are findings of their own designs, not universal claims that order books contain no information.

The complete broader code test suite has been rerun in the current audit. This is not a fresh re-extraction and refit of every historical model or an independent certification of the old Sharpe ratios. The relevant reports and limitations remain in the shared research repository.

## The new question

Suppose we have minute observations. We can either sum their quote-transition contributions over five minutes or discard the intermediate states and calculate one contribution from the two endpoints. Are those the same variable? If not, does a high contemporaneous price fit warn us?

The answer is no to both. The paper gives the exact difference, an endpoint ambiguity example, a Gaussian illustration in which equal high fit coexists with sign disagreement, and a paired audit across 483 stocks. It uses the same underlying observed paths and starting-depth denominator, so the measurement comparison does not depend on choosing a winning forecast model.

## What changed—and what did not

| Dimension | Earlier work | Current paper |
|---|---|---|
| Scientific aim | Predict returns and test possible trading value | Audit the consequences of replacing an observed path by endpoints |
| Main evidence | Forecast losses, rankings, portfolios and costs | Algebra, paired sign disagreements, fit diagnostics and a limited forward check |
| Main horizon | Overnight/next-day; minute follow-ups | One–thirty minutes, with five-minute measurement primary |
| Cross-stock modeling | Central to the original idea | Not the object of this paper |
| Deeper-book information | Integrated or compared in prior designs | Level one used for every stock |
| Sample | Multiple historical/modern experiments | 483 training-eligible stocks, 2017–2019 |
| Strongest finding | Short-lived information, without validated net alpha | 17.03% material five-minute sign disagreement in 2019 |
| Success criterion | Incremental, repeatable benefit beyond controls and costs | Consequential, correctly characterized measurement difference |
| Principal uncertainty | Predictive robustness and economic usefulness | Distinctness from prior measurement work and broader relevance |

The data limitation is unchanged: neither minute snapshots nor derived files reveal the full event path. The new study can establish an additional loss from coarsening already observed data; it cannot establish error relative to true event flow. The study years were already explored, so chronology does not turn the current paper into a preregistered confirmatory experiment.

## What is stronger now

The core result is direct, easy to interpret and independently reproduced across all primary cached paths. The empirical headline does not rely on an optimized trading rule. The identities can be checked exactly. The manuscript separates measurement, explanation and forecasting and reports the bounded-feature qualification rather than overstating the raw frozen-queue gap.

## What is not established

A measurement problem is not automatically a publishable contribution. Endpoint non-additivity is elementary, and generalized snapshot OFI is relevant prior work. We have not implemented a common-data GOFI benchmark or validated against an event-level sample. The paper therefore needs expert novelty feedback, especially from Mihai, before making a strong originality claim. A useful conference discussion is a more realistic immediate objective than claiming a new profitable strategy.

## What to tell Mihai

“We could not establish a robust incremental trading result with the snapshot archive. We found a narrower measurement issue we can document carefully: replacing the observed minute path by endpoints changes the sign in about one in six material five-minute comparisons, and contemporaneous fit does not diagnose the difference. We have a short paper with the exact comparison, controls and limitations. Is that audit useful and sufficiently distinct from the snapshot OFI work you know?”

That is a research question for his judgment, not a request to endorse a positive backtest.

## Earlier source documents

These links preserve the broader research record:

- [Folder and original-code audit](https://github.com/RM1505/math279_project/blob/research/ofi-snapshot-study-2026/replication_2026/LITERATURE_AND_AUDIT.md)
- [Minute-snapshot reconstruction](https://github.com/RM1505/math279_project/blob/research/ofi-snapshot-study-2026/replication_2026/REPORT.md)
- [Ranking follow-up](https://github.com/RM1505/math279_project/blob/research/ofi-snapshot-study-2026/replication_2026/RANKING_REPORT.md)
- [Next-day reconstruction](https://github.com/RM1505/math279_project/blob/research/ofi-snapshot-study-2026/replication_2026/NEXT_DAY_REPORT.md)
- [Decision-stage audit and earlier-period comparison](https://github.com/RM1505/math279_project/blob/research/ofi-snapshot-study-2026/replication_2026/DECISION_REPORT.md)
