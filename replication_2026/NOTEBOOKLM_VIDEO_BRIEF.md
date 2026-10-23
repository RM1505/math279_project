# What snapshot sampling does to order flow imbalance
## A detailed source and production brief for NotebookLM

**Authors:** Brendan Connelly and Rigel Mummers.  
**Project:** Snapshot OFI measurement study.  
**Status:** Research manuscript, not an accepted publication and not a validated trading strategy.  
**Audience:** Someone who understands basic statistics but may never have seen an order book. Build toward the mathematics rather than assuming market-microstructure knowledge.

## Instructions for the video

Please make a substantial, carefully paced explanation of this whole research story. Do not reduce it to a promotional abstract or a short list of percentages. Explain why we started, what the data contain, what the original trading work did and did not establish, why we changed questions, how the new measurements are constructed, what the proofs say, what we actually observed, and what remains uncertain. If the video format cannot fit that responsibly in one episode, divide the explanation into a coherent series rather than skipping the limitations or the mathematical intuition.

Aim for the depth of a 20–30 minute seminar if the available format allows it. Use ordinary language first, then notation. Whenever a formula appears, explain what each symbol means and work through a numerical example. Revisit the same small order book instead of introducing a new visual language in every scene. Show calculations, not only decorative stock-price charts.

The central question is: **If we throw away observations inside a time interval and apply the same formula to the two endpoints, are we still measuring the same thing?** The answer is generally no. The second question is: **Would a strong relationship with the price move necessarily reveal that measurement problem?** Again, no.

Do not frame the paper as discovering that all OFI research is wrong, that all order flow is uninformative, or that there is a new profitable strategy. Do not describe the minute sum as true event OFI. Do not say a conference has accepted the paper. Do not manufacture a quote or endorsement from Mihai Cucuringu or the authors. Refer to the project as “we” only if narrating explicitly on the authors’ behalf; otherwise say “the authors.” A clear, curious, candid tone is closer to Brendan’s writing than a dramatic investment pitch.

Treat the facts and equations below as source material. The suggested scenes are presentation suggestions, not additional empirical evidence. Any drawn price paths are illustrative unless specifically labeled as real data. If the manuscript changes, use the manuscript and released result tables as the numerical authority, and flag conflicts rather than blending incompatible versions.

## 1. The research story in plain language

We began with a more ambitious forecasting question: can intraday order-book information help predict overnight or next-day stock returns, including effects from one stock’s order flow on another stock? The older project used multi-level aggregation, time-decay weighting, cross-asset matrices, sector restrictions and denoising. Some historical backtests looked attractive. Those historical numbers were not validated merely by appearing in an earlier report.

The later reconstruction made the evaluation more explicit: chronological fitting, matching the baseline and book models on the same observations, checking trading spreads, ranking only large predictions, testing other periods, examining exposure and concentration, and distinguishing a stock-specific signal from common-market movements. It did not establish a reliable incremental net trading edge. There was modest short-lived minute queue information, but ranking did not rescue the tested strategies after spreads. A daily positive-looking result weakened under attribution, multiple-testing and earlier-period checks.

That did not make the data useless. It suggested a different question, one we could answer directly with the observations already present: what happens to the measured imbalance when an observed minute path is compressed into two endpoints?

This paper is therefore a change of scientific question, not a reinterpretation of the negative trading results as a positive strategy. Instead of asking whether a complex model extracts alpha, it asks whether two apparently similar definitions of a predictor are interchangeable. They often are not. That is a measurement result with implications for how snapshot studies are described and checked.

## 2. Start by explaining an order book

A limit order book contains standing offers to buy and sell. The **best bid** is the highest displayed buying price. The **best ask** is the lowest displayed selling price. The **bid size** and **ask size** are the displayed quantities at those prices. The midpoint is the average of the best bid and best ask; the spread is their difference.

For example, a bid of 100 with 10 shares and an ask of 101 with 20 shares is a valid, noncrossed top-of-book state. A market buy normally consumes sell-side liquidity; a limit order adds displayed quantity; a cancellation removes it. But a pair of snapshots does not label which combination of events caused the observed change.

Distinguish three objects:

1. **Queue imbalance:** a state, often `(bid size − ask size)/(bid size + ask size)` at one moment.
2. **A quote-transition OFI contribution:** a signed function of two successive observed quote states, incorporating price and quantity changes.
3. **Aggregated OFI:** the sum of contributions over a specified observation path.

These are not synonyms. A large standing bid queue is not a count of buys during an interval. A quote-transition contribution is not a trade-volume imbalance. A continuous-book snapshot is not a closing-auction imbalance feed.

Suggested visual: show two price ladders with quantities, highlight only the best level, and animate an order addition, cancellation and price move. Then freeze the screen at minute marks and show that several intervening events are hidden.

## 3. What data we have—and what we do not have

The local archive contains snapshots every sixty seconds. Some files contain one book level and some contain ten. Large processed files contain derived features; their size does not mean the missing event stream can be recovered from them. There is no complete sub-minute sequence in this study.

The paper uses 483 US stocks selected using coverage in 2017 and follows available data through 2019. Selection uses 2017 only, but the original archive itself may still have survivorship or coverage bias. Of the selected stocks, 378 have one-level source files and 105 have ten-level source files. **Every paper comparison uses only level one**, so the depth groups do not constitute a controlled experiment in adding deeper information.

The primary measurement has 25,475,162 valid five-minute blocks over 358,807 stock-days. A stock-day means one stock on one trading date; it is not a separate trading date. Blocks do not overlap, begin at 10:00, and end by 15:55. Shortened and exceptional sessions were removed upstream. An additional 135 candidate five-minute blocks fail the stated within-block validity or extreme-return checks. This exclusion count is conditional on the already constructed cache, not the number of raw records ever excluded during preprocessing.

A five-minute block contains six minute snapshots and five observed transitions. The paper also examines one-, two-, ten- and thirty-minute blocks and a five-minute grid shifted two minutes. At one minute, the two constructions coincide by definition.

The minute sum is a **finer observed reference**. We do not know the complete event path between minutes, and the experiment cannot measure either construction’s error relative to event-level OFI. It measures an additional loss caused by discarding observations that were already available.

Suggested visual: a timeline with event dots, minute snapshots and five-minute endpoints. Use different colors for “unobserved to begin with” and “observed, then discarded.” Those are two different information losses.

## 4. The quote-transition formula

For one bid-side transition, use price `p` and displayed quantity `q`:

\[
B(p_0,q_0;p_1,q_1)=q_1\mathbf{1}\{p_1\ge p_0\}-q_0\mathbf{1}\{p_1\le p_0\}.
\]

Explain the indicator as an on/off switch. There are three cases:

| Bid-price change | Contribution | Intuition within this convention |
|---|---:|---|
| Price rises | `+q₁` | Count the quantity displayed at the new higher bid. |
| Price falls | `−q₀` | Remove the quantity associated with the old best bid. |
| Price stays fixed | `q₁−q₀` | Count the net displayed size change at that price. |

For the ask, apply this same operator to **negative ask prices**, then subtract the result. The complete contribution is

\[
e(x_0,x_1)=B(b_0,q^b_0;b_1,q^b_1)-B(-a_0,q^a_0;-a_1,q^a_1).
\]

This convention makes an upward ask move positive and a downward ask move negative. Check the signs carefully in any animation. With a fixed ask price, an increase in ask size contributes negatively to net imbalance.

The standard event-level construction sums such contributions over observed book events. Applying it to minute quote states is an adaptation: individual minute transitions can hide many events.

## 5. The two quantities being compared

Let the block’s observations be `x₀,…,xₙ`. Let

\[
D_0=(q^b_0+q^a_0)/2>0
\]

be the mean displayed depth at the block start. The paper compares

\[
F_n=\frac{\sum_{j=1}^n e(x_{j-1},x_j)}{D_0},\qquad C_n=\frac{e(x_0,x_n)}{D_0}.
\]

`F` is the minute sum; `C` is the endpoint contribution. Both use the **same** starting-depth denominator. This is intentional: it isolates discarding intermediate states. Summing individually normalized minute contributions would introduce changing weights and would be a different experiment.

The key distinction is between summing retained contributions, which is additive by construction, and recalculating the transition formula on a new pair of endpoints, which is not generally additive. “OFI aggregation is invalid” would be an incorrect summary. Correct addition of already observed contributions is exactly the practical remedy.

### A numerical example with opposite signs

Use bid prices `100, 101, 102, 99`, constant bid size 10, and a fixed ask of 104 with size 10. The book stays noncrossed. The bid contributions are `+10, +10, −10`; the ask contributions are zero. Thus the sum is `+10`. The endpoint bid moves from 100 to 99, so its contribution is `−10`. With `D₀=10`, `F=+1` and `C=−1`.

Both calculations are correct applications of their respective definitions. The disagreement is caused by using different retained information. There is no need to invoke a forecasting model, fitted coefficient or trading cost to produce it.

This three-transition example is an illustration, not a five-minute empirical observation. If a five-transition illustration is desired, append two unchanged states at bid 99 and ask 104; they contribute zero and leave `F=+1`, `C=−1`.

## 6. The exact aggregation-error identity

One can rewrite the bid operator as

\[
B=(q_1-q_0)+q_0\mathbf1\{p_1>p_0\}-q_1\mathbf1\{p_1<p_0\}.
\]

When these expressions are summed along a path, all intermediate quantity differences cancel. That is telescoping. What remains after subtracting the endpoint contribution is

\[
E^B=\sum_{j=1}^n[q_{j-1}\mathbf1\{p_j>p_{j-1}\}-q_j\mathbf1\{p_j<p_{j-1}\}]
-q_0\mathbf1\{p_n>p_0\}+q_n\mathbf1\{p_n<p_0\}.
\]

Apply the same identity to negative ask prices, obtaining `Eᵃ`. Then

\[
F_n-C_n=(E^b-E^a)/D_0.
\]

The proof is a case check and cancellation, not an economic assumption. If the price stays fixed throughout, sizes telescope and there is no aggregation error. But monotone prices can still create magnitude errors: several upward transitions count several displayed queues, while the endpoint formula counts only the final queue. Do not say that reversals are the only source of error.

Suggested visual: cancel the successive `−q_j` and `+q_j` terms, while keeping the price-direction terms highlighted. Contrast a flat-price path, a monotone path and a reversing path.

## 7. What endpoint observations cannot identify

Keep bid size equal to `q`. Compare two paths:

- `p, p+δ, p+2δ, p`, giving summed bid contribution `+q`;
- `p, p−δ, p−2δ, p`, giving summed bid contribution `−q`.

Keep the ask fixed above both paths and its size fixed. Both paths have exactly the same starting and ending book state. An endpoint-only calculation cannot tell which occurred. Its endpoint contribution is zero in either case.

For any endpoint-only estimate `u`, at least one of `|u−q|` and `|u+q|` is at least `q`. Otherwise the two possible values, separated by `2q`, would both be less than `q` away from the same estimate. Repeating closed paths makes the ambiguity larger if the number of hidden intermediate observations is unrestricted.

This does **not** say a fixed five-minute grid has arbitrarily large error for fixed queues and a fixed number of transitions. Nor does it rule out statistical estimation under additional assumptions. It says endpoints alone do not universally recover the observed path sum. More sophisticated endpoint transformations cannot create information absent from the endpoints.

## 8. Why explaining the current price move is not enough

It is tempting to validate a flow variable by checking whether it fits the price move. But the quote-transition formula already uses price changes. This is not automatically wrong—contemporaneous price impact is a legitimate research question. It does mean explanatory fit is not independent evidence that two flow measurements agree or that either forecasts a future return.

The paper gives a deliberately simple mathematical example. Let minute log returns `X_j` be independent Gaussian draws with mean zero and variance `σ²`, with `σ>0`. Let the midpoint remain positive through a geometric random walk. Bid and ask are fixed proportions below and above it. Both queues have the same fixed size.

In this construction,

\[
R_n=\sum_j X_j,\quad F_n=2\sum_j\operatorname{sign}(X_j),\quad C_n=2\operatorname{sign}(R_n).
\]

The minute sum counts signed minute directions. The endpoint feature uses only the direction of the total return. Both have population contemporaneous linear-regression `R²=2/π≈63.66%` for every block length.

For the minute-sum result, explain:

- `Cov(ΣX_j, Σsign(X_j)) = n E|X|`;
- `Var(ΣX_j)=nσ²`;
- `Var(Σsign(X_j))=n`;
- a zero-mean Gaussian has `E|X|=σ√(2/π)`.

Putting these into squared correlation yields `2/π`. For the endpoint result, squared correlation between a Gaussian return and its sign is `(E|R|)²/E[R²]`, also `2/π`.

Despite equal fit, the probability that the two features have opposite signs approaches

\[
\arccos\sqrt{2/\pi}/\pi\approx20.60\%.
\]

The joint central limit theorem gives the limiting correlation `√(2/π)` between the standardized sum of returns and sum of return signs. The opposite-sign probability of that bivariate normal limit gives the formula. At even finite block lengths, the sign sum can be zero; the simulation reports those ties separately. It would be misleading to claim a simple monotone finite-length disagreement sequence from the asymptotic formula.

With 100,000 simulated paths and 1,001 increments, observed disagreement is 20.52%. This simulation checks the mathematical illustration. It does not calibrate a realistic market model. Following independent returns have zero covariance with either feature. The example is not evidence that real orders have no impact, and it is not a causal decomposition of real price formation.

Suggested visual: two gauges both reading about 64% explanatory fit, while their directional arrows disagree on a visible fraction of paths. Label this clearly as a synthetic model.

## 9. How the empirical disagreement rate is calculated

A comparison is “material” only if **both** `|F|` and `|C|` exceed 0.1 starting-depth units. This is not 0.1 dollars or a 10% price change. It is a dimensionless threshold after the stated depth normalization.

For each stock-day, divide opposite-sign material windows by all material windows. Average the stock rates within a date, then average dates equally. A stock-day with no material denominator is omitted from that conditional average. This weighting gives stocks and dates balanced influence; it is not simply the ratio of all pooled counts.

In 2019, the primary filter retains 95.91% of valid five-minute windows. There are 8,091,184 material comparisons, of which 1,377,269 have opposite signs. Their pooled ratio is close to but not exactly the headline 17.03%, because the headline uses equal-stock and equal-date weighting.

Uncertainty uses 2,000 circular bootstrap samples with 20-day blocks. Dates are resampled jointly across stocks, preserving cross-stock dependence within the sampled dates and some serial dependence within blocks. Millions of windows are not treated as millions of independent observations. This bootstrap does not remove all dependence assumptions or adjust for every exploratory research decision.

## 10. The main empirical findings

### Material sign disagreement in 2019

| Block length | Opposite signs | 95% day-block interval |
|---:|---:|---:|
| 1 minute | 0.00% | 0.00–0.00% |
| 2 minutes | 11.79% | 11.51–12.09% |
| 5 minutes | 17.03% | 16.82–17.24% |
| 10 minutes | 19.49% | 19.28–19.68% |
| 30 minutes | 21.61% | 21.36–21.87% |

The one-minute zero is a construction check, not an empirical discovery. At five minutes, disagreement is approximately one in six retained comparisons. This is a sign disagreement rate, not a forecasting error rate or percentage of profitable trades.

Additional checks:

- Five-minute rates: 15.98% in 2017 and 17.08% in 2018.
- Moving the five-minute grid two minutes: 17.27% in 2019.
- All 483 stock-level rates are positive in 2019; median 17.64%, 10th percentile 13.56%, 90th percentile 19.59%.
- One-level-source group: 16.77%; ten-level-source group: 17.97%. Both use level one; do not interpret this as a depth treatment effect.
- Lowering the threshold to 0.001 or 0.01 gives 17.91% and 17.80%.
- Bootstrap block lengths 10 and 40 give intervals 16.87–17.22% and 16.79–17.26% for the primary five-minute rate.

The findings are stable across these checks. That does not mean the study is representative of all markets or that these checks were all specified before any data were seen.

### Same-block explanatory fit

| Feature | 2019 five-minute R² |
|---|---:|
| Minute sum `F` | 23.38% |
| Endpoint contribution `C` | 24.25% |
| Endpoint with sizes frozen at block start `Z` | 54.32% |
| `tanh(F)` | 37.45% |
| `tanh(C)` | 49.88% |
| `sign(F)` | 32.31% |
| `sign(C)` | 48.72% |
| `sign(current return)` | 52.49% |

The last five transformations were investigated after the raw comparison. Their exploratory status must appear in the explanation, not be hidden in a footnote.

The frozen-size control is

\[
Z=\frac{q^b_0\operatorname{sign}(b_n-b_0)+q^a_0\operatorname{sign}(a_n-a_0)}{D_0}.
\]

It removes within-block queue-size changes but keeps price changes. It is not a future prediction and does not isolate an exogenous supply-and-demand shock.

The jump from 24.25% for raw endpoint OFI to 49.88% for bounded endpoint OFI is essential. Much of the apparent advantage of freezing queues depends on the feature’s functional form and sensitivity to magnitude. We cannot interpret the whole raw 24%-versus-54% gap as an economic effect of removing queue updates. The frozen feature still has about 8.86% lower MSE than bounded endpoint OFI in the matched comparison, but that also compares different nonlinear functions of observed quotes. It is not a causal result.

The return-sign benchmark intentionally uses the outcome. Its purpose is diagnostic. Calling its 52.49% fit a predictive result would be leakage by definition.

## 11. Fitting chronology, forward prediction and the audit correction

For each stock and scale, models train on 2017, select between two fixed ridge penalties using 2018, refit through 2018, and score 2019. Feature clipping and standardization are fitted on training data. The metric is one minus a ratio of equally aggregated model and training-mean benchmark losses, not an average of individual stock R² values.

However, these dates were already examined during earlier project work. A chronological split is useful protection against fitting on future observations; it is not proof that the research hypothesis was chosen without looking at the test years. The paper explicitly acknowledges this distinction.

For the next-five-minute return, the baseline uses current and previous block returns, endpoint spread, depth and queue imbalance, and trailing five- and thirty-minute returns. The tested additions are the minute sum or endpoint contribution. Their average incremental effects are tiny and negative in the reported comparison. Do not turn a statistically detectable loss difference into an economically important trading conclusion.

During the release audit, an inconsistency was found in the forward-sample mask: it screened the following-block return for extreme values but did not also apply the current-block extreme-return screen used by the measurement analysis. The filter was corrected and affected models rerun. The release audit document and current tables give the correction’s exact scope. This correction does not change the primary sign-disagreement counts. Explain it as a concrete example of checking the implementation rather than suggesting the original code was flawless.

This limited forward comparison does not prove that no OFI variable, cross-asset model or nonlinear method can ever predict returns. It answers a specific incremental question on these snapshots, controls and horizons.

## 12. What we were doing before, and how this differs

| Dimension | Earlier project and reconstruction | Current paper |
|---|---|---|
| Main question | Can book information predict future returns or support trading? | Does endpoint substitution preserve an already observed imbalance measure? |
| Primary object | Forecasts, cross-impact matrices, rankings and portfolios | Two deterministic constructions on the same quote path |
| Main horizons | Overnight/next-day in the original project; minute follow-ups in reconstruction | One to thirty minutes, primarily five-minute measurement blocks |
| Modeling emphasis | Decay kernels, structured matrices, denoising, multiple forecast families | Exact operator identity, identification example, transparent regressions |
| Success criterion | Increment beyond controls and costs, repeated across periods | Quantified disagreement and a defensible interpretation of fit |
| Main result | No validated incremental net trading edge from the reconstructed search | Material construction differences survive the stated checks |
| Claim to avoid | “An attractive backtest proves alpha” | “A measurement result proves trading value or overturns prior OFI research” |

Rigel’s ranking idea was reasonable: perhaps predicted magnitudes are poorly calibrated but the largest predictions identify useful trades. In the minute test, selecting the top 20 absolute forecasts did not solve the cost problem. A representative cross-book top-20 selection had average predicted magnitude 2.13 basis points versus average entry spread 9.26 basis points; tested independent round trips lost 7.71 basis points per selected stock-minute. A basis point is 0.01 percentage point. Ranking can preferentially select volatile, illiquid or expensive stocks rather than an executable edge. These figures belong to the earlier ranking study, not the current measurement experiment.

The broader decision study also examined an apparent +19.51-basis-point daily result. It weakened after common-market/sector attribution, concentration and multiple-testing checks, and did not establish a repeatable increment in a separate earlier period. Do not compare that portfolio return directly with an R² or sign-disagreement percentage; the units and estimands are different.

The current paper does not validate the original reported Sharpe ratios. It avoids relying on them. Its strongest result is direct and does not require selecting a winning forecast model.

## 13. What prior research already established

- **Cont, Kukanov and Stoikov (2014), The Price Impact of Order Book Events:** defines and sums event contributions and relates them to contemporaneous prices. Correctly retained increments are additive. This paper does not challenge that construction. https://doi.org/10.1093/jjfinec/nbt003
- **Su, Sun, Li and Yuan (2021), The Price Impact of Generalized Order Flow Imbalance:** explicitly addresses snapshot limitations and multi-tick changes using a generalized construction. This is the closest overlap. The current paper does not implement or outperform their estimator. https://arxiv.org/abs/2112.02947
- **Cont, Cucuringu and Zhang (2023), Cross-Impact of Order Flow Imbalance in Equity Markets:** distinguishes contemporaneous impact from lagged forecasting and studies integrated levels and cross-asset effects. The current paper’s question is temporal endpoint substitution. https://arxiv.org/abs/2112.13213
- **Gould and Bonart, Queue Imbalance as a One-Tick-Ahead Price Predictor:** studies predictive information in the book state, a different object and horizon. https://arxiv.org/abs/1512.03492
- **Xu, Gould and Howison, Multi-Level Order-Flow Imbalance in a Limit Order Book:** studies contemporaneous fit using more book levels. https://arxiv.org/abs/1907.06230

The proposed contribution is the combined empirical measurement audit and interpretation, not the invention of OFI, recognition that snapshots miss events, generic additive aggregation, or Gaussian sign identities. Literature searches cannot prove originality. A common-data generalized-OFI comparison and feedback from researchers in this area would strengthen the publication case.

## 14. The practical remedy

Before discarding an observed path, retain its summed bid contributions and summed reversed-ask contributions. Also retain the first and last timestamp and quote state. For normalization, retain the needed depth quantities: the starting quote for starting-depth normalization, or a depth sum and increment count for mean-depth normalization.

Adjacent chunks can then be merged by adding the summaries, provided their shared boundary time and state match and there is no gap. The shared observation is a boundary, not an extra duplicated transition. The implementation checks these conditions and is tested against direct aggregation.

This is a straightforward data-engineering practice, not a new compression algorithm. Its value is preventing avoidable loss. It cannot reconstruct the unobserved sub-minute events.

Suggested visual: compress a path into a small summary card. Merge two cards and recover exactly the sum obtained before compression. Contrast that with keeping just two endpoint pictures.

## 15. How the work was checked, and what those checks cannot establish

The release audit recomputed five-minute contributions directly from cached states across all 483 stocks and compared the resulting counts with saved outputs. It reproduced the primary sample and material counts. Direct summation and the original cumulative-sum implementation differed by at most about `1.73×10⁻¹¹` in normalized flow on that check. Selected models were refitted, and saved prediction losses were replayed independently of fitting.

The raw precision audit checks 18 original stock-day files across six stocks and all three years. Queue sizes had been reconstructed from float32 log-depth and imbalance. Four raw exact-zero flows became tiny cached nonzero values; the largest sampled normalized-flow difference was approximately 0.000122. That is why the headline uses a material threshold of 0.1 and reports smaller-threshold sensitivities. It is a sampled precision check, not a bound for every raw record.

The broader research test suite and the paper-specific tests cover signs, path identities, training-only preprocessing, future-state isolation, matching samples, merging and statistical routines. Passing tests does not prove every possible defect absent. It also cannot establish novelty, eliminate survivorship bias, supply missing event records or guarantee independent reproducibility without the data.

Raw data and per-stock fitted outputs are not redistributed. The paper repository contains source, synthetic tests and compact result tables. Readers can inspect the calculations and rerun the data-free checks; full empirical reruns require the original input archive or compatible licensed data.

## 16. A suggested chapter plan

1. **The puzzle:** show opposite arrows from the same observed interval. Ask why both calculations can be internally correct.
2. **The original goal:** explain daily prediction, the ranking idea and why trading evidence remained weak.
3. **What a snapshot contains:** introduce the book, state versus flow, and two kinds of missing information.
4. **Calculate one transition:** work through up, down and unchanged prices on both book sides.
5. **Sum versus endpoints:** do the `+10,+10,−10` example and normalize it.
6. **Why this is unavoidable:** explain the telescoping identity and identical-endpoint paths.
7. **The misleading validation shortcut:** derive the Gaussian example and equal 2/π explanatory fit.
8. **The actual study:** sample, chronology, material threshold, balanced weighting and bootstrap.
9. **Results:** animate the scale table, then compare raw, bounded and frozen explanatory features. Keep the bounded result prominent.
10. **The forecasting check:** distinguish current-price explanation from next-return prediction and explain the corrected mask.
11. **What to retain:** demonstrate the mergeable summary and its limits.
12. **What we can responsibly say:** a concrete measurement result, a plausible research note, unresolved novelty and event-level validation.

Finish with the scientific takeaway, not an investment slogan: **Specify what was observed, preserve what can be added, and do not use fit to answer a different measurement question.**

## 17. Final accuracy checklist for the narration

- “483 stocks” is the study universe; “358,807 stock-days” is not that many independent dates.
- The archive is sampled every minute; it is not a complete event feed.
- Both constructions use the same starting depth.
- Material means both absolute features exceed 0.1.
- 17.03% is a conditional, balanced sign-disagreement rate; it is not an accuracy score.
- 54.32% and 49.88% are same-block R² values; neither is future forecast accuracy.
- The tanh/sign checks are post-result; the full research history is exploratory.
- Source-depth groups both use the top level.
- The Gaussian example is an illustration, not an estimated market model.
- Current and following returns are different objects; the release filter correction must not be hidden if discussing audit history.
- The new measurement result does not rescue earlier trading backtests.
- No accepted paper, guaranteed novelty, causal mechanism or profitable strategy has been established.

## Sources to upload alongside this brief

Upload the current `paper/SNAPSHOT_OFI_PAPER.pdf`, this briefing, `docs/AUDIT_AND_READINESS.md`, and `docs/COMPARISON_WITH_EARLIER_WORK.md`. The compact tables in `results/` are the numerical source. The original broader research remains at https://github.com/RM1505/math279_project/tree/research/ofi-snapshot-study-2026/replication_2026. The separate paper repository is https://github.com/brendanconnelly/snapshot-ofi-paper; its private status may prevent automatic URL ingestion, so uploading the documents is the reliable handoff.
