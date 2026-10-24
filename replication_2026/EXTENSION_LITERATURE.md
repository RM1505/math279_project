# Literature and usage audit — 24 October 2026

This is a targeted literature review, not a systematic review or proof of novelty. Sources are distinguished by what was actually available. The existing [initial review](RESOLUTION_LITERATURE.md) records the preceding search; the new results replace its statement that no generalized benchmark had yet been implemented.

## Claim-by-claim assessment

| Claim | Earlier evidence | Present contribution and limit |
|---|---|---|
| OFI adds transition contributions over time | [Cont, Kukanov and Stoikov (2014)](https://arxiv.org/html/1011.6402v3) defines the event contributions and their interval sum. | No novelty claimed for addition or the operator. We compare a sum on a retained minute path with a single transition on its endpoints. |
| Snapshots miss intermediate events; multi-tick moves require care | [Su, Sun, Li and Yuan (2021)](https://arxiv.org/pdf/2112.02947), full text and equations (9)–(10) inspected, explicitly addresses snapshot limitations and generalized OFI. | Closest prior work. We now compare aggregation stability on a bounded common raw sample, distinguishing literal tick-grid coverage from a price-aligned adaptation. This is not replication of their price-fit estimates or log-GOFI. |
| Proper summation is used in published research | [Ghysels and Nguyen (2019), p.17](https://www.econstor.eu/bitstream/10419/239078/1/1691798193.pdf) explicitly accumulates tick-frequency snapshot transitions into one-minute OFI. | A counterexample to any claim that snapshot papers generally substitute coarse endpoints. The relevant primary-source page was returned in full in indexed search text; direct publisher access returned 429 and direct repository access a bot challenge. We did not claim full-paper retrieval. |
| Multiple levels can add explanatory information | [Xu, Gould and Howison (2019)](https://arxiv.org/abs/1907.06230). | Spatial level integration differs from this temporal comparison. Our source-depth groups in the main study both use level one. |
| Explanation and prediction are different | [Cont, Cucuringu and Zhang (2023)](https://arxiv.org/html/2112.13213) treats contemporaneous and lagged cross-impact separately. | We do not claim that distinction is new or that own-stock minute results refute cross-impact. |
| Static queue state can be predictive | [Gould and Bonart (2015)](https://arxiv.org/abs/1512.03492). | Queue imbalance is a state statistic, not the same temporal operator as OFI. |
| This exact measurement audit has not previously appeared | No exact match found by the targeted searches below. | Absence from search results is weak evidence. Candidate distinction is the combined paired measurement/ranking/coverage evidence, not the underlying principle. Expert review remains needed. |

## Generalized-OFI versions must not be conflated

The implemented reference is the 2021 arXiv preprint, equations (9)–(10), whose abstract discusses CSI 500 examples and log-GOFI. A later [2022 SSRN version](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4234479), *The Price Impact of Generalized Order Flow Imbalance: Evidence from Chinese Financial Market*, has an indexed abstract describing CSI 300 examples and different plain-GOFI fit figures. Only that abstract/metadata was available; direct access returned 403. We do not assume identical estimators, datasets or estimates across versions. Neither source gives permission to describe our bounded US comparison as a full replication.

## Usage checks

1. The current original local `src/ofi.py` computes successive minute quote contributions, normalizes by current mean best size, and `compress_day_df_fast` sums those contributions with exponential time weights. The checked hash and exact scope are in `results/extension/original_pipeline_audit.json`. This is evidence about the current helper, not every historical run or the upstream feed. It is not the coarse-endpoint substitution studied in the new paper.
2. A public [OFI feature implementation, pinned revision](https://github.com/poojitha376sp/order-flow-imbalance/blob/60b02458a85eec01cef1e115081cf9553d501b48/src/features/ofi.py#L371) resamples `e_n` and multi-level contributions with sum in `bucket_ofi`; the relevant lines 371–402 were directly inspected. This confirms that this aggregation stage retains accumulated contributions. It does not audit the entire application or certify its trading claims.
3. Ghysels–Nguyen provides the published correct-summation example above. Additional software search results described rolling sums, but no broader prevalence estimate is inferred from them.

We did not establish that endpoint substitution is common in deployed systems or published work. Ambiguous phrases such as “snapshot OFI” and “one-minute OFI” are insufficient to diagnose a mistake: they can describe a sum of many fine observations. A paper should specify both the observation grid and aggregation window.

## Search record

Queries used on 24 October 2026 included:

- `"order flow imbalance" "snapshot" aggregation sampling generalized`
- `"order flow imbalance" "non-additive" OR "endpoints" OR "temporal aggregation"`
- `"order flow imbalance" "temporal aggregation" sampling`
- `"order flow imbalance" "sampling frequency" bias`
- `"order flow imbalance" "snapshot" "sum" github resample`
- `"order flow imbalance" "endpoint" "aggregation"`
- `"Ghysels" "Nguyen" "order flow imbalance" "one-minute"`
- `"Price Discovery of a Speculative Asset" OFI snapshots`

Primary sources were prioritized. Marketing explanations, search snippets using OFI to mean trade-volume imbalance, and unrelated timestamp/venue problems were not treated as direct precedents or evidence of implementation faults. Search-engine coverage, alternate terminology, inaccessible full texts, unpublished work and version differences limit the review. No citation-count claim or first-ever claim is made.

## Importance beyond OFI

The mathematical issue is a familiar distinction between path-dependent quantities and endpoint changes. Log returns telescope; realized variation and many activity measures do not. That analogy helps explain the problem but is not new empirical evidence for other data products. We have measured this particular OFI substitution in this archive. We have not measured an industry-wide compression defect or established the size of any analogous error in another market.

The practical recommendation is to preserve the statistic's needed contributions before removing observations, identify the grid and normalization, and report sensitivity of the actual downstream decision. No change to the standard event-OFI definition is required.
