# Follow-up novelty check — 20 October 2026

**Assessment: useful reconstruction and a possible narrow extension; distinct publication-level novelty is not established.** The generic findings—book imbalance predicts short returns, signals decay, and execution costs can overwhelm them—are established. This review found a closer closing-session precedent than the initial audit identified.

## Closest overlaps

| Primary source | Relevant prior result | Difference from our study |
| --- | --- | --- |
| Cont, Cucuringu & Chao Zhang, *Cross-impact of order flow imbalance in equity markets* (2023), sections 4–5 ([full text](https://arxiv.org/html/2112.13213)) | Cross-asset OFI adds short-horizon prediction; the effect decays. They exclude opening/closing half-hours and suggest studying the last 15–30 minutes. Their forecast portfolio trades only when predictions exceed spreads and reports economic gains. | We use endpoint proxies and queue states, simple market factors, annual fits and explicit return controls. Our unconditional quote-cost losses do not refute their different strategy. Their suggested extension is motivation, not proof of an unoccupied research question. |
| Lu, Reinert & Cucuringu, *Trade co-occurrence, trade flow decomposition and conditional order imbalance in equity markets* (2024), §8.3 / appendix time-of-day analysis and cost appendix ([full text](https://arxiv.org/html/2209.10334)) | Explicitly separates 09:30–10:00, 10:00–15:30 and 15:30–16:00, finding different predictive contributions across trade categories and periods. A single-sort strategy loses at costs above 2 bps; selected double-sort strategies remain profitable under their assumed costs. | Uses classified executed trades and next-day open-to-close outcomes, rather than our sampled resting queues predicting the next minute within each session. Nevertheless, “imbalance information differs near the close” already overlaps with Mihai’s work. |
| Albers, Cucuringu, Howison & Shestopaloff, *To Make, or to Take, That Is the Question* (arXiv preprint, 2025), §6 ([full text](https://arxiv.org/html/2502.18625v1)) | Live Bitcoin-perpetual experiments show that simple imbalance strategies can fail despite predictive information: taker fees erase gains; maker fills suffer adverse selection. The paper also develops a more sophisticated reversal approach. | Different market, event resolution and execution experiment. It directly anticipates the general “predictive but unprofitable” lesson. This is a plausible match to the remembered project, not a confirmed identification. |
| Yichi Zhang, Cucuringu, Shestopaloff & Zohren, *ClusterLOB* (2025 preprint; published 2026) ([preprint](https://arxiv.org/html/2504.20349v1)) | Clusters individual market-by-order events and constructs group-specific imbalance signals, evaluated using thirty-minute buckets and trading strategies. | Different information and methodology; our snapshots cannot reproduce its event classifications. It is additional evidence that conditional imbalance signals are an active, populated research area. |
| Baltussen, Da & Soebhag, *End-of-Day Reversal* (April 2025 manuscript) ([author PDF](https://academicweb.nd.edu/~zda/EOD.pdf)) | Finds cross-sectional reversals in the final half-hour related to earlier intraday performance. | Motivates our return-since-open control. That control does not eliminate every nonlinear reversal, liquidity or inventory explanation. |

## What can still be claimed

Our specific comparison is a one-minute **queue-state increment beyond return-history controls**, contrasting core and closing sessions on matched stock-days, with a secondary return-since-open control. Its headline closing advantage is about 0.110 percentage points of relative MSE improvement. This is narrower than an OFI-alpha discovery: the strongest finding is queue state, while the endpoint-flow increment is small and specification-sensitive.

I did not locate an exact match to that entire design in this targeted public-source search. That absence does not establish novelty. Different aggregation, estimator or sample alone is usually an empirical variation rather than a substantive new contribution. The 2017–2019 period also overlaps the motivating cross-impact study and is not a fresh historical replication period.

## Recommended decision

Do not present this as a novel discovery yet. Describe it as a careful reconstruction with a candidate closing-session extension. Before expanding the model family, test whether the closing advantage survives comparisons within similar spread/tick-size, depth and volatility conditions, and a separate period already available in the archive. Including liquidity controls in a pooled baseline does not establish that the session difference is independent of liquidity composition or changing forecast-error variance. Those would be new, explicitly exploratory tests—not additional evidence already obtained.

The useful question for Mihai is whether this exact conditional, within-session one-minute comparison has already been explored by his group, particularly relative to the Lu and Albers papers. Public literature cannot rule out an unpublished student project. No claim is made here about which coauthor was the student remembered by Brendan.

## Search scope and limits

Checked Mihai’s current [publication list](https://math.ucla.edu/~mihai/index.html), the primary texts above, and targeted searches for closing-half-hour imbalance prediction and transaction-cost failures. The list also identifies Sitaru, Calinescu & Cucuringu’s *Order Flow Decomposition for Price Impact Analysis in Equity Limit Order Books* (ICAIF 2023); full text was not successfully retrieved in this pass, so it remains an explicit overlap check rather than a paper whose conclusions are asserted here. This was a brief targeted review, not an exhaustive citation census. No new empirical models were run and no follow-up email was sent.

## Next-day follow-up clarification

The later reconstruction tests next-session, overnight and combined price returns using daily summaries of minute snapshots. That is a different horizon from the completed minute study, but daily prediction and portfolio sorts are themselves established in Lu, Reinert and Cucuringu. The available snapshots cannot reconstruct their trade-co-occurrence classifications.

A subsequent check of the [Sitaru, Calinescu and Cucuringu SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4572510) confirms its use of market-by-order event types to decompose OFI on 100 stocks over three years, reporting forward-looking statistical and economic improvement. Thus it should not be described as simply another negative trading result. Full-text implementation and cost assumptions remain unchecked. No exact-replication or novelty claim follows from a different snapshot aggregation or model family.
