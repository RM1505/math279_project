# OFI research: current paper and earlier experiments

## Current paper

**What snapshot sampling does to order flow imbalance** is the current nine-page LaTeX manuscript. The paper audits endpoint substitution on the same observed minute paths; it does not claim a new trading strategy or event-level replication.

- [Read the PDF](SNAPSHOT_OFI_PAPER.pdf) · [LaTeX source](SNAPSHOT_OFI_PAPER.tex) · [Overleaf package](SNAPSHOT_OFI_LATEX.zip)
- [Audit and readiness](PAPER_AUDIT.md): all-universe primary measurement recomputation, a corrected forward filter, tests and remaining limitations.
- [Comparison with earlier work](PAPER_COMPARISON.md)
- [Detailed NotebookLM video brief](NOTEBOOKLM_VIDEO_BRIEF.md)
- [Dedicated paper repository](https://github.com/brendanconnelly/snapshot-ofi-paper), initially private for coauthor review. This shared branch retains the paper and evidence for existing collaborators.
- [Protocol](RESOLUTION_PROTOCOL.md), [theory](RESOLUTION_THEORY.md), [prior work](RESOLUTION_LITERATURE.md), and [full measurement report](RESOLUTION_REPORT.md).

The release audit corrected a missing current-return cutoff in the forward mask. Six KR training windows across four grids were affected; the measurement findings and rounded five-minute conclusions are unchanged. See the audit for exact scope. The PDF and current tables incorporate the fix. The older four-page [Markdown draft](SNAPSHOT_OFI_PAPER.md), Word file and its builder are historical, superseded artifacts retained for provenance.

### October 24 extension

The rewritten nine-page paper adds top-20 selection instability (51.71% overlap), negative endpoint-reconstruction results, and a bounded generalized-OFI comparison that improves aggregation stability with substantial depth censoring. The current original daily helper already sums weighted minute contributions; the endpoint issue is not a demonstrated explanation of the earlier strategy's weakness.

- [Extension findings and limits](EXTENSION_REPORT.md)
- [Literature and usage audit](EXTENSION_LITERATURE.md)
- [Extension protocol](EXTENSION_PROTOCOL.md) and [research log](RESEARCH_LOG.md)

The dedicated paper repository contains the runnable extension implementation. This shared branch mirrors the manuscript, briefing and compact extension outputs so the joint research record remains current.

## Earlier forecasting and trading work

These studies answer different questions and remain part of the research record. They are not evidence that the new measurement result is profitable.

1. [Decision memo](DECISION_REPORT.md): daily replication, exposure/concentration checks, earlier-period comparison, risk and relative-return extensions. No validated incremental net alpha.
2. [Next-day study](NEXT_DAY_REPORT.md): restores the original daily horizon and tests matched baselines and ranking rules.
3. [Minute ranking follow-up](RANKING_REPORT.md): directly tests Rigel's top-20 proposal; the tested rules do not overcome spreads.
4. [Minute-snapshot reconstruction](REPORT.md): modest short-lived queue information, with execution and cost limitations.
5. [Original folder/code and literature audit](LITERATURE_AND_AUDIT.md), [novelty review](NOVELTY_REVIEW.md), and [historical pilot](PILOT_README.md).

## Data and reproduction

The archive audit covered 516 archives; the 2017–2019 inventory contained 377,826 stock-day members. The main measurement study uses 483 training-eligible stocks at level one. The October 24 extension adds a bounded ten-level generalized comparison for 105 stocks on four dates. Raw archives, caches, per-stock forecasts and fitted parameters remain local. Compact aggregate results are under `full_study/`.

For a focused runnable package, use the dedicated repository's reproduction guide. This broader branch intentionally preserves the original experiment history and protocols. Do not interpret a chronological split as an untouched research holdout or rerun a changed specification over stale per-stock outputs. The optional [event-data request](DATA_REQUEST.md) concerns a different, stronger validation question.
