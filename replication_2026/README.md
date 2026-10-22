# OFI research: snapshot measurement, forecasts and replication

**Short paper:** [Read the four-page paper](SNAPSHOT_OFI_PAPER.md) or [download the editable version](SNAPSHOT_OFI_PAPER.docx). The paper develops the measurement result, checks its interpretation and states the limits.

**Supporting study:** [Snapshot measurement research note](RESOLUTION_REPORT.md) tests when endpoint OFI disagrees with summed minute increments. It includes an exact aggregation-error identity, fixed-queue price-input controls, a 483-stock resolution audit and an aggregation-preserving utility. This is a concrete methods-paper candidate; novelty and publishability remain unconfirmed. See also [theory](RESOLUTION_THEORY.md), [protocol](RESOLUTION_PROTOCOL.md) and [closest prior work](RESOLUTION_LITERATURE.md).

**Earlier decision study:** [Decision memo for the Mihai discussion](DECISION_REPORT.md) covers the earlier-period replication, exposure and concentration checks, relative-return/liquidity extensions, risk forecasts, and matched closing-session analysis. The recommendation is to stop the broad daily snapshot-alpha search and discuss the negative reconstruction honestly.

The next-day follow-up is complete: start with [NEXT_DAY_REPORT.md](NEXT_DAY_REPORT.md). It restores the original daily horizon, compares seven model families with matched price-history baselines, and tests top-20 and long-10/short-10 rankings. No positive primary book increment is established; all tested next-session ranking variants have negative average returns after spreads. [RANKING_REPORT.md](RANKING_REPORT.md) separately answers the minute-horizon ranking suggestion. [NOVELTY_REVIEW.md](NOVELTY_REVIEW.md) explains the substantial overlap with prior work.

The full minute-snapshot study is also complete. Start with [the research report](REPORT.md), which presents the actual findings, uncertainty, trading-cost diagnostics, and limitations. This supersedes the bounded pilot in [PILOT_README.md](PILOT_README.md).

The run audited 516 archives, considered 377,826 stock-day files for 2017–2019, and scored 42,655,513 one-minute observations across 483 stocks in 2019, including 105 genuine ten-level names. Original data and prior project outputs remain unchanged.

Queue imbalance adds modest predictive information beyond return history, with a larger increment near the close. This is a snapshot-data adaptation, not an exact event-OFI replication or validation of the old Sharpe ratios. Observed spread costs overwhelm the tested unconditional trading signals.

- [Report and results](REPORT.md)
- [Primary specification](PRIMARY_PROTOCOL.md) and [secondary specification](PROTOCOL.md)
- [Folder and literature audit](LITERATURE_AND_AUDIT.md)
- Main results: `full_study/summary.csv`, `inference.csv`, `stock_heterogeneity.csv`
- Additional open-to-current return control: `full_study/session_control_summary/`
- Input provenance: `full_study/archives.csv`, `members.csv.gz`, `cache/*_audit.csv`
- Reproduction commands and methodological caveats are in the report.

[Event-data acquisition](DATA_REQUEST.md) is optional future work. It is not needed to use or reproduce this completed study.
