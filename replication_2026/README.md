# OFI reconstruction and closing-session study

The full minute-snapshot study is complete. Start with [the research report](REPORT.md), which presents the actual findings, uncertainty, trading-cost diagnostics, and limitations. This supersedes the bounded pilot in [PILOT_README.md](PILOT_README.md).

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
