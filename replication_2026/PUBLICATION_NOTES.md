# Shared research branch

Published to RM1505/math279_project on 20 October 2026 at Brendan’s request. Start with [REPORT.md](REPORT.md). The report and audit describe the local research run before publication; statements that no external repository was modified refer to that run.

This branch includes source, tests, compact aggregate tables and the figure. Raw books, caches, per-stock score files, member inventories and large outputs stay local. References to those files in the report document local provenance, not files bundled here.

For reproduction, use Python and dependency versions recorded in full_study/implementation_manifest.json, create the project .venv, and point RAW in study.py to your local archive directory. Original local paths are retained in the audited source. Full extraction and fitting stages must run before saved-model replay checks. The report generator expects the secondary run too. Summary CSVs were intentionally included despite broad ignore rules, which protect against accidentally committing large regenerated results.

These are minute-snapshot adaptations, not exact event-level OFI replication or validation of historical Sharpe claims.
