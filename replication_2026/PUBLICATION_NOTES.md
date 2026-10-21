# Shared research branch

Published to `RM1505/math279_project` at Brendan's request. Start with [NEXT_DAY_REPORT.md](NEXT_DAY_REPORT.md), then [RANKING_REPORT.md](RANKING_REPORT.md). The original [REPORT.md](REPORT.md) covers the completed minute-snapshot reconstruction; its statements about no external publication refer to the earlier local run.

The branch includes source, tests, compact aggregate tables and figures. Raw books, inventories, minute caches, daily panels, fitted model objects and individual predictions remain local. Report references to these local provenance files do not mean they are bundled in this public repository.

For reproduction, use the dependency versions in the implementation manifests, create a project environment, and point `RAW` in `study.py` at the existing raw archives. Follow the extraction and analysis commands in the reports in order. Daily models require the original archive inventories and 2017–2019 caches; later daily features are rebuilt from raw 2020–2021 snapshots. Existing output files are reused, so isolate or clear stale generated outputs when changing the specification. Summary files are explicitly included despite broad ignore rules that protect against committing large outputs.

These are snapshot-data adaptations, not exact event-level OFI or trade-co-occurrence replications. No historical Sharpe claim is validated. There is no established new positive research contribution or trading strategy from these results.
