# Research log

## 24 October 2026 — extension completed

1. Fixed a bounded extension protocol before reading its new outcomes: selection stability, endpoint reconstruction, generalized-OFI coverage, preservation and prior-use audit. These remain exploratory because the archive was previously studied.
2. Ran all 483 stocks for 2019 top-k selection and endpoint reconstruction, with chronological fitting. Retained every specified k and scaling variant and both fitted reconstruction models, including negative results.
3. Implemented two explicitly distinguished generalized conventions after inspecting the 2021 equations. Extracted only four selected raw stock-days per ten-level stock, one archive at a time. Checked raw timestamps and recorded source hashes locally.
4. Observed different material denominators in the first generalized comparison. Added and labeled a post-result all-four-material comparison, reran the same sample, and used that paired result in the manuscript. Also reported the unfiltered four-date standard rate and coverage. No temporal bootstrap was presented for four dates.
5. Checked the current original daily helper and a pinned public implementation. Both retain sums of contributions. Added a published example of correct summation and removed any implication of widespread misuse.
6. Hardened chunk preservation with spacing metadata and explicit normalization methods. Added formula, finite-depth, grid and ranking tests. Documented that arbitrary later weights or partial-chunk queries require additional retained information.
7. Independently replayed 1,704 timestamp/scaling/k selection calculations from direct transition sums on four dates. Refit ten reconstruction models for five fixed stocks and replayed their raw-endpoint references; compared all their 2019 daily losses and sign counts. Confirmed that the complete reconstruction sample reproduces the original headline counts.
8. Rewrote the nine-page LaTeX paper in Brendan's direct, measured style. Reframed the contribution around selection consequences and remedies, retained qualifications about nonlinear fit, and expanded the closest-prior comparison. Updated the briefing, audit, reproduction instructions and release records.

The comparison protocol is in [EXTENSION_PROTOCOL.md](EXTENSION_PROTOCOL.md), findings in [EXTENSION_REPORT.md](EXTENSION_REPORT.md), and novelty/usage evidence in [EXTENSION_LITERATURE.md](EXTENSION_LITERATURE.md). Machine-readable results and audit records are in `results/extension/`.

## Stopping rule

The specified extension family is complete. More optimization on the same explored years would not resolve distinctness or event-level accuracy. Those require expert novelty feedback or independent data. This record does not label those unresolved questions as solved or claim a formal submission has occurred.
