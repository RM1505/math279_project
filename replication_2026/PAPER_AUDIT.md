# Release audit and readiness assessment

**Audit date:** October 22, 2026.  
**Assessment:** Ready for coauthor and Mihai review as a carefully qualified measurement paper. The data-free package is runnable and the primary empirical result has been independently recomputed from the complete cached study universe. This is not certification of novelty, event-level validity, the original trading claims, or acceptance readiness independent of author approval.

## What was actually audited

| Workstream | Work performed in this audit | What this does not establish |
|---|---|---|
| Primary measurement | Directly summed five-minute increments across all 483 cached stocks, independently of the production prefix-sum calculation; compared every material stock-day count | Does not observe events between snapshots or independently reacquire the entire raw archive |
| Sample and weighting | Reproduced all 25,475,162 valid primary windows, 358,807 stock-days, 135 exclusions, annual balanced rates and 2019 material counts | Does not eliminate archive-level selection or survivorship bias |
| Mathematics | Rechecked the bid/ask sign convention, telescoping identity, endpoint ambiguity, Gaussian squared-correlation argument and limiting sign probability | Elementary identities and correctness are not proof of originality |
| Models | Refit 15 selected primary same-block models and compared parameters, penalties and validation loss; replayed 210 saved models against 7,560 daily losses | Not a complete independent reimplementation/refit of every historical model |
| Raw precision | Reran the 18-stock-day audit from original archives across six stocks and three years | A selected sample, not an exhaustive raw-data error bound |
| Broad research tests | Reran all 39 tests after adding a forward-filter regression test | Tests are not a new empirical replication of every older strategy |
| Paper checks | Checked all scale/fit table rows, new robustness numbers and scope statements against saved results | Internal consistency alone is not independent evidence about the market |
| Repository | Separated paper source, compact outputs, tests and documentation from raw data and older experiments; checked portable paths and the standalone package | The repository cannot reproduce the full empirical study without compatible data |
| Earlier conclusions | Compared original manuscript claims with reconstruction, ranking and decision reports | No independent validation of the original reported Sharpe ratios is claimed |

The main direct-sum audit reproduced 8,436,055 valid five-minute windows in 2019, 8,091,184 material comparisons and 1,377,269 opposite-sign comparisons. The balanced rate is 0.17031980599049115. The direct-versus-prefix calculation differed by at most approximately 1.73e-11 normalized units, without changing any material count. See `results/release_audit.json`.

## A real issue found and corrected

The initial forward-sample mask required valid current quotes and a non-extreme following-block return, but omitted the current-block extreme-return exclusion used by the measurement sample. That meant an extreme current return could enter as a predictor during fitting even though the measurement sample excluded that interval.

The all-universe scan found six affected training windows, all for **KR on October 11, 2017**: two one-minute windows, two two-minute windows, one primary five-minute window, and one shifted five-minute window. No additional stock was affected on the tested grids. The return jumps were extreme source observations, not evidence of a tradeable market move.

The correction requires finite current returns with absolute log return below 0.2 in the forward mask as well. The old per-stock and aggregate results were preserved locally before regeneration. KR's models were refitted, aggregate summaries regenerated, and a regression test added that would fail under the old mask. The current paper documents the correction.

**Impact:** measurement counts and explanatory results are unchanged. The rounded five-minute forward numbers and substantive negative-increment conclusion are unchanged. Across all reported grids, the maximum absolute R² change is approximately 4.442e-6, or 0.0004442 percentage points. The exact changes and affected windows are recorded in `results/forward_filter_fix.json`. Small impact does not make the omitted check acceptable; it is corrected rather than dismissed.

## Other improvements made

- Clarified that the headline material cutoff followed the raw precision audit; it is not presented as a pristine preregistered threshold.
- Made the ridge loss normalization and unpenalized intercept explicit.
- Kept the 49.88% bounded endpoint fit prominent alongside the 54.32% frozen-size result.
- Removed ambiguous “current paper” navigation pointing to both the obsolete Word draft and the current LaTeX manuscript.
- Made data paths configurable in the focused repository and documented the exact coverage rule, input schema and output-reuse behavior.
- Preserved original source hashes separately from hashes of the path-adapted package.
- Added a detailed comparison with the earlier project and a NotebookLM handoff that does not conflate fit, forecasts and profits.

## Results that remain supported

1. Endpoint substitution changes the measured direction in 17.03% of material five-minute comparisons in the balanced 2019 estimand, with a 95% interval of 16.82–17.24%.
2. The discrepancy increases on the coarser grids considered and is present across the stated years, phase, stocks and source-depth groups.
3. High contemporaneous fit can coexist with substantial disagreement between measurements, both in the synthetic example and in the empirical diagnostics.
4. Bounding endpoint OFI closes much of the raw frozen-versus-endpoint fit gap; the full raw gap is not a causal effect of queue updates.
5. Neither tested five-minute forward addition establishes a positive incremental loss improvement over its matched baseline.
6. Retaining additive contributions before dropping observations prevents additional endpoint-substitution loss, but does not recover hidden events.

## Readiness boundaries

**Ready now:** a nine-page coauthor-review manuscript, mathematical arguments, auditable aggregate evidence, data-free tests, reproducibility instructions, and an explanatory-video source brief.

**Still needed before formal submission:** both authors' final approval, current affiliations and presenting/corresponding-author details, confirmation of any data-use obligations, and the chosen venue's submission requirements/payment. The existing conference plan is a recommendation, not a submitted application.

**Still scientifically unresolved:** whether the combined audit is sufficiently distinct from generalized snapshot OFI and other temporal-measurement work; how either measure compares with true event-level OFI; how broadly the result generalizes beyond this archive. An expert novelty review and a common-data GOFI comparison would strengthen the case. The one-level majority constrains that benchmark; it does not justify pretending the comparison was performed.

**Not ready to claim:** a profitable strategy, validated original daily Sharpe ratios, universal failure of OFI prediction, a causal decomposition of price impact, or a guaranteed publishable contribution.

## How to read the provenance files

- `results/implementation_manifest.json` records the shared study source versions after correction.
- `results/release_audit.json` records the complete primary-cache audit and selected refits.
- `results/forward_filter_fix.json` records the correction and its numerical impact.
- `results/raw_verification.csv` and `results/fit_replay.csv` record sampled raw and saved-fit replay checks.
- `RELEASE_MANIFEST.json` hashes the focused repository's deliverables and adapted code. Adapted source hashes legitimately differ from the historical study hashes.

The old backups remain local and were not mixed into the current tables. The shared research repository preserves the broader history. The new private repository is a focused presentation of the paper, not a rewrite of that history into a clean confirmatory experiment.
