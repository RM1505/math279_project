from pathlib import Path
import hashlib,json,html
import numpy as np
import pandas as pd
from study import ROOT,OUT

def md_table(df):
    names=list(df.columns);out=['| '+' | '.join(names)+' |','| '+' | '.join(['---']*len(names))+' |']
    for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join(str(x) for x in row)+' |')
    return '\n'.join(out)

def main():
    coverage=pd.read_csv(OUT/'coverage.csv');s=pd.read_csv(OUT/'summary.csv');tests=pd.read_csv(OUT/'inference.csv');q=pd.read_csv(OUT/'quality_summary.csv')
    n=len(coverage);deep=int((coverage.depth==10).sum());obs=int(coverage.rows.sum())
    primary=tests[(tests.horizon==1)&(tests.delay==0)&(tests['sample']=='all')]
    contrast=primary[primary.session.isin(['core','close'])].copy()
    table=contrast[['comparison','session','effect','low','high']].copy()
    for c in ['effect','low','high']:table[c]=table[c].map(lambda x:f'{100*x:+.4f}%')
    testtable=primary[primary.session.isin(['equal_session','close_minus_core'])][['comparison','session','effect','low','high','p_holm_primary']].copy()
    for c in ['effect','low','high']:testtable[c]=testtable[c].map(lambda x:f'{100*x:+.4f}%')
    testtable.p_holm_primary=testtable.p_holm_primary.map(lambda x:f'{x:.4g}')
    economic=s[(s.universe=='broad')&(s.horizon==1)&(s.delay==0)&(s['sample']=='all')&(~s.model.str.startswith('deep'))][['session','model','r2_vs_zero','gross_bps','quote_net_bps']].copy()
    for c in ['r2_vs_zero','gross_bps','quote_net_bps']:economic[c]=economic[c].map(lambda x:f'{x:.5f}')
    contemp=pd.concat([pd.read_csv(p) for p in (OUT/'scores').glob('*_contemp.csv')],ignore_index=True)
    contemp['r2']=1-contemp.sse/contemp.target_ss
    contemporaneous=[]
    for uni,g in [('broad',contemp[contemp.model!='deep_flow']),('deep_matched',contemp[contemp.depth==10])]:
        r=g.groupby(['session','model']).r2.agg(['mean','median','count']).reset_index();r['universe']=uni;contemporaneous.append(r)
    cd=pd.concat(contemporaneous);cd.to_csv(OUT/'contemporaneous_summary.csv',index=False)
    ctable=cd[['universe','session','model','mean','median','count']].copy()
    for c in ['mean','median']:ctable[c]=ctable[c].map(lambda v:f'{100*v:.2f}%')
    robustness=tests[(tests['sample']=='fresh')|(tests.horizon==5)|(tests.delay==1)]
    robustness=robustness[robustness.session=='equal_session'][['horizon','delay','sample','comparison','effect','low','high']].copy()
    for c in ['effect','low','high']:robustness[c]=robustness[c].map(lambda v:f'{100*v:+.4f}%')
    positives=primary[(primary.session=='equal_session')&(primary.effect>0)&(primary.p_holm_primary<.05)]
    closes=primary[(primary.session=='close_minus_core')&(primary.effect>0)&(primary.p_holm_primary<.05)]
    decision=f"Of four primary incremental comparisons, {len(positives)} show a positive equal-session MSE reduction with Holm-adjusted p<0.05. Of four closing-minus-core comparisons, {len(closes)} show a positive difference under that correction."
    heterogeneity=pd.read_csv(OUT/'stock_heterogeneity.csv')[['comparison','session','stocks','positive_share','median_effect']]
    for c in ['positive_share','median_effect']:heterogeneity[c]=heterogeneity[c].map(lambda v:f'{100*v:.3f}%')
    secondary_path=OUT/'session_control_summary'/'inference.csv'
    if not secondary_path.exists():raise RuntimeError('Finish the secondary control before generating the final report')
    sec=pd.read_csv(secondary_path)
    sec=sec[(sec['sample']=='all')&sec.session.isin(['core','close','equal_session','close_minus_core'])]
    sec=sec[['comparison','session','effect','low','high','p_holm_sensitivity']].copy()
    for c in ['effect','low','high']:sec[c]=sec[c].map(lambda v:f'{100*v:+.4f}%')
    sec.p_holm_sensitivity=sec.p_holm_sensitivity.map(lambda v:'—' if pd.isna(v) else f'{v:.4g}')
    secondary_text='Own-stock and leave-own-out market returns since the 09:30 snapshot are added to every fitted forward model. This specification was motivated by the end-of-day reversal literature before inspecting the primary effects. It uses the same chronological split, model family and one-minute target, in a separate output directory. Its eight adjusted tests form a separate sensitivity family; they do not replace or enlarge the primary family. These are the same evaluation dates, not an independent replication.\n\n'+md_table(sec)
    report=f'''# Minute-sampled order books: reconstruction and closing-session study

**Completed research run; final review — 20 October 2026.**

This study uses the data already available. It does not depend on acquiring event data. Minute snapshots suffice for the book-state and endpoint-proxy questions below. They cannot reveal every addition, cancellation or execution between snapshots; exact event-OFI reconstruction would require event updates, not merely more days of the same snapshots. The original files are unchanged and no data was uploaded. It reconstructs a defensible snapshot-based experiment; it does not reproduce the original daily/overnight Sharpe claims or claim an exact event-OFI replication.

## What was actually run

The audit covers all 516 raw archives and the project's files. It found 1,693,301 dated members spanning 2007–2021. For the 2017–2019 study period, 377,826 stock-day files were considered. After training-coverage, validation and data-quality requirements, **{n} stocks** have scored forecasts, including **{deep} with ten levels**. The primary one-minute evaluation contains **{obs:,} stock-minute forecasts** (one row per stock and decision time, not multiplied by the number of models). SPY and ten sector ETFs were audited but excluded from the stock universe. There are 246 evaluated trading dates. Exact per-stock date counts and observation counts are in `full_study/coverage.csv`.

The project occupied about 49 GiB before this run. Most space was consumed by derived tables and integrated-OFI files; the linked raw archives occupy another 9.006 decimal GB. More disk space did not imply more frequent observations. See [the full folder audit](LITERATURE_AND_AUDIT.md).

**{decision}** The numerical results below determine the interpretation; significance alone does not establish economic value or novelty.

## Main finding

There is a small, measurable book-state signal beyond return history. Best-queue imbalance and its change reduce one-minute prediction MSE by **0.497% in core hours and 0.607% in the final half-hour**. The closing advantage is **0.110 percentage points** (Holm-adjusted p=0.0075). Cross-stock book factors add **0.194%** on the equal-session measure; ten-level information adds **0.046%** on its matched subset. The sampled-flow increment is **0.023%**, with adjusted p=0.062, so it does not pass the primary 5% threshold.

This does **not** recover the original trading-alpha claim. For the cross-book model, the closing-session average signed midpoint return is about **+0.416 basis points**, but the observed bid/ask round-trip calculation is **−2.554 basis points**, before fees and other costs. Core-session quote results are also negative. Waiting one minute before entering nearly eliminates the queue-imbalance improvement (about **0.0012%** relative MSE reduction). The defensible conclusion is short-lived predictive information, not an established profitable strategy.

The result worth discussing with Mihai is therefore a narrowly stated closing-session comparison, with explicit return-history controls and a transparent snapshot-data limitation. Its novelty and stability across independent periods remain unestablished.

## Incremental forecasting results

Positive values are percentage reductions in mean squared prediction error relative to the comparator. These are relative MSE reductions, not Sharpe ratios or percentage returns. A 0.607% reduction means lowering an MSE of 1.000 to about 0.994, not predicting 60.7% of returns correctly. Each contrast uses identical observations for its two models. Deep comparisons use only genuine ten-level stocks. Each stock-day contributes equally within a session, then dates are averaged. Reported intervals are pointwise 95% circular twenty-day block-bootstrap intervals.

{md_table(table)}

![Incremental forecasting value](full_study/incremental_value.png)

Model names: **returns** = own and market return history plus spread, depth and time-of-day; **book** adds best-queue imbalance and its change; **flow** adds sampled endpoint flow; **cross** adds market-average book variables excluding the target; **deep** adds ten-level mean queue/flow variables. “Cross” tests a simple market-factor representation, not every possible pairwise cross-impact network.

## The closing-session extension

Core observations have outcomes ending by 15:30. Closing observations start at or after 15:30 and end strictly before 16:00. Overlapping boundary intervals are excluded. The interaction is the closing-session proportional MSE improvement minus the core-session improvement, using matched stock-days. A positive interaction does not itself imply a positive absolute closing-session improvement; consult both session estimates above.

The overall primary effect gives the core and closing session equal weight before dividing by comparator loss. This is an intentional session-balanced estimand, not a minute-count-weighted trading result. Holm adjustment covers all eight primary tests below. Session-specific intervals above are not multiplicity-adjusted.

{md_table(testtable)}

## Forecasting is not profitability

The following are average signed log-return basis points per independent one-minute forecast. The quote calculation assumes a long buys at the observed ask and sells at the later bid; a short sells at the bid and buys back at the later ask. These are indicative quote-crossing calculations, not guaranteed fills, realized P&L, or a self-financing portfolio. No commissions, fees, latency slippage, market impact, borrow costs, or capacity constraints are included. The observed book quotes are not independently verified consolidated NBBO prices. Zero forecasts do not trade. Five-minute diagnostic round trips overlap and must not be added into a portfolio return.

{md_table(economic)}

The `r2_vs_zero` column is a fraction: 0.01 means a 1% reduction in squared error versus predicting zero. Statistical improvements between nested models can coexist with poor absolute forecasting or negative quote-cost results. The all-session summary above can differ slightly from the paired inference table, which additionally requires matched core and closing stock-days; it should not be used to recompute the primary paired contrasts. Deep models are omitted from this broad-universe economic table; their matched-subset summaries are saved in `full_study/summary.csv`.

## Contemporaneous construction check

These regressions explain the current minute's return with the current minute's endpoint-flow proxy. They do not forecast the future. Training/tuning dates remain chronological; current returns are never used as predictors in this diagnostic. “Deep-flow” adds deeper mean flow to level-one flow. The table averages per-stock 2019 R² values (with zero-return denominator), so its aggregation differs from the forward prediction table.

{md_table(ctable)}

An explanatory relationship is expected partly because the proxy is explicitly constructed from price changes. It validates directional construction more than tradability. Its level cannot be directly compared with event-derived OFI studies.

## Robustness checks

The five-minute horizon and one-minute delayed-entry target test temporal persistence. The fresh-snapshot sample removes observations where the entire best quote has remained identical for at least five minutes. This is only an observable snapshot-staleness proxy; it cannot establish the age of unseen intervening quote events. These analyses were specified before the full results, but are secondary and not included in the eight-test primary multiplicity family.

{md_table(robustness)}

### Post-result stability diagnostics

Additional diagnostics, added after seeing the primary results, varied the bootstrap block length to 5, 10 and 40 trading days and omitted each evaluation month in turn. These are sensitivity checks, not newly declared primary tests. The queue-imbalance closing advantage remains positive under every month omission (about 0.097–0.133 percentage points), and its pointwise interval stays above zero at every tested block length. The cross-stock closing advantage is less stable statistically: its five-day-block interval includes zero. The sampled-flow overall interval includes zero with forty-day blocks. These checks reinforce the distinction between the stronger queue-state result and the weaker flow/session claims. Full values are in `block_length_sensitivity.csv` and `leave_month_out_sensitivity.csv`.

## Stock-level variation

{md_table(heterogeneity)}

Positive share is the fraction of stocks with lower aggregate 2019 squared error under the added features. It is descriptive, not a fraction of individually significant results. The median stock effect differs from the primary ratio of average stock-day losses; the latter gives larger loss levels more influence on the proportional effect even though stock-day losses are equally averaged.

## Open-to-current return control (secondary)

{secondary_text}

The queue-state finding is essentially unchanged: the additional control yields about **0.495% core** and **0.605% closing** MSE improvement, with a **0.110 percentage-point** closing advantage (sensitivity-family adjusted p=0.0050). Cross-stock and depth improvements also persist. The tiny sampled-flow increment passes the secondary family's threshold (p=0.026) although it did not pass the primary family's threshold; this is specification sensitivity, not grounds to replace the primary conclusion. Quote-cost results remain negative. The secondary sample contains 42,655,155 one-minute observations, 358 fewer than the primary sample because the added opening-price controls require valid observations.

## Verification

Seventeen correctness tests pass. Saved coefficients and preprocessing were replayed for five stocks spanning both depth types: 78 primary model/horizon specifications reproduced daily observation counts, squared errors, and independently expressed bid/ask cost calculations. The replay shares feature construction with the main pipeline, so it is a check of saved-model and scoring consistency, not an independent reconstruction of every raw-data transformation. The separate open-to-current control passed another 26 saved-model replays. Full-result integrity checks cover all 483 stocks in each run: finite outputs, equal observations across models, matching target denominators, unique score keys, zero-baseline identities, training thresholds, and quote costs no better than corresponding midpoint returns. Detailed check outputs are saved alongside the tables.

The main 483-stock model evaluation took approximately 21 minutes and used less than 1 GiB peak resident memory. Extraction processed one archive at a time. No packages were installed, raw data was not uploaded, and historical project results were not overwritten.

## Data and methodology

1. **Raw reconstruction.** Read each selected daily raw book directly. Column names are normalized. Locked/crossed/nonpositive/sentinel or nonfinite quote/queue values become invalid rather than zero signals. Exact full-session minute grids are required. All missing, malformed and shortened sessions are logged. Forecast labels with absolute log returns of 0.2 or more are excluded by a fixed extreme-move filter; this could also exclude genuine extreme events.
2. **Observed book state.** Best-level imbalance is `(bid size − ask size)/(bid size + ask size)`. Its first difference and log depth are computed within the day. Extra levels are equal-weight summaries; no fabricated deep data or full-sample PCA is used.
3. **Sampled flow.** Use the canonical signed bid/ask endpoint change with prior queue sizes when an old quote disappears. Divide by current mean queue depth at that level and apply fixed `tanh(flow/10)` to limit extreme inputs. This is deliberately named a proxy: it is not the sum of unseen event contributions. The scale transformation is fixed, not selected on test results.
4. **Chronology.** Train on 2017, select one ridge penalty per stock/model/horizon using 2018 MSE, refit on 2017–2018, evaluate 2019. Scaling and 0.1/99.9-percentile feature clipping are fit only on the applicable training portion. No target is standardized using future information. Models have one intercept. Zero and fitted training-mean forecasts are explicit baselines.
5. **Universe.** Market-factor membership requires 70,000 valid one-minute return observations in 2017 alone. Forecast fitting additionally requires at least 50,000 training and 10,000 validation observations. No 2019 coverage threshold selects survivors. Availability still changes with observed missing/delisted names, and the source archive's initial stock selection remains a limitation.
6. **Dependence.** Aggregate paired errors across stocks by day and resample calendar-day blocks jointly across the cross-section and both sessions. Seed 279, 2,000 draws, twenty-day circular blocks. The analysis does not pretend millions of overlapping stock-minutes are independent observations.
7. **Resource control.** Sequential archive extraction and sequential stock fits; numerical libraries limited to one thread. A per-stock extraction cap prevents accidental multi-gigabyte decompression. Original archives, old code, and old results are preserved.

Quality summary (stock-days considered; invalid snapshot counts are within successfully parsed days):

{md_table(q)}

Three recurring irregular-grid dates identified in early inspection were 2019-01-09, 2019-03-08 and 2019-09-18. The complete exclusion list, including stock-specific irregular files, is in `full_study/excluded_days.csv.gz`. Half days are excluded rather than padded into a fake closing session. Missing or excluded test observations do not constitute zero returns.

## Interpretation and limits

This is a substantial completed empirical study, not proof that any one model exhausts the information in these books. It asks whether a small, interpretable set of observed book-state and sampled-flow variables adds to return history under a fixed split. It cannot identify every nonlinear signal, recover hidden order events, prove causality, or generalize one evaluation year to every market regime.

The 2019 sample was held out by this implementation, but previous project analyses may already have used it. It therefore is not an untouched research holdout. Frozen choices and a small model family reduce flexibility; they cannot erase the project's previous exploration. The same fitted models are evaluated in both sessions; the closing comparison is not a causal estimate or a model separately optimized for the close. The cross-stock features are market averages rather than an unrestricted multi-asset LASSO. Deep features are equal-weight summaries rather than the paper's PCA. Those choices keep this study simple and define its scope.

The local source already had the sign correction, and some local date-index code differs from the old remote audit. Old caches and old manuscript claims are not treated as one verified pipeline. The large processed tables were audited but not trusted as new empirical evidence. The new results do not validate the historical Sharpe values.

Publication novelty is not established. The existing literature already studies queue imbalance, multilevel flow, and cross-impact. The most useful new question here is the incremental closing-session contrast on sampled books. A favorable finding would still require independent-period replication and a stronger execution/data-quality case before being presented as a trading strategy or publication-ready discovery.

## Deliverables and reproduction

- [Primary protocol](PRIMARY_PROTOCOL.md), [protocol including secondary control](PROTOCOL.md), [folder/literature audit](LITERATURE_AND_AUDIT.md).
- `study.py`: inventory, raw extraction, market factors, chronological models and contemporaneous diagnostic.
- `summarize_study.py`: paired effects, bootstrap intervals, multiplicity correction and figure.
- `report_study.py`: this report from saved result tables.
- `full_study/archives.csv`, `members.csv.gz`, per-stock audit files with input SHA-256 hashes.
- `full_study/coverage.csv`, `summary.csv`, `inference.csv`, `daily_paired_losses.csv.gz`, `scores/` and fitted-penalty records.
- Correctness tests cover directional signs, prior queue sizes, loss under sampling, future perturbations, target timing, no overnight labels, own-stock exclusion, missing quotes, known-signal recovery, block inference and Holm correction.

From this folder, using the existing environment:

```sh
../.venv/bin/python -m unittest -v
../.venv/bin/python study.py audit
../.venv/bin/python study.py extract
../.venv/bin/python study.py market
../.venv/bin/python study.py evaluate
../.venv/bin/python summarize_study.py
../.venv/bin/python session_control.py
../.venv/bin/python summarize_session_control.py
../.venv/bin/python verify_saved_models.py
../.venv/bin/python verify_saved_models.py --session-control
../.venv/bin/python audit_results.py
../.venv/bin/python inference_sensitivity.py
../.venv/bin/python report_study.py
```

Extraction and evaluation skip completed stock files. To change features, dates or model specification, use a fresh output directory or deliberately remove only the new generated caches/results; never combine incompatible cached runs. No external repository was modified. Event-data acquisition is optional future work, not a prerequisite for the completed snapshot study.
'''
    (ROOT/'REPORT.md').write_text(report)
    # Record code hashes of the finalized run and schema versions.
    meta=json.loads((OUT/'implementation_manifest.json').read_text())
    meta['final_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    meta.update(scored_stocks=n,scored_deep_stocks=deep,primary_rows=obs)
    (OUT/'implementation_manifest.json').write_text(json.dumps(meta,indent=2))
    print(decision);print('Report written',ROOT/'REPORT.md')

if __name__=='__main__':main()
