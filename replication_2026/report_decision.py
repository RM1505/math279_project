"""Decision memo from completed frozen extensions; no model selection here."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/ofi_mpl'
import json,hashlib,platform
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decision_models import O
ROOT=Path(__file__).resolve().parent

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    modern=pd.read_csv(O/'modern/inference.csv');history=pd.read_csv(O/'historical/inference.csv');hp=pd.read_csv(O/'historical/portfolio_summary.csv');hi=pd.read_csv(O/'historical/portfolio_increment.csv');stress=pd.read_csv(O/'candidate_stress_tests.csv');closing=pd.read_csv(O/'closing_matched_inference.csv');verification=pd.read_csv(O/'verification.csv');correction=json.loads((O/'historical/primary_correction.json').read_text())
    primary=history.query("kind=='original' and span=='all'").iloc[0];trade=hp.query("kind=='original' and partition=='test' and span=='all' and group=='book' and rule=='top20'").iloc[0];inc=hi.query("kind=='original' and partition=='test' and rule=='top20'").iloc[0]
    rows=[]
    for partition in ['modern','historical']:
        q=pd.read_csv(O/partition/'portfolio_summary.csv');delta=pd.read_csv(O/partition/'portfolio_increment.csv');choices=json.loads((O/partition/'validation_selection.json').read_text())
        for choice in choices:
            if choice['family']=='cash':rows.append([partition,choice['target'],'Cash','0.00','—','—']);continue
            target=choice['target'];fam=choice['family'];rule=choice['rule'];r=q.query("kind=='relative' and target==@target and family==@fam and rule==@rule and group=='book' and partition=='test' and span=='all'").iloc[0];d=delta.query("kind=='relative' and target==@target and family==@fam and rule==@rule and partition=='test'").iloc[0]
            rows.append([partition,target,fam+' / '+rule,f'{r.net_bps:+.2f}',f'[{r.low:+.2f}, {r.high:+.2f}]',f'{d.increment_bps:+.2f} [{d.low:+.2f}, {d.high:+.2f}]'])
    strategytable=table(['Period','Target','Validation choice','Test net bps/cycle','95% interval','Book minus baseline bps [interval]'],rows)
    rows=[]
    for kind in ['relative','risk']:
        for r in modern.query('kind==@kind and span=="all"').itertuples():
            h=history.query('kind==@kind and span=="all" and target==@r.target and family==@r.family').iloc[0]
            rows.append([kind,r.target,r.family,f'{100*r.effect:+.3f}% (p={r.p_holm:.3f})',f'{100*h.effect:+.3f}% (p={h.p_holm:.3f})'])
    modeltable=table(['Question','Target','Family','2020–2021 increment','2015–2016 increment'],rows)
    hrows=[]
    for group in ['base','book']:
        for r in hp.query("kind=='original' and partition=='test' and group==@group and rule=='top20'").itertuples():hrows.append([group,r.span,r.days,f'{r.net_bps:+.2f}',f'[{r.low:+.2f}, {r.high:+.2f}]',f'{r.net_exposure:+.2f}'])
    histtable=table(['Model','Period','Days','Net bps/cycle','95% interval','Mean net exposure'],hrows)
    stressrows=[[r.check,f'{r.mean:+.2f}',f'[{r.low:+.2f}, {r.high:+.2f}]'] for r in stress.itertuples() if r.check in ['net_bps','market_adjusted_bps','sector_adjusted_bps','net_after_5bp_side','omit_best_5_days','omit_best_10_days']]
    stresstable=table(['Diagnostic','Mean bps/cycle','95% day-block interval'],stressrows)
    audit_files=list((O/'historical/daily').glob('*_audit.csv'));audits=pd.concat([pd.read_csv(f,usecols=['status']) for f in audit_files]);auditcounts=audits.status.value_counts().rename_axis('status').reset_index(name='stock_days');auditcounts.to_csv(O/'historical/extraction_quality.csv',index=False)
    universe=len(list((O/'historical/daily').glob('*.csv')))-len(audit_files)
    risks=[]
    for period in ['modern','historical']:
        m=pd.read_csv(O/period/'daily_metrics.csv');m=m.query("kind=='risk' and partition=='test'")
        for key,g in m.groupby(['target','family','group']):risks.append(dict(period=period,target=key[0],family=key[1],group=key[2],improvement_vs_persistence=1-g.loss.mean()/g.persistence_loss.mean()))
    pd.DataFrame(risks).to_csv(O/'risk_vs_persistence.csv',index=False)
    # Show the central threat to the observed positive result and period comparison.
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    chosen=stress.set_index('check').loc[['net_bps','market_adjusted_bps','sector_adjusted_bps','omit_best_5_days']];y=np.arange(4)
    axes[0].errorbar(chosen['mean'],y,xerr=np.vstack([chosen['mean']-chosen.low,chosen.high-chosen['mean']]),fmt='o',capsize=3,color='#24677a');axes[0].set_yticks(y,['Reported net return','Less dollar market exposure','Less dollar sector exposure','Omit best five days']);axes[0].invert_yaxis();axes[0].set_title('Stress tests of the 2020–2021 lead');axes[0].set_xlabel('Mean bps per holding cycle')
    h=hp.query("kind=='original' and partition=='test' and span=='all' and rule=='top20'").set_index('group').loc[['base','book']]
    axes[1].errorbar(h.net_bps,np.arange(2),xerr=np.vstack([h.net_bps-h.low,h.high-h.net_bps]),fmt='o',capsize=3,color='#24677a');axes[1].set_yticks(np.arange(2),['Baseline','Baseline + book']);axes[1].invert_yaxis();axes[1].set_title('Same combined-logistic rule: 2015–2016');axes[1].set_xlabel('Mean net bps per holding cycle')
    for ax in axes:ax.axvline(0,color='gray',lw=1);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Decision-stage evidence · individual 95% day-block intervals\nExposure subtraction is diagnostic attribution, not a tradable hedge',fontsize=11);fig.savefig(O/'decision_results.png',dpi=170);plt.close(fig)
    text=f'''# Research decision memo — 21 October 2026

**Recommendation: stop the broad next-day snapshot-alpha search, and give Mihai a transparent negative research update rather than a discovery pitch.** There is a careful reconstruction here, but no validated incremental trading signal or established publishable contribution. Do not discard the work; use it to ask whether a narrowly scoped negative replication or measurement study is worthwhile. Continuing the same model search without a new mechanism and independent evaluation would be hard to justify.

This memo is for deciding what to tell Mihai, not for selecting the prettiest backtest. The original next-day question, the minute closing extension, Rigel's ranking proposal, relative-return forecasts, liquidity filtering and a separate risk pivot have now been tested. Prior positive-looking results are treated as exploratory, not as fresh validation.

## The apparent daily trading lead

The previously highlighted combined overnight-plus-day logistic top20 portfolio earned +19.51 bps after crossing spreads in 2020–2021. That number alone is not sufficient evidence of book alpha.

{stresstable}

The original result has unadjusted p≈0.030, but Holm p=1.000 across the 84 originally reported portfolio variants. This correction covers that documented family, not every adaptive research choice made in the project. Removing the five best days reduces the mean to +10.67 bps with an interval crossing zero; removing ten reduces it to +4.47 bps. These are ex-post concentration diagnostics, not alternative trading rules.

Dollar market attribution removes about 16.55 of the 19.51 bps; dollar sector attribution leaves about 1.51 bps. The market component decomposes into roughly 2.99 bps from average exposure and 13.56 bps from the covariance of changing exposure with market returns. Therefore this is not merely a constant long-market bet: common-return timing could in principle contain information. The subtraction does not prove that timing is spurious. It does show that a stock-specific ranking-alpha interpretation is unsupported. These benchmarks use realized same-universe outcomes and static sectors, not an executable hedged strategy, a beta-adjusted factor model or a causal alpha estimate.

## Separate earlier chronological experiment

We extracted {len(audits):,} additional archived stock-days for {universe} training-eligible stocks in 2011–2016. Universe eligibility uses at least 200 valid signal days in 2011, with no test-year coverage screen. The original archive's composition may still be survivorship-biased. Train 2011–2013, select the same two preset settings on 2014, refit through 2014, test 2015–2016. The protocol was fixed before inspecting these new test results. This is a separate earlier-period replication, not a prospective holdout or a guarantee that the original project never examined these dates.

For the original combined-logistic rule, the book increment in prediction loss is {100*primary.effect:+.3f}% (95% interval [{100*primary.low:+.3f}%, {100*primary.high:+.3f}%]; two-comparison Holm p={correction['loss_p_holm']:.3f}). The top20 book-minus-baseline portfolio increment is {inc.increment_bps:+.2f} bps per cycle (interval [{inc.low:+.2f}, {inc.high:+.2f}]; Holm p={correction['portfolio_p_holm']:.3f}).

{histtable}

Each increment compares book and baseline on identical observations within that comparison; loss inference uses 497 days and portfolio inference uses 493 common scorable days. The historical book portfolio becomes about −0.42 bps after dollar market attribution and −1.46 bps after dollar sector attribution; these remain diagnostic rather than executable hedges. A strategy's raw positive return is not sufficient if the baseline matches or exceeds it. Missing selected outcomes make a day unscorable, rather than cash or a replacement trade. Common complete-day evaluation remains a selection limitation. Overnight/combined returns are unadjusted price returns: absolute overnight log gaps >=0.20 are excluded, but dividends and corporate actions are not independently verified. Fees, impact, borrow and financing remain additional limitations.

## Pushes beyond raw-return forecasting

Relative-return targets subtract the same day's cross-stock target mean during training. This uses future information only to define labels; it never enters signal features. Models are ridge, spline ridge and logistic, each with matched baseline/book features. Four preset portfolio rules are top20 signed and long10/short10, unrestricted or limited to the lower half of signal-time spreads. Current spreads determine eligibility before future outcomes are inspected. Cash competes with the twelve book model/rule combinations per target on validation net return. No test outcome selects a winning rule.

{strategytable}

The historical validation-selected combined spline strategy has a small positive paired point estimate (+3.49 bps), with an individual interval just above zero, but its own net-return interval includes zero. Its centered two-sided paired bootstrap p is about 0.0505, and it does not survive correction across the 24 relative-return portfolio increments. None of those positive paired increments passes that correction in either period. This isolated result does not meet the joint requirements of beating cash, beating the baseline and repeating across periods. Portfolio mean tests cover 48 variants per period; paired increments cover 24.

The risk pivot separately predicts next-session log realized volatility, log price range and log mean spread. Baselines include current values and trailing own-history averages, in addition to the price/liquidity controls; both ridge and boosted trees are tested. Risk prediction is not trading alpha. The next session's risk quantities end at its 15:50 feature cutoff.

Below, positive percentages mean reduced loss from adding book features. Regression loss is MSE, including log-risk MSE; classification loss is log loss. Each six-test relative-return or risk family receives its own Holm correction within each period. These separate families are exploratory and are not a universal correction for the entire research history.

{modeltable}

Full yearly estimates, all portfolio variants and their corrected tests are included in the compact result tables. [Risk-versus-persistence results](full_study/decision/risk_vs_persistence.csv) check whether the fitted risk forecasts beat simply carrying today's value forward. A positive result in only one period or one selected model is not cross-period validation.

## Does the minute closing result offer a distinct fallback?

A new diagnostic replays the original saved 2019 baseline/book forecasts. Within each stock-day, core and close are matched on stock-specific training-defined spread terciles, trailing volatility terciles and depth halves. Identical common-bin weights retain about 91.5% of eligible closing observations across 118,814 stock-days.

The matched queue increment is approximately 0.564% relative MSE improvement in core hours and 0.609% at the close. The difference is **0.045 percentage points**, with an individual interval approximately [−0.0003, +0.0912] points and p≈0.055, versus the earlier unmatched 0.110-point headline. On a common error denominator the absolute-loss benefit is larger near the close and its interval is positive; that is a different estimand and does not restore a robust relative closing advantage. Matching is an exploratory conditioning exercise, not a causal design or a new independent period. The previously demonstrated one-minute-delay collapse and spread-cost losses still apply.

Thus the short-horizon queue signal remains real in the tested sample, but a distinctive closing-session story is weaker after addressing composition and error scaling. This should not be presented as a validated profitable strategy or as an established novel mechanism.

## What to communicate to Mihai

The substantive update is: minute queue information is measurable but short-lived; ranking and a broader daily model search did not establish incremental net alpha; the one attractive daily backtest weakened under exposure, multiple-testing and separate-period checks; and controlling liquidity conditions weakened the distinctive closing advantage. This is a snapshot adaptation, not a failed exact replication or a refutation of his event/trade-based results.

Ask whether that negative/measurement result answers a worthwhile question for his group, or whether to close this direction and choose a different problem. Do not ask him to validate a claimed profitable strategy. A useful meeting and a publishable positive paper are different thresholds; this work supports the former, not the latter.

## Novelty and what would justify continuing

The [targeted literature review](DECISION_LITERATURE.md) finds direct prior work on daily imbalance/market-excess returns, order-book forecasting of volatility and liquidity, and imbalance volatility. A change of estimator, horizon or snapshot aggregation is not by itself an original contribution. Minute queues cannot reproduce event-type OFI or trade-co-occurrence labels.

The decision standard is evidence beyond matching controls, repeatability across periods, robustness to costs/exposure/concentration, and a narrow contribution distinguishable from prior work. The responsible discussion with Mihai should separate: (1) a careful reconstruction and negative directional-alpha result; (2) any explicitly exploratory feature of the tables above; and (3) a new research question that would need his judgment. “We found a profitable OFI strategy” is not supported by the original +19.51 bps backtest.

Do not keep enlarging the model search on 2020–2021 to manufacture a winner. A fresh hypothesis suggested by these diagnostics needs an independent evaluation plan; a risk pivot needs its own novelty and utility case. This work is informative even if the correct decision is to stop the current daily snapshot-alpha route.

## Audit and reproducibility

All 31 research tests pass. The {len(verification)} new saved final models replay {verification.replayed.sum():,} test predictions with maximum absolute discrepancy {verification.max_error.max():.2g}. Extraction is sequential with a 600 MB per-stock temporary-data cap; numerical fits use one thread. Raw archives are preserved. Models, individual forecasts, full panels and per-stock provenance stay local; source, protocols and compact summaries can be shared on the existing research branch.

Run `historical_daily.py`, `decision_audit.py`, `decision_models.py modern`, `decision_models.py historical`, `closing_match.py`, `historical_attribution.py`, `verify_decision.py`, the test suite, then `report_decision.py`. Existing fitted outputs are reused: isolate stale outputs before changing a specification. The earlier next-day extraction and minute-study caches must already exist. Source hashes are saved in the decision implementation manifest.

![Decision-stage evidence](full_study/decision/decision_results.png)
'''
    (ROOT/'DECISION_REPORT.md').write_text(text)
    files=['DECISION_PROTOCOL.md','historical_daily.py','decision_models.py','decision_audit.py','closing_match.py','historical_attribution.py','test_decision.py','verify_decision.py','report_decision.py']
    manifest=dict(date='2026-10-21',python=platform.python_version(),models=len(verification),replayed=int(verification.replayed.sum()),historical_stocks=universe,historical_archived_stock_days=len(audits),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files});(O/'implementation_manifest.json').write_text(json.dumps(manifest,indent=2));print('Decision memo and figure written')
if __name__=='__main__':main()
