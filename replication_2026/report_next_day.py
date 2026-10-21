"""Generate a reproducible research summary and static figure from saved evaluations."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/ofi_mpl'
import json,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn,scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from next_day_study import D,FAMILIES
ROOT=Path(__file__).resolve().parent
NAMES={'ridge':'Ridge','additive':'Spline ridge','boost':'Boosted trees','forest':'Random forest','neural':'Small neural net','logistic':'Logistic classifier','class_boost':'Boosted classifier'}
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    inf=pd.read_csv(D/'inference.csv');ps=pd.read_csv(D/'portfolio_summary.csv');ms=pd.read_csv(D/'model_summary.csv');dep=pd.read_csv(D/'depth_inference.csv');coverage=pd.read_csv(D/'target_coverage.csv');verification=pd.read_csv(D/'verification.csv')
    primary=inf.query("target=='intraday' and period=='all'").set_index('family').loc[FAMILIES]
    rows=[]
    for fam,r in primary.iterrows():rows.append([NAMES[fam],f'{100*r.effect:+.3f}%',f'[{100*r.low:+.3f}%, {100*r.high:+.3f}%]',f'{r.p_holm:.3f}'])
    primary_table=table(['Model','Book improvement in loss','95% day-block interval','Holm p'],rows)
    rows=[]
    for target in ['overnight','combined']:
        for r in inf.query('target==@target and period=="all"').itertuples():rows.append([target,NAMES[r.family],f'{100*r.effect:+.3f}%',f'[{100*r.low:+.3f}%, {100*r.high:+.3f}%]',f'{r.p_holm:.3f}'])
    secondary_table=table(['Target','Model','Book improvement in loss','95% interval','Holm p'],rows)
    rows=[]
    for fam in FAMILIES:
        vals=[]
        for target,rule in [('intraday','top20'),('intraday','long10_short10'),('overnight','top20'),('combined','top20')]:
            r=ps.query('family==@fam and target==@target and rule==@rule and group=="book" and period=="all"').iloc[0];vals.append(f'{r.net_bps:+.2f}')
        rows.append([NAMES[fam]]+vals)
    portfolio_table=table(['Book model','Next-session top20','Next-session L10/S10','Overnight top20','Combined top20'],rows)
    rows=[]
    for fam in FAMILIES:
        r=ms.query('family==@fam and group=="book" and target=="intraday" and period=="all"').iloc[0]
        rows.append([NAMES[fam],f'{r.rank_ic:.4f}',f'{100*r.accuracy:.2f}%', '—' if pd.isna(r.r2_vs_zero) else f'{100*r.r2_vs_zero:+.3f}%',f'{100*r.improvement_vs_control:+.3f}%'])
    control_table=table(['Book model','Mean rank IC','Direction accuracy','MSE gain vs zero','Loss gain vs own-mean / prior'],rows)
    p=ps.query('target=="intraday" and group=="book" and rule=="top20" and period=="all"').set_index('family').loc[FAMILIES]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');yy=np.arange(len(FAMILIES))
    for ax,point,lo,hi,title,xlabel in [(axes[0],100*primary.effect,100*primary.low,100*primary.high,'Does the book improve prediction?','Relative loss improvement (%)'),(axes[1],p.net_bps,p.low,p.high,'Does top-20 ranking cover spreads?','Mean next-session net return (bps)')]:
        ax.errorbar(point,yy,xerr=np.vstack([point-lo,hi-point]),fmt='o',color='#25617c',capsize=3);ax.axvline(0,color='#888888',lw=1);ax.set_yticks(yy,[NAMES[f] for f in FAMILIES]);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Next-session snapshot study · 2020–2021\n95% day-block intervals; predictive loss is MSE or classifier log loss',fontsize=12);fig.savefig(D/'next_day_results.png',dpi=180);plt.close(fig)
    notes=[]
    for f in sorted((D/'models').glob('*.json')):
        m=json.loads(f.read_text())
        if m['warnings']:notes.append(f.stem)
    sources=['daily_data.py','next_day_study.py','summarize_next_day.py','depth_next_day.py','verify_next_day.py','report_next_day.py','test_next_day.py','NEXT_DAY_PROTOCOL.md']
    manifest={'date':'2026-10-21','python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},'models_with_convergence_warnings':notes,'model_count':len(verification),'replayed_predictions':int(verification.replayed.sum()),'max_replay_error':float(verification.max_error.max())}
    (D/'implementation_manifest.json').write_text(json.dumps(manifest,indent=2))
    positive=inf.query('period=="all" and target!="intraday" and effect>0 and p_holm<.05')
    secondary_conclusion='No secondary-target book increment passes its multiplicity correction.' if positive.empty else 'Secondary increments passing their separate multiplicity correction: '+', '.join(r.target+' / '+NAMES[r.family] for r in positive.itertuples())+'. These do not change the primary next-session conclusion.'
    text=f'''# Next-day snapshot reconstruction and ranking — 21 October 2026

The completed next-session study does not establish incremental predictive value from these book features, or a profitable top-20 rule after spreads. This corrects the horizon of the earlier minute study: those minute results were not an answer to the original next-day question. The conclusions below concern this bounded experiment, not every possible forecasting method.

## What was tested

Signals stop at 15:50. The primary target is **next trading day's 09:35–15:55 price return**. Secondary targets are today's 15:55 to next 09:35 (overnight), and today's 15:55 to next 15:55 (combined). These are observable near-open/near-close proxies, not auction returns. Daily book features summarize queue imbalance, endpoint flow, persistence, early/late pressure, spread-scaled imbalance, pressure with/without price movement, and market/sector pressure. Sampled endpoint flow is not reconstructed event OFI.

Five regressions and two direction classifiers each have a matched baseline and book version: 42 final fitted models. Baselines include return history, volatility, price, spread, depth, market/sector return context and trailing target means. Classifiers predict positive raw returns; they do not directly predict profit after costs. Settings were chosen on 2019 after fitting 2017–2018, then refitted through 2019 and evaluated on 2020–2021. Features use only matured outcomes and training-only preprocessing. At most two preset settings per family were considered. All forecasts and final model objects are retained locally.

The universe was chosen from 2017 coverage: 483 stocks, including 105 genuine ten-level names. The test sample contains 111,767 valid primary outcomes in 2020 and 69,524 in 2021, with 450 scored signal dates. Available names fall from 483 in 2020 to 380 in early 2021 and 271 by December; the archive is not a balanced panel. No test-coverage screen removes unavailable names from the original universe. The 2020–2021 extraction records 9,515 irregular stock-day grids, 1,221 shortened/exceptional sessions and 11 insufficient signal histories. The full year/target coverage is in [target_coverage.csv](full_study/next_day/target_coverage.csv).

## Primary predictive finding

Positive values below mean lower loss from adding book information to the matched model. Regression loss is MSE; classifier loss is log loss. Cross-stock losses are averaged within each day before inference, and 20-day circular blocks are resampled 2,000 times. Confidence intervals are individual 95% intervals; Holm p-values correct seven primary comparisons.

{primary_table}

Only spline ridge has a positive pooled point estimate, about 0.085% relative MSE improvement, with an interval crossing zero. The small neural model deteriorates, but its fixed 30-iteration optimization limit was reached without convergence; this is evidence about this restricted implementation, not neural networks generally. No positive primary increment is established after correction. The matched ten-level ridge and boosting increments also have intervals crossing zero and Holm p=1.000; deeper summaries do not rescue this specification. Separate yearly estimates and the depth comparison are saved in [inference.csv](full_study/next_day/inference.csv) and [depth_inference.csv](full_study/next_day/depth_inference.csv).

## Simple controls matter

{control_table}

The last column compares regressions with the stock's matured trailing 60-observation target mean (training mean fallback), and classifiers with the training positive-return frequency. Every regression's pooled next-session MSE is worse than a zero-return forecast, despite some improving on the noisier rolling-mean control. Direction accuracy near 50% and small positive rank correlations are not evidence of a usable trading edge. Training-mean, zero and rolling controls are retained in the daily control output; all model versions are in [model_summary.csv](full_study/next_day/model_summary.csv).

## Rigel's ranking idea

Ranking avoids treating an imperfect forecast as a precise return estimate, but it still needs useful ordering and enough realized movement to pay the spread. Top20 chooses the twenty largest absolute forecasts, uses their predicted signs and assigns each 1/20 of fixed gross notional. Classifier confidence is probability minus 0.5. The separate L10/S10 rule takes the highest ten and shorts the lowest ten at 1/20 each. Both choose names **before** checking future return availability. A selected missing outcome makes the day unscorable; it is not assigned zero return or replaced with another name.

The following are average basis points per portfolio holding cycle after crossing quoted entry/exit spreads, before fees, impact, financing and borrow costs. Book-model rows are all reported; none is selected using test profits.

{portfolio_table}

For the primary target, every baseline and book variant has a negative pooled net mean under both ranking rules. The book top20 variants are also all negative separately in 2020 and 2021. Spline ridge, for example, earns about 9.09 bps gross but loses 8.57 bps net per next-session cycle; ordering alone does not bridge the spread. Its paired improvement over the baseline top20 rule is about 5.09 bps with an interval crossing zero. All seven primary top20 book-minus-baseline intervals include zero.

Primary top20 comparisons use 435 common scorable days across all fourteen model variants, versus 450 prediction-scoring days. Common-day selection depends on future quote availability and can bias the economic sample; missing outcomes are not harmless cash. Complete unmatched records and exclusions remain local. [Portfolio summaries](full_study/next_day/portfolio_summary.csv) include individual intervals, yearly results and an additional 1 bp-per-side charge. [Paired portfolio increments](full_study/next_day/portfolio_increment.csv) are descriptive, not multiplicity-adjusted trading discoveries. Returns use exact arithmetic cash/share accounting for shorts and fixed initial gross notional. They are not annualized Sharpe ratios or compounded portfolio wealth. Combined cycles charge a fresh round trip, without netting consecutive exposures.

The earlier minute ranking study likewise found all 27 tested variants net negative, including explicit forecast-versus-spread gates; see [RANKING_REPORT.md](RANKING_REPORT.md). That result and the next-session result are separate tests.

## Secondary horizons

{secondary_conclusion}

{secondary_table}

One descriptive result deserves follow-up without being promoted to a discovery: combined-horizon logistic top20 with book features averages **+19.51 bps net per cycle**, with an individual 95% interval of [+2.01, +36.35] bps. Its baseline already earns +9.59 bps, and the paired book increment is +9.92 bps with an interval of [−9.17, +29.02] bps. The book variant averages +25.13 bps in 2020 and +13.58 bps in 2021, but these are not independent new validation samples after inspecting the pooled experiment. Its average net long exposure is 0.24, versus 0.92 for the baseline, so the comparison also changes directional exposure. The corresponding long10/short10 book portfolio is −0.88 bps per cycle. This is an exploratory candidate among many reported strategies, not demonstrated incremental book alpha; portfolio intervals are unadjusted for that search. A genuinely new evaluation period and verified corporate-action treatment would be needed before upgrading the claim.

These targets use unadjusted price returns. Absolute overnight log gaps of at least 0.20 are removed: 68 flagged test observations in 2020 and 8 in 2021. This is a crude contamination screen that also removes genuine large moves; it is not corporate-action adjustment. Dividends and corporate actions were not independently verified, so overnight/combined economic results are weaker evidence than the intraday primary test. Fourteen secondary increments receive a separate Holm correction, not the primary seven-test correction.

## Checks, limits and interpretation

All 29 research tests pass, including signal-cutoff invariance, matured labels, missing-session alignment, training-only preprocessing, missing classifier labels and explicit short cash accounting. All 42 saved models replay 97 sampled predictions each ({int(verification.replayed.sum()):,} total), with maximum absolute error {verification.max_error.max():.2g}. Model convergence warnings are preserved in [implementation_manifest.json](full_study/next_day/implementation_manifest.json); the neural runs are optimization-limited. Primary sensitivity checks excluding absolute log returns >=0.10 or March 2020 do not reveal a positive increment with an individual interval excluding zero. These are descriptive robustness checks, not alternative selected primary samples.

The calendar is inferred from archive dates with at least 100 names; absent archive-wide sessions cannot be recovered. Static sector labels are not verified point-in-time; missing sectors map to Unknown. Future daily quality failures can remove targets even when the required execution quotes might otherwise exist. There is no independent dividends/corporate-action file, execution replay, borrow availability, or market-impact model. The archive was explored before this implementation, so 2020–2021 are held out from this code's fitting choices, not guaranteed untouched research data. Coarse snapshots cannot reproduce within-minute order events or trade classifications.

This is a useful, reproducible negative reconstruction and a bounded extension, not yet a distinct positive paper contribution. Daily imbalance prediction and portfolio sorts already appear in [Lu, Reinert and Cucuringu](https://arxiv.org/html/2209.10334); event-type OFI decomposition appears in [Sitaru, Calinescu and Cucuringu](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4572510). Our snapshot features cannot exactly reproduce either construction. See [the novelty review](NOVELTY_REVIEW.md) for overlap and the limits of the literature search. The defensible conclusion is that minute queue predictability does not automatically survive aggregation into these next-day forecasts, and selecting large forecasts does not establish net profitability. More model search on the same test period would require a new exploratory label, not a stronger holdout claim.

## Reproduce

Use the existing environment and raw-input path described in the original [report](REPORT.md). The minute raw-derived caches and archive inventories must exist first. Run, in order:

```sh
python daily_data.py
python next_day_study.py panel
python next_day_study.py fit
python depth_next_day.py
python summarize_next_day.py
python verify_next_day.py
python -m unittest discover
python report_next_day.py
```

Extraction/fitting stages resume existing files; remove or isolate stale outputs before changing a specification. Full daily summaries, input hashes, panel, models and forecasts stay local under `full_study/next_day/`; the research branch includes source, tests, compact aggregate results and the figure. Numerical fits use one thread. The daily panel is about 422 MB, and the observed fit process used about 1.4 GB RAM. No new event-data purchase is needed to reproduce this snapshot study.

![Next-session predictive increments and ranked net returns](full_study/next_day/next_day_results.png)
'''
    (ROOT/'NEXT_DAY_REPORT.md').write_text(text)
    print('Wrote next-day report, manifest and figure')
if __name__=='__main__':main()
