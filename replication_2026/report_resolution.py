"""Generate the research note directly from completed aggregate outputs."""
import json,hashlib,platform
from pathlib import Path
import numpy as np,pandas as pd
from resolution_study import R,OUT

def main():
    cov=json.loads((R/'coverage.json').read_text());assert cov['stocks']==483
    s=pd.read_csv(R/'sensitivity_summary.csv');f=pd.read_csv(R/'model_summary.csv');c=pd.read_csv(R/'model_contrasts.csv');raw=pd.read_csv(R/'raw_verification.csv');replay=pd.read_csv(R/'fit_replay.csv');stock=pd.read_csv(R/'material_stock_summary.csv');dep=pd.read_csv(R/'material_depth_summary.csv');null=pd.read_csv(R/'aggregation_null.csv')
    def rate(h=5,year=2019,offset=0):return s[(s.h==h)&(s.year==year)&(s.offset==offset)&(s.tolerance==.1)&(s.metric=='opposite_rate')].iloc[0]
    def pct(v):return f'{100*v:.4f}%' if 0<abs(v)<.001 else f'{100*v:.2f}%'
    def interval(v):return f'{pct(v.estimate)} [{pct(v.low)}, {pct(v.high)}]'
    trans=pd.read_csv(R/'transform_summary.csv')
    transform_table='\n'.join(f'| {r.model} | {pct(r.r2)} | [{pct(r.low)}, {pct(r.high)}] |' for r in trans.itertuples())
    five=rate();long=rate(30);model=f[(f.h==5)&(f.offset==0)].set_index('model');contrast=c[(c.h==5)&(c.offset==0)]
    years='\n'.join(f'| {y} | {interval(rate(year=y))} |' for y in [2017,2018,2019])
    scales='\n'.join(f'| {h} | {interval(rate(h))} |' for h in [1,2,5,10,30])
    models='\n'.join(f'| {label} | {pct(model.loc[k,"r2"])} | [{pct(model.loc[k,"low"])}, {pct(model.loc[k,"high"])}] |' for k,label in [('same_fine','Summed minute increments: same-block explanation'),('same_coarse','Endpoint increment: same-block explanation'),('same_frozen','Frozen queue sizes: same-block explanation'),('lagged_coarse','Previous-block endpoint pressure: separate control sample'),('next_base','Next-block forecast: baseline'),('next_fine','Next-block forecast: baseline + minute-sum'),('next_coarse','Next-block forecast: baseline + endpoint')])
    contrasts='\n'.join(f'| {r.comparison} | {pct(r.relative_mse_gain)} | [{pct(r.low)}, {pct(r.high)}] | {r.holm_primary:.4f} |' for r in contrast.itertuples())
    material=dep[dep.year==2019];n=int(material.both_material.sum());opp=int(material.opposite.sum())
    text=f'''# When snapshot OFI changes meaning: a measurement study

**21 October 2026 — exploratory research-note candidate.** This is the strongest distinct direction found so far. It establishes an aggregation problem in the available data and supplies a practical diagnostic. It does **not** establish a publishable novelty claim, profitable strategy, or a failure of correctly computed event OFI.

The concrete result: replacing a five-minute observed quote path with its endpoints reverses the sign of measured pressure in **{pct(five.estimate)}** of materially nonzero comparisons (95% day-block interval {pct(five.low)}–{pct(five.high)}). At thirty minutes the rate is **{pct(long.estimate)}**. A simple fixed-queue null shows that two measures can each achieve 63.66% contemporaneous explanatory R² while disagreeing on signs in about 20.6% of long windows. High explanatory fit therefore does not certify measurement fidelity. In the empirical five-minute test, freezing within-block queue-size changes raises explanatory R² from **24.25% to 54.32%**. However, bounding ordinary endpoint OFI raises its fit to **49.88%**, so much of the raw gap reflects functional form. The frozen control retains price changes by construction; it is not a tradable forecast.

## What was implemented

For each block, compare (1) the sum of its observed one-minute quote-transition increments, (2) the standard operator applied only to the two endpoint snapshots, and (3) the endpoint operator with displayed queue sizes frozen at their initial values. All three use the same positive starting mean depth. The first is a finer *snapshot reference*, not true event flow. The third retains observed price changes, so it is a price-input control rather than evidence of no market activity.

The study covers **483 stocks**, 2017–2019, and **{cov['windows_primary']:,} eligible five-minute windows** across {cov['stock_days_primary']:,} stock-days. It excludes {cov['excluded_primary']:,} five-minute windows for invalid observations or the stated extreme-return filter. Other block sizes are 1, 2, 10 and 30 minutes; the five-minute grid is also shifted by two minutes. Blocks lie between 10:00 and 15:55; shortened/exceptional sessions were removed upstream. Only best-level queues are used even for ten-level source books.

## The measured disagreement survives materiality and phase checks

“Material” means the absolute value of **each** normalized measure exceeds 0.1 starting-depth units. Rates average stocks equally within dates, then dates equally. Intervals resample whole dates in circular 20-day blocks, with 2,000 draws, preserving contemporaneous stock dependence. They describe uncertainty within this archive, not representativeness of all markets.

| Block length, minutes | Opposite signs in 2019; 95% interval |
|---|---|
{scales}

| Year | Five-minute opposite signs; 95% interval |
|---|---|
{years}

The two-minute phase shift gives **{interval(rate(offset=2))}** in 2019. Across stocks, the median five-minute disagreement rate is {pct(stock[stock.year==2019].rate.median())}; every stock with a reported rate has some disagreements. Both one-level and ten-level source groups exhibit the effect. There are {opp:,} opposite-sign instances among {n:,} material five-minute comparisons in 2019; this pooled count ratio differs slightly from the equal-stock/day estimand above. Material comparisons retain {pct(n/material.windows.sum())} of valid windows.

![Resolution audit](full_study/resolution/resolution.png)

The left panel uses the materiality threshold; the right panel shows same-block explanatory fit. Neither panel establishes tradable alpha.

## Explanatory fit and genuine forward forecasts

Models train on 2017, choose ridge strength on 2018, refit through 2018, and evaluate 2019. All clipping and scaling use training data. These are chronologically separated evaluations, but the archive has already been explored extensively, so 2019 is not a fresh untouched research holdout.

The forward baseline includes current and previous block returns, current spread/depth/queue imbalance and trailing 5/30-minute returns. The target begins at the current endpoint and ends one block later. No future outcomes enter its features. Explanatory models intentionally use information from the return interval itself.

| Five-minute specification | 2019 R² versus training mean | 95% interval |
|---|---|---|
{models}

R² is 1 minus the ratio of equal-stock/day average model loss to training-mean benchmark loss. It is not the mean of stock-level R² values. The shifted-pressure negative control has fewer windows and is not a paired comparison with the contemporaneous models.

| Paired contrast (positive favors first model) | Relative MSE improvement | 95% interval | Holm p, four primary contrasts |
|---|---|---|---|
{contrasts}

A frozen-size model retaining substantial fit shows that price inputs alone can generate strong association. It does not estimate a causal percentage of true flow information. In the primary five-minute forward test, adding either pressure measure slightly worsens the baseline loss (roughly 0.004% relative MSE). The statistically detectable deterioration is economically tiny. These forecasts do not rescue the alpha hypothesis. All scales and model outcomes are retained in the accompanying tables.

### Bounded-transform check: an important qualification

After seeing the primary fit comparison, I tested fixed tanh and sign transforms of both pressure measures, plus the sign of the current return itself, using the same chronological fitting procedure across all 483 stocks. Tanh uses the original starting-depth units with scale one; its scale was not tuned. This is a **post-result descriptive sensitivity**, not an additional pre-specified primary test.

| Five-minute single-feature model | 2019 explanatory R² | 95% interval |
|---|---|---|
{transform_table}

Bounding endpoint OFI raises its fit from 24.25% to 49.88%, near the frozen-queue control's 54.32%. Therefore the raw fit gap must **not** be interpreted entirely as an economic effect of removing queue changes. Even the sign of the return alone gives 52.49% fit; this feature intentionally contains the outcome and serves only as an explanatory benchmark. The robust point is that large same-block fit can coexist with severe path-measurement disagreement, not that all OFI information is mechanically explained or that frozen queues constitute a superior trading model.

## Mathematical and practical contribution

[The derivation](RESOLUTION_THEORY.md) gives an exact difference between the sum of observed increments and the endpoint calculation. It also constructs two paths with identical endpoints and opposite summed flow; without path restrictions, no endpoint-only estimator can recover both. A Gaussian fixed-queue calculation gives identical 2/π explanatory R² for the fine and endpoint measures while their sign-disagreement probability approaches arccos(√(2/π))/π. The supplied simulation uses 100,000 paths at each scale, including 1,001 increments, and obtains {pct(null[null.minutes==1001].opposite_nonzero.iloc[0])} disagreement at that long scale.

The useful remedy is simple: preserve summed observed bid/ask increments and depth/count summaries before discarding intermediate observations. The supplied utility merges adjacent chunks, checks their shared boundary and rejects gaps. It reproduces the direct sum exactly in the unit test. This is ordinary additive aggregation, not a novel compression algorithm, and it cannot recover sub-minute activity never recorded.

## Verification and numerical precision

All 17 relevant measurement, original-study and statistical tests pass. Seven targeted tests cover the identity, flat-price telescoping, endpoint ambiguity, merging, the Gaussian control, future-state isolation and gap rejection. The identity was also checked on every eligible empirical block; maximum normalized discrepancy was {cov['max_algebra_error']:.3g}. Saved parameters were independently replayed for **{len(replay)} models and {int(replay.days.sum()):,} stock-day losses**, with maximum loss difference {replay.max_mse_error.max():.3g}.

An original-file audit covers 18 stock-days from six stocks and all three years. Their cached bid/ask prices match exactly. Reconstructing queue sizes from float32 log-depth and imbalance creates small errors: maximum sampled normalized five-minute flow error {raw.max_normalized_flow_error.max():.6g}. Four exact raw zeros became tiny nonzero cached values. Accordingly the original near-zero threshold results remain available but are not the headline; sensitivities at 0.001, 0.01 and 0.1 are all reported. This sample check is not a universal numerical error bound.

## Is this worth taking to Mihai?

**Yes, as a concrete methods-paper proposal, with the novelty question still open.** The contribution would be the combination of a transparent identification argument, a large within-path aggregation audit, price-input controls and a reproducible preservation tool. This is more specific than another weak/untradeable OFI signal.

The closest prior work is [Su et al. (2021)](https://arxiv.org/pdf/2112.02947), which already recognizes snapshot limitations and proposes generalized OFI using deeper levels. [Cont–Kukanov–Stoikov](https://arxiv.org/html/1011.6402v3) already defines OFI as a sum of event contributions. We cannot claim that sampling loss, summing increments, or contemporaneous-versus-forward distinctions are new. [The literature memo](RESOLUTION_LITERATURE.md) states the overlap explicitly. We have not implemented or outperformed generalized OFI here.

A sensible pitch is: *We found that endpoint substitution changes the sign of snapshot pressure surprisingly often, and that strong price-fit statistics do not detect this measurement failure. We have an exact decomposition, a 483-stock audit and an aggregation-preserving implementation. Is this specific combination new enough for a short methods note?*

The unresolved publication hurdle is whether this is useful new evidence rather than a familiar implementation pitfall. An event-level benchmark would strengthen the scope but is not needed to establish the narrower observed-path facts. The available minute data cannot measure the amount of information lost below one minute, reconstruct cancellations/trades, identify causality, or validate a general correction to event OFI. Archive composition may also be subject to survivorship selection.

## Reproduction

From this directory, with the existing 2017–2019 caches and archive inventories:

```bash
../.venv/bin/python -m unittest test_resolution
../.venv/bin/python resolution_study.py
../.venv/bin/python verify_resolution.py
../.venv/bin/python resolution_sensitivity.py
../.venv/bin/python resolution_heterogeneity.py
../.venv/bin/python replay_resolution.py
../.venv/bin/python resolution_null.py
../.venv/bin/python resolution_transform.py
../.venv/bin/python summarize_resolution.py
../.venv/bin/python report_resolution.py
```

Outputs are under `full_study/resolution/`. Per-stock files are resumable; use a fresh output directory if changing the specification. Raw data, caches, per-stock losses and saved model parameters remain local; source, tests, protocol, theory, compact aggregate results and figures are shared.
'''
    (Path(__file__).parent/'RESOLUTION_REPORT.md').write_text(text)
    sources=['resolution_study.py','resolution_transform.py','resolution_tools.py','resolution_null.py','resolution_sensitivity.py','resolution_heterogeneity.py','summarize_resolution.py','verify_resolution.py','replay_resolution.py','report_resolution.py','test_resolution.py','RESOLUTION_PROTOCOL.md']
    manifest=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,stocks=cov['stocks'],hashes={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    (R/'implementation_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Wrote RESOLUTION_REPORT.md')
if __name__=='__main__':main()
