"""Check manuscript tables and added descriptive estimates against saved outputs."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parent
R=P/'full_study/resolution'
s=(P/'SNAPSHOT_OFI_PAPER.tex').read_text(); checks=[]
def contains(label,expected):
    assert expected in s,(label,expected)
    checks.append({'claim':label,'checked_text':expected})
a=pd.read_csv(R/'sensitivity_summary.csv').query("year == 2019 and offset == 0 and tolerance == 0.1 and metric == 'opposite_rate'")
for row in a.itertuples():
    contains(f'scale table h={row.h}',f'{row.h}&{100*row.estimate:.2f}&[{100*row.low:.2f}, {100*row.high:.2f}]')
m=pd.read_csv(R/'model_summary.csv').query('h == 5 and offset == 0').set_index('model')
for key in ['same_fine','same_coarse','same_frozen']:
    row=m.loc[key]; contains(key,f'&{100*row.r2:.2f}&[{100*row.low:.2f}, {100*row.high:.2f}]')
for row in pd.read_csv(R/'transform_summary.csv').itertuples():
    contains(row.model,f'&{100*row.r2:.2f}&[{100*row.low:.2f}, {100*row.high:.2f}]')
r=json.loads((R/'paper_robustness.json').read_text());f=r['frozen_vs_tanh_coarse']
contains('paired gain',f"{100*f['relative_mse_gain']:.2f}\\%")
contains('paired interval',f"{100*f['low']:.2f}--{100*f['high']:.2f}\\%")
contains('matched windows',f"{f['matched_windows']:,}")
for b in r['sign_block_sensitivity']:
    contains(f"bootstrap block {b['block_days']}",f"{100*b['low']:.2f}--{100*b['high']:.2f}")
for q,v in r['stock_rate_quantiles'].items():contains('stock quantile '+q,f'{100*v:.2f}\\%')
for key in ['next_base','next_fine','next_coarse']:contains(key,f"{100*m.loc[key,'r2']:.4f}\\%")
for row in pd.read_csv(R/'model_contrasts.csv').query('h == 5 and offset == 0').itertuples():
    if row.comparison.startswith('next'):
        contains(row.comparison,f'{-100*row.relative_mse_gain:.4f}\\%')
        contains(row.comparison+' interval',f'[{100*row.low:.4f},{100*row.high:.4f}]')
for phrase in ['Neither is event-level OFI','does not make 2019 an untouched research holdout','post-result descriptive','pending coauthor review','0.001 and 0.999']:
    contains('scope',phrase)
extension=R/'extension'
a=pd.read_csv(extension/'selection_summary.csv')
for mode,label in [('raw','Starting-depth units'),('training_scaled','Common training scale')]:
    g=a.query('k==20 and scaling==@mode').set_index('metric');v=g.loc['overlap'];o=g.loc['intersection_opposite']
    contains('top20 '+mode,f"{label}&{100*v.estimate:.2f} [{100*v.low:.2f}, {100*v.high:.2f}]&{100*o.estimate:.2f}")
    for k in [5,50]:
        v=a.query('k==@k and scaling==@mode and metric=="overlap"').iloc[0]
        contains(f'top{k} {mode}',f'{100*v.estimate:.2f}\\%')
g=pd.read_csv(extension/'gofi_summary.csv').query('comparison=="generalized_endpoint_vs_sum"').set_index('convention')
for v in pd.read_csv(extension/'gofi_paired_summary.csv').itertuples():
    label={'price_aligned':'Price-aligned','contiguous_ticks':'Tick grid'}[v.convention]
    contains('gofi '+v.convention,f'{label}&{100*g.loc[v.convention,"retention"]:.2f}&{v.material:,}&{v.nonempty_stock_days}&{100*v.standard_rate:.2f}&{100*v.generalized_rate:.2f}')
    contains('gofi full standard',f'{100*v.full_standard_rate:.2f}\\%')
for v in pd.read_csv(extension/'reconstruction_summary.csv').itertuples():
    contains('reconstruction sign '+v.model,f'{100*v.sign_error:.2f}\\%')
    if v.model!='raw_endpoint':
        contains('reconstruction loss '+v.model,f'{-100*v.mse_gain_vs_raw:.2f}\\%')
        contains('reconstruction interval '+v.model,f'{-100*v.mse_gain_high:.2f}--{-100*v.mse_gain_low:.2f}\\%')
for phrase in ['not predicted returns or realized trades','four dates do not support a reliable temporal confidence interval']:
    contains('extension scope',phrase)
(R/'latex_fact_check.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n')
print(f'Passed {len(checks)} LaTeX table, robustness and scope checks')
