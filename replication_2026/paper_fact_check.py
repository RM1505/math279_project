"""Cross-check the short paper's empirical claims against saved result tables."""
from pathlib import Path
import json
import pandas as pd
from resolution_study import R
p=Path(__file__).parent/'SNAPSHOT_OFI_PAPER.md';text=p.read_text();checks=[]
def claim(label,value,printed):
    assert printed in text,(label,'missing from paper');checks.append(dict(claim=label,value=value,paper=printed))
s=pd.read_csv(R/'sensitivity_summary.csv')
def rate(h,year=2019,offset=0,tol=.1):return s[(s.h==h)&(s.year==year)&(s.offset==offset)&(s.tolerance==tol)&(s.metric=='opposite_rate')].iloc[0]
for label,row in [('five-minute 2019',rate(5)),('thirty-minute 2019',rate(30)),('five-minute 2017',rate(5,2017)),('five-minute 2018',rate(5,2018)),('phase shift',rate(5,offset=2)),('threshold 0.001',rate(5,tol=.001))]:claim(label,float(row.estimate),f'{100*row.estimate:.2f}%')
row=rate(5);assert f'{100*row.low:.2f}–{100*row.high:.2f}%' in text
m=pd.read_csv(R/'model_summary.csv');m=m[(m.h==5)&(m.offset==0)].set_index('model')
for label in ['same_fine','same_coarse','same_frozen']:claim(label,float(m.loc[label,'r2']),f"{100*m.loc[label,'r2']:.2f}%")
t=pd.read_csv(R/'transform_summary.csv').set_index('model')
for label in ['tanh_coarse','sign_return']:claim(label,float(t.loc[label,'r2']),f"{100*t.loc[label,'r2']:.2f}%")
c=pd.read_csv(R/'model_contrasts.csv');c=c[(c.h==5)&(c.offset==0)].set_index('comparison')
for label in ['next_fine vs next_base','next_coarse vs next_base']:
    value=float(c.loc[label,'relative_mse_gain']);assert value<0;claim(label,value,f'{-100*value:.4f}%')
coverage=json.loads((R/'coverage.json').read_text())
for key in ['stocks','windows_primary','stock_days_primary','excluded_primary']:claim(key,coverage[key],f'{coverage[key]:,}')
d=pd.read_csv(R/'material_depth_summary.csv');d=d[d.year==2019]
for key in ['opposite','both_material']:claim(key,int(d[key].sum()),f'{int(d[key].sum()):,}')
claim('material retention',d.both_material.sum()/d.windows.sum(),f'{100*d.both_material.sum()/d.windows.sum():.2f}%')
n=pd.read_csv(R/'aggregation_null.csv');v=n[n.minutes==1001].opposite_nonzero.iloc[0];claim('long null disagreement',float(v),f'{100*v:.2f}%')
f=pd.read_csv(R/'fit_replay.csv');claim('replayed models',len(f),str(len(f)));claim('replayed daily losses',int(f.days.sum()),f'{int(f.days.sum()):,}')
raw=pd.read_csv(R/'raw_verification.csv');assert len(raw)==18;assert raw.raw_zero_false_nonzero.sum()==4
assert 'N(0, σ²)' in text and 'σ > 0' in text
assert 'after seeing the raw comparison' in text
assert 'does not make 2019 an untouched research holdout' in text
assert 'float32 log-depth' in text
Path('full_study/resolution/paper_fact_check.json').write_text(json.dumps(dict(status='passed',claims=checks,raw_files=18),indent=2));print('Passed',len(checks),'empirical claim checks plus interval and scope checks')
