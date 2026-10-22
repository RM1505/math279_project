"""Post-result manuscript checks; descriptive, not additional confirmatory tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent/'full_study/resolution'
def resamples(n, block, draws=2000):
    rng=np.random.default_rng(279)
    starts=rng.integers(0,n,(draws,int(np.ceil(n/block))))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
d=pd.read_csv(P/'sensitivity_daily.csv.gz')
d=d.query('h == 5 and offset == 0 and year == 2019 and tolerance == 0.1').sort_values('date')
a=d.opposite_rate.to_numpy(); rows=[]
for b in [10,20,40]:
    ci=np.quantile(a[resamples(len(a),b)].mean(1),[.025,.975])
    rows.append(dict(block_days=b,estimate=float(a.mean()),low=float(ci[0]),high=float(ci[1]),days=len(a)))
parts=[]
for f in sorted((P/'transforms').glob('*.csv.gz')):
    t=pd.read_csv(f).query("model == 'tanh_coarse'")
    s=pd.read_csv(P/'stock'/f.name).query('h == 5 and offset == 0').dropna(subset=['same_frozen_mse'])
    m=t.merge(s[['ticker','date','same_frozen_n','same_frozen_mse','same_frozen_benchmark']],on=['ticker','date'],validate='one_to_one')
    assert len(m)==len(t)==len(s)
    assert np.array_equal(m.n,m.same_frozen_n)
    assert np.allclose(m.benchmark,m.same_frozen_benchmark,rtol=1e-12,atol=0)
    parts.append(m)
m=pd.concat(parts); daily=m.groupby('date')[['mse','same_frozen_mse']].mean()
x=daily.same_frozen_mse.to_numpy(); y=daily.mse.to_numpy(); ix=resamples(len(x),20)
gain=1-x.mean()/y.mean(); ci=np.quantile(1-x[ix].mean(1)/y[ix].mean(1),[.025,.975])
s=pd.read_csv(P/'material_stock_summary.csv').query('year == 2019')
out={'classification':'post-result descriptive sensitivity; no additional confirmatory p-values', 'sign_block_sensitivity':rows,'frozen_vs_tanh_coarse':{'relative_mse_gain':float(gain),'low':float(ci[0]),'high':float(ci[1]),'stock_days':len(m),'dates':len(daily),'stocks':m.ticker.nunique(),'matched_windows':int(m.n.sum())},'stock_rate_quantiles':{str(q):float(s.rate.quantile(q)) for q in [.1,.5,.9]},'all_stock_rates_positive':bool((s.rate>0).all())}
(P/'paper_robustness.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
