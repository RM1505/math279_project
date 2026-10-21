"""Cross-year and stock coverage checks for material five-minute disagreements."""
import numpy as np,pandas as pd
from resolution_study import R,OUT
from summarize_study import bootstrap_ratio

def main():
    parts=[]
    for p in (R/'sensitivity').glob('*.csv.gz'):
        d=pd.read_csv(p);parts.append(d[(d.h==5)&(d.offset==0)&(d.tolerance==.1)])
    d=pd.concat(parts);d['rate']=d.opposite/d.both.replace(0,np.nan);d['year']=d.date.str[:4].astype(int)
    arcs=pd.read_csv(OUT/'archives.csv').set_index('ticker');d['depth']=d.ticker.map(arcs.depth)
    s=d.groupby(['ticker','year']).agg(rate=('rate','mean'),opposite=('opposite','sum'),both=('both','sum'),n=('n','sum')).reset_index();s.to_csv(R/'material_stock_summary.csv',index=False)
    rows=[]
    for (year,depth),g in d.groupby(['year','depth']):
        v=g.groupby('date').rate.mean().dropna().to_numpy();p,l,u,_,_=bootstrap_ratio(v,np.ones(len(v)));rows.append(dict(year=year,depth=depth,stocks=g.ticker.nunique(),estimate=p,low=l,high=u,opposite=int(g.opposite.sum()),both_material=int(g.both.sum()),windows=int(g.n.sum())))
    pd.DataFrame(rows).to_csv(R/'material_depth_summary.csv',index=False)
    print(s.groupby('year').rate.agg(['median','min','max']).to_string());print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
