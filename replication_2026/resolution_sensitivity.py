"""Numerical sign sensitivity, frozen before inspecting the full-study estimates."""
import numpy as np,pandas as pd
from resolution_study import OUT,CACHE,R,blocks

def main():
    dest=R/'sensitivity';dest.mkdir(exist_ok=True)
    for i,tk in enumerate(np.load(OUT/'market.npz')['universe']):
        p=dest/f'{tk}.csv.gz'
        if p.exists():continue
        with np.load(CACHE/f'{tk}.npz') as f:x=f['x'];dates=f['dates']
        rows=[]
        for h,offset in [(1,0),(2,0),(5,0),(10,0),(30,0),(5,2)]:
            z=blocks(x,dates,h,offset)
            for j,date in enumerate(dates):
                g=z['good'][j]&np.isfinite(z['ret'][j])&(abs(z['ret'][j])<.2);a=z['fine'][j,g];b=z['coarse'][j,g]
                if not len(a):continue
                for tol in [.001,.01,.1]:
                    both=(abs(a)>tol)&(abs(b)>tol);fn=abs(a)>tol
                    rows.append(dict(ticker=tk,date=date,h=h,offset=offset,tolerance=tol,n=len(a),both=int(both.sum()),opposite=int(np.sum(both&(a*b<0))),fine_nonzero=int(fn.sum()),coarse_zero_fine_nonzero=int(np.sum(fn&(abs(b)<=tol)))))
        pd.DataFrame(rows).to_csv(p,index=False)
        if i%50==0:print('Sensitivity',i+1,flush=True)
    d=pd.concat([pd.read_csv(p) for p in dest.glob('*.csv.gz')]);d['year']=d.date.str[:4];d['opposite_rate']=d.opposite/d.both.replace(0,np.nan);d['zero_rate']=d.coarse_zero_fine_nonzero/d.fine_nonzero.replace(0,np.nan)
    day=d.groupby(['h','offset','year','tolerance','date'])[['opposite_rate','zero_rate']].mean().reset_index();day.to_csv(R/'sensitivity_daily.csv.gz',index=False)
    from summarize_study import bootstrap_ratio
    rows=[]
    for k,g in day.groupby(['h','offset','year','tolerance']):
        for col in ['opposite_rate','zero_rate']:
            v=g[col].dropna().to_numpy();p,l,u,_,_=bootstrap_ratio(v,np.ones(len(v)));rows.append(dict(h=k[0],offset=k[1],year=k[2],tolerance=k[3],metric=col,estimate=p,low=l,high=u))
    pd.DataFrame(rows).to_csv(R/'sensitivity_summary.csv',index=False);print('Sensitivity complete',flush=True)
if __name__=='__main__':main()
