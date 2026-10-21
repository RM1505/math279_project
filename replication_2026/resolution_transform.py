"""Post-result bounded-transform check; descriptive, not a new primary test."""
import numpy as np,pandas as pd
from resolution_study import OUT,CACHE,R,blocks,fit_predict
from summarize_study import bootstrap_ratio

def main():
    dest=R/'transforms';dest.mkdir(exist_ok=True)
    for i,tk in enumerate(np.load(OUT/'market.npz')['universe']):
        p=dest/f'{tk}.csv.gz'
        if p.exists():continue
        with np.load(CACHE/f'{tk}.npz') as f:x=f['x'];dates=f['dates']
        z=blocks(x,dates,5,0);good=z['good']&np.isfinite(z['ret'])&(abs(z['ret'])<.2);rows=[]
        configs={'tanh_fine':np.tanh(z['fine']),'tanh_coarse':np.tanh(z['coarse']),'sign_fine':np.sign(z['fine']),'sign_coarse':np.sign(z['coarse']),'sign_return':np.sign(z['ret'])}
        for name,xx in configs.items():
            f=fit_predict(xx[...,None],z['ret'],z['date'],good)
            if f is None:continue
            pred,mean,_=f
            for j in np.flatnonzero(dates>='2019-01-01'):
                ok=np.isfinite(pred[j]);y=z['ret'][j,ok]
                if ok.any():rows.append(dict(ticker=tk,date=dates[j],model=name,n=int(ok.sum()),mse=float(np.mean((pred[j,ok]-y)**2)),benchmark=float(np.mean((y-mean)**2))))
        pd.DataFrame(rows).to_csv(p,index=False)
        if i%100==0:print('Transform sensitivity',i+1,flush=True)
    d=pd.concat([pd.read_csv(p) for p in dest.glob('*.csv.gz')]);day=d.groupby(['model','date'])[['mse','benchmark']].mean().reset_index();rows=[]
    for name,g in day.groupby('model'):
        p,l,u,_,_=bootstrap_ratio((g.benchmark-g.mse).to_numpy(),g.benchmark.to_numpy());rows.append(dict(model=name,r2=p,low=l,high=u,days=len(g),stocks=d[d.model==name].ticker.nunique()))
    pd.DataFrame(rows).to_csv(R/'transform_summary.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
