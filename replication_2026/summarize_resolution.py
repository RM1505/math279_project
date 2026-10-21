"""Equal-stock/equal-day measurement summaries; paired forward-loss contrasts."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/ofi_mpl'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from resolution_study import R
from summarize_study import bootstrap_ratio,holm

def main():
    files=sorted((R/'stock').glob('*.csv.gz'))
    d=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    d['year']=d.date.str[:4].astype(int)
    for name,num,den in [('opposite_rate','opposite','both_nonzero'),('zero_loss_rate','coarse_zero_fine_nonzero','fine_nonzero'),('material_opposite_rate','material_opposite','material')]:
        d[name]=d[num]/d[den].replace(0,np.nan)
    d['relative_squared_error']=d.error_ss/d.fine_ss.replace(0,np.nan)
    cols=['opposite_rate','zero_loss_rate','material_opposite_rate','corr','relative_squared_error']
    day=d.groupby(['h','offset','year','date'])[cols].mean().reset_index()
    day.to_csv(R/'measurement_daily.csv.gz',index=False)
    rows=[]
    for key,g in day.groupby(['h','offset','year']):
        for c in cols:
            v=g[c].dropna().to_numpy();p,l,u,_,_=bootstrap_ratio(v,np.ones(len(v)))
            if c=='relative_squared_error':p,l,u=np.sqrt([p,l,u])
            rows.append(dict(h=key[0],offset=key[1],year=key[2],metric=c,estimate=p,low=l,high=u,days=len(v)))
    pd.DataFrame(rows).to_csv(R/'measurement_summary.csv',index=False)
    d.groupby(['ticker','h','offset'])[cols].mean().to_csv(R/'measurement_by_stock.csv')
    fits=[];contrasts=[]
    for key,g in d[d.year==2019].groupby(['h','offset']):
        for model in ['same_fine','same_coarse','same_frozen','next_base','next_fine','next_coarse','lagged_coarse']:
            m=model+'_mse';b=model+'_benchmark'
            z=g[['date',m,b]].dropna().groupby('date')[[m,b]].mean()
            p,l,u,_,_=bootstrap_ratio((z[b]-z[m]).to_numpy(),z[b].to_numpy())
            fits.append(dict(h=key[0],offset=key[1],model=model,r2=p,low=l,high=u,days=len(z)))
        for alt,base in [('same_coarse','same_fine'),('same_frozen','same_coarse'),('next_fine','next_base'),('next_coarse','next_base')]:
            a=alt+'_mse';b=base+'_mse';na=alt+'_n';nb=base+'_n'
            z=g[['date',a,b,na,nb]].dropna();assert (z[na]==z[nb]).all()
            z=z.groupby('date')[[a,b]].mean();p,l,u,pv,_=bootstrap_ratio((z[b]-z[a]).to_numpy(),z[b].to_numpy())
            contrasts.append(dict(h=key[0],offset=key[1],comparison=alt+' vs '+base,relative_mse_gain=p,low=l,high=u,p=pv,days=len(z)))
    c=pd.DataFrame(contrasts);c['holm_primary']=np.nan;primary=(c.h==5)&(c.offset==0);c.loc[primary,'holm_primary']=holm(c.loc[primary,'p'])
    c.to_csv(R/'model_contrasts.csv',index=False);f=pd.DataFrame(fits);f.to_csv(R/'model_summary.csv',index=False)
    maxerr=max(json.loads(p.read_text())['max_algebra_error'] for p in (R/'stock').glob('*.json'))
    coverage=dict(stocks=len(files),stock_days_primary=len(d[(d.h==5)&(d.offset==0)]),windows_primary=int(d[(d.h==5)&(d.offset==0)].n.sum()),excluded_primary=int(d[(d.h==5)&(d.offset==0)].excluded.sum()),max_algebra_error=maxerr)
    (R/'coverage.json').write_text(json.dumps(coverage,indent=2))
    rng=np.random.default_rng(279);r=rng.normal(size=1000000);s=np.sign(r);future=rng.normal(size=len(r));null=dict(draws=len(r),gaussian_theory=2/np.pi,contemporaneous_r2=float(np.corrcoef(r,s)[0,1]**2),forward_r2=float(np.corrcoef(future,s)[0,1]**2));(R/'null.json').write_text(json.dumps(null,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    z=pd.read_csv(R/'sensitivity_summary.csv');z=z[(z.year==2019)&(z.offset==0)&(z.tolerance==.1)&(z.metric=='opposite_rate')]
    ax[0].plot(z.h,100*z.estimate,'o-');ax[0].fill_between(z.h,100*z.low,100*z.high,alpha=.2);ax[0].set(xlabel='Snapshot spacing (minutes)',ylabel='Opposite signs, both material (%)',title='Endpoint versus summed minute imbalance',xticks=[1,2,5,10,30])
    for model,label in [('same_fine','Summed minute'),('same_coarse','Endpoint'),('same_frozen','Frozen queues')]:
        z=f[(f.offset==0)&(f.model==model)];ax[1].plot(z.h,100*z.r2,'o-',label=label)
    ax[1].set(xlabel='Block length (minutes)',ylabel='Contemporaneous R² (%)',title='Same-block explanation, not forecasting',xticks=[1,2,5,10,30]);ax[1].legend();fig.savefig(R/'resolution.png',dpi=170);plt.close(fig)
    print(json.dumps(coverage));print(c[primary].to_string(index=False));print(f[(f.h==5)&(f.offset==0)].to_string(index=False))
if __name__=='__main__':main()
