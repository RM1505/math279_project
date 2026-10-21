"""Exploratory within-stock-day liquidity matching of the saved closing effect."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json
import numpy as np
import pandas as pd
from study import OUT,CACHE,make_design,targets,infer
from summarize_study import bootstrap_ratio
O=OUT/'decision'
def main():
    market=np.load(OUT/'market.npz');rows=[];counts=[]
    for i,tk in enumerate(market['universe']):
        with np.load(CACHE/f'{tk}.npz') as f:x=f['x'];dates=f['dates'];depth=int(f['depth'])
        vol=pd.DataFrame(x[:,:,3]**2).T.rolling(30,min_periods=20).mean().T.to_numpy()**.5
        train=dates<'2018-01-01';time=np.arange(391);trainvalid=train[:,None]&(time[None,:]>=30)&np.isfinite(x[:,:,6])&np.isfinite(vol)&np.isfinite(x[:,:,7])
        cuts=[np.quantile(z[trainvalid],q) for z,q in [(x[:,:,6],[1/3,2/3]),(vol,[1/3,2/3]),(x[:,:,7],[.5])]]
        test=dates>='2019-01-01';x=x[test];vol=vol[test];dates=dates[test];z=make_design(x,dates,market);y=targets(x,1,0);t=np.broadcast_to(time,y.shape)
        valid=np.isfinite(z[:,:,:17 if depth==10 else 15]).all(2)&np.isfinite(y)&(t>=30)&(t+1<390)&(abs(y)<.2)
        pars=json.loads((OUT/'scores'/f'{tk}_parameters.json').read_text());errors=[]
        for name in ['returns','book']:
            p=pars[name+'_h1_delay0'];fit=tuple(np.array(p[k]) for k in ['clip_low','clip_high','center','scale'])+(p['target_mean'],np.array(p['coefficients']));pred=infer(fit,z[valid][:,p['columns']]);err=np.full(y.shape,np.nan);err[valid]=(y[valid]-pred)**2;errors.append(err)
        bins=np.digitize(x[:,:,6],cuts[0])*6+np.digitize(vol,cuts[1])*2+np.digitize(x[:,:,7],cuts[2])
        for j,date in enumerate(dates):
            mask=valid[j]&np.isfinite(vol[j]);weights=[];stats=[]
            for cell in range(18):
                core=mask&(time<360)&(bins[j]==cell);close=mask&(time>=360)&(bins[j]==cell)
                weight=min(core.sum(),close.sum())
                if weight==0:continue
                weights.append(weight);stats.append([errors[0][j,core].mean(),(errors[0][j,core]-errors[1][j,core]).mean(),errors[0][j,close].mean(),(errors[0][j,close]-errors[1][j,close]).mean()])
            if not weights:continue
            v=np.average(stats,axis=0,weights=weights);rows.append(dict(ticker=tk,date=date,core_base=v[0],core_gain=v[1],close_base=v[2],close_gain=v[3],matched_minutes=sum(weights),eligible_close=int((mask&(time>=360)).sum())))
        if i%40==0:print('Matched',i+1,'/',len(market['universe']),flush=True)
    r=pd.DataFrame(rows);r.to_csv(O/'closing_matched_stock_days.csv.gz',index=False);d=r.groupby('date')[['core_base','core_gain','close_base','close_gain']].mean();d.to_csv(O/'closing_matched_days.csv')
    result=[]
    for label,a,b in [('core',d.core_gain,d.core_base),('close',d.close_gain,d.close_base),('close_minus_core_common_scale',d.close_gain-d.core_gain,(d.close_base+d.core_base)/2)]:
        point,lo,hi,p,_=bootstrap_ratio(a.to_numpy(),b.to_numpy());result.append(dict(check=label,effect=point,low=lo,high=hi,p=p,days=len(d),stock_days=len(r),matched_close_fraction=r.matched_minutes.sum()/r.eligible_close.sum()))
    cp,_,_,_,cv=bootstrap_ratio(d.close_gain.to_numpy(),d.close_base.to_numpy());bp,_,_,_,bv=bootstrap_ratio(d.core_gain.to_numpy(),d.core_base.to_numpy());delta=cp-bp;draw=cv-bv;lo,hi=np.quantile(draw,[.025,.975]);pv=(1+np.sum(abs(draw-delta)>=abs(delta)))/(len(draw)+1);result.append(dict(check='close_minus_core_relative',effect=delta,low=lo,high=hi,p=pv,days=len(d),stock_days=len(r),matched_close_fraction=r.matched_minutes.sum()/r.eligible_close.sum()))
    pd.DataFrame(result).to_csv(O/'closing_matched_inference.csv',index=False);print(pd.DataFrame(result).to_string(index=False))
if __name__=='__main__':main()
