"""Fixed exploratory ranking rules; raw-data caches and frozen original estimates."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,time,argparse
import numpy as np
import pandas as pd
from study import OUT,CACHE,make_design,targets,prepare_ridge,solve_prepared,fit_ridge,infer,ALPHAS
R=OUT/'ranking';MODELS=['returns','cross','scaled_book']
# All data used in selection are current-time fields; outcomes are separate.
FIELDS=['returns','cross','scaled_book','y','long_net','short_net','spread','exit_spread','mid']

def build():
    R.mkdir(exist_ok=True)
    m=np.load(OUT/'market.npz');stocks=m['universe'].tolist();dates=m['dates'];dates=dates[dates>='2018-01-01'];lookup={d:i for i,d in enumerate(dates)}
    shape=(len(stocks),len(dates),359,len(FIELDS))
    path=R/'panel.npy'
    if path.exists():a=np.lib.format.open_memmap(path,mode='r+')
    else:a=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=shape);a[:]=np.nan
    done=json.loads((R/'build_progress.json').read_text()) if (R/'build_progress.json').exists() else []
    np.savez(R/'index.npz',stocks=stocks,dates=dates)
    for i,tk in enumerate(stocks):
        if tk in done:continue
        with np.load(CACHE/f'{tk}.npz') as d:x=d['x'];dd=d['dates'];depth=int(d['depth'])
        z=make_design(x,dd,m);z=np.dstack([z,x[:,:,6]*x[:,:,8]/2])
        y=targets(x,1,0);t=np.broadcast_to(np.arange(391),y.shape)
        current=np.isfinite(z[:,:,:17 if depth==10 else 15]).all(2)&np.isfinite(z[:,:,-1])&(t>=30)&(t<389)
        good=current&np.isfinite(y)&(abs(y)<.2)
        tr=good&(dd[:,None]<'2018-01-01');va=good&(dd[:,None]>='2018-01-01')&(dd[:,None]<'2019-01-01')
        params=json.loads((OUT/'scores'/f'{tk}_parameters.json').read_text());preds=[];fitrecord={}
        for model in MODELS:
            cols=list(range(10)) if model=='returns' else list(range(15))
            if model=='scaled_book':cols=cols+[17]
            if model=='scaled_book':
                prep=prepare_ridge(z[tr][:,cols],y[tr].astype(float));loss=[np.mean((infer(solve_prepared(prep,alpha),z[va][:,cols])-y[va])**2) for alpha in ALPHAS]
                alpha=ALPHAS[int(np.argmin(loss))];f=fit_ridge(z[tr|va][:,cols],y[tr|va].astype(float),alpha)
            else:
                v=params[model+'_h1_delay0'];alpha=v['alpha'];loss=v['validation_mse']
                f=tuple(np.array(v[k]) for k in ['clip_low','clip_high','center','scale'])+(v['target_mean'],np.array(v['coefficients']))
            vf=fit_ridge(z[tr][:,cols],y[tr].astype(float),alpha)
            p=np.full(y.shape,np.nan)
            for period,fit in [('2018',vf),('2019',f)]:
                mask=current&np.char.startswith(dd[:,None],period);p[mask]=infer(fit,z[mask][:,cols])
            preds.append(p)
            fitrecord[model]={'alpha':alpha,'columns':cols,'validation_mse':loss,'validation_fit':[np.asarray(v).tolist() for v in vf],'test_fit':[np.asarray(v).tolist() for v in f]}
        with np.errstate(invalid='ignore',divide='ignore'):
            ln=np.full(y.shape,np.nan);sn=ln.copy()
            ln[:,:-1]=np.log(x[:,1:,1]/x[:,:-1,2])*1e4;sn[:,:-1]=np.log(x[:,:-1,1]/x[:,1:,2])*1e4
        yy=y.copy();yy[abs(yy)>=.2]=np.nan
        exitsp=np.roll(x[:,:,6],-1,axis=1)*1e4
        data=np.stack([*preds,yy,ln,sn,x[:,:,6]*1e4,exitsp,x[:,:,0]],axis=-1)
        keep=dd>='2018-01-01';ix=[lookup[v] for v in dd[keep]]
        a[i,ix]=data[keep,30:389];a.flush()
        (R/f'{tk}_fits.json').write_text(json.dumps(fitrecord))
        done.append(tk);(R/'build_progress.json').write_text(json.dumps(done))
        if i%20==0:print('Built',i+1,'/',len(stocks),tk,flush=True)
    print('Panel complete',shape,flush=True)

def select_weights(pred,spread,k,kind):
    # shapes minute x stock; stable ticker ordering resolves ties.
    valid=np.isfinite(pred)&np.isfinite(spread)&(spread>0)&(pred!=0)
    score=abs(pred)
    if kind!='magnitude':
        multiple=1 if kind=='edge1' else 2
        valid &= abs(pred)*1e4>multiple*spread
        score=abs(pred)*1e4-spread
    score=np.where(valid,score,-np.inf)
    order=np.argsort(-score,axis=1,kind='stable')[:,:k]
    w=np.zeros_like(pred,dtype=float);ii=np.arange(len(pred))[:,None]
    w[ii,order]=np.where(valid[ii,order],np.sign(pred[ii,order])/k,0)
    return w

def account(w,y,spread,exit_spread,ln,sn):
    chosen=w!=0
    bad=chosen&(~np.isfinite(y)|~np.isfinite(ln)|~np.isfinite(sn)|~np.isfinite(exit_spread))
    if bad.any():return None,int(bad.sum())
    gross=np.sum(np.where(chosen,w*np.expm1(y)*1e4,0),axis=1)
    previous=np.zeros_like(w);previous[1:]=np.where(chosen[:-1],w[:-1]*np.exp(y[:-1]),0)
    change=abs(w-previous)
    if ((change>0)&~np.isfinite(spread)).any():return None,int(((change>0)&~np.isfinite(spread)).sum())
    costs=np.sum(np.where(change>0,change*spread/2,0),axis=1)
    liquidation=np.sum(np.where(chosen[-1],abs(w[-1])*np.exp(y[-1])*exit_spread[-1]/2,0))
    costs[-1]+=liquidation
    turnover=change.sum(axis=1);turnover[-1]+=np.sum(np.where(chosen[-1],abs(w[-1])*np.exp(y[-1]),0))
    rt=np.sum(np.where(w>0,abs(w)*ln,np.where(w<0,abs(w)*sn,0)),axis=1)
    return {'gross':gross,'net':gross-costs,'roundtrip':rt,'turnover':turnover,'positions':chosen.sum(axis=1)},0

def evaluate():
    idx=np.load(R/'index.npz');dates=idx['dates'];a=np.load(R/'panel.npy',mmap_mode='r')
    assert len(json.loads((R/'build_progress.json').read_text()))==len(idx['stocks'])
    rows=[];excluded=[]
    for di,date in enumerate(dates):
        v=np.asarray(a[:,di],dtype=float).transpose(1,0,2)
        if not np.isfinite(v[:,:,:3]).any():continue
        for mi,model in enumerate(MODELS):
            for k in [5,20,50]:
                for kind in ['magnitude','edge1','edge2']:
                    w=select_weights(v[:,:,mi],v[:,:,6],k,kind)
                    result,bad=account(w,v[:,:,3],v[:,:,6],v[:,:,7],v[:,:,4],v[:,:,5])
                    if result is None:excluded.append({'date':date,'model':model,'k':k,'rule':kind,'unscorable':bad});continue
                    for ss,mask in [('all',np.ones(359,dtype=bool)),('core',np.arange(359)<330),('close',np.arange(359)>=330)]:
                        row={'date':date,'year':int(date[:4]),'model':model,'k':k,'rule':kind,'session':ss}
                        row.update({key:float(value[mask].sum()) for key,value in result.items()})
                        row['active_minutes']=int((result['positions'][mask]>0).sum());row['minutes']=int(mask.sum())
                        row['predicted_bps']=float(np.sum(np.where(w[mask]!=0,abs(w[mask])*abs(v[mask,:,mi])*1e4,0)))
                        row['entry_spread_bps']=float(np.sum(np.where(w[mask]!=0,abs(w[mask])*v[mask,:,6],0)))
                        row['absolute_net_exposure']=float(abs(w[mask].sum(axis=1)).sum())
                        rows.append(row)
        if di%50==0:print('Scored',date,flush=True)
    d=pd.DataFrame(rows);d.to_csv(R/'daily.csv',index=False);pd.DataFrame(excluded).to_csv(R/'excluded.csv',index=False)
    summarize(d)

def block_ci(a,draws=2000):
    rng=np.random.default_rng(279);n=len(a);start=rng.integers(0,n,(draws,int(np.ceil(n/20))))
    ix=((start[:,:,None]+np.arange(20))%n).reshape(draws,-1)[:,:n]
    return np.quantile(a[ix].mean(1),[.025,.975])

def summarize(d):
    # All candidate rules are compared on the same complete dates.
    counts=d[d.session=='all'].groupby('date').size()
    common=counts[counts==len(MODELS)*9].index
    excluded_dates=sorted(set(d.date)-set(common))
    (R/'common_day_exclusions.json').write_text(json.dumps(excluded_dates,indent=2))
    d=d[d.date.isin(common)].copy()
    rows=[]
    for key,g in d.groupby(['year','model','k','rule','session']):
        lo,hi=block_ci(g.net.to_numpy());r=dict(zip(['year','model','k','rule','session'],key))
        r.update(days=len(g),daily_net_bps=g.net.mean(),low=lo,high=hi,daily_gross_bps=g.gross.mean(),daily_roundtrip_bps=g.roundtrip.mean(),daily_turnover=g.turnover.mean(),avg_positions=g.positions.sum()/g.minutes.sum(),active_minutes=g.active_minutes.sum(),quote_net_per_roundtrip_bps=g.roundtrip.sum()*r['k']/g.positions.sum() if g.positions.sum()>0 else np.nan)
        positions=g.positions.sum()
        r.update(predicted_per_trade_bps=g.predicted_bps.sum()*r['k']/positions if positions else np.nan,entry_spread_per_trade_bps=g.entry_spread_bps.sum()*r['k']/positions if positions else np.nan,avg_absolute_net_exposure=g.absolute_net_exposure.sum()/g.minutes.sum())
        for fee in [.1,.5,1.]:r['daily_net_fee_'+str(fee)+'_bps']=(g.net-fee*g.turnover).mean()
        rows.append(r)
    s=pd.DataFrame(rows);s.to_csv(R/'summary.csv',index=False)
    va=s[(s.year==2018)&(s.session=='all')].sort_values('daily_net_bps',ascending=False)
    best=va.iloc[0]
    selected={'rule':'cash','validation_daily_net_bps':0.0} if best.daily_net_bps<=0 else {'model':best.model,'k':int(best.k),'rule':best.rule,'validation_daily_net_bps':best.daily_net_bps}
    (R/'selected_rule.json').write_text(json.dumps(selected,indent=2))
    print('Validation selection',selected,flush=True)
    print(s[(s.year==2019)&(s.k==20)&(s.session=='all')].to_string(index=False),flush=True)
    print('Best 2019 after looking (not validation-selected):',s[(s.year==2019)&(s.session=='all')].sort_values('daily_net_bps',ascending=False).head(3).to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','evaluate']);a=p.parse_args();build() if a.stage=='build' else evaluate()
