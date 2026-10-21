"""Paired day-level inference and separately evaluated next-day portfolios."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from next_day_study import D,TARGETS,FAMILIES,classifier,loss
from summarize_study import bootstrap_ratio,holm
from ranking_study import block_ci

def weights(p,kind):
    n=len(p);w=np.zeros(n);ok=np.isfinite(p)
    if kind=='top20':
        ids=np.flatnonzero(ok&(p!=0));ids=ids[np.argsort(-abs(p[ids]),kind='stable')[:20]];w[ids]=np.sign(p[ids])/20
    else:
        ids=np.flatnonzero(ok)
        if len(ids)<20:return w
        ids=ids[np.argsort(p[ids],kind='stable')];w[ids[-10:]]=.05;w[ids[:10]]=-.05
    return w

def portfolio(g,p,kind):
    w=weights(p,kind);chosen=w!=0
    if chosen.any() and not np.isfinite(g[['y','long_net','short_net']].to_numpy()[chosen]).all():return None
    y=g.y.to_numpy();ln=g.long_net.to_numpy()/1e4;sn=g.short_net.to_numpy()/1e4
    gross=np.sum(np.where(chosen,w*np.expm1(y),0))*1e4
    net=np.sum(np.where(w>0,abs(w)*np.expm1(ln),np.where(w<0,abs(w)*-np.expm1(-sn),0)))*1e4
    return dict(gross_bps=gross,net_bps=net,positions=int(chosen.sum()),gross_exposure=float(abs(w).sum()),net_exposure=float(w.sum()),oneway_turnover=float(np.sum(np.where(w>0,abs(w)*(1+np.exp(ln)),np.where(w<0,abs(w)*(1+np.exp(-sn)),0)))))

def main():
    metrics=[];port=[];missing=[];comparisons=[];control_rows=[];sensitivity=[]
    for target in TARGETS:
        controls=pd.read_csv(D/'predictions'/f'{target}_controls.csv.gz')
        for date,g in controls.groupby('signal_date'):
            observed=g[np.isfinite(g.y)]
            if len(observed)==0:continue
            for name in ['zero','training_mean','trailing_mean']:
                control_rows.append(dict(target=target,date=date,control=name,n=len(observed),mse=float(((observed.y-observed[name])**2).mean())))
        for family in FAMILIES:
            pair={}
            for group in ['base','book']:
                name=f'{target}_{family}_{group}';d=pd.read_csv(D/'predictions'/f'{name}.csv.gz')
                assert d[['ticker','signal_date']].equals(controls[['ticker','signal_date']])
                assert np.isfinite(d.prediction).all(),name
                valid=np.isfinite(d.y);y=d.y.to_numpy();p=d.prediction.to_numpy()
                d['loss']=loss(y,p,family)
                if classifier(family):
                    d['control_loss']=loss(y,controls.training_probability.to_numpy(),family);d['brier']=(p-(y>0))**2;d['accuracy']=((p>.5)==(y>0)).astype(float)
                else:
                    d['control_loss']=(y-controls.trailing_mean.to_numpy())**2;d['brier']=np.nan;d['accuracy']=((p>0)==(y>0)).astype(float)
                d['zero_loss']=y*y
                for date,g in d.groupby('signal_date',sort=True):
                    observed=g[np.isfinite(g.y)]
                    if len(observed)>=20:
                        ic=spearmanr(observed.prediction,observed.y).statistic if observed.prediction.std()>1e-14 else np.nan
                        metrics.append(dict(target=target,family=family,group=group,date=date,n=len(observed),loss=observed.loss.mean(),control_loss=observed.control_loss.mean(),zero_loss=observed.zero_loss.mean(),rank_ic=ic,brier=observed.brier.mean(),accuracy=observed.accuracy.mean()))
                    score=g.prediction.to_numpy()-(.5 if classifier(family) else 0)
                    for kind in ['top20','long10_short10']:
                        result=portfolio(g,score,kind)
                        if result is None:missing.append(dict(target=target,family=family,group=group,date=date,rule=kind))
                        else:port.append(dict(target=target,family=family,group=group,date=date,rule=kind,**result))
                pair[group]=d
            # Paired errors on EXACT identical observations; outcome validity shared.
            b=pair['base'];z=pair['book'];np.testing.assert_allclose(b.y,z.y,equal_nan=True)
            temp=pd.DataFrame({'date':b.signal_date,'baseline':b.loss,'gain':b.loss-z.loss});temp=temp[np.isfinite(temp.baseline)]
            day=temp.groupby('date')[['baseline','gain']].mean()
            if target=='intraday':
                for label,keep in [('abs_return_below_10pct',np.isfinite(b.y)&(abs(b.y)<.1)),('exclude_march_2020',np.isfinite(b.y)&~b.signal_date.str.startswith('2020-03'))]:
                    sd=temp.loc[temp.index.intersection(b.index[keep])].groupby('date')[['baseline','gain']].mean()
                    point,lo,hi,p,_=bootstrap_ratio(sd.gain.to_numpy(),sd.baseline.to_numpy())
                    sensitivity.append(dict(family=family,check=label,days=len(sd),effect=point,low=lo,high=hi,p_unadjusted=p))
            for period in ['all','2020','2021']:
                g=day if period=='all' else day[day.index.str.startswith(period)]
                point,lo,hi,p,_=bootstrap_ratio(g.gain.to_numpy(),g.baseline.to_numpy())
                comparisons.append(dict(target=target,family=family,period=period,days=len(g),effect=point,low=lo,high=hi,p=p))
        print('Summarized next-day',target,flush=True)
    pd.DataFrame(control_rows).to_csv(D/'control_daily.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(D/'intraday_sensitivity.csv',index=False)
    m=pd.DataFrame(metrics);m.to_csv(D/'daily_metrics.csv',index=False)
    m['year']=m.date.str[:4]
    summaries=[]
    for period in ['all','2020','2021']:
        g=m if period=='all' else m[m.year==period]
        s=g.groupby(['target','family','group'])[['loss','control_loss','zero_loss','rank_ic','brier','accuracy']].mean().reset_index();s['period']=period;s['improvement_vs_control']=1-s.loss/s.control_loss;s['r2_vs_zero']=np.where(s.family.isin(['logistic','class_boost']),np.nan,1-s.loss/s.zero_loss);summaries.append(s)
    pd.concat(summaries).to_csv(D/'model_summary.csv',index=False)
    c=pd.DataFrame(comparisons);c['p_holm']=np.nan
    for primary in [True,False]:
        mask=(c.period=='all')&((c.target=='intraday') if primary else (c.target!='intraday'))
        c.loc[mask,'p_holm']=holm(c.loc[mask,'p'])
    c.to_csv(D/'inference.csv',index=False)
    pd.DataFrame(missing).to_csv(D/'unscorable_portfolios.csv',index=False)
    p=pd.DataFrame(port);p.to_csv(D/'portfolio_daily_unmatched.csv',index=False)
    allports=[];paired=[];dates_info=[]
    for (target,rule),g in p.groupby(['target','rule']):
        counts=g.groupby('date').size();dates=counts[counts==len(FAMILIES)*2].index
        # Exclude dates with no scored target anywhere (e.g., final archive day).
        dates=dates.intersection(m[m.target==target].date.unique());g=g[g.date.isin(dates)].copy();g['year']=g.date.str[:4]
        dates_info.append(dict(target=target,rule=rule,days=len(dates),all_dates=sorted(dates)))
        allports.append(g)
        for period in ['all','2020','2021']:
            z=g if period=='all' else g[g.year==period]
            for fam,h in z.groupby('family'):
                w=h.pivot(index='date',columns='group',values='net_bps');a=(w.book-w.base).to_numpy();lo,hi=block_ci(a)
                paired.append(dict(target=target,rule=rule,period=period,family=fam,days=len(w),book_minus_base_bps=a.mean(),low=lo,high=hi))
    p=pd.concat(allports);p.to_csv(D/'portfolio_daily.csv',index=False);pd.DataFrame(paired).to_csv(D/'portfolio_increment.csv',index=False);(D/'portfolio_dates.json').write_text(json.dumps(dates_info,indent=2))
    ps=[]
    for period in ['all','2020','2021']:
        z=p if period=='all' else p[p.year==period]
        for key,g in z.groupby(['target','family','group','rule']):
            lo,hi=block_ci(g.net_bps.to_numpy());r=dict(zip(['target','family','group','rule'],key));r.update(period=period,days=len(g),gross_bps=g.gross_bps.mean(),net_bps=g.net_bps.mean(),low=lo,high=hi,avg_positions=g.positions.mean(),avg_net_exposure=g.net_exposure.mean(),net_after_1bp_per_side=(g.net_bps-g.oneway_turnover).mean());ps.append(r)
    pd.DataFrame(ps).to_csv(D/'portfolio_summary.csv',index=False)
    print(c[c.period=='all'].to_string(index=False),flush=True)
if __name__=='__main__':main()
