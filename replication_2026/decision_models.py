"""Frozen relative-return and risk extensions, with an earlier chronological check."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import next_day_study as n
from summarize_next_day import portfolio
from summarize_study import bootstrap_ratio,holm
from historical_daily import H
O=n.D.parent/'decision'
FAMILIES=['ridge','additive','logistic'];RULES=['top20','long10_short10','liquid_top20','liquid_long10_short10']

def masks(df,y,period):
    a,b,c=('2014-01-01','2015-01-01','2017-01-01') if period=='historical' else ('2019-01-01','2020-01-01','2022-01-01')
    valid=np.isfinite(y)
    tr=valid&(df.signal_date<a)&(df.target_date<a)
    va=(df.signal_date>=a)&(df.signal_date<b)&(df.target_date<b)
    te=(df.signal_date>=b)&(df.signal_date<c)&(df.target_date<c)
    return tr,va,te

def data(period):
    directory=H if period=='historical' else n.D
    if not (directory/'panel.pkl').exists():
        old=n.D;n.D=directory
        try:n.make_panel()
        finally:n.D=old
    df=pd.read_pickle(directory/'panel.pkl');fc=json.loads((directory/'feature_columns.json').read_text())
    ix=pd.MultiIndex.from_frame(df[['ticker','target_date']]);lookup=df.set_index(['ticker','signal_date'])
    for target in ['intraday','combined']:df[target+'_relative']=df[target+'_y']-df.groupby('signal_date')[target+'_y'].transform('mean')
    for name,source in [('vol','rv'),('range','price_range'),('spread','spread_mean')]:
        v=df[source].where(df[source]>0);df[name+'_current']=np.log(v)
        future=lookup[source].reindex(ix).to_numpy();df[name+'_risk']=np.log(np.where(future>0,future,np.nan))
        for window in [5,20]:
            ordered=df.sort_values(['ticker','signal_date']);roll=ordered.groupby('ticker')[name+'_current'].transform(lambda s:s.rolling(window,min_periods=max(2,window//2)).mean());df[name+'_mean'+str(window)]=roll.reindex(df.index)
    return df,fc

def fit_one(df,cols,y,family,period,dest):
    tr,va,te=masks(df,y,period);valid=np.isfinite(y);vv=va&valid;tv=tr|vv
    x=df[cols].to_numpy(float);scores=[];warns=[]
    for setting in n.settings(family):
        fit,w,clip=n.train_model(x[tr],y[tr],family,setting);scores.append(float(n.loss(y[vv],n.predict(fit,x[vv],family),family).mean()));warns+=w
    chosen=n.settings(family)[np.argmin(scores)]
    fit,w,clip=n.train_model(x[tr],y[tr],family,chosen);warns+=w;pv=n.predict(fit,x[va],family)
    fit,w,clip=n.train_model(x[tv],y[tv],family,chosen);warns+=w;pt=n.predict(fit,x[te],family)
    result=pd.concat([pd.DataFrame({'row':np.flatnonzero(va),'period':'validation','prediction':pv}),pd.DataFrame({'row':np.flatnonzero(te),'period':'test','prediction':pt})],ignore_index=True)
    result.to_csv(dest.with_suffix('.csv.gz'),index=False)
    with open(dest.with_suffix('.pkl'),'wb') as f:pickle.dump(dict(fit=fit,columns=cols,family=family),f)
    dest.with_suffix('.json').write_text(json.dumps(dict(setting=int(chosen),validation_losses=scores,train=int(tr.sum()),validation=int(vv.sum()),refit=int(tv.sum()),test=int(te.sum()),warnings=warns,columns=cols),indent=2))
    print('Fit',period,dest.name,flush=True)

def main(period):
    out=O/period/'models';out.mkdir(parents=True,exist_ok=True);df,fc=data(period)
    jobs=[]
    if period=='historical':jobs += [('combined','original','logistic',df.combined_y.to_numpy())]
    for target in ['intraday','combined']:
        for family in FAMILIES:jobs.append((target,'relative',family,df[target+'_relative'].to_numpy()))
    for target in ['vol','range','spread']:
        for family in ['ridge','boost']:jobs.append((target,'risk',family,df[target+'_risk'].to_numpy()))
    for target,kind,family,y in jobs:
        extra=[target+'_past_mean20',target+'_past_mean60',target+'_past_vol20'] if kind!='risk' else [r+s for r in ['vol','range','spread'] for s in ['_current','_mean5','_mean20']]
        for group in ['base','book']:
            dest=out/f'{target}_{kind}_{family}_{group}'
            if dest.with_suffix('.csv.gz').exists():continue
            fit_one(df,fc['base']+extra+(fc['book'] if group=='book' else []),y,family,period,dest)
    evaluate(period,df,jobs)

def evaluate(period,df,jobs):
    folder=O/period;metrics=[];ports=[];missing=[];inference=[]
    for target,kind,family,y in jobs:
        pair={}
        for group in ['base','book']:
            p=pd.read_csv(folder/'models'/f'{target}_{kind}_{family}_{group}.csv.gz');z=df.iloc[p.row].copy().reset_index(drop=True);z['prediction']=p.prediction;z['partition']=p.period;z['loss']=n.loss(y[p.row],p.prediction.to_numpy(),family)
            if kind=='risk':z['persistence_loss']=(y[p.row]-z[target+'_current'])**2
            for (partition,date),g in z.groupby(['partition','signal_date']):
                good=np.isfinite(g.loss)
                if good.any():metrics.append(dict(target=target,kind=kind,family=family,group=group,partition=partition,date=date,n=int(good.sum()),loss=g.loc[good,'loss'].mean(),persistence_loss=g.loc[good,'persistence_loss'].mean() if kind=='risk' else np.nan))
                if kind=='risk':continue
                q=g.rename(columns={target+'_y':'y',target+'_long':'long_net',target+'_short':'short_net'})
                score=g.prediction.to_numpy()-(.5 if n.classifier(family) else 0)
                for rule in RULES if kind=='relative' else RULES[:2]:
                    s=score.copy();base_rule=rule.replace('liquid_','')
                    if rule.startswith('liquid_'):s[g.spread.to_numpy()>g.spread.median()]=np.nan
                    result=portfolio(q,s,base_rule)
                    keys=dict(target=target,kind=kind,family=family,group=group,partition=partition,date=date,rule=rule)
                    if result is None:missing.append(keys)
                    else:ports.append(keys|result)
            pair[group]=z
        b=pair['base'];book=pair['book'];assert b[['ticker','signal_date','partition']].equals(book[['ticker','signal_date','partition']])
        day=pd.DataFrame(dict(date=b.signal_date,partition=b.partition,baseline=b.loss,gain=b.loss-book.loss));day=day[day.partition=='test'].groupby('date')[['baseline','gain']].mean().dropna()
        for span in ['all']+sorted(set(day.index.str[:4])):
            z=day if span=='all' else day[day.index.str.startswith(span)];point,lo,hi,p,_=bootstrap_ratio(z.gain.to_numpy(),z.baseline.to_numpy());inference.append(dict(target=target,kind=kind,family=family,span=span,days=len(z),effect=point,low=lo,high=hi,p=p))
    m=pd.DataFrame(metrics);m.to_csv(folder/'daily_metrics.csv',index=False);r=pd.DataFrame(inference);r['p_holm']=np.nan
    for kind in ['risk','relative']:
        ix=(r.kind==kind)&(r.span=='all');r.loc[ix,'p_holm']=holm(r.loc[ix,'p'])
    r.to_csv(folder/'inference.csv',index=False)
    p=pd.DataFrame(ports);p.to_csv(folder/'portfolios_unmatched.csv',index=False);pd.DataFrame(missing).to_csv(folder/'missing_portfolios.csv',index=False)
    summaries=[];paired=[];selection=[];matched=[]
    for (kind,target,partition),g in p.groupby(['kind','target','partition']):
        expected=24 if kind=='relative' else 4
        counts=g.groupby('date').size();dates=counts[counts==expected].index;g=g[g.date.isin(dates)].copy();matched.append(g)
        for (family,group,rule),z in g.groupby(['family','group','rule']):
            for span in ['all']+sorted(set(z.date.str[:4])):
                zz=z if span=='all' else z[z.date.str.startswith(span)];point,lo,hi,pval,_=bootstrap_ratio(zz.net_bps.to_numpy(),np.ones(len(zz)))
                summaries.append(dict(kind=kind,target=target,partition=partition,family=family,group=group,rule=rule,span=span,days=len(zz),net_bps=point,low=lo,high=hi,p=pval,gross_bps=zz.gross_bps.mean(),net_exposure=zz.net_exposure.mean(),after_1bp_side=(zz.net_bps-zz.oneway_turnover).mean()))
        for (family,rule),z in g.groupby(['family','rule']):
            pivot=z.pivot(index='date',columns='group',values='net_bps');delta=(pivot.book-pivot.base).to_numpy();point,lo,hi,pval,_=bootstrap_ratio(delta,np.ones(len(delta)));paired.append(dict(kind=kind,target=target,partition=partition,family=family,rule=rule,days=len(pivot),increment_bps=point,low=lo,high=hi,p=pval))
        if partition=='validation' and kind=='relative':
            scores=g[g.group=='book'].groupby(['family','rule']).net_bps.mean();key=scores.idxmax();best=float(scores.loc[key]);selection.append(dict(target=target,family=key[0] if best>0 else 'cash',rule=key[1] if best>0 else 'cash',validation_net_bps=max(0,best),best_risky_validation_net_bps=best))
    pd.concat(matched).to_csv(folder/'portfolios_matched.csv',index=False);s=pd.DataFrame(summaries);s['p_holm']=np.nan
    for partition in ['validation','test']:
        ix=(s.kind=='relative')&(s.partition==partition)&(s.span=='all');s.loc[ix,'p_holm']=holm(s.loc[ix,'p'])
    s.to_csv(folder/'portfolio_summary.csv',index=False);paired_frame=pd.DataFrame(paired);paired_frame['p_holm']=np.nan
    for partition in ['validation','test']:
        ix=(paired_frame.kind=='relative')&(paired_frame.partition==partition);paired_frame.loc[ix,'p_holm']=holm(paired_frame.loc[ix,'p'])
    paired_frame.to_csv(folder/'portfolio_increment.csv',index=False);(folder/'validation_selection.json').write_text(json.dumps(selection,indent=2))
    if period=='historical':
        a=r.query("kind=='original' and span=='all'").iloc[0];b=pd.DataFrame(paired).query("kind=='original' and partition=='test' and rule=='top20'").iloc[0];corrected=holm([a.p,b.p]);(folder/'primary_correction.json').write_text(json.dumps(dict(loss_p_holm=float(corrected[0]),portfolio_p_holm=float(corrected[1])),indent=2))
    print('Evaluation complete',period,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('period',choices=['modern','historical']);args=ap.parse_args();main(args.period)
