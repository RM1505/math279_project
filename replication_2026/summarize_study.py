"""Streaming aggregation of paired day-stock losses; dependence-aware inference."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/ofi_mpl'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import OUT,CACHE

SCORES=OUT/'scores'
PAIRS=[('book','returns'),('flow','book'),('cross','flow'),('deep','flow')]

def bootstrap_ratio(a,b,block=20,draws=2000):
    # Same draw for numerator and denominator. Calendar days resampled jointly.
    rng=np.random.default_rng(279); n=len(a)
    starts=rng.integers(0,n,(draws,int(np.ceil(n/block))))
    ix=(starts[:,:,None]+np.arange(block))%n;ix=ix.reshape(draws,-1)[:,:n]
    v=a[ix].mean(axis=1)/b[ix].mean(axis=1)
    point=a.mean()/b.mean(); lo,hi=np.quantile(v,[.025,.975])
    # Center bootstrap distribution for a two-sided null test (with finite correction).
    p=(1+np.sum(np.abs(v-point)>=abs(point)))/(draws+1)
    return point,lo,hi,p,v

def holm(p):
    p=np.array(p);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return out

def main():
    daily_parts=[];pair_parts=[];coverage=[];stockparts=[]
    for i,p in enumerate(sorted(SCORES.glob('*.csv.gz'))):
        try:d=pd.read_csv(p)
        except pd.errors.EmptyDataError:continue
        if d.empty:continue
        d['mse']=d.sse/d.n;d['y2']=d.target_ss/d.n
        coverage.append(dict(ticker=d.ticker.iloc[0],depth=d.depth.iloc[0],dates=d.date.nunique(),rows=int(d[(d.model=='returns')&(d.horizon==1)&(d.delay==0)&(d['sample']=='all')].n.sum())))
        for universe in ['broad']+(['deep'] if d.depth.iloc[0]==10 else []):
            z=d.copy();z['universe']=universe
            keys=['universe','date','session','horizon','delay','model','sample']
            z['stock_days']=1
            daily_parts.append(z.groupby(keys)[['mse','y2','gross_bps','quote_net_bps','stock_days','n']].sum())
            # Save stock-level outcomes for heterogeneity and diagnostics.
            ss=z.groupby(['universe','session','horizon','delay','model','sample']).agg(sse=('sse','sum'),target_ss=('target_ss','sum'),n=('n','sum'),gross_bps=('gross_bps','mean'),quote_net_bps=('quote_net_bps','mean')).reset_index()
            ss['ticker']=d.ticker.iloc[0];stockparts.append(ss)
        index=['date','session','horizon','delay','sample']
        wide=d.pivot(index=index,columns='model',values='mse')
        for alt,base in PAIRS:
            if alt not in wide:continue
            z=wide[[alt,base]].dropna().reset_index()
            matched=z.groupby(['date','horizon','delay','sample']).session.transform('nunique')==2
            z=z[matched].copy();z['gain']=z[base]-z[alt];z['baseline_loss']=z[base]
            z['comparison']=alt+' - '+base;z['stocks']=1
            pair_parts.append(z.groupby(index+['comparison'])[['gain','baseline_loss','stocks']].sum())
        if i%100==0: print('Summarized',i+1,flush=True)
    daily=pd.concat(daily_parts).groupby(level=list(range(7))).sum().reset_index()
    for col in ['mse','y2','gross_bps','quote_net_bps']:daily[col]/=daily.stock_days
    daily.to_csv(OUT/'daily_metrics.csv.gz',index=False)
    summary=daily.groupby(['universe','session','horizon','delay','model','sample'])[['mse','y2','gross_bps','quote_net_bps']].mean().reset_index()
    summary['r2_vs_zero']=1-summary.mse/summary.y2
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    stocks=pd.concat(stockparts)
    stocks.to_csv(OUT/'stock_summary.csv.gz',index=False)
    heterogeneity=[]
    for alt,base in PAIRS:
        uni='deep' if alt=='deep' else 'broad'
        select=stocks[(stocks.universe==uni)&(stocks.horizon==1)&(stocks.delay==0)&(stocks['sample']=='all')]
        for session,g in select.groupby('session'):
            w=g.pivot(index='ticker',columns='model',values='sse')
            effect=(w[base]-w[alt])/w[base]
            effect=effect.replace([np.inf,-np.inf],np.nan).dropna()
            heterogeneity.append(dict(comparison=alt+' - '+base,session=session,stocks=len(effect),positive_share=float((effect>0).mean()),median_effect=effect.median(),q10=effect.quantile(.1),q90=effect.quantile(.9)))
    pd.DataFrame(heterogeneity).to_csv(OUT/'stock_heterogeneity.csv',index=False)
    pairs=pd.concat(pair_parts).groupby(level=list(range(6))).sum().reset_index()
    pairs['gain']/=pairs.stocks;pairs['baseline_loss']/=pairs.stocks
    pairs.to_csv(OUT/'daily_paired_losses.csv.gz',index=False)
    rows=[]
    for (h,delay,sample,comparison),g in pairs.groupby(['horizon','delay','sample','comparison']):
        piv=g.pivot(index='date',columns='session',values=['gain','baseline_loss']).dropna()
        session_results={}
        for ss in ['core','close']:
            a=piv['gain'][ss].to_numpy();b=piv['baseline_loss'][ss].to_numpy()
            point,lo,hi,p,v=bootstrap_ratio(a,b)
            session_results[ss]=(point,v)
            rows.append(dict(horizon=h,delay=delay,sample=sample,comparison=comparison,session=ss,days=len(piv),effect=point,low=lo,high=hi,p=p))
        # Overall: equal weight to each session's stock-day average, explicitly report this estimand.
        a=piv['gain'].mean(axis=1).to_numpy();b=piv['baseline_loss'].mean(axis=1).to_numpy()
        point,lo,hi,p,v=bootstrap_ratio(a,b)
        rows.append(dict(horizon=h,delay=delay,sample=sample,comparison=comparison,session='equal_session',days=len(piv),effect=point,low=lo,high=hi,p=p))
        point=session_results['close'][0]-session_results['core'][0]
        v=session_results['close'][1]-session_results['core'][1]
        lo,hi=np.quantile(v,[.025,.975]);p=(1+np.sum(abs(v-point)>=abs(point)))/(len(v)+1)
        rows.append(dict(horizon=h,delay=delay,sample=sample,comparison=comparison,session='close_minus_core',days=len(piv),effect=point,low=lo,high=hi,p=p))
    tests=pd.DataFrame(rows);tests['p_holm_primary']=np.nan
    primary=(tests.horizon==1)&(tests.delay==0)&(tests['sample']=='all')&tests.session.isin(['equal_session','close_minus_core'])
    tests.loc[primary,'p_holm_primary']=holm(tests.loc[primary,'p'])
    tests.to_csv(OUT/'inference.csv',index=False)
    # Summarize QA without reading raw files again.
    audit_parts=[]
    for p in CACHE.glob('*_audit.csv'):
        try:audit_parts.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:continue
    audit=pd.concat(audit_parts,ignore_index=True)
    audit.groupby('status').agg(stock_days=('ticker','size'),invalid_snapshots=('invalid_l1','sum')).to_csv(OUT/'quality_summary.csv')
    audit[audit.status!='ok'].to_csv(OUT/'excluded_days.csv.gz',index=False)
    audit.groupby('ticker').agg(stock_days=('date','size'),invalid_l1=('invalid_l1','sum'),invalid_levels=('invalid_levels','sum')).to_csv(OUT/'quality_by_stock.csv')
    primary_results=tests[(tests.horizon==1)&(tests.delay==0)&(tests['sample']=='all')&tests.session.isin(['core','close'])]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True,sharex=True)
    order=[a+' - '+b for a,b in PAIRS]
    labels=['Queue imbalance beyond returns','Sampled flow beyond queue imbalance','Cross-stock book information','Ten-level depth (matched subset)']
    for ax,ss in zip(axs,['core','close']):
        g=primary_results[primary_results.session==ss].set_index('comparison').loc[order]
        val=g.effect.to_numpy()*100;lo=g.low.to_numpy()*100;hi=g.high.to_numpy()*100
        ax.errorbar(val,np.arange(4),xerr=[np.maximum(0,val-lo),np.maximum(0,hi-val)],fmt='o',color='#236b82',capsize=4)
        ax.axvline(0,color='#888',lw=1);ax.set_yticks(np.arange(4),labels);ax.set_title('Core hours' if ss=='core' else 'Final half-hour');ax.set_xlabel('Reduction in prediction MSE (%)');ax.grid(axis='x',alpha=.2)
    axs[0].invert_yaxis();fig.suptitle('Minute-sampled books: incremental one-minute forecasting value\n2019 evaluation; 95% day-block intervals',fontsize=12);fig.tight_layout();fig.savefig(OUT/'incremental_value.png',dpi=180);plt.close(fig)
    print(tests[primary].to_string(index=False))
    print('Coverage',len(coverage),'stocks; one-minute scored rows',sum(x['rows'] for x in coverage))

if __name__=='__main__':main()
