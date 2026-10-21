"""Exploratory stress tests of the already-observed combined-horizon lead."""
import json
import numpy as np
import pandas as pd
from daily_data import D
from summarize_next_day import weights
from summarize_study import bootstrap_ratio,holm
from ranking_study import block_ci
O=D.parent/'decision';O.mkdir(exist_ok=True)
def mean_test(a):
    a=np.asarray(a);a=a[np.isfinite(a)];point,lo,hi,p,_=bootstrap_ratio(a,np.ones(len(a)));return dict(n=len(a),mean=point,low=lo,high=hi,p=p)
def main():
    daily=pd.read_csv(D/'portfolio_daily.csv');rows=[]
    for keys,g in daily.groupby(['target','family','group','rule']):rows.append(dict(zip(['target','family','group','rule'],keys))|mean_test(g.net_bps))
    r=pd.DataFrame(rows);r['p_holm_84']=holm(r.p);r.to_csv(O/'original_portfolio_multiplicity.csv',index=False)
    panel=pd.read_pickle(D/'panel.pkl').set_index(['ticker','signal_date'])
    f=pd.read_csv(D/'predictions/combined_logistic_book.csv.gz');b=pd.read_csv(D/'predictions/combined_logistic_base.csv.gz');common=set(daily.query("target=='combined' and family=='logistic' and group=='book' and rule=='top20'").date)
    records=[];trades=[]
    for date,g in f.groupby('signal_date'):
        if date not in common:continue
        idx=g.index;w=weights(g.prediction.to_numpy()-.5,'top20');ww=weights(b.loc[idx,'prediction'].to_numpy()-.5,'top20');chosen=w!=0
        raw=np.expm1(g.y.to_numpy());ln=np.expm1(g.long_net.to_numpy()/1e4);sn=-np.expm1(-g.short_net.to_numpy()/1e4)
        pnl=np.where(w>0,w*ln,np.where(w<0,-w*sn,0))*1e4
        q=panel.reindex(pd.MultiIndex.from_arrays([g.ticker,g.signal_date]));sectors=q.sector.to_numpy();rv=q.rv.to_numpy();sp=q.spread.to_numpy();target=g.y.to_numpy()
        market=np.nanmean(raw)*1e4;sector_mean=pd.Series(raw).groupby(pd.Series(sectors)).transform('mean').to_numpy()*1e4
        # These are realized benchmark attribution, not execution-feasible hedges.
        market_adjusted=np.sum(pnl)-w.sum()*market;sector_adjusted=np.sum(pnl)-np.sum(np.where(chosen,w*sector_mean,0))
        records.append(dict(date=date,net_bps=pnl.sum(),net_exposure=w.sum(),market_bps=market,market_adjusted_bps=market_adjusted,sector_adjusted_bps=sector_adjusted,selected_spread_bps=np.mean(sp[chosen])*1e4,selected_rv=np.mean(rv[chosen]),net_after_5bp_side=pnl.sum()-5*np.sum(np.where(w>0,abs(w)*(2+ln),np.where(w<0,abs(w)*(2-sn),0)))))
        for j in np.flatnonzero(chosen):trades.append(dict(date=date,ticker=g.ticker.iloc[j],weight=w[j],net_contribution_bps=pnl[j],log_return=target[j],sector=sectors[j]))
    d=pd.DataFrame(records);t=pd.DataFrame(trades);d.to_csv(O/'candidate_daily_audit.csv',index=False);t.to_csv(O/'candidate_contributions.csv',index=False)
    timing={'market_component_bps':float((d.net_exposure*d.market_bps).mean()),'constant_average_exposure_bps':float(d.net_exposure.mean()*d.market_bps.mean()),'dynamic_covariance_component_bps':float((d.net_exposure*d.market_bps).mean()-d.net_exposure.mean()*d.market_bps.mean()),'interpretation':'Ex-post decomposition; sample-average exposure is not a validation-fixed strategy.'};(O/'candidate_exposure_decomposition.json').write_text(json.dumps(timing,indent=2))
    checks=[]
    for col in ['net_bps','market_adjusted_bps','sector_adjusted_bps','net_after_5bp_side']:checks.append(dict(check=col,**mean_test(d[col])))
    for n in [1,5,10]:
        chosen=d.nlargest(n,'net_bps').index;checks.append(dict(check=f'omit_best_{n}_days',**mean_test(d.drop(chosen).net_bps)))
        stocks=t.groupby('ticker').net_contribution_bps.sum().nlargest(n).index
        # Ex-post attribution: remove these contributions, leaving cash; not a new tradable rule.
        z=t[~t.ticker.isin(stocks)].groupby('date').net_contribution_bps.sum().reindex(d.date,fill_value=0)
        checks.append(dict(check=f'remove_best_{n}_stocks_contributions',**mean_test(z)))
    for threshold in [.05,.1]:
        bad=set(t.loc[abs(t.log_return)>=threshold,'date']);checks.append(dict(check=f'omit_selected_absolute_log_return_ge_{threshold}',**mean_test(d.loc[~d.date.isin(bad),'net_bps'])))
    for year,g in d.groupby(d.date.str[:4]):checks.append(dict(check='year_'+year,**mean_test(g.net_bps)))
    pd.DataFrame(checks).to_csv(O/'candidate_stress_tests.csv',index=False)
    t.groupby('ticker').net_contribution_bps.agg(['sum','count']).sort_values('sum',ascending=False).to_csv(O/'candidate_stock_concentration.csv')
    print(pd.DataFrame(checks).to_string(index=False));print(r.query("target=='combined' and family=='logistic' and group=='book' and rule=='top20'").to_string(index=False))
if __name__=='__main__':main()
