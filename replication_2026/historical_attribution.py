"""Descriptive exposure attribution of the frozen earlier-period original rule."""
import numpy as np,pandas as pd
from decision_models import O,data
from summarize_next_day import weights
from decision_audit import mean_test

def main():
    df,_=data('historical');folder=O/'historical';common=pd.read_csv(folder/'portfolios_matched.csv');dates=set(common.query("kind=='original' and partition=='test' and rule=='top20'").date);rows=[]
    for group in ['base','book']:
        p=pd.read_csv(folder/'models'/f'combined_original_logistic_{group}.csv.gz');p=p[p.period=='test'];q=df.iloc[p.row].copy();q['prediction']=p.prediction.to_numpy()
        for date,g in q.groupby('signal_date'):
            if date not in dates:continue
            w=weights(g.prediction.to_numpy()-.5,'top20');chosen=w!=0;raw=np.expm1(g.combined_y.to_numpy());net=np.where(w>0,w*np.expm1(g.combined_long.to_numpy()/1e4),np.where(w<0,w*np.expm1(-g.combined_short.to_numpy()/1e4),0)).sum()*1e4
            market=np.nanmean(raw)*1e4;sec=pd.Series(raw).groupby(pd.Series(g.sector.to_numpy())).transform('mean').to_numpy()*1e4
            rows.append(dict(group=group,date=date,net_bps=net,market_adjusted_bps=net-w.sum()*market,sector_adjusted_bps=net-np.where(chosen,w*sec,0).sum(),net_exposure=w.sum()))
    d=pd.DataFrame(rows);d.to_csv(folder/'original_exposure_daily.csv',index=False);out=[]
    for measure in ['net_bps','market_adjusted_bps','sector_adjusted_bps']:
        for group,g in d.groupby('group'):out.append(dict(group=group,measure=measure,**mean_test(g[measure])))
        pivot=d.pivot(index='date',columns='group',values=measure);out.append(dict(group='book_minus_base',measure=measure,**mean_test(pivot.book-pivot.base)))
    pd.DataFrame(out).to_csv(folder/'original_exposure_summary.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__':main()
