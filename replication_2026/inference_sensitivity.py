"""Post-result inference diagnostics, not new primary tests or model selection."""
import pandas as pd
import numpy as np
from study import OUT
from summarize_study import bootstrap_ratio

def main():
    d=pd.read_csv(OUT/'daily_paired_losses.csv.gz')
    d=d[(d.horizon==1)&(d.delay==0)&(d['sample']=='all')]
    rows=[];omissions=[]
    for name,g in d.groupby('comparison'):
        w=g.pivot(index='date',columns='session',values=['gain','baseline_loss']).dropna()
        for block in [5,10,20,40]:
            sessions={}
            for ss in ['core','close']:
                sessions[ss]=bootstrap_ratio(w['gain'][ss].to_numpy(),w['baseline_loss'][ss].to_numpy(),block=block)
            pt=sessions['close'][0]-sessions['core'][0];draw=sessions['close'][4]-sessions['core'][4]
            lo,hi=np.quantile(draw,[.025,.975])
            rows.append(dict(comparison=name,block=block,session='close_minus_core',effect=pt,low=lo,high=hi))
            pt,lo,hi,_,_=bootstrap_ratio(w['gain'].mean(axis=1).to_numpy(),w['baseline_loss'].mean(axis=1).to_numpy(),block=block)
            rows.append(dict(comparison=name,block=block,session='equal_session',effect=pt,low=lo,high=hi))
        months=w.index.str[:7]
        for month in sorted(set(months)):
            z=w[months!=month]
            a=z['gain'].mean()/z['baseline_loss'].mean()
            omissions.append(dict(comparison=name,omitted_month=month,equal_session=z['gain'].to_numpy().mean()/z['baseline_loss'].to_numpy().mean(),close_minus_core=a['close']-a['core']))
    pd.DataFrame(rows).to_csv(OUT/'block_length_sensitivity.csv',index=False)
    pd.DataFrame(omissions).to_csv(OUT/'leave_month_out_sensitivity.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(omissions).groupby('comparison')[['equal_session','close_minus_core']].agg(['min','max']))
if __name__=='__main__':main()
