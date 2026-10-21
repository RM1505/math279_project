"""Prespecified matched genuine-depth sensitivity for next-session returns."""
import json
import numpy as np
import pandas as pd
from next_day_study import D,split_masks,train_model,predict,loss,settings
from summarize_study import bootstrap_ratio,holm

def main():
    df=pd.read_pickle(D/'panel.pkl');df=df[df.depth==10].reset_index(drop=True)
    fc=json.loads((D/'feature_columns.json').read_text());cols=fc['base']+fc['book']+['intraday_past_mean20','intraday_past_mean60','intraday_past_vol20']
    extra=['deep_obi_day','deep_obi_late','deep_flow_day','deep_flow_late'];y=df.intraday_y.to_numpy();tr,va,te=split_masks(df,y);tv=tr|va
    out=[]
    for fam in ['ridge','boost']:
        preds={}
        for group in ['book','depth']:
            x=df[cols+(extra if group=='depth' else [])].to_numpy(float);scores=[]
            for setting in settings(fam):
                fit,w,b=train_model(x[tr],y[tr],fam,setting);scores.append(float(loss(y[va],predict(fit,x[va],fam),fam).mean()))
            chosen=settings(fam)[np.argmin(scores)];fit,w,b=train_model(x[tv],y[tv],fam,chosen);preds[group]=predict(fit,x[te],fam)
        d=df.loc[te,['ticker','signal_date','intraday_y']].copy();d['book']=preds['book'];d['depth']=preds['depth'];d.to_csv(D/f'depth_{fam}_predictions.csv.gz',index=False)
        d['baseline']=(d.intraday_y-d.book)**2;d['gain']=d.baseline-(d.intraday_y-d.depth)**2
        day=d.groupby('signal_date')[['baseline','gain']].mean().dropna()
        for period in ['all','2020','2021']:
            z=day if period=='all' else day[day.index.str.startswith(period)];point,lo,hi,p,_=bootstrap_ratio(z.gain.to_numpy(),z.baseline.to_numpy());out.append(dict(family=fam,period=period,stocks=df.ticker.nunique(),days=len(z),effect=point,low=lo,high=hi,p=p))
        print('Depth complete',fam,flush=True)
    r=pd.DataFrame(out);r['p_holm']=np.nan;mask=r.period=='all';r.loc[mask,'p_holm']=holm(r.loc[mask,'p']);r.to_csv(D/'depth_inference.csv',index=False)
if __name__=='__main__':main()
