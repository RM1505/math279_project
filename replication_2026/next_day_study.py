"""Chronological next-day price-return regressions/classifiers with paired baselines."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,pickle,warnings,argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier,RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from daily_data import D
from study import ROOT
from summarize_study import bootstrap_ratio,holm
BASE=['ret_day','ret_5','ret_30','ret_60','rv','price_range','log_price','spread','spread_mean','log_depth','depth_change']
BOOK=['obi_day','obi_morning','obi_afternoon','obi_late','obi_last','obi_std','flow_day','flow_morning','flow_afternoon','flow_late','flow_last','flow_std','obi_positive_fraction','obi_afternoon_minus_morning','obi_persistence','weighted_mid_displacement','pressure_return_interaction','pressure_without_move']
TARGETS=['intraday','overnight','combined']
FAMILIES=['ridge','additive','boost','forest','neural','logistic','class_boost']

def make_panel():
    cal=json.loads((D/'calendar.json').read_text());frames=[]
    sector=pd.read_csv(ROOT.parent/'data/ticker_sectors.csv').drop_duplicates('ticker').set_index('ticker').sector.fillna('Unknown')
    for f in sorted((D/'daily').glob('*.csv')):
        if f.stem.endswith('_audit'):continue
        q=pd.read_csv(f).set_index('date').reindex(cal);q['ticker']=f.stem;q['sector']=sector.get(f.stem,'Unknown')
        q['signal_date']=q.index;q['target_date']=pd.Series(cal,index=cal).shift(-1)
        with np.errstate(divide='ignore',invalid='ignore'):
            cc=np.log(q.signal_mid/q.signal_mid.shift(1));cc=cc.where(abs(cc)<.2)
            q['last_overnight']=np.log(q.open_mid/q.entry_mid.shift(1));q['last_overnight']=q.last_overnight.where(abs(q.last_overnight)<.2)
            for n in [1,5,20,60]:q['return_history_'+str(n)]=cc.rolling(n,min_periods=max(1,int(.8*n))).sum()
            q['return_vol20']=cc.rolling(20,min_periods=15).std()
            for target in TARGETS:
                if target=='intraday':eb=q.open_bid.shift(-1);ea=q.open_ask.shift(-1);xb=q.entry_bid.shift(-1);xa=q.entry_ask.shift(-1)
                elif target=='overnight':eb=q.entry_bid;ea=q.entry_ask;xb=q.open_bid.shift(-1);xa=q.open_ask.shift(-1)
                else:eb=q.entry_bid;ea=q.entry_ask;xb=q.entry_bid.shift(-1);xa=q.entry_ask.shift(-1)
                y=np.log((xb+xa)/(eb+ea));long=np.log(xb/ea)*1e4;short=np.log(eb/xa)*1e4
                valid=np.isfinite(eb)&np.isfinite(ea)&np.isfinite(xb)&np.isfinite(xa)&(ea>eb)&(xa>xb)&(eb>0)&(xb>0)
                gap=np.log(q.open_mid.shift(-1)/q.entry_mid)
                q[target+'_large_gap']=np.isfinite(gap)&(abs(gap)>=.2)
                if target!='intraday':valid &= abs(gap)<.2
                q[target+'_y']=y.where(valid);q[target+'_long']=long.where(valid);q[target+'_short']=short.where(valid)
                lag=1 if target=='overnight' else 2
                for n in [20,60]:q[target+'_past_mean'+str(n)]=q[target+'_y'].shift(lag).rolling(n,min_periods=max(5,n//2)).mean()
                q[target+'_past_vol20']=q[target+'_y'].shift(lag).rolling(20,min_periods=10).std()
        frames.append(q.reset_index(drop=True))
    df=pd.concat(frames,ignore_index=True)
    book=list(BOOK)
    base=BASE+['last_overnight','return_vol20']+['return_history_'+str(n) for n in [1,5,20,60]]
    # Same-time context uses current signals, never target returns.
    for col in ['ret_day','return_history_5','rv','obi_afternoon','flow_day']:
        for name,group in [('market',['signal_date']),('sector',['signal_date','sector'])]:
            total=df.groupby(group)[col].transform('sum');count=df.groupby(group)[col].transform('count')-df[col].notna()
            val=(total-df[col].fillna(0))/count.where(count>=3)
            new=name+'_'+col;df[new]=val
            (base if col in ['ret_day','return_history_5','rv'] else book).append(new)
    for col in ['ret_day','rv','spread','obi_day','flow_day','obi_persistence']:
        df[col+'_rank']=df.groupby('signal_date')[col].rank(pct=True)
        (base if col in ['ret_day','rv','spread'] else book).append(col+'_rank')
    for sec in sorted(df.sector.unique()):
        name='sector_is_'+sec;df[name]=(df.sector==sec).astype(float);base.append(name)
    valid_signal=np.isfinite(df.ret_day)&np.isfinite(df.signal_mid)&np.isfinite(df.spread)
    df=df[valid_signal].sort_values(['signal_date','ticker']).reset_index(drop=True)
    df.to_pickle(D/'panel.pkl');(D/'feature_columns.json').write_text(json.dumps({'base':base,'book':book},indent=2))
    quality=[]
    for year,g in df.groupby(df.signal_date.str[:4]):
        for target in TARGETS:quality.append({'year':year,'target':target,'signals':len(g),'valid_targets':int(g[target+'_y'].notna().sum()),'large_overnight_gap_flags':int(g[target+'_large_gap'].sum()),'stocks':g.ticker.nunique()})
    pd.DataFrame(quality).to_csv(D/'target_coverage.csv',index=False)
    print('Panel',len(df),len(base),len(book),flush=True)

class Prep:
    def fit(self,x):
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);self.median=np.nanmedian(x,axis=0)
        self.median=np.nan_to_num(self.median);x=np.where(np.isfinite(x),x,self.median)
        self.lo,self.hi=np.quantile(x,[.001,.999],axis=0);x=np.clip(x,self.lo,self.hi)
        self.mean=x.mean(0);self.std=x.std(0);self.keep=self.std>1e-10;self.std=np.where(self.std>1e-10,self.std,1)
        return self
    def transform(self,x):
        x=np.where(np.isfinite(x),x,self.median);return ((np.clip(x,self.lo,self.hi)-self.mean)/self.std)[:,self.keep]

def classifier(family):return family in ['logistic','class_boost']
def settings(family):return [0,1] if family in ['ridge','additive','boost','logistic','class_boost'] else [0]
def estimator(family,setting,n):
    if family in ['ridge','additive']:return Ridge(alpha=n*[.01,.1][setting])
    if family=='logistic':return LogisticRegression(C=[.01,1][setting],max_iter=200,solver='lbfgs')
    if family in ['boost','class_boost']:
        cls=HistGradientBoostingClassifier if classifier(family) else HistGradientBoostingRegressor
        return cls(max_iter=80,max_leaf_nodes=[7,15][setting],min_samples_leaf=150,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=279)
    if family=='forest':return RandomForestRegressor(n_estimators=32,max_depth=5,min_samples_leaf=200,max_features=.5,max_samples=.5,n_jobs=1,random_state=279)
    if family=='neural':return MLPRegressor(hidden_layer_sizes=(24,8),max_iter=30,batch_size=1024,learning_rate_init=.001,alpha=.01,shuffle=False,early_stopping=False,n_iter_no_change=5,random_state=279)
    raise ValueError(family)

def train_model(x,y,family,setting):
    prep=Prep().fit(x);z=prep.transform(x);spline=None
    if family=='additive':spline=SplineTransformer(n_knots=3,degree=2,knots='uniform',include_bias=False);z=spline.fit_transform(z)
    bound=np.quantile(y,[.001,.999]);yy=(y>0).astype(int) if classifier(family) else np.clip(y,*bound)*100
    model=estimator(family,setting,len(y))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(z,yy)
    return (prep,spline,model),[str(w.message) for w in caught],bound

def predict(fit,x,family):
    prep,spline,model=fit;z=prep.transform(x)
    if spline is not None:z=spline.transform(z)
    return model.predict_proba(z)[:,1] if classifier(family) else model.predict(z)/100

def loss(y,p,family):
    if classifier(family):
        yy=y>0;p=np.clip(p,1e-8,1-1e-8);return np.where(np.isfinite(y),-(yy*np.log(p)+(1-yy)*np.log(1-p)),np.nan)
    return (y-p)**2

def split_masks(df,y):
    valid=np.isfinite(y)
    tr=valid&(df.signal_date<'2019-01-01')&(df.target_date<'2019-01-01')
    va=valid&(df.signal_date>='2019-01-01')&(df.signal_date<'2020-01-01')&(df.target_date<'2020-01-01')
    te=df.signal_date>='2020-01-01'
    return tr,va,te

def fit_all():
    df=pd.read_pickle(D/'panel.pkl');fc=json.loads((D/'feature_columns.json').read_text());(D/'predictions').mkdir(exist_ok=True);(D/'models').mkdir(exist_ok=True)
    for target in TARGETS:
        y=df[target+'_y'].to_numpy();valid=np.isfinite(y)
        tr,va,te=split_masks(df,y);tv=tr|va
        trailing=[target+'_past_mean20',target+'_past_mean60',target+'_past_vol20']
        metadata=df.loc[te,['ticker','signal_date','target_date','depth',target+'_y',target+'_long',target+'_short']].rename(columns={target+'_y':'y',target+'_long':'long_net',target+'_short':'short_net'}).copy()
        for family in FAMILIES:
            for group in ['base','book']:
                name=f'{target}_{family}_{group}';dest=D/'predictions'/f'{name}.csv.gz'
                if dest.exists():continue
                cols=fc['base']+trailing+(fc['book'] if group=='book' else [])
                x=df[cols].to_numpy(dtype=float);scores=[];warnings_seen=[]
                for setting in settings(family):
                    fit,warn,bound=train_model(x[tr],y[tr],family,setting);p=predict(fit,x[va],family);scores.append(float(np.mean(loss(y[va],p,family))));warnings_seen+=warn
                chosen=settings(family)[int(np.argmin(scores))]
                fit,warn,bound=train_model(x[tv],y[tv],family,chosen);warnings_seen+=warn
                p=predict(fit,x[te],family);result=metadata.copy();result['prediction']=p;result.to_csv(dest,index=False)
                with open(D/'models'/f'{name}.pkl','wb') as f:pickle.dump({'fit':fit,'columns':cols,'family':family},f)
                (D/'models'/f'{name}.json').write_text(json.dumps({'target':target,'family':family,'group':group,'columns':cols,'setting':chosen,'validation_losses':scores,'training_rows':int(tr.sum()),'validation_rows':int(va.sum()),'refit_rows':int(tv.sum()),'predicted_rows':int(te.sum()),'training_target_clip':bound.tolist(),'warnings':warnings_seen},indent=2))
                print('Fit',name,'setting',chosen,'validation',min(scores),'warnings',len(warnings_seen),flush=True)
        controls=metadata.copy();controls['zero']=0.;controls['training_mean']=y[tv].mean();controls['trailing_mean']=df.loc[te,target+'_past_mean60'].fillna(y[tv].mean()).to_numpy();controls['training_probability']=float((y[tv]>0).mean());controls.to_csv(D/'predictions'/f'{target}_controls.csv.gz',index=False)
    print('All next-day fits complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['panel','fit']);a=p.parse_args();make_panel() if a.stage=='panel' else fit_all()
