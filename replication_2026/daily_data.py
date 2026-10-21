"""Daily snapshot summaries with explicit signal and execution timestamps."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,hashlib,tempfile,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import py7zr
from study import OUT,CACHE,RAW,features,EXCLUDE
D=OUT/'next_day'; SIGNAL=380; ENTRY=385; OPEN=5
MORE_EXCLUDE={'2020-11-27','2020-12-24','2021-11-26'}

def daily_features(x):
    # ONLY x[:381] may influence signal features. Future execution fields are separate.
    h=x[:SIGNAL+1];row={}
    if np.isfinite(h[:,0]).sum()<350 or not np.isfinite(h[-1,:11]).all():return None
    with np.errstate(invalid='ignore',divide='ignore'):
        row.update(ret_day=np.log(h[-1,0]/h[OPEN,0]),ret_5=h[-1,4],ret_30=h[-1,5],ret_60=np.log(h[-1,0]/h[-61,0]),rv=np.sqrt(np.nansum(h[OPEN+1:,3]**2)),price_range=np.log(np.nanmax(h[:,0])/np.nanmin(h[:,0])),log_price=np.log(h[-1,0]),spread=h[-1,6],spread_mean=np.nanmean(h[:,6]),log_depth=h[-1,7],depth_change=h[-1,7]-h[OPEN,7])
    for name,col in [('obi',8),('flow',10)]:
        for label,start,end in [('day',OPEN,SIGNAL+1),('morning',OPEN,150),('afternoon',150,SIGNAL+1),('late',350,SIGNAL+1)]:row[f'{name}_{label}']=np.nanmean(h[start:end,col])
        row[f'{name}_last']=h[-1,col];row[f'{name}_std']=np.nanstd(h[:,col])
    row['obi_positive_fraction']=np.nanmean(h[OPEN:,8]>0)
    row['obi_afternoon_minus_morning']=row['obi_afternoon']-row['obi_morning']
    row['obi_persistence']=row['obi_day']/(row['obi_std']+.01)
    row['weighted_mid_displacement']=h[-1,6]*h[-1,8]/2
    row['pressure_return_interaction']=row['obi_afternoon']*row['ret_day']
    row['pressure_without_move']=row['obi_afternoon']/(.001+abs(row['ret_day']))
    for name,col in [('deep_obi',11),('deep_flow',12)]:
        row[name+'_day']=np.nanmean(h[OPEN:,col]) if np.isfinite(h[:,col]).any() else np.nan
        row[name+'_late']=np.nanmean(h[350:,col]) if np.isfinite(h[:,col]).any() else np.nan
    row.update(signal_mid=h[-1,0],open_mid=x[OPEN,0],open_bid=x[OPEN,1],open_ask=x[OPEN,2],entry_mid=x[ENTRY,0],entry_bid=x[ENTRY,1],entry_ask=x[ENTRY,2])
    return row

def build():
    D.mkdir(exist_ok=True);(D/'daily').mkdir(exist_ok=True)
    arcs=pd.read_csv(OUT/'archives.csv').set_index('ticker');members=pd.read_csv(OUT/'members.csv.gz');stocks=np.load(OUT/'market.npz')['universe'].tolist()
    allcal=members[members.date.between('2017-01-01','2021-12-31')].groupby('date').ticker.nunique();cal=allcal[allcal>=100].index.tolist();cal=[d for d in cal if d!='2018-12-05'];(D/'calendar.json').write_text(json.dumps(cal))
    selected=members[members.date.between('2020-01-01','2021-12-31')]
    for i,tk in enumerate(stocks):
        dest=D/'daily'/f'{tk}.csv'
        if dest.exists():continue
        rows=[];audit=[]
        with np.load(CACHE/f'{tk}.npz') as data:
            for date,x in zip(data['dates'],data['x']):
                with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);r=daily_features(x)
                if r:rows.append(dict(ticker=tk,date=date,depth=int(arcs.loc[tk,'depth']),**r))
        files=selected[selected.ticker==tk]
        if files.uncompressed.sum()>400_000_000:raise ValueError('Per-stock extraction cap')
        with tempfile.TemporaryDirectory(prefix='ofi_daily_') as tmp:
            names=files.member.tolist()
            if names:
                with py7zr.SevenZipFile(RAW/arcs.loc[tk,'archive']) as z:z.extract(path=tmp,targets=names)
            for m in files.itertuples():
                rec={'ticker':tk,'date':m.date,'status':'ok'}
                if m.date in MORE_EXCLUDE|EXCLUDE:rec['status']='short_or_exceptional_session';audit.append(rec);continue
                f=Path(tmp)/m.member;rec['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
                try:
                    x,bad1,badall,depth=features(pd.read_csv(f));rec.update(invalid_l1=bad1,invalid_levels=badall)
                    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);r=daily_features(x)
                    if r:rows.append(dict(ticker=tk,date=m.date,depth=depth,**r))
                    else:rec['status']='insufficient_signal_history'
                except ValueError as e:rec['status']=str(e)
                audit.append(rec)
        pd.DataFrame(rows).to_csv(dest,index=False);pd.DataFrame(audit).to_csv(D/'daily'/f'{tk}_audit.csv',index=False)
        if i%10==0:print('Daily summaries',i+1,'/',len(stocks),tk,len(rows),flush=True)
    print('Daily extraction complete',flush=True)
if __name__=='__main__':build()
