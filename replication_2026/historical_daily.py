"""Bounded sequential extraction for a separate earlier chronological experiment."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,tempfile,hashlib,warnings
from pathlib import Path
import pandas as pd
import py7zr
from daily_data import daily_features
from study import OUT,RAW,features,ETF_SYMBOLS
H=OUT/'decision'/'historical'
SHORT={'2011-11-25','2012-07-03','2012-11-23','2012-12-24','2013-07-03','2013-11-29','2013-12-24','2014-07-03','2014-11-28','2014-12-24','2015-11-27','2015-12-24','2016-11-25'}
def main():
    (H/'daily').mkdir(parents=True,exist_ok=True);(H/'excluded_universe').mkdir(exist_ok=True)
    m=pd.read_csv(OUT/'members.csv.gz');a=pd.read_csv(OUT/'archives.csv').set_index('ticker');m=m[m.date.between('2011-01-01','2016-12-31')]
    cal=m.groupby('date').ticker.nunique();cal=cal[cal>=100].index.tolist();cal=[d for d in cal if d not in {'2012-10-29','2012-10-30'}];(H/'calendar.json').write_text(json.dumps(cal))
    counts=m[m.date.str.startswith('2011')].groupby('ticker').size();stocks=sorted(set(counts[counts>=200].index)-ETF_SYMBOLS)
    for i,tk in enumerate(stocks):
        dest=H/'daily'/f'{tk}.csv';excluded=H/'excluded_universe'/f'{tk}.csv'
        if dest.exists() or excluded.exists():continue
        rows=[];audit=[];q=m[m.ticker==tk]
        if q.uncompressed.sum()>600_000_000:raise ValueError('Per-stock temporary extraction cap exceeded')
        with tempfile.TemporaryDirectory(prefix='ofi_history_') as tmp:
            with py7zr.SevenZipFile(RAW/a.loc[tk,'archive']) as z:z.extract(path=tmp,targets=q.member.tolist())
            for r in q.itertuples():
                rec=dict(ticker=tk,date=r.date,status='ok')
                if r.date in SHORT:rec['status']='short_session';audit.append(rec);continue
                f=Path(tmp)/r.member;rec['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
                try:
                    x,bad,allbad,depth=features(pd.read_csv(f));rec.update(invalid_l1=bad,invalid_levels=allbad)
                    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);v=daily_features(x)
                    if v:rows.append(dict(ticker=tk,date=r.date,depth=depth,**v))
                    else:rec['status']='insufficient_signal'
                except ValueError as e:rec['status']=str(e)
                audit.append(rec)
        d=pd.DataFrame(rows);eligible=len(d[d.date.str.startswith('2011')])>=200 if len(d) else False
        d.to_csv(dest if eligible else excluded,index=False);pd.DataFrame(audit).to_csv((dest if eligible else excluded).with_name(tk+'_audit.csv'),index=False)
        print(i+1,'/',len(stocks),tk,len(d),'eligible',eligible,flush=True)
    print('Historical extraction complete',flush=True)
if __name__=='__main__':main()
