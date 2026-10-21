"""Check cached queue reconstruction against original archive records."""
import tempfile,json
from pathlib import Path
import numpy as np,pandas as pd,py7zr
from study import OUT,CACHE,RAW
from resolution_study import R,ofi

def main():
    members=pd.read_csv(OUT/'members.csv.gz');arcs=pd.read_csv(OUT/'archives.csv').set_index('ticker');rows=[]
    for tk in ['AAL','AAPL','AMZN','BAC','MSFT','SPG']:
        with np.load(CACHE/f'{tk}.npz') as f:x=f['x'].astype(float);dates=f['dates']
        chosen=[next(d for d in dates if d.startswith(y)) for y in ['2017','2018','2019']]
        m=members[(members.ticker==tk)&members.date.isin(chosen)]
        with tempfile.TemporaryDirectory(prefix='ofi_resolution_verify_') as tmp:
            with py7zr.SevenZipFile(RAW/arcs.loc[tk,'archive']) as z:z.extract(path=tmp,targets=m.member.tolist())
            for r in m.itertuples():
                raw=pd.read_csv(Path(tmp)/r.member).rename(columns=lambda s:s.strip());v=x[np.flatnonzero(dates==r.date)[0]]
                qb=raw.bid_size_1.to_numpy(float);qa=raw.ask_size_1.to_numpy(float);b=raw.bid_1.to_numpy(float);a=raw.ask_1.to_numpy(float)
                dep=np.exp(v[:,7]);rb=dep*(1+v[:,8]);ra=dep*(1-v[:,8]);good=np.isfinite(rb)&np.isfinite(ra)
                starts=np.arange(30,381,5);ends=starts+5
                fine=np.array([ofi(b[s:e],qb[s:e],a[s:e],qa[s:e],b[s+1:e+1],qb[s+1:e+1],a[s+1:e+1],qa[s+1:e+1]).sum() for s,e in zip(starts,ends)])/((qb[starts]+qa[starts])/2)
                rec=np.array([ofi(v[s:e,1],rb[s:e],v[s:e,2],ra[s:e],v[s+1:e+1,1],rb[s+1:e+1],v[s+1:e+1,2],ra[s+1:e+1]).sum() for s,e in zip(starts,ends)])/dep[starts]
                ok=np.isfinite(rec)&np.isfinite(fine)
                rows.append(dict(ticker=tk,date=r.date,integer_sizes=bool(np.all(qb==np.rint(qb)) and np.all(qa==np.rint(qa))),max_size_error=float(np.max(np.maximum(abs(rb[good]-qb[good]),abs(ra[good]-qa[good])))),max_relative_size_error=float(np.max(np.maximum(abs(rb[good]/qb[good]-1),abs(ra[good]/qa[good]-1)))),max_normalized_flow_error=float(np.max(abs(rec[ok]-fine[ok]))),raw_zero_count=int(np.sum((fine==0)&ok)),raw_zero_false_nonzero=int(np.sum((fine==0)&ok&(abs(rec)>1e-8))),price_exact=bool(np.array_equal(v[good,1],b[good]) and np.array_equal(v[good,2],a[good]))))
        print('Verified',tk,flush=True)
    pd.DataFrame(rows).to_csv(R/'raw_verification.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
