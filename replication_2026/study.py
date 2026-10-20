"""Minute-snapshot study: raw inputs, fixed chronology, modest linear models."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[k]='1'
import argparse, hashlib, json, re, tempfile, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import py7zr
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'full_study'; CACHE=OUT/'cache'
RAW=Path('/Users/brendanconnelly/Documents/raw')
# Known shortened/exceptional sessions; do not let padded snapshots mimic a late session.
EXCLUDE={'2017-07-03','2017-11-24','2018-07-03','2018-11-23','2018-12-24','2018-12-05','2019-07-03','2019-11-29','2019-12-24'}
COLS=['mid','bid','ask','r1','r5','r30','spread','logdepth','obi','dobi','flow','deep_obi','deep_flow','stale']

def features(df):
    d=df.rename(columns=lambda s:s.strip()); levels=(len(d.columns)-1)//4
    sec=pd.to_timedelta(d.time).dt.total_seconds().to_numpy()
    if len(d)!=391 or not np.array_equal(sec,np.arange(34200,57601,60)):
        raise ValueError('irregular_time_grid')
    a=d[[f'ask_{k}' for k in range(1,levels+1)]].to_numpy(float)
    b=d[[f'bid_{k}' for k in range(1,levels+1)]].to_numpy(float)
    qa=d[[f'ask_size_{k}' for k in range(1,levels+1)]].to_numpy(float)
    qb=d[[f'bid_size_{k}' for k in range(1,levels+1)]].to_numpy(float)
    valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(qa)&np.isfinite(qb)&(a>b)&(b>0)&(a<999999999)&(qa>0)&(qb>0)
    mid=(a[:,0]+b[:,0])/2; mid[~valid[:,0]]=np.nan
    with np.errstate(divide='ignore',invalid='ignore'):
        obi=(qb-qa)/(qb+qa)
        f=np.full_like(b,np.nan)
        bid=np.where(b[1:]>b[:-1],qb[1:],np.where(b[1:]<b[:-1],-qb[:-1],qb[1:]-qb[:-1]))
        ask=np.where(a[1:]>a[:-1],qa[:-1],np.where(a[1:]<a[:-1],-qa[1:],qa[:-1]-qa[1:]))
        f[1:]=(bid+ask)/((qb[1:]+qa[1:])/2)
        f[1:][~(valid[1:]&valid[:-1])]=np.nan
        obi[~valid]=np.nan
        rr=[]
        for h in [1,5,30]:
            r=np.full(391,np.nan); r[h:]=np.log(mid[h:]/mid[:-h]); rr.append(r)
        spread=(a[:,0]-b[:,0])/mid
        depth=np.log((qa[:,0]+qb[:,0])/2)
    dob=np.r_[np.nan,np.diff(obi[:,0])]
    stale=np.zeros(391)
    for t in range(1,391):
        same=(a[t,0]==a[t-1,0])&(b[t,0]==b[t-1,0])&(qa[t,0]==qa[t-1,0])&(qb[t,0]==qb[t-1,0])
        stale[t]=stale[t-1]+1 if same else 0
    deepobi=obi.mean(axis=1) if levels==10 else np.full(391,np.nan)
    deepflow=np.tanh(f/10).mean(axis=1) if levels==10 else np.full(391,np.nan)
    out=np.column_stack([mid,b[:,0],a[:,0],*rr,spread,depth,obi[:,0],dob,np.tanh(f[:,0]/10),deepobi,deepflow,stale])
    out[~valid[:,0],:]=np.nan
    return out.astype('float32'),int((~valid[:,0]).sum()),int((~valid).sum()),levels

def audit():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    records=[]
    for p in ROOT.parent.rglob('*'):
        if p.is_file() and '.git/' not in str(p) and '.venv/' not in str(p) and 'full_study/' not in str(p):
            records.append({'path':str(p.relative_to(ROOT.parent)),'bytes':p.stat().st_size,'suffix':p.suffix})
    pd.DataFrame(records).to_csv(OUT/'project_file_inventory.csv',index=False)
    arcs=[]; members=[]
    for i,p in enumerate(sorted(RAW.glob('*.7z'))):
        m=re.search(r'__([^_]+)_.*_(\d+)_(\d+)\.7z$',p.name)
        if not m: raise ValueError(f'Unrecognized archive {p}')
        ticker,depth,interval=m.groups()
        with py7zr.SevenZipFile(p) as z:
            infos=z.list()
        arcs.append(dict(ticker=ticker,depth=int(depth),interval=int(interval),archive=p.name,bytes=p.stat().st_size,members=len(infos)))
        for info in infos:
            dm=re.search(r'_(\d{4}-\d{2}-\d{2})_',info.filename)
            if dm:
                members.append(dict(ticker=ticker,date=dm.group(1),member=info.filename,uncompressed=info.uncompressed))
        if i%50==0: print('Audited',i+1,'archives',flush=True)
    pd.DataFrame(arcs).to_csv(OUT/'archives.csv',index=False)
    pd.DataFrame(members).to_csv(OUT/'members.csv.gz',index=False)
    print('Audit complete:',len(arcs),'archives',len(members),'members',flush=True)

def extract(limit=None):
    arcs=pd.read_csv(OUT/'archives.csv'); members=pd.read_csv(OUT/'members.csv.gz')
    selected=members[(members.date>='2017-01-01')&(members.date<'2020-01-01')]
    if arcs.ticker.duplicated().any(): raise ValueError('Duplicate ticker archives need explicit resolution')
    for i,arc in arcs.iterrows():
        if limit is not None and i>=limit: break
        target=CACHE/f'{arc.ticker}.npz'
        if target.exists():continue
        files=selected[selected.ticker==arc.ticker].sort_values('date')
        audits=[]; days=[]; arrays=[]
        if files.uncompressed.sum()>400_000_000: raise ValueError('Per-stock extraction limit exceeded')
        with tempfile.TemporaryDirectory(prefix='ofi_study_') as td:
            names=files.member.tolist()
            if names:
                with py7zr.SevenZipFile(RAW/arc.archive) as z: z.extract(path=td,targets=names)
            for row in files.itertuples():
                rec=dict(ticker=arc.ticker,date=row.date,member=row.member,status='ok',invalid_l1=0,invalid_levels=0)
                if row.date in EXCLUDE:
                    rec['status']='short_or_exceptional_session'; audits.append(rec); continue
                p=Path(td)/row.member
                data=p.read_bytes(); rec['sha256']=hashlib.sha256(data).hexdigest()
                try:
                    df=pd.read_csv(p); x,bad1,badall,levels=features(df)
                    rec.update(rows=len(df),invalid_l1=bad1,invalid_levels=badall)
                    if levels!=arc.depth:raise ValueError('depth_mismatch')
                    arrays.append(x); days.append(row.date)
                except ValueError as e:rec['status']=str(e)
                audits.append(rec)
        if arrays:
            np.savez_compressed(target,x=np.stack(arrays),dates=np.array(days),depth=arc.depth)
        else: np.savez_compressed(target,x=np.empty((0,391,len(COLS)),dtype='float32'),dates=np.array([],dtype='U10'),depth=arc.depth)
        pd.DataFrame(audits).to_csv(CACHE/f'{arc.ticker}_audit.csv',index=False)
        if i%10==0 or i==len(arcs)-1:print('Extracted',i+1,'/',len(arcs),arc.ticker,len(days),'days',flush=True)
    print('Extraction complete',flush=True)

# Global means use each timestamp's available valid features; leave own stock out.
MCOLS=[3,4,5,8,10]
ETF_SYMBOLS={'SPY','XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY'}
def market():
    arcs=pd.read_csv(OUT/'archives.csv'); universe=[]
    for row in arcs.itertuples():
        if row.ticker in ETF_SYMBOLS:continue
        with np.load(CACHE/f'{row.ticker}.npz') as d:
            dates=d['dates']; x=d['x']
            # Membership depends ONLY on 2017 coverage. No future survivor screen.
            train=dates<'2018-01-01'
            n=int(np.isfinite(x[train,:,3]).sum())
            if n>=200*350:universe.append(row.ticker)
    dates=sorted(set(np.concatenate([np.load(CACHE/f'{t}.npz')['dates'] for t in universe]).tolist()))
    sums=np.zeros((len(dates),391,len(MCOLS))); counts=np.zeros_like(sums,dtype='int32'); lookup={d:i for i,d in enumerate(dates)}
    for tk in universe:
        with np.load(CACHE/f'{tk}.npz') as d:
            ix=[lookup[s] for s in d['dates']]; x=d['x'][:,:,MCOLS]
            sums[ix]+=np.nan_to_num(x); counts[ix]+=np.isfinite(x)
    np.savez_compressed(OUT/'market.npz',sums=sums,counts=counts,dates=np.array(dates),universe=np.array(universe))
    print('Training-eligible universe:',len(universe),flush=True)

def make_design(x,dates,mkt):
    lookup={d:i for i,d in enumerate(mkt['dates'])}; ix=[lookup[d] for d in dates]
    own=x[:,:,MCOLS].astype(float); counts=mkt['counts'][ix]-np.isfinite(own)
    sums=mkt['sums'][ix]-np.nan_to_num(own)
    cross=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>=20)
    phase=np.broadcast_to(np.arange(391)/390,x.shape[:2])
    design=np.dstack([x[:,:,3:8],cross[:,:,:3],phase,phase**2,x[:,:,8:11],cross[:,:,3:],x[:,:,11:13]])
    # indices 0..4: own r1 r5 r30 spread logdepth; 5..7: market returns;
    # 8..9 time; 10 OBI, 11 dOBI, 12 flow; 13 market OBI,14 market flow;15..16 deep.
    return design

GROUPS={'returns':list(range(10)),'book':list(range(12)),'flow':list(range(13)),
        'cross':list(range(15)),'deep':list(range(13))+[15,16],'deep_cross':list(range(17))}
ALPHAS=[.001,.01,.1,1.]

def prepare_ridge(x,y):
    lo,hi=np.quantile(x,[.001,.999],axis=0); z=np.clip(x,lo,hi)
    mu=z.mean(axis=0); sd=z.std(axis=0); sd=np.where(sd>1e-10,sd,1.)
    z=(z-mu)/sd; ym=y.mean()
    return (lo,hi,mu,sd,ym), z.T@z/len(z), z.T@(y-ym)/len(z)

def solve_prepared(prepared,alpha):
    state,gram,rhs=prepared
    return (*state,np.linalg.solve(gram+alpha*np.eye(len(rhs)),rhs))

def fit_ridge(x,y,alpha):
    return solve_prepared(prepare_ridge(x,y),alpha)

def infer(fit,x):
    lo,hi,mu,sd,ym,beta=fit
    return ym+((np.clip(x,lo,hi)-mu)/sd)@beta

def targets(x,h,delay):
    y=np.full(x.shape[:2],np.nan)
    with np.errstate(invalid='ignore',divide='ignore'):
        y[:,:391-h-delay]=np.log(x[:,h+delay:,0]/x[:,delay:391-h,0])
    return y

def contemporary(tk,x,dates,cross,depth):
    rows=[]; y=x[:,:,3].astype(float)
    t=np.broadcast_to(np.arange(391),y.shape)
    configs={'own_flow':[10],'cross_flow':[10,14]}
    # Combined array uses raw feature flow at 10 and market flow at 14.
    z=np.dstack([x[:,:,10],cross[:,:,14],x[:,:,12]])
    configs={'own_flow':[0],'cross_flow':[0,1]}
    if depth==10:configs['deep_flow']=[0,2]
    valid=np.isfinite(z[:,:,:3 if depth==10 else 2]).all(axis=2)&np.isfinite(y)&(t>=30)&(t<390)&(np.abs(y)<.2)
    tr=valid&(dates[:,None]<'2018-01-01');va=valid&(dates[:,None]>='2018-01-01')&(dates[:,None]<'2019-01-01');te=valid&(dates[:,None]>='2019-01-01')
    if tr.sum()<50000 or va.sum()<10000 or te.sum()==0:return
    for name,cols in configs.items():
        prep=prepare_ridge(z[tr][:,cols],y[tr]);loss=[]
        for a in ALPHAS:loss.append(np.mean((infer(solve_prepared(prep,a),z[va][:,cols])-y[va])**2))
        a=ALPHAS[int(np.argmin(loss))];f=fit_ridge(z[tr|va][:,cols],y[tr|va],a)
        pred=infer(f,z[te][:,cols]);yy=y[te];ss=np.where(t[te]>=360,'close','core')
        for session in ['core','close']:
            keep=ss==session
            rows.append(dict(ticker=tk,depth=depth,session=session,model=name,n=int(keep.sum()),sse=float(((pred[keep]-yy[keep])**2).sum()),target_ss=float((yy[keep]**2).sum()),alpha=a))
    pd.DataFrame(rows).to_csv(OUT/'scores'/f'{tk}_contemp.csv',index=False)

def evaluate(limit=None):
    mkt=np.load(OUT/'market.npz'); universe=mkt['universe'].tolist()
    (OUT/'scores').mkdir(exist_ok=True)
    for ii,tk in enumerate(universe):
        if limit is not None and ii>=limit:break
        dest=OUT/'scores'/f'{tk}.csv.gz'
        if dest.exists():continue
        with np.load(CACHE/f'{tk}.npz') as data:
            x=data['x']; dates=data['dates']; depth=int(data['depth'])
        ds=make_design(x,dates,mkt); contemporary(tk,x,dates,ds,depth); models=dict(GROUPS)
        if depth!=10:models={k:v for k,v in models.items() if not k.startswith('deep')}
        rows=[]; fits=[]; parameters={}
        for h,delay in [(1,0),(5,0),(1,1)]:
            y=targets(x,h,delay); t=np.broadcast_to(np.arange(391),y.shape)
            end=t+h+delay
            # Same feature/target eligibility for all models on each stock.
            needed=17 if depth==10 else 15
            valid=np.isfinite(ds[:,:,:needed]).all(axis=2)&np.isfinite(y)&(t>=30)&(end<390)&(np.abs(y)<.2)
            tr=valid&(dates[:,None]<'2018-01-01'); va=valid&(dates[:,None]>='2018-01-01')&(dates[:,None]<'2019-01-01')
            te=valid&(dates[:,None]>='2019-01-01')
            if tr.sum()<50000 or va.sum()<10000 or te.sum()==0:continue
            a=ds[tr]; b=ds[va]; c=ds[te]; yt=y[tr].astype(float); yv=y[va].astype(float); yy=y[te].astype(float)
            tv=tr|va; allx=ds[tv]; ally=y[tv].astype(float)
            pred={'zero':np.zeros(len(yy)),'mean':np.full(len(yy),ally.mean())}
            for name,cols in models.items():
                losses=[]
                prepared=prepare_ridge(a[:,cols],yt)
                for alpha in ALPHAS:
                    f=solve_prepared(prepared,alpha); losses.append(np.mean((infer(f,b[:,cols])-yv)**2))
                alpha=ALPHAS[int(np.argmin(losses))]
                f=fit_ridge(allx[:,cols],ally,alpha); pred[name]=infer(f,c[:,cols])
                parameters[f'{name}_h{h}_delay{delay}']={'columns':cols,'alpha':alpha,'validation_mse':losses,'clip_low':f[0].tolist(),'clip_high':f[1].tolist(),'center':f[2].tolist(),'scale':f[3].tolist(),'target_mean':float(f[4]),'coefficients':f[5].tolist()}
                fits.append(dict(ticker=tk,horizon=h,delay=delay,model=name,alpha=alpha,train=int(tr.sum()),validation=int(va.sum()),test=int(te.sum())))
            daygrid=np.broadcast_to(dates[:,None],y.shape)[te]; timegrid=t[te]
            session=np.where(timegrid>=360,'close',np.where(timegrid+h+delay<=360,'core','boundary'))
            fresh=x[:,:,13][te]<5
            # Quotes for hypothetical, independent round trips, not a portfolio P&L.
            entrybid=np.roll(x[:,:,1],-delay,axis=1)[te]; entryask=np.roll(x[:,:,2],-delay,axis=1)[te]
            exitbid=np.roll(x[:,:,1],-(h+delay),axis=1)[te]; exitask=np.roll(x[:,:,2],-(h+delay),axis=1)[te]
            groups=pd.DataFrame({'date':daygrid,'session':session,'fresh':fresh})
            indices=groups.groupby(['date','session']).indices
            for name,pp in pred.items():
                error=(yy-pp)**2; sign=np.sign(pp)
                gross=sign*yy*1e4
                with np.errstate(invalid='ignore',divide='ignore'):
                    net=np.where(sign>0,np.log(exitbid/entryask),np.where(sign<0,np.log(entrybid/exitask),0))*1e4
                for (date,ss),ix in indices.items():
                    if ss=='boundary':continue
                    for sample in ['all','fresh']:
                        ids=ix if sample=='all' else ix[fresh[ix]]
                        if len(ids)<5:continue
                        rows.append(dict(ticker=tk,depth=depth,date=date,session=ss,horizon=h,delay=delay,model=name,sample=sample,
                            n=len(ids),sse=error[ids].sum(),target_ss=(yy[ids]**2).sum(),
                            gross_bps=gross[ids].mean(),quote_net_bps=net[ids].mean(),
                            sum_y=yy[ids].sum(),sum_p=pp[ids].sum(),sum_yp=(yy[ids]*pp[ids]).sum(),sum_pp=(pp[ids]**2).sum()))
        pd.DataFrame(rows).to_csv(dest,index=False,compression='gzip')
        pd.DataFrame(fits).to_csv(OUT/'scores'/f'{tk}_fits.csv',index=False)
        (OUT/'scores'/f'{tk}_parameters.json').write_text(json.dumps(parameters))
        if ii%10==0 or ii==len(universe)-1:print('Evaluated',ii+1,'/',len(universe),tk,flush=True)
    print('Evaluation complete',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','extract','market','evaluate']);p.add_argument('--limit',type=int)
    a=p.parse_args();start=time.time()
    if a.stage=='audit':audit()
    elif a.stage=='extract':extract(a.limit)
    elif a.stage=='market':market()
    else:evaluate(a.limit)
    print('Elapsed seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
