"""Bounded snapshot proxy pilot; deliberately NOT an event-level OFI replication."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse, hashlib, json, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import py7zr

ROOT = Path(__file__).resolve().parent
MODELS = ["zero", "mean", "returns", "own", "cross", "returns_own", "combined"]

def flows(book):
    """Canonical prior-queue convention; rows must be consecutive within one day.
    With snapshots this is an endpoint proxy, not the sum of intervening events.
    """
    d = book.rename(columns=lambda c: c.strip())
    b = d[[f"bid_{k}" for k in range(1,11)]].to_numpy(float)
    a = d[[f"ask_{k}" for k in range(1,11)]].to_numpy(float)
    qb = d[[f"bid_size_{k}" for k in range(1,11)]].to_numpy(float)
    qa = d[[f"ask_size_{k}" for k in range(1,11)]].to_numpy(float)
    if not (np.isfinite(np.stack([a,b,qa,qb])).all() and (a>b).all()
            and (b>0).all() and (qa>0).all() and (qb>0).all()):
        raise ValueError("Invalid/missing/crossed level: reject day rather than silently impute")
    bid = np.where(b[1:]>b[:-1],qb[1:],np.where(b[1:]<b[:-1],-qb[:-1],qb[1:]-qb[:-1]))
    ask = np.where(a[1:]>a[:-1],qa[:-1],np.where(a[1:]<a[:-1],-qa[1:],qa[:-1]-qa[1:]))
    # Endpoint mean depth across ten levels, explicitly a proxy normalization.
    depth = ((qb[1:]+qa[1:])/2).mean(axis=1)
    f = np.vstack([np.full(10,np.nan),(bid+ask)/depth[:,None]])
    return f, (b[:,0]+a[:,0])/2

def integrate(train, current):
    # Fit each stock separately, using only training rows. Never overwrite history.
    center = train.mean(axis=0)
    _,_,vt = np.linalg.svd(train-center,full_matrices=False)
    w = vt[0]
    if w.sum()<0: w=-w
    w = w / np.abs(w).sum()
    return train@w, current@w

def predict_day(levels, mids, horizon=1, delay=0, window=30, alpha=30.):
    """levels shape (minutes,stocks,10). Only fully matured labels enter training."""
    n,s,_=levels.shape
    returns=np.vstack([np.full(s,np.nan),np.diff(np.log(mids),axis=0)])
    y=np.full_like(mids,np.nan)
    y[:n-horizon-delay]=np.log(mids[horizon+delay:]/mids[delay:n-horizon])
    out=[]
    for t in range(window+horizon+delay,n-horizon-delay):
        end=t-horizon-delay  # last label ends at decision timestamp t
        ix=np.arange(end-window+1,end+1)
        if not (np.isfinite(levels[ix]).all() and np.isfinite(levels[t]).all()
                and np.isfinite(returns[ix]).all() and np.isfinite(y[ix]).all()):
            continue
        train=np.empty((window,s)); current=np.empty(s)
        for j in range(s): train[:,j],current[j]=integrate(levels[ix,j],levels[t,j])
        for j in range(s):
            target=y[ix,j]; mean=target.mean()
            features={"returns":(returns[ix],returns[t]),
                      "own":(train[:,j:j+1],current[j:j+1]),
                      "cross":(train,current),
                      "returns_own":(np.column_stack([returns[ix],train[:,j]]),np.r_[returns[t],current[j]]),
                      "combined":(np.column_stack([returns[ix],train]),np.r_[returns[t],current])}
            preds={"zero":0.,"mean":mean}
            for name,(x,z) in features.items():
                mu=x.mean(axis=0); sd=x.std(axis=0); sd=np.where(sd>1e-12,sd,1.)
                x=(x-mu)/sd; z=(z-mu)/sd
                beta=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@(target-mean))
                preds[name]=mean+z@beta
            out.append(dict(minute=t,ticker_index=j,target=y[t,j],last_train_target_end=end+delay+horizon,
                            target_start=t+delay,target_end=t+delay+horizon,**preds))
    return pd.DataFrame(out)

def summarize(pred):
    rows=[]
    for (h,delay,session),g in pred.groupby(["horizon","delay","session"]):
        for model in MODELS:
            err=(g.target-g[model])**2
            base=(g.target-g["mean"])**2
            rows.append(dict(horizon=h,delay=delay,session=session,model=model,n=len(g),
                             mse=err.mean(),r2_vs_mean=1-err.sum()/base.sum()))
    return pd.DataFrame(rows)

def paired_blocks(pred):
    # Resample contiguous five-day blocks of daily cross-section-aggregated losses.
    # Exploratory intervals only: twenty days cannot establish publication evidence.
    rng=np.random.default_rng(279)
    rows=[]
    for (h,delay,session),g in pred.groupby(["horizon","delay","session"]):
        for alternative,baseline in [("combined","returns"),("combined","returns_own"),("cross","own")]:
            d=g.assign(gain=(g.target-g[baseline])**2-(g.target-g[alternative])**2)
            daily=d.groupby("date").gain.mean().to_numpy(); n=len(daily)
            if n<5: continue
            draws=[]
            for _ in range(1000):
                starts=rng.integers(0,n,size=int(np.ceil(n/5)))
                ids=np.concatenate([np.arange(k,k+5)%n for k in starts])[:n]
                draws.append(daily[ids].mean())
            low,high=np.quantile(draws,[.025,.975])
            rows.append(dict(horizon=h,delay=delay,session=session,alternative=alternative,baseline=baseline,
                             days=n,mse_gain=daily.mean(),ci_low=low,ci_high=high))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--days",type=int,default=20); ap.add_argument("--start",default="2017-01-03")
    ap.add_argument("--tickers",nargs="+",default=["AAPL","MSFT","JPM","JNJ","XOM"])
    ap.add_argument("--out",type=Path,default=ROOT/"outputs"); args=ap.parse_args()
    if not 1<=args.days<=40 or len(args.tickers)>10: raise ValueError("Pilot capped at 40 days and 10 stocks")
    args.out.mkdir(exist_ok=True); started=time.time()
    inventory=[]
    for p in sorted(args.raw.glob("*.7z")):
        tail=p.stem.split("_")
        inventory.append(dict(archive=p.name,bytes=p.stat().st_size,depth_label=tail[-2],interval_label=tail[-1]))
    pd.DataFrame(inventory).to_csv(args.out/"archive_inventory.csv",index=False)
    selected={}; dates=None
    for tk in args.tickers:
        matches=list(args.raw.glob(f"*__{tk}_*_10_60.7z"))
        if len(matches)!=1: raise ValueError(f"Expected one ten-level archive for {tk}, got {len(matches)}")
        with py7zr.SevenZipFile(matches[0]) as z:
            infos={i.filename:i.uncompressed for i in z.list() if i.filename.endswith(".csv")}
        bydate={n.split("_")[1]:n for n in infos if n.split("_")[1]>=args.start}
        dates=set(bydate) if dates is None else dates & set(bydate)
        selected[tk]=(matches[0],bydate,infos)
    dates=sorted(dates)[:args.days]
    if len(dates)!=args.days: raise ValueError("Not enough common dates")
    books={}; audits=[]; members=[]
    for tk,(arc,bydate,infos) in selected.items():
        names=[bydate[d] for d in dates]
        if sum(infos[n] for n in names)>100_000_000: raise ValueError("Extraction exceeds 100 MB per name")
        print(f"Extracting {tk}: {len(names)} days",flush=True)
        with tempfile.TemporaryDirectory(prefix="ofi_pilot_") as td:
            with py7zr.SevenZipFile(arc) as z: z.extract(path=td,targets=names)
            for date,name in zip(dates,names):
                p=Path(td)/name; data=p.read_bytes()
                members.append(dict(ticker=tk,date=date,archive=arc.name,member=name,sha256=hashlib.sha256(data).hexdigest()))
                df=pd.read_csv(p); seconds=pd.to_timedelta(df.time).dt.total_seconds().to_numpy()
                delta=np.diff(seconds)
                audits.append(dict(ticker=tk,date=date,rows=len(df),first=df.time.iloc[0],last=df.time.iloc[-1],
                                   min_step=float(delta.min()),max_step=float(delta.max()),
                                   sixty_second_share=float((delta==60).mean())))
                # Deliberately exclude early closes / missing or irregular samples.
                if len(df)!=391 or seconds[0]!=34200 or seconds[-1]!=57600 or not (delta==60).all():
                    raise ValueError(f"Irregular session {tk} {date}; inspect before proceeding")
                books[tk,date]=flows(df)
    pd.DataFrame(audits).to_csv(args.out/"sample_audit.csv",index=False)
    pd.DataFrame(members).to_csv(args.out/"input_manifest.csv",index=False)
    predictions=[]
    for date in dates:
        levels=np.stack([books[tk,date][0] for tk in args.tickers],axis=1)
        mids=np.stack([books[tk,date][1] for tk in args.tickers],axis=1)
        for h,delay in [(1,0),(5,0),(1,1)]:
            p=predict_day(levels,mids,h,delay)
            p["ticker"]=[args.tickers[j] for j in p.ticker_index]; p["date"]=date
            p["horizon"]=h; p["delay"]=delay
            # Core target entirely before 15:30; close target strictly before 16:00.
            p["session"]=np.where((p.minute>=30)&(p.target_end<=360),"core",
                                  np.where((p.minute>=360)&(p.target_end<390),"close","exclude"))
            predictions.append(p[p.session!="exclude"])
        print(f"Scored {date}",flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(args.out/"predictions.csv.gz",index=False)
    summarize(pred).to_csv(args.out/"metrics.csv",index=False)
    paired_blocks(pred).to_csv(args.out/"paired_loss_intervals.csv",index=False)
    metadata=dict(status="SNAPSHOT_PROXY_NOT_REPLICATION",tickers=args.tickers,dates=dates,
                  train_window=30,ridge_alpha=30,seed=279,untouched_holdout_claim=False,
                  model_note="One-minute lag only; PCA and scaling fitted separately at every origin; fixed ridge penalty.",
                  input_note="60-second endpoint snapshots; missing within-minute events are irrecoverable.",
                  runtime_seconds=time.time()-started,numpy=np.__version__,pandas=pd.__version__,py7zr=py7zr.__version__)
    (args.out/"run.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
if __name__=="__main__": main()
