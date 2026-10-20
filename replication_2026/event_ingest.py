"""Stream standard LOBSTER message/orderbook pairs into right-closed minute bins.
Canonical previous-queue OFI convention. This is ingestion, not a full replication.
Run per stock/day. Input prices retain vendor integer scaling; log returns cancel it.
"""
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

def event_contribution(previous,current):
    # LOBSTER row: ask price, ask size, bid price, bid size, repeated by level.
    old=np.asarray(previous,float).reshape(-1,4)
    new=np.asarray(current,float).reshape(-1,4)
    a,qa,b,qb=new.T; ao,qao,bo,qbo=old.T
    bid=np.where(b>bo,qb,np.where(b<bo,-qbo,qb-qbo))
    ask=np.where(a>ao,qao,np.where(a<ao,-qa,qao-qa))
    return bid+ask

def aggregate(messages,books,levels=10,chunksize=100000,max_events=5000000):
    """Reject malformed data, halts, sentinel levels, or missing initial quote.
    Must begin with a valid state at/before 09:30; no cross-day carry permitted.
    Auction-cross messages seed/update book state but do not enter OFI sums.
    No zero-padding: bins without events have NaN normalized OFI.
    """
    sums=np.zeros((390,levels)); depth=np.zeros(390); counts=np.zeros(390,int)
    mids=np.full(391,np.nan); previous=None; last_time=-np.inf; seen=0
    with pd.read_csv(messages,header=None,chunksize=chunksize) as mr, \
         pd.read_csv(books,header=None,chunksize=chunksize) as br:
        for mc,bc in itertools.zip_longest(mr,br):
            if mc is None or bc is None or len(mc)!=len(bc): raise ValueError("Message/book row mismatch")
            if bc.shape[1]!=4*levels or mc.shape[1]<6: raise ValueError("Unexpected LOBSTER schema")
            seen+=len(mc)
            if seen>max_events: raise ValueError("Event limit reached; increase only after checking resources")
            for message,book in zip(mc.to_numpy(),bc.to_numpy(float)):
                t=float(message[0]); kind=int(message[1])
                if not np.isfinite(t) or t<last_time: raise ValueError("Nonmonotonic time")
                last_time=t
                if t>57600: continue
                if kind==7: raise ValueError("Halt present: needs explicit halt handling")
                q=book.reshape(levels,4)
                if not (np.isfinite(q).all() and (q>0).all() and (q[:,0]>q[:,2]).all()
                        and (q[:,0]<999999999).all()): raise ValueError("Invalid/sentinel/crossed book")
                mid=(q[0,0]+q[0,2])/2
                # State at each minute boundary: update that and all later unfilled
                # boundaries by final forward-fill below. Boundary event belongs left.
                boundary=int(np.ceil((t-34200)/60))
                if t<=34200:
                    mids[0]=mid; previous=book.copy(); continue
                if previous is None or not np.isfinite(mids[0]):
                    raise ValueError("Need a valid opening state at/before 09:30")
                bucket=boundary-1
                if kind!=6:  # Exclude auction crosses from continuous-book event sums.
                    sums[bucket]+=event_contribution(previous,book)
                    depth[bucket]+=(q[:,1]+q[:,3]).mean()/2
                    counts[bucket]+=1
                mids[boundary]=mid
                previous=book.copy()
    if not np.isfinite(mids[0]): raise ValueError("No opening state")
    mids=pd.Series(mids).ffill().to_numpy()
    denom=np.divide(depth,counts,out=np.full(390,np.nan),where=counts>0)
    normalized=np.divide(sums,denom[:,None])
    result=pd.DataFrame(normalized,columns=[f"ofi_{k}" for k in range(1,levels+1)])
    result.insert(0,"end_seconds",np.arange(34260,57601,60))
    result["events"]=counts; result["mean_depth"]=denom
    result["mid_start"]=mids[:-1]; result["mid_end"]=mids[1:]
    result["log_return"]=np.log(mids[1:]/mids[:-1])
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--messages",type=Path,required=True)
    p.add_argument("--books",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); aggregate(a.messages,a.books).to_csv(a.output,index=False)
