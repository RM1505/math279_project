"""Controlled coarsening of observed quote paths; never calls snapshots event data."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,argparse
from dataclasses import dataclass
import numpy as np,pandas as pd
from study import OUT,CACHE,fit_ridge,infer
R=OUT/'resolution';SCALES=[(1,0),(2,0),(5,0),(10,0),(30,0),(5,2)]

# Signed bid-side operator. Ask contribution equals the same operator with reversed price.
def side_increment(p0,q0,p1,q1):return np.where(p1>=p0,q1,0)-np.where(p1<=p0,q0,0)
def ofi(b0,qb0,a0,qa0,b1,qb1,a1,qa1):return side_increment(b0,qb0,b1,qb1)-side_increment(-a0,qa0,-a1,qa1)
def side_error(p,q):
    d=np.diff(p,axis=-1);return np.sum(np.where(d>0,q[...,:-1],0)-np.where(d<0,q[...,1:],0),axis=-1)-np.where(p[...,-1]>p[...,0],q[...,0],0)+np.where(p[...,-1]<p[...,0],q[...,-1],0)
@dataclass
class FlowSummary:
    bid_flow:float
    ask_flow:float
    depth_sum:float
    increments:int
    def merge(self,other):return FlowSummary(self.bid_flow+other.bid_flow,self.ask_flow+other.ask_flow,self.depth_sum+other.depth_sum,self.increments+other.increments)
    @property
    def net(self):return self.bid_flow-self.ask_flow
    @property
    def normalized(self):return self.net/(self.depth_sum/self.increments) if self.increments else np.nan

def blocks(x,dates,h,offset):
    x=x.astype(float);b=x[:,:,1];a=x[:,:,2];depth=np.exp(x[:,:,7]);qb=depth*(1+x[:,:,8]);qa=depth*(1-x[:,:,8]);mid=(a+b)/2
    inc=ofi(b[:,:-1],qb[:,:-1],a[:,:-1],qa[:,:-1],b[:,1:],qb[:,1:],a[:,1:],qa[:,1:])
    starts=np.arange(30+offset,386-h,h);ends=starts+h
    valid=np.isfinite(np.stack([b,a,qb,qa],axis=-1)).all(-1)&(qb>0)&(qa>0)&(b<a)
    prefix=np.pad(np.cumsum(np.where(np.isfinite(inc),inc,0),axis=1),((0,0),(1,0)))
    invalid=np.pad(np.cumsum(~valid,axis=1),((0,0),(1,0)))
    good=(invalid[:,ends+1]-invalid[:,starts])==0
    normalizer=depth[:,starts]
    fine=(prefix[:,ends]-prefix[:,starts])/normalizer
    coarse=ofi(b[:,starts],qb[:,starts],a[:,starts],qa[:,starts],b[:,ends],qb[:,ends],a[:,ends],qa[:,ends])/normalizer
    frozen=(qb[:,starts]*np.sign(b[:,ends]-b[:,starts])+qa[:,starts]*np.sign(a[:,ends]-a[:,starts]))/normalizer
    ret=np.log(mid[:,ends]/mid[:,starts]);lag=np.log(mid[:,starts]/mid[:,starts-h]) if h<=30 else ret*np.nan
    # Forward target ends at the next block endpoint; aligned within each session.
    nxt=ends+h;futuregood=nxt<=385;clipped=np.minimum(nxt,390)
    forward=np.log(mid[:,clipped]/mid[:,ends]);forwardgood=good&futuregood[None,:]&((invalid[:,clipped+1]-invalid[:,ends])==0)
    baseline=np.stack([ret,lag,x[:,ends,6],x[:,ends,7],x[:,ends,8],x[:,ends,4],x[:,ends,5]],axis=-1)
    forwardgood &= np.isfinite(baseline).all(-1)&np.isfinite(forward)&(abs(forward)<.2)
    # Verify the exact path correction on all valid blocks with unnormalized quantities.
    err=np.empty_like(fine)
    for j,(s,e) in enumerate(zip(starts,ends)):err[:,j]=(side_error(b[:,s:e+1],qb[:,s:e+1])-side_error(-a[:,s:e+1],qa[:,s:e+1]))/normalizer[:,j]
    maximum=float(np.max(abs((fine-coarse-err)[good]))) if good.any() else 0
    if maximum>1e-7:raise AssertionError(('algebra error',maximum))
    dategrid=np.broadcast_to(dates[:,None],good.shape)
    return dict(fine=fine,coarse=coarse,frozen=frozen,ret=ret,forward=forward,baseline=baseline,good=good,forwardgood=forwardgood,date=dategrid,algebra_error=maximum)

def fit_predict(x,y,date,good):
    tr=good&(date<'2018-01-01');va=good&(date>='2018-01-01')&(date<'2019-01-01');te=good&(date>='2019-01-01')
    if min(tr.sum(),va.sum(),te.sum())<100:return None
    scores=[]
    for alpha in [.01,1.]:
        f=fit_ridge(x[tr],y[tr],alpha);scores.append(np.mean((infer(f,x[va])-y[va])**2))
    alpha=[.01,1.][np.argmin(scores)];f=fit_ridge(x[tr|va],y[tr|va],alpha);p=np.full(y.shape,np.nan);p[te]=infer(f,x[te]);mean=float(y[tr|va].mean())
    return p,mean,dict(alpha=alpha,validation_mse=list(map(float,scores)),fit=[np.asarray(z).tolist() for z in f])

def run(limit=None):
    (R/'stock').mkdir(parents=True,exist_ok=True);stocks=np.load(OUT/'market.npz')['universe'];stocks=stocks if limit is None else stocks[:limit]
    for i,tk in enumerate(stocks):
        dest=R/'stock'/f'{tk}.csv.gz'
        if dest.exists():continue
        with np.load(CACHE/f'{tk}.npz') as f:x=f['x'];dates=f['dates'];depth=int(f['depth'])
        rows=[];models={};maxerr=0
        for h,offset in SCALES:
            z=blocks(x,dates,h,offset);maxerr=max(maxerr,z['algebra_error']);good=z['good']&np.isfinite(z['ret'])&(abs(z['ret'])<.2)
            fg=z['forwardgood'];dd=z['date'];fits={}
            # Within-block explanation; uses within-block outcomes by construction, NOT a forecast.
            for name in ['fine','coarse','frozen']:
                f=fit_predict(z[name][...,None],z['ret'],dd,good)
                if f:fits['same_'+name]=f
            for name in ['base','fine','coarse']:
                xx=z['baseline'] if name=='base' else np.concatenate([z['baseline'],z[name][...,None]],axis=-1)
                f=fit_predict(xx,z['forward'],dd,fg)
                if f:fits['next_'+name]=f
            # Last completed block alone is a negative control for explaining this block.
            shifted=np.roll(z['coarse'],1,axis=1);shiftok=good&np.roll(good,1,axis=1);shiftok[:,0]=False
            f=fit_predict(shifted[...,None],z['ret'],dd,shiftok)
            if f:fits['lagged_coarse']=f
            models[f'{h}_{offset}']={k:v[2] for k,v in fits.items()}
            for j,date in enumerate(dates):
                g=good[j];fine=z['fine'][j,g];coarse=z['coarse'][j,g];nonzero=(abs(fine)>1e-8)&(abs(coarse)>1e-8);material=abs(fine)>.1
                if not len(fine):continue
                row=dict(ticker=tk,depth=depth,date=date,h=h,offset=offset,n=len(fine),excluded=int((~g).sum()),opposite=int(np.sum((fine*coarse<0)&nonzero)),both_nonzero=int(nonzero.sum()),fine_nonzero=int(np.sum(abs(fine)>1e-8)),coarse_zero_fine_nonzero=int(np.sum((abs(coarse)<=1e-8)&(abs(fine)>1e-8))),material=int(material.sum()),material_opposite=int(np.sum(material&(fine*coarse<0))),fine_ss=float(np.sum(fine*fine)),error_ss=float(np.sum((fine-coarse)**2)),corr=float(np.corrcoef(fine,coarse)[0,1]) if min(np.std(fine),np.std(coarse))>1e-10 else np.nan)
                if date>='2019-01-01':
                    for name,(pred,mean,_) in fits.items():
                        yy=z['forward'][j] if name.startswith('next_') else z['ret'][j];ok=np.isfinite(pred[j])&np.isfinite(yy);row[name+'_n']=int(ok.sum());row[name+'_mse']=float(np.mean((yy[ok]-pred[j,ok])**2)) if ok.any() else np.nan;row[name+'_benchmark']=float(np.mean((yy[ok]-mean)**2)) if ok.any() else np.nan
                rows.append(row)
        pd.DataFrame(rows).to_csv(dest,index=False);(R/'stock'/f'{tk}.json').write_text(json.dumps(dict(ticker=tk,max_algebra_error=maxerr,models=models)))
        if i%10==0:print('Resolution',i+1,'/',len(stocks),tk,flush=True)
    print('Resolution run complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();run(a.limit)
