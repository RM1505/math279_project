"""Independent cached-data audit of the primary measurement and release provenance."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from resolution_study import OUT,CACHE,R,blocks,fit_predict

def main():
    universe=np.load(OUT/'market.npz')['universe']; rows=[]; errors=[]; models=[]
    starts=np.arange(30,381,5); ends=starts+5
    for ii,tk in enumerate(universe):
        with np.load(CACHE/f'{tk}.npz') as f:x=f['x'].astype(float);dates=f['dates']
        b=x[:,:,1];a=x[:,:,2];d=np.exp(x[:,:,7]);qb=d*(1+x[:,:,8]);qa=d*(1-x[:,:,8])
        valid=np.isfinite(np.stack([b,a,qb,qa],-1)).all(-1)&(b>0)&(b<a)&(qb>0)&(qa>0)
        def transition(p0,q0,p1,q1):
            return np.where(p1>p0,q1,np.where(p1<p0,-q0,q1-q0))
        inc=transition(b[:,:-1],qb[:,:-1],b[:,1:],qb[:,1:])-transition(-a[:,:-1],qa[:,:-1],-a[:,1:],qa[:,1:])
        fine=np.stack([inc[:,s:e].sum(1)/d[:,s] for s,e in zip(starts,ends)],1)
        coarse=(transition(b[:,starts],qb[:,starts],b[:,ends],qb[:,ends])-transition(-a[:,starts],qa[:,starts],-a[:,ends],qa[:,ends]))/d[:,starts]
        ret=np.log((a[:,ends]+b[:,ends])/(a[:,starts]+b[:,starts]))
        quoteok=np.stack([valid[:,s:e+1].all(1) for s,e in zip(starts,ends)],1)
        good=quoteok&np.isfinite(ret)&(abs(ret)<.2)
        saved=pd.read_csv(R/'sensitivity'/f'{tk}.csv.gz').query('h==5 and offset==0 and tolerance==0.1').set_index('date')
        z=blocks(x,dates,5,0)
        assert np.array_equal(z['good'],quoteok),tk
        err=float(np.max(np.abs(z['fine'][good]-fine[good])));errors.append(err)
        assert err<1e-7,(tk,err)
        np.testing.assert_allclose(z['coarse'][good],coarse[good],atol=1e-9,rtol=1e-12)
        for j,date in enumerate(dates):
            keep=good[j];material=keep&(abs(fine[j])>.1)&(abs(coarse[j])>.1)
            both=int(material.sum());opposite=int((material&(fine[j]*coarse[j]<0)).sum());n=int(keep.sum())
            if n:
                v=saved.loc[date];assert (n,both,opposite)==(v.n,v.both,v.opposite),(tk,date)
                rows.append(dict(ticker=tk,date=date,n=n,both=both,opposite=opposite,excluded=int((~keep).sum()),forward_extreme_current=int((z['forwardgood'][j]&(abs(ret[j])>=.2)).sum())))
        if tk in ['AAL','AAPL','AMZN','BAC','GS']:
            cfg=json.loads((R/'stock'/f'{tk}.json').read_text())['models']['5_0']
            for name in ['fine','coarse','frozen']:
                fit=fit_predict(z[name][...,None],z['ret'],z['date'],good)[2]; old=cfg['same_'+name]
                assert fit['alpha']==old['alpha']
                for v,w in zip(fit['fit'],old['fit']):np.testing.assert_allclose(v,w,rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(fit['validation_mse'],old['validation_mse'],rtol=1e-10,atol=1e-16)
                models.append(tk+':same_'+name)
        if ii%100==0:print('Audited primary cached paths',ii+1,'of',len(universe),flush=True)
    t=pd.DataFrame(rows);t['year']=t.date.str[:4];t['rate']=t.opposite/t.both.replace(0,np.nan)
    rates=t.groupby(['year','date']).rate.mean().groupby('year').mean()
    manifest=json.loads((R/'implementation_manifest.json').read_text())
    hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest['hashes'].items()};assert all(hashes.values()),hashes
    result=dict(status='passed',stocks=len(universe),stock_days=len(t),windows=int(t.n.sum()),excluded=int(t.excluded.sum()),max_direct_vs_prefix_error=max(errors),material_rates=rates.to_dict(),year2019_counts=t[t.year=='2019'][['n','both','opposite']].sum().astype(int).to_dict(),forward_extreme_current_blocks=int(t.forward_extreme_current.sum()),refitted_models=len(models),source_hashes_match=hashes,scope='Independent direct five-minute sums and all material counts across cached universe; selected refits. Raw-file provenance is separately sampled, not exhaustively re-extracted.')
    (R/'release_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
