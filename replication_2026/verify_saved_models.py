"""Independently replay saved parameters, checking daily error and quote-cost totals."""
import json
import numpy as np
import pandas as pd
from study import OUT,CACHE,make_design,targets,infer

def main():
    import sys
    control='--session-control' in sys.argv
    folder=OUT/('scores_session_control' if control else 'scores')
    checks=[];m=np.load(OUT/'market.npz')
    for tk in ['AAL','AAPL','ADI','AMGN','BAC']:
        p=folder/f'{tk}_parameters.json'
        if not p.exists():continue
        params=json.loads(p.read_text());actual=pd.read_csv(folder/f'{tk}.csv.gz')
        with np.load(CACHE/f'{tk}.npz') as d:x=d['x'];dates=d['dates'];depth=int(d['depth'])
        ds=make_design(x,dates,m)
        if control:
            with np.load(OUT/'open_return_market.npz') as om:
                lookup={d:i for i,d in enumerate(om['dates'])};ix=[lookup[d] for d in dates]
                with np.errstate(invalid='ignore',divide='ignore'):own=np.log(x[:,:,0]/x[:,0:1,0])
                count=om['counts'][ix]-np.isfinite(own)
                cross=np.divide(om['sums'][ix]-np.nan_to_num(own),count,out=np.full_like(own,np.nan,dtype=float),where=count>=20)
                ds=np.dstack([own,cross,ds])
        for key,v in params.items():
            model,tail=key.rsplit('_h',1);h,delay=tail.split('_delay');h=int(h);delay=int(delay)
            y=targets(x,h,delay);t=np.broadcast_to(np.arange(391),y.shape)
            good=(dates[:,None]>='2019-01-01')&np.isfinite(ds[:,:,(slice(None, (17 if depth==10 else 15)+(2 if control else 0)))]).all(axis=2)&np.isfinite(y)&(t>=30)&(t+h+delay<390)&(abs(y)<.2)
            fit=tuple(np.array(v[k]) for k in ['clip_low','clip_high','center','scale'])+(v['target_mean'],np.array(v['coefficients']))
            pred=infer(fit,ds[good][:,v['columns']]);yy=y[good];tt=t[good];dd=np.broadcast_to(dates[:,None],y.shape)[good]
            ss=np.where(tt>=360,'close',np.where(tt+h+delay<=360,'core','boundary'))
            # Independent quote-cost expression, sign direction checked explicitly.
            eb=np.roll(x[:,:,1],-delay,axis=1)[good];ea=np.roll(x[:,:,2],-delay,axis=1)[good]
            xb=np.roll(x[:,:,1],-(h+delay),axis=1)[good];xa=np.roll(x[:,:,2],-(h+delay),axis=1)[good]
            net=np.zeros(len(pred),dtype='float32');long=pred>0;short=pred<0
            net[long]=np.log(xb[long]/ea[long])*1e4;net[short]=np.log(eb[short]/xa[short])*1e4
            frame=pd.DataFrame({'date':dd,'session':ss,'error':(yy-pred)**2,'quote_net_bps':net})
            frame=frame[frame.session!='boundary']
            replay=frame.groupby(['date','session']).agg(sse=('error','sum'),n=('error','size'),quote_net_bps=('quote_net_bps','mean'))
            saved=actual[(actual.model==model)&(actual.horizon==h)&(actual.delay==delay)&(actual['sample']=='all')].set_index(['date','session'])
            replay=replay.loc[saved.index]
            np.testing.assert_allclose(replay.sse,saved.sse,rtol=1e-10,atol=1e-14)
            np.testing.assert_array_equal(replay.n,saved.n)
            np.testing.assert_allclose(replay.quote_net_bps,saved.quote_net_bps,rtol=2e-6,atol=2e-6)
            checks.append({'ticker':tk,'model':key,'groups':len(replay),'passed':True})
        print('Replayed',tk,flush=True)
    pd.DataFrame(checks).to_csv(OUT/('session_control_replay_checks.csv' if control else 'parameter_replay_checks.csv'),index=False)
    print('Passed',len(checks),'saved-model replays')
if __name__=='__main__':main()
