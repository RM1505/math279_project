"""Replay saved fits independently of model fitting and compare daily losses."""
import json
import numpy as np,pandas as pd
from resolution_study import R,CACHE,blocks,infer

def main():
    rows=[]
    for tk in ['AAL','AAPL','AMZN','BAC','GS']:
        cfg=json.loads((R/'stock'/f'{tk}.json').read_text());saved=pd.read_csv(R/'stock'/f'{tk}.csv.gz')
        with np.load(CACHE/f'{tk}.npz') as f:x=f['x'];dates=f['dates']
        for key,models in cfg['models'].items():
            h,offset=map(int,key.split('_'));z=blocks(x,dates,h,offset);good=z['good']&np.isfinite(z['ret'])&(abs(z['ret'])<.2)
            for name,m in models.items():
                if name.startswith('next_'):
                    feature=name[5:];xx=z['baseline'] if feature=='base' else np.concatenate([z['baseline'],z[feature][...,None]],axis=-1);ok=z['forwardgood'];y=z['forward']
                elif name=='lagged_coarse':
                    xx=np.roll(z['coarse'],1,axis=1)[...,None];ok=good&np.roll(good,1,axis=1);ok[:,0]=False;y=z['ret']
                else:xx=z[name[5:]][...,None];ok=good;y=z['ret']
                fit=tuple(np.asarray(v) for v in m['fit']);errors=[]
                for j in np.flatnonzero(dates>='2019-01-01')[::7]:
                    if not ok[j].any():continue
                    pred=infer(fit,xx[j,ok[j]]);mse=np.mean((y[j,ok[j]]-pred)**2);v=saved[(saved.date==dates[j])&(saved.h==h)&(saved.offset==offset)][name+'_mse'].iloc[0];errors.append(abs(mse-v))
                maximum=max(errors);assert maximum<1e-14,(tk,key,name,maximum)
                rows.append(dict(ticker=tk,h=h,offset=offset,model=name,days=len(errors),max_mse_error=maximum))
    pd.DataFrame(rows).to_csv(R/'fit_replay.csv',index=False);print('Replayed',len(rows),'fits;',sum(r['days'] for r in rows),'stock-day model losses')
if __name__=='__main__':main()
