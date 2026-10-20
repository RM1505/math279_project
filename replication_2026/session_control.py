"""Secondary literature-motivated control for open-to-current intraday returns.
Added before inspection of main results. Primary pipeline remains unchanged.
"""
import numpy as np
from pathlib import Path
import study
from study import OUT,CACHE

def build():
    m=np.load(OUT/'market.npz');dates=m['dates'];lookup={d:i for i,d in enumerate(dates)}
    sums=np.zeros((len(dates),391));counts=np.zeros_like(sums,dtype='int32')
    for tk in m['universe']:
        with np.load(CACHE/f'{tk}.npz') as d:
            x=d['x'];ix=[lookup[v] for v in d['dates']]
            with np.errstate(invalid='ignore',divide='ignore'):r=np.log(x[:,:,0]/x[:,0:1,0])
            sums[ix]+=np.nan_to_num(r);counts[ix]+=np.isfinite(r)
    np.savez_compressed(OUT/'open_return_market.npz',sums=sums,counts=counts,dates=dates)

def main():
    if not (OUT/'open_return_market.npz').exists():build()
    # Reuse the reviewed evaluation implementation with a different output location,
    # feature function, and one forecast horizon. No primary result files are touched.
    m=np.load(OUT/'open_return_market.npz');lookup={d:i for i,d in enumerate(m['dates'])}
    old=study.make_design
    def design(x,dates,mkt):
        original=old(x,dates,mkt)
        with np.errstate(invalid='ignore',divide='ignore'):own=np.log(x[:,:,0]/x[:,0:1,0])
        ix=[lookup[v] for v in dates];count=m['counts'][ix]-np.isfinite(own)
        cross=np.divide(m['sums'][ix]-np.nan_to_num(own),count,out=np.full_like(own,np.nan,dtype=float),where=count>=20)
        # Put additional controls first, preserving the broad/deep contiguous masks.
        return np.dstack([own,cross,original])
    source=Path(study.__file__).read_text()
    start=source.index('def evaluate(limit=None):');end=source.index('\ndef main():',start)
    body=source[start:end].replace('def evaluate(limit=None):','def evaluate_control(limit=None):')
    body=body.replace("OUT/'scores'","OUT/'scores_session_control'")
    body=body.replace('contemporary(tk,x,dates,ds,depth); ','')
    body=body.replace('[(1,0),(5,0),(1,1)]','[(1,0)]')
    body=body.replace('needed=17 if depth==10 else 15','needed=19 if depth==10 else 17')
    # Constant mapping transforms feature indices; no changes to estimators/labels.
    namespace=dict(study.__dict__);namespace['make_design']=design
    namespace['GROUPS']={name:[0,1]+[i+2 for i in columns] for name,columns in study.GROUPS.items()}
    exec(compile(body,'session_control_evaluation','exec'),namespace)
    namespace['evaluate_control']()

if __name__=='__main__':main()
