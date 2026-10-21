"""An algebraic quote-path null, not an equilibrium order-arrival model."""
import numpy as np,pandas as pd
from resolution_study import R

def main():
    rng=np.random.default_rng(279);rows=[]
    for n in [1,2,5,10,30,1001]:
        ys=[];fs=[]
        for _ in range(10):
            r=rng.normal(size=(10000,n));ys.append(r.sum(1));fs.append(np.sign(r).sum(1))
        y=np.concatenate(ys);fine=np.concatenate(fs);coarse=np.sign(y);nonzero=fine!=0
        rows.append(dict(minutes=n,draws=len(y),fine_r2=float(np.corrcoef(y,fine)[0,1]**2),coarse_r2=float(np.corrcoef(y,coarse)[0,1]**2),opposite_nonzero=float(np.mean((fine[nonzero]*coarse[nonzero])<0)),fine_zero_rate=float(np.mean(~nonzero)),r2_theory=2/np.pi,asymptotic_disagreement=np.arccos(np.sqrt(2/np.pi))/np.pi))
    pd.DataFrame(rows).to_csv(R/'aggregation_null.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
