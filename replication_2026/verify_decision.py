"""Replay all final decision-stage models on saved test-row coordinates."""
import pickle
import numpy as np
import pandas as pd
from decision_models import O,data
from next_day_study import predict

def main():
    rows=[]
    for period in ['modern','historical']:
        df,fc=data(period)
        for f in sorted((O/period/'models').glob('*.pkl')):
            with open(f,'rb') as stream:m=pickle.load(stream)
            p=pd.read_csv(f.with_suffix('.csv.gz'));assert np.isfinite(p.prediction).all(),f;assert not p[['row','period']].duplicated().any(),f;p=p[p.period=='test'];ids=np.unique(np.linspace(0,len(p)-1,61,dtype=int));sample=p.iloc[ids];actual=predict(m['fit'],df.iloc[sample.row][m['columns']].to_numpy(float),m['family']);error=float(np.max(abs(actual-sample.prediction)));assert error<1e-9,(f,error);rows.append(dict(period=period,model=f.stem,replayed=len(ids),max_error=error))
    assert len(rows)==50, len(rows)
    pd.DataFrame(rows).to_csv(O/'verification.csv',index=False);print('Verified',len(rows),'models;',sum(r['replayed'] for r in rows),'predictions')
if __name__=='__main__':main()
