"""Replay saved predictions and verify chronological metadata for every daily model."""
import json,pickle,sys
import numpy as np
import pandas as pd
from next_day_study import D,Prep,predict,split_masks,TARGETS,FAMILIES
# Runs produced by next_day_study.py serialize its local Prep class.
sys.modules['__main__'].Prep=Prep

def main():
    df=pd.read_pickle(D/'panel.pkl');out=[]
    for target in TARGETS:
        tr,va,te=split_masks(df,df[target+'_y'].to_numpy());test=df.loc[te].reset_index(drop=True)
        ids=np.unique(np.linspace(0,len(test)-1,97,dtype=int))
        for family in FAMILIES:
            for group in ['base','book']:
                name=f'{target}_{family}_{group}'
                with open(D/'models'/f'{name}.pkl','rb') as f:model=pickle.load(f)
                saved=pd.read_csv(D/'predictions'/f'{name}.csv.gz');meta=json.loads((D/'models'/f'{name}.json').read_text())
                assert len(saved)==len(test)==meta['predicted_rows']
                assert meta['refit_rows']==int((tr|va).sum())
                assert saved[['ticker','signal_date']].equals(test[['ticker','signal_date']])
                np.testing.assert_allclose(saved.y,test[target+'_y'],atol=1e-14,equal_nan=True)
                actual=predict(model['fit'],test.iloc[ids][model['columns']].to_numpy(float),family)
                delta=np.max(abs(actual-saved.prediction.to_numpy()[ids]));assert delta<1e-10,(name,delta)
                assert np.isfinite(saved.prediction).all()
                out.append(dict(model=name,rows=len(saved),replayed=len(ids),max_error=delta,warnings=len(meta['warnings'])))
    pd.DataFrame(out).to_csv(D/'verification.csv',index=False);print('Verified',len(out),'models and',sum(x['replayed'] for x in out),'saved predictions')
if __name__=='__main__':main()
