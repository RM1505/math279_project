"""Validate all saved aggregate results without fitting or changing estimates."""
import json
import numpy as np
import pandas as pd
from study import OUT

def main():
    record={}
    for name in ['scores','scores_session_control']:
        stocks=0;rows=0;specs=0
        for path in sorted((OUT/name).glob('*.csv.gz')):
            d=pd.read_csv(path)
            assert len(d)>0,path
            keys=['date','session','horizon','delay','sample']
            assert not d.duplicated(keys+['model']).any(),path
            assert np.isfinite(d.select_dtypes('number')).all().all(),path
            assert (d.n>=5).all() and (d.sse>=0).all(),path
            assert d.date.between('2019-01-01','2019-12-31').all(),path
            assert d.groupby(keys).n.nunique().eq(1).all(),path
            assert d.groupby(keys).target_ss.nunique().eq(1).all(),path
            z=d[d.model=='zero']
            np.testing.assert_allclose(z.sse,z.target_ss,rtol=1e-12)
            assert (z.quote_net_bps==0).all() and (z.gross_bps==0).all(),path
            assert (d.quote_net_bps<=d.gross_bps+1e-4).all(),path
            expected=8 if d.depth.iloc[0]==10 else 6
            assert d.groupby(keys).model.nunique().eq(expected).all(),path
            params=json.loads(path.with_name(path.name.replace('.csv.gz','_parameters.json')).read_text())
            fit=pd.read_csv(path.with_name(path.name.replace('.csv.gz','_fits.csv')))
            assert len(fit)==len(params),path
            assert (fit.train>=50000).all() and (fit.validation>=10000).all(),path
            stocks+=1;rows+=len(d);specs+=len(params)
        assert stocks==483,(name,stocks)
        record[name]={'stocks':stocks,'aggregate_rows':rows,'fitted_specifications':specs,'checks_passed':True}
        print(name,record[name],flush=True)
    (OUT/'result_integrity_checks.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
