import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from decision_models import masks,data
class DecisionTests(unittest.TestCase):
    def test_historical_maturity_boundaries(self):
        d=pd.DataFrame({'signal_date':['2013-12-30','2013-12-31','2014-12-30','2014-12-31','2015-01-02','2016-12-30'],'target_date':['2013-12-31','2014-01-02','2014-12-31','2015-01-02','2015-01-05','2017-01-03']})
        tr,va,te=masks(d,np.ones(6),'historical');np.testing.assert_array_equal(tr,[1,0,0,0,0,0]);np.testing.assert_array_equal(va,[0,0,1,0,0,0]);np.testing.assert_array_equal(te,[0,0,0,0,1,0])
    def test_missing_risk_session_is_not_skipped_and_no_future_feature(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);d=pd.DataFrame({'ticker':['A']*3,'signal_date':['2020-01-02','2020-01-06','2020-01-07'],'target_date':['2020-01-03','2020-01-07','2020-01-08'],'intraday_y':[.01,.02,.03],'combined_y':[.02,.03,.04],'rv':[.01,.02,.03],'price_range':[.02,.03,.04],'spread_mean':[.001,.002,.003]});d.to_pickle(root/'panel.pkl');(root/'feature_columns.json').write_text(json.dumps({'base':[],'book':[]}))
            with patch('next_day_study.D',root):
                a,_=data('modern');self.assertTrue(np.isnan(a.loc[0,'vol_risk']));self.assertAlmostEqual(a.loc[1,'vol_risk'],np.log(.03))
                d.loc[2,'rv']=9;d.to_pickle(root/'panel.pkl');b,_=data('modern')
                for col in ['vol_current','vol_mean5','vol_mean20']:np.testing.assert_allclose(a.loc[:1,col],b.loc[:1,col],equal_nan=True)
if __name__=='__main__':unittest.main()
