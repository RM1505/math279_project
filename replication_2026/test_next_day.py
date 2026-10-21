import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from daily_data import daily_features
from next_day_study import Prep,split_masks,make_panel,loss

class NextDayTests(unittest.TestCase):
    def test_missing_classification_outcome(self):
        self.assertTrue(np.isnan(loss(np.array([np.nan]),np.array([.6]),'logistic')[0]))
    def test_portfolio_cash_accounting_and_missing_selection(self):
        from summarize_next_day import portfolio
        g=pd.DataFrame({'y':np.log([1.1,.9]),'long_net':1e4*np.log([109/101,89/101]),'short_net':1e4*np.log([99/111,99/91])})
        p=portfolio(g,np.array([1.,-1.]),'top20')
        expected=((109/101-1)+(1-91/99))/20*1e4
        self.assertAlmostEqual(p['net_bps'],expected)
        self.assertAlmostEqual(p['oneway_turnover'],((1+109/101)+(1+91/99))/20)
        g.loc[0,'y']=np.nan
        self.assertIsNone(portfolio(g,np.array([1.,-1.]),'top20'))
    def x(self,base=100.):
        x=np.ones((391,14));x[:,0]=base+np.arange(391)*.001;x[:,1]=x[:,0]-.01;x[:,2]=x[:,0]+.01
        x[:,3:6]=.0001;x[:,6]=.0002;x[:,7]=5;x[:,8:13]=.1
        return x
    def test_signal_cutoff_excludes_execution_prices(self):
        x=self.x();a=daily_features(x);x[381:,0:13]*=2;b=daily_features(x)
        excluded={'entry_mid','entry_bid','entry_ask'}
        for key in set(a)-excluded:self.assertAlmostEqual(a[key],b[key],msg=key)
        self.assertNotEqual(a['entry_mid'],b['entry_mid'])
    def test_target_maturity_at_splits(self):
        d=pd.DataFrame({'signal_date':['2018-12-28','2018-12-31','2019-12-30','2019-12-31','2020-01-02'],'target_date':['2018-12-31','2019-01-02','2019-12-31','2020-01-02','2020-01-03']})
        tr,va,te=split_masks(d,np.ones(5))
        np.testing.assert_array_equal(tr,[True,False,False,False,False]);np.testing.assert_array_equal(va,[False,False,True,False,False]);np.testing.assert_array_equal(te,[False,False,False,False,True])
    def test_preprocessing_training_only(self):
        x=np.array([[1.,np.nan],[2.,4.],[3.,6.]])
        p=Prep().fit(x);before=p.transform(np.array([[10.,5.]]));p.transform(np.array([[1e10,-1e10]]))
        np.testing.assert_equal(before,p.transform(np.array([[10.,5.]])))
    def test_missing_session_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);d=root/'study';(d/'daily').mkdir(parents=True);(root/'data').mkdir()
            pd.DataFrame({'ticker':['AAA'],'sector':[np.nan]}).to_csv(root/'data/ticker_sectors.csv',index=False)
            dates=pd.bdate_range('2019-01-01',periods=18).strftime('%Y-%m-%d').tolist();(d/'calendar.json').write_text(json.dumps(dates))
            rows=[]
            for i,date in enumerate(dates):
                if i==1:continue
                rows.append(dict(ticker='AAA',date=date,depth=10,**daily_features(self.x(100+i))))
            pd.DataFrame(rows).to_csv(d/'daily/AAA.csv',index=False)
            with patch('next_day_study.D',d),patch('next_day_study.ROOT',root/'research'):
                make_panel();p=pd.read_pickle(d/'panel.pkl').set_index('signal_date')
                self.assertEqual(p.loc[dates[0],'target_date'],dates[1]);self.assertTrue(np.isnan(p.loc[dates[0],'intraday_y']))
                baseline=p.loc[dates[-2],'intraday_past_mean20']
                rows[-2]['entry_mid']*=1.1;rows[-2]['entry_bid']*=1.1;rows[-2]['entry_ask']*=1.1
                pd.DataFrame(rows).to_csv(d/'daily/AAA.csv',index=False);make_panel();q=pd.read_pickle(d/'panel.pkl').set_index('signal_date')
                self.assertAlmostEqual(baseline,q.loc[dates[-2],'intraday_past_mean20'])
if __name__=='__main__':unittest.main()
