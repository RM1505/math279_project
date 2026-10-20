import unittest
import numpy as np
import pandas as pd
from study import features,targets,make_design,fit_ridge,infer,MCOLS

class StudyTests(unittest.TestCase):
    def book(self,levels=10):
        n=391; d={'time':pd.to_timedelta(np.arange(34200,57601,60),unit='s').astype(str).str[-8:]}
        for k in range(1,levels+1):
            d[f'ask_{k}']=100000+np.arange(n)*100+k*100
            d[f'bid_{k}']=100000+np.arange(n)*100-k*100
            d[f'ask_size_{k}']=np.full(n,100)
            d[f' bid_size_{k}']=np.full(n,100)
        return pd.DataFrame(d)
    def test_direction_and_observed_imbalance(self):
        x,*_=features(self.book());np.testing.assert_allclose(x[1:,10],np.tanh(.2),rtol=1e-6)
        np.testing.assert_allclose(x[:,8],0)
    def test_future_changes_cannot_alter_features(self):
        d=self.book(); x,*_=features(d)
        d.loc[200:,'ask_size_1']*=10; y,*_=features(d)
        np.testing.assert_allclose(x[:200],y[:200])
    def test_missing_and_one_level(self):
        d=self.book(1);d.loc[50,'bid_1']=0
        x,bad,*_=features(d);self.assertEqual(bad,1);self.assertTrue(np.isnan(x[50]).all());self.assertTrue(np.isnan(x[:,11]).all())
    def test_horizon_day_boundaries(self):
        x=np.zeros((2,391,14));x[0,:,0]=np.exp(np.arange(391)/1000);x[1,:,0]=10*np.exp(np.arange(391)/1000)
        y=targets(x,5,1);np.testing.assert_allclose(y[:,:385],.005);self.assertTrue(np.isnan(y[:,385:]).all())
    def test_leave_own_out(self):
        x=np.zeros((1,391,14));x[:,:,MCOLS]=2
        m={'dates':np.array(['2017-01-03']),'sums':np.full((1,391,5),22.),'counts':np.full((1,391,5),21)}
        z=make_design(x,np.array(['2017-01-03']),m)
        np.testing.assert_allclose(z[:,:,5:8],1);np.testing.assert_allclose(z[:,:,13:15],1)
    def test_fit_uses_only_supplied_training(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(100,3));y=x[:,0]+rng.normal(size=100)
        f=fit_ridge(x[:70],y[:70],.1);p=infer(f,x[70:])
        y[70:]=1e10
        np.testing.assert_allclose(p,infer(fit_ridge(x[:70],y[:70],.1),x[70:]))

if __name__=='__main__':unittest.main()
