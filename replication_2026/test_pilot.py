import unittest
import numpy as np
import pandas as pd
from pilot import flows, predict_day

class Correctness(unittest.TestCase):
    def book(self,prices):
        d={}
        for k in range(1,11):
            d[f"bid_{k}"]=np.array(prices)-k
            d[f"ask_{k}"]=np.array(prices)+k
            d[f"bid_size_{k}"]=np.full(len(prices),100.)
            d[f"ask_size_{k}"]=np.full(len(prices),100.)
        return pd.DataFrame(d)
    def test_price_direction(self):
        for prices,expected in [([100,101,102],2),([102,101,100],-2),([100,100,100],0)]:
            f,_=flows(self.book(prices)); self.assertTrue(np.isnan(f[0]).all())
            np.testing.assert_allclose(f[1:],expected)
    def test_previous_queue_used(self):
        d=self.book([100,101]); d.loc[0,"ask_size_1"]=300
        f,_=flows(d); self.assertEqual(f[1,0],4.)
    def test_future_perturbation(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(90,2,10)); m=100*np.exp(rng.normal(0,.001,(90,2)).cumsum(0))
        p=predict_day(x,m,horizon=5,delay=1)
        xx=x.copy(); mm=m.copy(); xx[60:]*=100; mm[60:]*=2
        q=predict_day(xx,mm,horizon=5,delay=1)
        cols=["mean","returns","own","cross","returns_own","combined"]
        np.testing.assert_allclose(p.loc[p.minute<60,cols],q.loc[q.minute<60,cols])
        self.assertTrue((p.last_train_target_end<=p.minute).all())
        self.assertTrue((p.target_start==p.minute+1).all())
        self.assertTrue((p.target_end==p.minute+6).all())
    def test_sampling_loses_flow(self):
        # Identical endpoints can conceal nonzero event OFI when queues change.
        d=self.book([100,101,100]); d.loc[1,"bid_size_1"]=300
        event,_=flows(d); endpoint,_=flows(d.iloc[[0,2]])
        self.assertNotEqual(event[1:,0].sum(),endpoint[1,0])

if __name__=="__main__": unittest.main()
