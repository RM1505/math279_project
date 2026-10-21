import unittest
import numpy as np
from ranking_study import select_weights,account

class RankingTests(unittest.TestCase):
    def test_ranking_scale_and_direction(self):
        p=np.array([[.001,-.003,.002,0.]])
        w=select_weights(p,np.ones_like(p),2,'magnitude')
        np.testing.assert_equal(w,[[0,-.5,.5,0]])
        np.testing.assert_equal(w,select_weights(p*10,np.ones_like(p),2,'magnitude'))
    def test_cost_gate_uses_cash_slots(self):
        p=np.array([[.0001,.0004,-.0002]])
        w=select_weights(p,np.array([[2.,3.,4.]]),2,'edge1')
        np.testing.assert_equal(w,[[0,.5,0]])
    def test_ranking_cannot_use_future(self):
        p=np.array([[.001,.002]])
        w=select_weights(p,np.ones_like(p),1,'magnitude')
        r,bad=account(w,np.array([[.01,np.nan]]),np.ones_like(p),np.ones_like(p),np.zeros_like(p),np.zeros_like(p))
        self.assertIsNone(r);self.assertEqual(bad,1)
    def test_exact_share_accounting_hold_and_reverse(self):
        # Verify against an explicit cash-and-share ledger at observed quotes.
        mids=np.array([100.,101.,100.5,102.]);sp=np.array([.02,.03,.02,.04])
        w=np.array([[.5],[-.5],[-.5]])
        y=np.log(mids[1:]/mids[:-1])[:,None]
        relative=sp/mids*1e4
        ln=np.log((mids[1:]-sp[1:]/2)/(mids[:-1]+sp[:-1]/2))[:,None]*1e4
        sn=np.log((mids[:-1]-sp[:-1]/2)/(mids[1:]+sp[1:]/2))[:,None]*1e4
        r,bad=account(w,y,relative[:-1,None],relative[1:,None],ln,sn)
        cash=0.;shares=0.
        for t in range(3):
            desired=w[t,0]/mids[t];delta=desired-shares
            cash-=delta*(mids[t]+np.sign(delta)*sp[t]/2);shares=desired
        cash+=shares*(mids[-1]-np.sign(shares)*sp[-1]/2)
        self.assertEqual(bad,0);self.assertAlmostEqual(r['net'].sum(),cash*1e4,places=9)
    def test_unchanged_flat_position_pays_spread_once(self):
        w=np.ones((3,1));y=np.zeros_like(w);s=np.ones_like(w)*2
        r,bad=account(w,y,s,s,-s,-s)
        self.assertAlmostEqual(r['net'].sum(),-2)
        self.assertAlmostEqual(r['roundtrip'].sum(),-6)
    def test_cash_missing_quotes_is_zero(self):
        w=np.zeros((3,2));v=np.full_like(w,np.nan)
        r,bad=account(w,v,v,v,v,v)
        self.assertEqual(bad,0);self.assertEqual(r['net'].sum(),0)
if __name__=='__main__':unittest.main()
