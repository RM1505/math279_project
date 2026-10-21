import unittest
import numpy as np
from resolution_study import side_increment,side_error,ofi,FlowSummary,blocks
class ResolutionTests(unittest.TestCase):
    def test_path_decomposition_random_books(self):
        r=np.random.default_rng(279);p=100+np.cumsum(r.integers(-2,3,(1000,31)),axis=1);q=r.uniform(1,1000,(1000,31))
        summed=side_increment(p[:,:-1],q[:,:-1],p[:,1:],q[:,1:]).sum(1);endpoint=side_increment(p[:,0],q[:,0],p[:,-1],q[:,-1]);np.testing.assert_allclose(summed-endpoint,side_error(p,q),atol=1e-10)
    def test_flat_price_telescopes(self):
        p=np.ones((3,5))*100;q=np.array([[10,20,5,30,12],[5,4,3,2,1],[1,2,3,4,5]])
        np.testing.assert_allclose(side_error(p,q),0)
    def test_same_endpoints_can_opposite_path_signs(self):
        p=np.array([[100,101,102,100],[100,98,99,100]]);q=np.ones_like(p)*10
        sums=side_increment(p[:,:-1],q[:,:-1],p[:,1:],q[:,1:]).sum(1);np.testing.assert_equal(sums,[10,10])
        p2=np.array([100,99,98,100]);self.assertEqual(side_increment(p2[:-1],q[0,:-1],p2[1:],q[0,1:]).sum(),-10)
        self.assertEqual(side_increment(100,10,100,10),0)
    def test_summary_merge_associativity(self):
        a=FlowSummary(5,3,100,1);b=FlowSummary(2,-1,200,2);c=FlowSummary(-2,4,50,1)
        self.assertEqual(a.merge(b).merge(c),a.merge(b.merge(c)));self.assertEqual(a.merge(b).net,5)
    def test_frozen_queue_null(self):
        rng=np.random.default_rng(279);r=rng.normal(size=500000);x=2*np.sign(r)
        self.assertAlmostEqual(np.corrcoef(r,x)[0,1]**2,2/np.pi,delta=.003)
    def test_future_states_cannot_change_current_pressure(self):
        x=np.ones((1,391,14));x[0,:,0]=100+np.arange(391)*.001;x[0,:,1]=x[0,:,0]-.01;x[0,:,2]=x[0,:,0]+.01;x[0,:,7]=np.log(100);x[0,:,8]=.2
        a=blocks(x,np.array(['2019-01-02']),5,0);x[:,36:,1:3]+=2;b=blocks(x,np.array(['2019-01-02']),5,0)
        for key in ['fine','coarse','frozen']:self.assertEqual(a[key][0,0],b[key][0,0])

class EncoderTests(unittest.TestCase):
    def test_chunked_matches_direct_and_rejects_gap(self):
        from resolution_tools import summarize_quotes
        t=np.arange(7)*60;q=np.array([[100+i%3,10+i,101+i%3,15-i] for i in range(7)],float)
        whole=summarize_quotes(t,q);left=summarize_quotes(t[:4],q[:4]);right=summarize_quotes(t[3:],q[3:])
        self.assertEqual(left.merge(right),whole)
        with self.assertRaises(ValueError):left.merge(summarize_quotes(t[4:],q[4:]))
        with self.assertRaises(ValueError):summarize_quotes(t[[0,2,3]],q[[0,2,3]])

if __name__=='__main__':unittest.main()
