import unittest
import numpy as np
from summarize_study import holm,bootstrap_ratio

class Statistics(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(holm([.04,.001,.03]),[.06,.003,.06])
    def test_constant_paired_improvement(self):
        effect,lo,hi,p,_=bootstrap_ratio(np.ones(80)*.02,np.ones(80),draws=200)
        self.assertAlmostEqual(effect,.02);self.assertAlmostEqual(lo,.02);self.assertAlmostEqual(hi,.02)
    def test_identical_forecasts(self):
        effect,lo,hi,p,_=bootstrap_ratio(np.zeros(80),np.ones(80),draws=200)
        self.assertEqual(effect,0);self.assertEqual(p,1)
    def test_known_forward_signal(self):
        from study import fit_ridge,infer
        rng=np.random.default_rng(279);feature=rng.normal(size=(3000,2));target=2*feature[:,0]+rng.normal(0,.1,3000)
        model=fit_ridge(feature[:2000],target[:2000],.001)
        pred=infer(model,feature[2000:]);self.assertLess(np.mean((pred-target[2000:])**2),.1)

if __name__=='__main__':unittest.main()
