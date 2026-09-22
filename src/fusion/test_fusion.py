import unittest
from .run import np, percentile, benign_threshold, macro_threshold, binary_metrics

class FusionTests(unittest.TestCase):
    def test_ecdf_ties_and_tails(self):
        np.testing.assert_array_equal(percentile(np.array([1.,2.,2.,4.]),np.array([0.,1.,2.,3.,4.,5.])),
                                      [0.,.25,.75,.75,1.,1.])

    def test_conservative_cap_with_ties(self):
        values=np.array([0.,0.,1.,1.,1.,2.,2.,2.,3.,3.])
        for cap in (.05,.1,.2,.5):
            self.assertLessEqual(np.count_nonzero(values>=benign_threshold(values,cap)),int(cap*len(values)))
        self.assertEqual(np.count_nonzero(values>=benign_threshold(values,.1)),0)

    def test_macro_threshold_matches_exhaustive(self):
        y=np.array([0,1,0,1,1,0]); scores=np.array([.1,.1,.3,.4,.7,.7])
        candidates=np.r_[np.nextafter(scores.max(),np.inf),np.unique(scores)[::-1]]
        best=max(candidates,key=lambda t:binary_metrics(y,(scores>=t).astype(float),.5,False)['macro_f1'])
        self.assertEqual(macro_threshold(y,scores),best)

    def test_or_union_can_exceed_component_cap(self):
        rf=np.array([True,False,False,False]); ae=np.array([False,True,False,False])
        self.assertEqual((rf|ae).mean(),.5)
        self.assertEqual(rf.mean(),.25)

if __name__=='__main__':
    unittest.main()
