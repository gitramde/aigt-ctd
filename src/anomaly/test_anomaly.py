import unittest
import numpy as np
from .metrics import benign_threshold,label_thresholds,metrics

class AnomalyTests(unittest.TestCase):
    def test_quantile_caps_handle_ties_and_small_samples(self):
        for values in (np.array([1.,2.,3.,4.]),np.ones(100),np.r_[np.zeros(950),np.ones(50)]):
            for cap in (.05,.01,.005,.001):
                r=benign_threshold(values,cap)
                self.assertLessEqual(int((values>=r['threshold']).sum()),int(np.floor(cap*len(values))))
    def test_raw_scores_are_not_treated_as_probabilities(self):
        y=np.array([0,0,1,1]);scores=np.array([10.,20.,1000.,2000.])
        r=metrics(y,scores,100.)
        self.assertEqual(r['recall'],1.);self.assertEqual(r['false_positive_rate'],0.)
        self.assertEqual(r['roc_auc'],1.);self.assertEqual(r['average_precision'],1.)
        for point in label_thresholds(y,scores): self.assertEqual(metrics(y,scores,point['threshold'])['f1'],1.)
    def test_label_aware_ties_prefer_higher_threshold(self):
        y=np.array([0,0,1,1]);scores=np.full(4,500.)
        points={r['criterion']:r['threshold'] for r in label_thresholds(y,scores)}
        self.assertGreater(points['maximum_macro_f1'],500.)
        self.assertEqual(points['maximum_binary_f1'],500.)

if __name__=='__main__': unittest.main()
