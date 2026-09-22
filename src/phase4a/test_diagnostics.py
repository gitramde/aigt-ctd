import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import ks_2samp
from .run import select_thresholds,threshold_candidates,counts_at
from .features import ks_sorted,psi

class DiagnosticsTests(unittest.TestCase):
    def test_threshold_selection_matches_exhaustive(self):
        y=np.array([0,1,0,1,0,1,0,0]);p=np.array([.1,.1,.2,.4,.4,.7,.8,.9])
        candidates=np.r_[np.nextafter(p.max(),np.inf),np.unique(p)[::-1]]
        expected=[counts_at(y,p,t) for t in candidates]
        selected=select_thresholds(y,p)
        for name,key in [('maximum_binary_f1','f1'),('maximum_macro_f1','macro_f1')]:
            row=next(r for r in selected if r['criterion']==name)
            self.assertEqual(row['threshold'],candidates[np.argmax([r[key] for r in expected])])
        for row in selected:
            cap=row['validation_fpr_cap']
            if cap is not None:
                self.assertLessEqual(row['validation_false_positive_rate'],cap)
                best=max(r['recall'] for r in expected if r['false_positive_rate']<=cap)
                self.assertEqual(row['validation_recall'],best)
    def test_ks_ties_and_constants(self):
        for a,b in [([0,0,1,2],[0,1,1,3]),([0,0],[1,1]),([0,0],[0,0])]:
            self.assertAlmostEqual(ks_sorted(np.sort(a),np.sort(b)),ks_2samp(a,b).statistic)
    def test_psi_constant_shift_and_missing(self):
        a=np.zeros(10)
        self.assertAlmostEqual(psi(a,a,10,10),0)
        self.assertGreater(psi(a,np.ones(10),10,10),1)
        self.assertGreater(psi(a,a[:5],10,10),0)

if __name__=='__main__': unittest.main()
