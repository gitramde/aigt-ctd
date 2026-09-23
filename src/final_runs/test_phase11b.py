"""Small synthetic no-fit tests for Phase11B safeguards and independent reporting."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from . import phase11b as p

class Phase11BTests(unittest.TestCase):
    def test_scope_and_dependency_order(self):
        self.assertEqual(p.ORDER,[('graph','edge_mlp'),('graph','gatv2'),
            ('graph_temporal','gat_transformer_L1'),('graph_temporal','gat_transformer_L8'),('graph','gatv2_self_only')])

    def test_wrong_dependency_seed_rejected(self):
        with patch.object(p,'read',return_value={'seed':123,'status':'complete'}):
            with self.assertRaisesRegex(RuntimeError,'Encoder seed'):p.dependency(42)

    def test_environment_mismatch_stops(self):
        with patch.object(p,'SPEC',{'runtime':{'software':{'python':'0.0.0'}}}):
            with self.assertRaisesRegex(RuntimeError,'dependency mismatch'):p.environment()

    def test_counts_independent(self):
        from .report11b import independent
        r=independent(np.array([0,0,1,1]),np.array([.1,.8,.4,.9]),.5)
        self.assertEqual([r[k] for k in ('TN','FP','FN','TP')],[1,1,1,1])
        self.assertEqual(r['macro_f1'],.5)

    def test_incomplete_aggregation_rejected(self):
        from .report11b import aggregate
        with self.assertRaisesRegex(RuntimeError,'Missing seeds'):
            aggregate(pd.DataFrame({'seed':[42],'model':['x'],'f1':[.5]}),['model'],['f1'])

    def test_thresholds_against_brute_force(self):
        from .metrics import thresholds,counts
        y=np.r_[np.zeros(200,dtype=int),np.ones(25,dtype=int)]
        score=np.random.default_rng(19).integers(0,15,len(y))/14
        candidates=np.r_[np.nextafter(score.max(),np.inf),np.unique(score)[::-1]]
        stats=[counts(y,score,t) for t in candidates]
        for r in thresholds(y,score,'graph'):
            if r['criterion']=='fixed_0_5':self.assertEqual(r['threshold'],.5)
            elif r['criterion']=='maximum_macro_f1':
                best=max(range(len(stats)),key=lambda i:(stats[i]['macro_f1'],candidates[i]))
                self.assertEqual(r['threshold'],candidates[best])
            else:
                best=max((i for i,s in enumerate(stats) if s['false_positive_rate']<=.01),
                    key=lambda i:(stats[i]['TP'],-stats[i]['FP'],candidates[i]))
                self.assertEqual(r['threshold'],candidates[best])
