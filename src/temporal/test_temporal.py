import unittest
import numpy as np
import torch
from .data import cohort_indices,sequence_batches
from .train import TemporalTransformer,config_for,seed

class SequenceTests(unittest.TestCase):
    def test_same_day_endpoints_and_final_labels(self):
        spans=[(0,100),(100,201)];ids=cohort_indices(spans,8)
        x=np.arange(201,dtype=np.float32)[:,None]
        for length in (1,16,32,64):
            for positions,b in sequence_batches(x,ids,length,batch_size=3):
                np.testing.assert_array_equal(b[:,-1,0],ids[positions])
                for target,seq in zip(ids[positions],b):
                    start=0 if target<100 else 100
                    self.assertGreaterEqual(seq[0,0],start)
                    self.assertLessEqual(seq[-1,0],target)
    def test_control_capacity_and_no_context(self):
        arch=dict(d_model=64,layers=2,heads=4,dropout=.1)
        seed();a=TemporalTransformer(config_for('a',32,arch))
        seed();b=TemporalTransformer(config_for('b',32,arch,control=True))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        b.eval();x=torch.randn(3,1,80)
        with torch.no_grad(): self.assertEqual(tuple(b(x).shape),(3,))
        self.assertEqual(b.position_offset,31)
        self.assertFalse(torch.equal(a.encoder.layers[0].linear1.weight,a.encoder.layers[1].linear1.weight))

if __name__=='__main__': unittest.main()
