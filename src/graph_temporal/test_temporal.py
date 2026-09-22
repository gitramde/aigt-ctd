import unittest
from .prepare import np, build_history
import pandas as pd

class CausalityTests(unittest.TestCase):
    def test_identity_completion_gap_and_partition(self):
        # Earlier flow 0 completes at current cutoff: it must be excluded.
        # Flow 1 shares only source; flow 2 shares neither endpoint.
        meta=pd.DataFrame(dict(second=[30,31,32,150,151,210],completion_us=[150_000_000,40_000_000,41_000_000,160_000_000,160_000_000,211_000_000],
            src=[1,1,9,1,8,1],dst=[2,3,9,2,8,2],partition=[0,0,0,0,0,1],source_row=[1,2,3,4,5,6],target_binary=[0]*6))
        targets,h,t,c=build_history(meta,np.arange(6))
        self.assertEqual(h[3,0],-1)  # missing intervening minute is not compressed
        self.assertEqual(h[3,1],1)  # role-consistent source fallback
        self.assertEqual(t[3,1],2)
        self.assertTrue((h[4]==-1).all()) # unrelated interactions never borrowed
        self.assertTrue((h[5]==-1).all()) # partition reset

    def test_pair_priority_and_completion_tie(self):
        meta=pd.DataFrame(dict(second=[30,31,32,90],completion_us=[40_000_000,40_000_000,50_000_000,100_000_000],
            src=[1,1,1,1],dst=[2,2,3,2],partition=[0]*4,source_row=[1,2,3,4],target_binary=[0]*4))
        _,h,t,_=build_history(meta,np.arange(4))
        self.assertEqual(h[3,0],1);self.assertEqual(t[3,0],1)

    def test_masked_tokens_cannot_change_prediction(self):
        import torch
        from .model import TemporalHead
        torch.manual_seed(42);torch.set_num_threads(2)
        model=TemporalHead(5).eval();x=torch.randn(2,4,5);mask=torch.tensor([[True,True,False,False],[True,True,True,False]])
        with torch.inference_mode():
            a=model(x,mask);x[mask]=12345;b=model(x,mask)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__': unittest.main()
