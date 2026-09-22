import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .data import GraphData,timestamp_units
from .model import GATv2Layer,EdgeModel


class GraphSafetyTests(unittest.TestCase):
    def test_self_only_control_removes_relational_input(self):
        torch.manual_seed(42)
        model=EdgeModel(dict(kind='self_only',hidden=8,heads=2,dropout=0.),3,2).eval()
        b=dict(nodes=torch.randn(3,8),edges=torch.tensor([[0,1],[1,2]]),edge_features=torch.randn(2,2),
            target_src=torch.tensor([0,1]),target_dst=torch.tensor([2,0]),x=torch.randn(2,3))
        with torch.no_grad():
            first=model(b)
            changed=dict(b,edges=torch.tensor([[2,0],[0,1]]),edge_features=torch.randn(2,2)*100)
            torch.testing.assert_close(first,model(changed),rtol=0,atol=0)

    def test_microsecond_and_nanosecond_timestamps_agree(self):
        values=np.array(['2018-02-20T10:30:00'],dtype='datetime64[us]')
        seconds,microseconds=timestamp_units(values)
        self.assertEqual(int(seconds[0]),int(pd.Timestamp('2018-02-20 10:30:00').timestamp()))
        np.testing.assert_array_equal(seconds,timestamp_units(values.astype('datetime64[ns]'))[0])
        self.assertEqual(int(microseconds[0]),int(seconds[0])*1_000_000)

    def test_attention_matches_dense_reference_and_backpropagates(self):
        torch.manual_seed(42);layer=GATv2Layer(8,2,3,0.)
        x=torch.randn(4,8,requires_grad=True);edges=torch.tensor([[0,1,2],[1,2,1]])
        attrs=torch.randn(3,3);actual=layer(x,edges,attrs)
        s=torch.cat([edges[0],torch.arange(4)]);d=torch.cat([edges[1],torch.arange(4)])
        a=torch.cat([attrs,torch.zeros(4,3)])
        left=layer.source(x).view(4,2,4);right=layer.destination(x).view(4,2,4)
        logits=(F.leaky_relu(left[s]+right[d]+layer.edge(a).view(-1,2,4),.2)*layer.attention).sum(-1)
        expected=[]
        for target in range(4):
            mask=d==target;weight=torch.softmax(logits[mask],dim=0)
            expected.append((weight[:,:,None]*left[s[mask]]).sum(0).reshape(-1))
        expected=torch.stack(expected)+layer.bias
        torch.testing.assert_close(actual,expected)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(layer.attention.grad.abs().sum()),0.)

    def test_prefix_excludes_unfinished_tied_and_future_flows(self):
        data=GraphData.__new__(GraphData)
        data.meta=pd.DataFrame(dict(second=[0,10,20,30,40],completion_us=[1_000_000,50_000_000,30_000_000,31_000_000,41_000_000],
            src=[0,1,2,0,3],dst=[1,2,0,2,0],source_row=[1,2,3,4,5],partition=[0]*5,
            bytes=[10.,20.,30.,40.,50.],protocol=[6]*5,src_port=[80]*5,dst_port=[80]*5))
        data.times=data.meta.second.to_numpy();data.src=data.meta.src.to_numpy();data.dst=data.meta.dst.to_numpy()
        data.ids=np.array([3]);data.y=np.array([1]);data.x=np.ones((1,3),dtype=np.float32);data.hx=np.ones((5,3),dtype=np.float32)
        group=dict(minute=0,partition='train',start=0,end=1)
        a=data.snapshot(group,1);np.testing.assert_array_equal(a['history_rows'],[0])
        data.meta.loc[4,'bytes']=1e12;data.meta.loc[4,'src']=999;data.src=data.meta.src.to_numpy()
        b=data.snapshot(group,1)
        for key in ('nodes','edges','edge_features','x'): np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__': unittest.main()
