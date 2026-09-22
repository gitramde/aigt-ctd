"""Identical capacity across L=1/4/8, with masked historical minute slots."""
import torch
from torch import nn

class TemporalHead(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.projection=nn.Linear(input_dim,64)
        self.position=nn.Parameter(torch.zeros(1,8,64))
        nn.init.normal_(self.position,std=.02)
        layer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=128,dropout=.1,
                                          activation='gelu',batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.classifier=nn.Linear(64,1)

    def forward(self,x,missing):
        # Every unmasked historical token is strictly before current cutoff.
        # Bidirectional attention within this already-causal context introduces no future data.
        z=self.projection(x)+self.position[:,-x.shape[1]:]
        return self.classifier(self.encoder(z,src_key_padding_mask=missing)[:,-1]).squeeze(-1)

def batch(representations,history,positions,length):
    import numpy as np
    positions=np.asarray(positions)
    historical=history[positions,:length-1][:,::-1]
    index=np.concatenate([historical,positions[:,None]],axis=1)
    missing=index<0
    x=np.array(representations[np.maximum(index,0)],dtype=np.float32)
    x[missing]=0
    return torch.from_numpy(x),torch.from_numpy(missing)
