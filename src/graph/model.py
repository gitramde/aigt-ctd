"""Dependency-light GATv2 with edge-aware dynamic attention and CPU scatter."""
import torch
from torch import nn
from torch.nn import functional as F


class GATv2Layer(nn.Module):
    def __init__(self,hidden,heads,edge_dim,dropout):
        super().__init__();self.heads=heads;self.channels=hidden//heads;self.dropout=dropout
        assert hidden%heads==0
        self.source=nn.Linear(hidden,hidden,bias=False)
        self.destination=nn.Linear(hidden,hidden,bias=False)
        self.edge=nn.Linear(edge_dim,hidden,bias=False)
        self.attention=nn.Parameter(torch.empty(heads,self.channels));nn.init.xavier_uniform_(self.attention)
        self.bias=nn.Parameter(torch.zeros(hidden))

    def forward(self,x,edges,edge_features):
        n=len(x);h=self.heads;c=self.channels
        loops=torch.arange(n,device=x.device)
        src=torch.cat([edges[0],loops]);dst=torch.cat([edges[1],loops])
        attrs=torch.cat([edge_features,torch.zeros(n,edge_features.shape[1],device=x.device)])
        source=self.source(x).view(n,h,c);destination=self.destination(x).view(n,h,c)
        logits=(F.leaky_relu(source[src]+destination[dst]+self.edge(attrs).view(-1,h,c),negative_slope=.2)*self.attention).sum(-1)
        index=dst[:,None].expand(-1,h)
        maximum=torch.full((n,h),-torch.inf,device=x.device).scatter_reduce_(0,index,logits.detach(),reduce='amax',include_self=True)
        exp=torch.exp(logits-maximum[dst]);denominator=torch.zeros((n,h),device=x.device).scatter_add_(0,index,exp)
        weights=F.dropout(exp/denominator[dst],p=self.dropout,training=self.training)
        out=torch.zeros((n,h,c),device=x.device).index_add_(0,dst,weights[:,:,None]*source[src])
        return out.reshape(n,h*c)+self.bias


class EdgeModel(nn.Module):
    def __init__(self,config,edge_dim,history_dim):
        super().__init__();self.kind=config['kind'];d=config['hidden'];p=config['dropout']
        if self.kind=='edge_mlp':
            width=config['mlp_width']
            self.classifier=nn.Sequential(nn.Linear(edge_dim,width),nn.GELU(),nn.Dropout(p),
                nn.Linear(width,width),nn.GELU(),nn.Dropout(p),nn.Linear(width,1))
        else:
            self.projection=nn.Linear(8,d)
            self.layers=nn.ModuleList([GATv2Layer(d,config['heads'],history_dim,p) for _ in range(2)])
            self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
            self.dropout=nn.Dropout(p)
            self.classifier=nn.Sequential(nn.Linear(edge_dim+2*d,d),nn.GELU(),nn.Dropout(p),nn.Linear(d,1))

    def forward(self,b):
        if self.kind=='edge_mlp': return self.classifier(b['x']).squeeze(-1)
        x=self.projection(b['nodes']);edges=b['edges'];features=b['edge_features']
        if self.kind=='self_only': edges=edges[:,:0];features=features[:0]
        for layer,norm in zip(self.layers,self.norms): x=norm(x+self.dropout(F.elu(layer(x,edges,features))))
        representation=torch.cat([x[b['target_src']],x[b['target_dst']],b['x']],dim=1)
        return self.classifier(representation).squeeze(-1)


def tensors(snapshot):
    return {k:torch.from_numpy(snapshot[k].copy()).to(dtype=torch.long if k in ('edges','target_src','target_dst') else torch.float32)
            for k in ('nodes','edges','edge_features','target_src','target_dst','x','y')}
