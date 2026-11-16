"""Isolated experimental A readout on frozen features; production PBSLM unchanged.

No M access. E-key vs repeated token-mean uses identical readout parameter count.
This replaces A output in experiments, not a residual production adapter or model
wide Online solution. Existing explicitly aligned E pretraining is a strong prior.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FrozenFeatureReadout(nn.Module):
    def __init__(self,base,feature='e_key'):
        super().__init__()
        if feature not in ('e_key','token_mean'):raise ValueError('unknown feature source')
        if base.training or base.use_mem or any(m.count_nonzero() for m in base.mem):
            raise ValueError('frozen-feature experiment requires eval M-disabled zero-M base')
        self.base=base;self.feature=feature;self.use_mem=False
        for p in base.parameters():p.requires_grad_(False)
        self.readout=nn.Linear(base.e_n*base.d,base.d).to(base.tok.weight.device)
        nn.init.normal_(self.readout.weight,std=.02);nn.init.zeros_(self.readout.bias)
        self.eval()

    @property
    def code(self):return self.base.code

    @property
    def mem(self):return self.base.mem

    def online_parameters(self):return list(self.readout.parameters())

    def features(self,x):
        if self.use_mem or self.base.use_mem or any(m.count_nonzero() for m in self.mem):
            raise ValueError('readout cannot access nonzero/enabled memory')
        with torch.no_grad():
            tokens=self.base.token_emb(x)
            if self.feature=='e_key':
                encoded,_=self.base.e_forward(tokens)
                keys=F.normalize(encoded.mean(2),dim=-1).permute(1,0,2)
            else:
                # Same padded input positions, same E*D dimension/readout capacity.
                keys=F.normalize(tokens.mean(1),dim=-1)[:,None,:].expand(-1,self.base.e_n,-1)
            return keys.reshape(x.shape[0],-1).detach()

    def forward(self,x,m=1,sample_noise=False):
        if m!=1 or sample_noise:raise ValueError('experimental readout supports deterministic m=1 only')
        z=F.normalize(self.readout(self.features(x)).float(),dim=-1)
        return {'z_a':z[:,None,:]}
