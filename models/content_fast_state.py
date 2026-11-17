"""Continuous content-addressed fast state; no entity-ID lookup/history rows.

M has shape key_dim x value_width. Delta update subtracts prediction at current key;
additive control omits subtraction. Standard fast-weight associative rule, not a
novel algorithm. Persistent activations include aggregate associations, not raw keys.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ContentFastState(nn.Module):
    def __init__(self,key_dim=16,values=8,rule='delta'):
        super().__init__()
        if rule not in ('delta','additive'):raise ValueError('unsupported write rule')
        self.key_dim=key_dim;self.values=values;self.rule=rule
        # Value-ID encoding is known-interface assistance, not language extraction.
        self.value_embedding=nn.Embedding(values,values)
        with torch.no_grad():self.value_embedding.weight.copy_(torch.eye(values))
        self.write_logit=nn.Parameter(torch.tensor(2.))
        self.read=nn.Sequential(nn.Linear(values,48),nn.Tanh(),nn.Linear(48,values+1))

    def initial(self,batch):return self.value_embedding.weight.new_zeros(batch,self.key_dim,self.values)

    def observe(self,state,key,value):
        key=F.normalize(key,dim=-1)
        target=self.value_embedding(value)
        prediction=torch.einsum('bd,bdv->bv',key,state)
        residual=target-prediction if self.rule=='delta' else target
        return state+self.write_logit.sigmoid()*torch.einsum('bd,bv->bdv',key,residual)

    def query(self,state,key):
        read=torch.einsum('bd,bdv->bv',F.normalize(key,dim=-1),state)
        return self.read(read)
