"""Learned continuous-key working state, no fixed entity-ID embedding.

Independent diagnostic candidate, not integrated PBSLM. Query receives only current
state and continuous key; class values0..7 and unknown class8. Keys freshly sampled
per episode; state alone persists, no original event/history is retained by model.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ContinuousKeyState(nn.Module):
    def __init__(self,key_dim=16,values=8,width=128):
        super().__init__()
        self.key_dim=key_dim;self.values=values;self.width=width
        self.value_embedding=nn.Embedding(values,24)
        self.update=nn.GRUCell(key_dim+24,width)
        self.read=nn.Sequential(nn.Linear(width+key_dim,width),nn.Tanh(),nn.Linear(width,values+1))

    def initial(self,batch):return self.value_embedding.weight.new_zeros(batch,self.width)

    def observe(self,state,key,value):
        return self.update(torch.cat((F.normalize(key,dim=-1),self.value_embedding(value)),-1),state)

    def query(self,state,key):
        return self.read(torch.cat((state,F.normalize(key,dim=-1)),-1))
