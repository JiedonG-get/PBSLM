"""Fixed-size learned recurrent working state; no raw event/history store.

Symbolic entity/value interface for mechanism tests, NOT language understanding.
Writes update state, queries read state without altering it. Weights trained offline;
eval recurrent activations update but parameters stay fixed.
"""
import torch
from torch import nn


class RecurrentWorkingState(nn.Module):
    def __init__(self,entities=8,values=8,width=96):
        super().__init__()
        self.entities=entities;self.values=values;self.width=width
        self.entity_embedding=nn.Embedding(entities,24)
        self.value_embedding=nn.Embedding(values,24)
        self.update=nn.GRUCell(48,width)
        self.read=nn.Sequential(nn.Linear(width+24,width),nn.Tanh(),nn.Linear(width,values))

    def initial(self,batch,device=None):
        return torch.zeros(batch,self.width,device=device or self.entity_embedding.weight.device)

    def observe(self,state,entities,values):
        return self.update(torch.cat((self.entity_embedding(entities),self.value_embedding(values)),-1),state)

    def query(self,state,entities):
        return self.read(torch.cat((state,self.entity_embedding(entities)),-1))
