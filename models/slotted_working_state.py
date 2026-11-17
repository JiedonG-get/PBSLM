"""96-float slotted recurrent states; learned and explicit-ID routing references.

Symbolic IDs supplied by task. Learned routing is NOT language entity extraction.
No raw history, all parameters fixed in evaluation, slot activations update.
"""
import torch
from torch import nn
import torch.nn.functional as F


class SlottedWorkingState(nn.Module):
    def __init__(self,entities=8,values=8,slots=8,slot_width=12,*,addressing='learned'):
        super().__init__()
        if addressing not in ('learned','explicit'):raise ValueError('unknown addressing mode')
        if addressing=='explicit' and slots!=entities:raise ValueError('explicit slots must match entities')
        self.entities=entities;self.values=values;self.slots=slots;self.slot_width=slot_width
        self.width=slots*slot_width;self.addressing=addressing
        self.entity_embedding=nn.Embedding(entities,24);self.value_embedding=nn.Embedding(values,24)
        self.keys=nn.Parameter(torch.randn(slots,24)*.1)
        self.cell=nn.GRUCell(48,slot_width)
        self.read=nn.Sequential(nn.Linear(slot_width+24,48),nn.Tanh(),nn.Linear(48,values))

    def initial(self,batch,device=None):
        return torch.zeros(batch,self.slots,self.slot_width,device=device or self.keys.device)

    def routing(self,entities):
        if self.addressing=='explicit':return F.one_hot(entities,self.slots).to(self.keys.dtype)
        return (self.entity_embedding(entities)@self.keys.T).softmax(-1)

    def observe(self,state,entities,values):
        inputs=torch.cat((self.entity_embedding(entities),self.value_embedding(values)),-1)
        candidate=self.cell(inputs[:,None,:].expand(-1,self.slots,-1).reshape(-1,48),state.reshape(-1,self.slot_width)).reshape_as(state)
        weights=self.routing(entities)[:,:,None]
        return state+weights*(candidate-state)

    def query(self,state,entities):
        pooled=torch.einsum('bs,bsd->bd',self.routing(entities),state)
        return self.read(torch.cat((pooled,self.entity_embedding(entities)),-1))
